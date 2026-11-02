"""Ghép pipeline end-to-end: nguồn audio (videoId, URL, file) -> JSON hợp âm.

    fetch_audio -> separate (cache ở CACHE_DIR/<song_id>/) -> track_beats
    -> run_chordmini + run_cnnlstm -> fusion.fuse -> analysis.estimate_key, analyze_structure
    -> JSON (xem `build_result`)

Mỗi bước dùng GPU tự lấy `gpu_lock()` riêng bên trong hàm của nó (separate, track_beats,
run_chordmini, run_cnnlstm), nên ở đây không lồng khóa; giữa các bước gọi `free_vram()`.
Kết quả cuối cache ở `CACHE_DIR/<song_id>/result.json` (cấu hình mặc định) hoặc
`result-<hash cấu hình>.json` (cấu hình khác), có rồi thì trả lại luôn, trừ khi `force`.
"""
from __future__ import annotations

import functools
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import soundfile as sf

from chordtrack import analysis, fusion, paths
from chordtrack import audio as audio_mod
from chordtrack import separation, symbols
from chordtrack.audio import read_json, write_json
from chordtrack.beats import track_beats
from chordtrack.chords.chordmini import run_chordmini
from chordtrack.chords.cnnlstm import run_cnnlstm
from chordtrack.gpu import free_vram
from chordtrack.symbols import harte_to_symbol
from chordtrack.types import BeatGrid, ChordModelOutput, ChordSegment, KeyEstimate, Section, Stems

log = logging.getLogger(__name__)

RESULT_VERSION = 1
UNCERTAIN_THRESHOLD = 0.5   # hợp âm có confidence nhỏ hơn mức này bị đánh dấu "uncertain"
STEM_NAMES = ("mix", "vocals", "instrumental", "bass", "no_bass")
CHORD_MODELS = ("chordmini-btc", "cnnlstm")
TIMING_KEYS = ("fetch", "separation", "beats", "chords", "fusion", "analysis", "total")


@functools.lru_cache(maxsize=1)
def code_revision() -> str:
    """Hash mã nguồn các bước sau tách nguồn. Mã đổi thì result.json cũ không dùng lại nữa
    (stem vẫn dùng lại vì nằm ở cache riêng)."""
    import chordtrack.beats
    import chordtrack.chords.chordmini
    import chordtrack.chords.cnnlstm

    h = hashlib.sha1()
    for mod in (fusion, analysis, symbols, chordtrack.beats, chordtrack.chords.chordmini,
                chordtrack.chords.cnnlstm):
        h.update(Path(mod.__file__).read_bytes())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()[:12]


@dataclass(frozen=True)
class PipelineConfig:
    use_separation: bool = True
    chord_stem: str = "instrumental"   # stem chạy model hợp âm
    beat_stem: str = "mix"             # stem chạy bắt nhịp

    def __post_init__(self):
        for name in (self.chord_stem, self.beat_stem):
            if name not in STEM_NAMES:
                raise ValueError(f"stem phải là một trong {STEM_NAMES}, nhận được {name!r}")

    def effective(self) -> "PipelineConfig":
        """Không tách nguồn thì mọi bước chạy trên mix."""
        if self.use_separation:
            return self
        return PipelineConfig(use_separation=False, chord_stem="mix", beat_stem="mix")

    def result_filename(self) -> str:
        if self.effective() == PipelineConfig():
            return "result.json"
        digest = hashlib.sha1(json.dumps(asdict(self.effective()), sort_keys=True).encode()).hexdigest()[:8]
        return f"result-{digest}.json"


def analyze(
    source: str,
    *,
    force: bool = False,
    use_separation: bool = True,
    chord_stem: str = "instrumental",
    beat_stem: str = "mix",
) -> dict:
    """Phân tích một bài, trả về dict JSON (xem `build_result`)."""
    config = PipelineConfig(use_separation, chord_stem, beat_stem).effective()
    timing: dict[str, float] = {}
    vram: dict[str, float | None] = {}
    t_start = time.perf_counter()

    # YouTube: biết song_id từ videoId nên xem cache trước khi lấy audio (không gọi mạng).
    if not force and not Path(source).expanduser().is_file():
        video_id = audio_mod.parse_video_id(source)
        cached = _cached_result(video_id, config) if video_id else None
        if cached is not None:
            return cached

    t0 = time.perf_counter()
    src = audio_mod.fetch(source)
    timing["fetch"] = time.perf_counter() - t0

    cache_dir = paths.CACHE_DIR / src.song_id
    result_path = cache_dir / config.result_filename()
    if not force:
        cached = _cached_result(src.song_id, config)
        if cached is not None:
            return cached

    # Tách nguồn (hoặc chỉ chuẩn hóa audio khi không tách)
    t0 = time.perf_counter()
    stats_path = cache_dir / separation.STATS_FILE
    stats_before = read_json(stats_path) or {}
    stems = _prepare_stems(src.path, cache_dir, config)
    timing["separation"] = time.perf_counter() - t0
    if config.use_separation:
        # Chỉ báo VRAM của bước thật sự chạy lần này, không lấy số cũ của stem đã cache.
        stats_after = read_json(stats_path) or {}
        for step in ("roformer", "demucs"):
            ran = stats_after.get(step) is not None and stats_after.get(step) != stats_before.get(step)
            vram[step] = stats_after[step].get("peak_vram_mb") if ran else None
    free_vram()

    duration = float(sf.info(str(stems.mix)).duration)

    t0 = time.perf_counter()
    grid, vram["beats"] = _measure_vram(track_beats, getattr(stems, config.beat_stem))
    timing["beats"] = time.perf_counter() - t0
    free_vram()

    t0 = time.perf_counter()
    chord_audio = getattr(stems, config.chord_stem)
    models: dict[str, ChordModelOutput] = {}
    models["chordmini"], vram["chordmini"] = _measure_vram(run_chordmini, chord_audio)
    free_vram()
    models["cnnlstm"], vram["cnnlstm"] = _measure_vram(run_cnnlstm, chord_audio)
    free_vram()
    timing["chords"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    bass_audio = stems.bass if config.use_separation else None
    segments = fusion.fuse(models, grid, bass_audio, duration)
    timing["fusion"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    key = analysis.estimate_key(segments)
    sections = analysis.analyze_structure(segments, grid, duration, key)
    timing["analysis"] = time.perf_counter() - t0
    timing["total"] = time.perf_counter() - t_start

    result = build_result(
        src, duration=duration, grid=grid, segments=segments, key=key, sections=sections,
        timing=timing, config=config, peak_vram_mb=vram,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_json(result_path, result)
    return result


def _cached_result(song_id: str, config: PipelineConfig) -> dict | None:
    path = paths.CACHE_DIR / song_id / config.result_filename()
    cached = read_json(path)
    if cached is None or cached.get("version") != RESULT_VERSION:
        return None
    if (cached.get("pipeline") or {}).get("code_revision") != code_revision():
        log.info("%s tạo bởi mã cũ, chạy lại", path)
        return None
    log.info("Dùng kết quả có sẵn %s", path)
    return cached


def _prepare_stems(audio_path: Path, cache_dir: Path, config: PipelineConfig) -> Stems:
    if config.use_separation:
        return separation.separate(audio_path, cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    mix = cache_dir / "mix.wav"
    if not mix.exists():
        separation.normalize_audio(audio_path, mix)
    # Không tách: chỉ có mix. Các stem khác trỏ về mix để tham số stem vẫn hợp lệ.
    return Stems(mix=mix, vocals=mix, instrumental=mix, bass=mix, no_bass=mix)


def _measure_vram(fn, *args):
    """Chạy fn(*args), trả về (kết quả, VRAM đỉnh MiB của tiến trình này hoặc None)."""
    try:
        import torch

        cuda = torch.cuda.is_available()
    except ImportError:
        cuda = False
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    out = fn(*args)
    peak = round(torch.cuda.max_memory_allocated() / 2**20, 1) if cuda else None
    return out, peak


def _confidence(c: float | None) -> float | None:
    return None if c is None else round(float(c), 3)


def _is_uncertain(c: float | None) -> bool:
    # Không có confidence (vd. đoạn "N" lấp khoảng trống) thì coi là không chắc.
    return c is None or c < UNCERTAIN_THRESHOLD


def chord_to_json(seg: ChordSegment, key: KeyEstimate | None) -> dict:
    return {
        "start": round(float(seg.start), 3),
        "end": round(float(seg.end), 3),
        "label": seg.label,
        "symbol": harte_to_symbol(seg.label, key),
        "roman": analysis.to_roman(seg.label, key) if key is not None and seg.label not in ("N", "X") else None,
        "confidence": _confidence(seg.confidence),
        "uncertain": _is_uncertain(seg.confidence),
    }


def section_to_json(sec: Section, key: KeyEstimate | None) -> dict:
    return {
        "start": round(float(sec.start), 3),
        "end": round(float(sec.end), 3),
        "label": sec.label,
        "progression": [harte_to_symbol(label, key) for label in sec.progression],
        "roman": list(sec.roman) if key is not None else None,
    }


def build_result(
    src: audio_mod.AudioSource,
    *,
    duration: float,
    grid: BeatGrid | None,
    segments: list[ChordSegment],
    key: KeyEstimate | None,
    sections: list[Section],
    timing: dict[str, float],
    config: PipelineConfig,
    peak_vram_mb: dict[str, Any] | None = None,
) -> dict:
    return {
        "version": RESULT_VERSION,
        "song_id": src.song_id,
        "source": src.to_json(),
        "duration": round(duration, 3),
        "tempo": round(float(grid.tempo), 1) if grid is not None else None,
        "key": None if key is None else {
            "tonic": key.tonic, "mode": key.mode, "confidence": _confidence(key.confidence),
        },
        "beats": [round(float(t), 3) for t in (grid.beats if grid is not None else [])],
        "downbeats": [round(float(t), 3) for t in (grid.downbeats if grid is not None else [])],
        "chords": [chord_to_json(s, key) for s in segments],
        "sections": [section_to_json(s, key) for s in sections],
        "timing": {k: round(float(timing.get(k, 0.0)), 2) for k in TIMING_KEYS},
        "pipeline": {
            "separation_model": separation.ROFORMER_MODEL if config.use_separation else None,
            "bass_model": separation.DEMUCS_MODEL if config.use_separation else None,
            "chord_models": list(CHORD_MODELS),
            "chord_stem": config.chord_stem,
            "beat_stem": config.beat_stem,
            "uncertain_threshold": UNCERTAIN_THRESHOLD,
            "peak_vram_mb": dict(peak_vram_mb or {}),
            "code_revision": code_revision(),
        },
    }

