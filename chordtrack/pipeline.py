"""Ghép pipeline end-to-end: nguồn audio (videoId, URL, file) -> JSON hợp âm.

    fetch_audio -> chuẩn hóa thành mix.wav (cache ở CACHE_DIR/<song_id>/) -> track_beats
    -> run_chordmini + run_cnnlstm -> fusion.fuse -> analysis.estimate_key, analyze_structure,
    estimate_key_regions -> JSON (xem `build_result`)

Mặc định **không tách nguồn**: model hợp âm và bắt nhịp chạy trên mix, bass lấy từ `bass_probs` của
CNN-LSTM. Tách nguồn (BS-Roformer + HTDemucs) chỉ thêm khoảng +0.1 điểm hợp âm đảo mà tốn khoảng
84 trên 89 giây mỗi bài (PLAN.md, "Kết quả đánh giá Kajiura"), nên chỉ bật khi `use_separation=True`;
khi đó stem bass được trộn thêm vào phân bố bass (`bass_source="both"`).

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
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import soundfile as sf

from chordtrack import analysis, fusion, paths
from chordtrack import audio as audio_mod
from chordtrack import separation, symbols, transpose
from chordtrack.audio import read_json, write_json
from chordtrack.beats import track_beats
from chordtrack.chords.chordmini import run_chordmini
from chordtrack.chords.cnnlstm import run_cnnlstm
from chordtrack.gpu import free_vram
from chordtrack.symbols import harte_to_symbol
from chordtrack.types import BeatGrid, ChordModelOutput, ChordSegment, KeyEstimate, KeyRegion, Section, Stems

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
    for mod in (fusion, analysis, symbols, transpose, chordtrack.beats, chordtrack.chords.chordmini,
                chordtrack.chords.cnnlstm):
        h.update(Path(mod.__file__).read_bytes())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()[:12]


@dataclass(frozen=True)
class PipelineConfig:
    use_separation: bool = False       # tách nguồn: tắt mặc định (PLAN.md, mục 2)
    chord_stem: str = "mix"            # stem chạy model hợp âm. Mix tốt hơn instrumental (PLAN.md, kết quả đợt 2)
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
    use_separation: bool = False,
    chord_stem: str = "mix",
    beat_stem: str = "mix",
    progress: Callable[[str, float], None] | None = None,
) -> dict:
    """Phân tích một bài, trả về dict JSON (xem `build_result`).

    `progress(step, fraction)`: nếu có, được gọi ở đầu (0.0) và cuối (1.0) mỗi bước
    ("fetch", "separation", "beats", "chords", "fusion", "analysis"). Có cache thì không gọi.
    """
    report = progress or (lambda step, fraction: None)
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

    report("fetch", 0.0)
    t0 = time.perf_counter()
    src = audio_mod.fetch(source)
    timing["fetch"] = time.perf_counter() - t0
    report("fetch", 1.0)

    cache_dir = paths.CACHE_DIR / src.song_id
    result_path = cache_dir / config.result_filename()
    if not force:
        cached = _cached_result(src.song_id, config)
        if cached is not None:
            return cached

    # Tách nguồn (hoặc chỉ chuẩn hóa audio khi không tách)
    report("separation", 0.0)
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
    report("separation", 1.0)

    duration = float(sf.info(str(stems.mix)).duration)

    report("beats", 0.0)
    t0 = time.perf_counter()
    grid, vram["beats"] = _measure_vram(track_beats, getattr(stems, config.beat_stem))
    timing["beats"] = time.perf_counter() - t0
    free_vram()
    report("beats", 1.0)

    report("chords", 0.0)
    t0 = time.perf_counter()
    chord_audio = getattr(stems, config.chord_stem)
    models: dict[str, ChordModelOutput] = {}
    models["chordmini"], vram["chordmini"] = _measure_vram(run_chordmini, chord_audio)
    free_vram()
    models["cnnlstm"], vram["cnnlstm"] = _measure_vram(run_cnnlstm, chord_audio)
    free_vram()
    timing["chords"] = time.perf_counter() - t0
    report("chords", 1.0)

    report("fusion", 0.0)
    t0 = time.perf_counter()
    bass_audio, fusion_params = _fusion_inputs(stems, config)
    segments = fusion.fuse(models, grid, bass_audio, duration, fusion_params)
    timing["fusion"] = time.perf_counter() - t0
    report("fusion", 1.0)

    report("analysis", 0.0)
    t0 = time.perf_counter()
    key = analysis.estimate_key(segments)
    sections = analysis.analyze_structure(segments, grid, duration, key)
    regions = analysis.estimate_key_regions(segments, grid, duration, [s.start for s in sections[1:]])
    key = main_key(key, regions)
    timing["analysis"] = time.perf_counter() - t0
    report("analysis", 1.0)
    timing["total"] = time.perf_counter() - t_start

    result = build_result(
        src, duration=duration, grid=grid, segments=segments, key=key, sections=sections,
        timing=timing, config=config, peak_vram_mb=vram, regions=regions,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_json(result_path, result)
    return result


def _fusion_inputs(stems: Stems, config: PipelineConfig) -> tuple[Path | None, fusion.FusionParams]:
    """(file stem bass, tham số fuse). Không tách nguồn: không có stem, bass chỉ từ CNN-LSTM. Có tách
    nguồn: trộn thêm chroma stem bass với trọng số `stem_weight`."""
    if not config.use_separation:
        return None, fusion.DEFAULT_PARAMS
    return stems.bass, replace(fusion.DEFAULT_PARAMS, bass_source="both")


def _cached_result(song_id: str, config: PipelineConfig) -> dict | None:
    path = paths.CACHE_DIR / song_id / config.result_filename()
    cached = read_json(path)
    if cached is None or cached.get("version") != RESULT_VERSION:
        return None
    meta = cached.get("pipeline") or {}
    if meta.get("code_revision") != code_revision():
        log.info("%s tạo bởi mã cũ, chạy lại", path)
        return None
    # Phòng khi tên file trùng nhưng cấu hình khác (vd. result.json do bản cũ ghi khi mặc định còn tách nguồn).
    if ((meta.get("separation_model") is not None) != config.use_separation
            or meta.get("chord_stem") != config.chord_stem or meta.get("beat_stem") != config.beat_stem):
        log.info("%s tạo với cấu hình khác, chạy lại", path)
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


def chord_to_json(seg: ChordSegment, key: KeyEstimate | None, key_index: int | None = None) -> dict:
    """Một hợp âm. `key` là giọng của vùng chứa hợp âm (chỉ số `key_index` trong `keys`);
    bậc La Mã và ký hiệu phím trắng tính theo giọng đó."""
    is_chord = seg.label not in ("N", "X")
    return {
        "start": round(float(seg.start), 3),
        "end": round(float(seg.end), 3),
        "label": seg.label,
        "symbol": harte_to_symbol(seg.label, key),
        "roman": analysis.to_roman(seg.label, key) if key is not None and is_chord else None,
        "white_key_symbol": transpose.white_key_symbol(seg.label, key) if key is not None else None,
        "key_index": key_index,
        "confidence": _confidence(seg.confidence),
        "uncertain": _is_uncertain(seg.confidence),
    }


def section_to_json(sec: Section, key: KeyEstimate | None, info: dict | None = None) -> dict:
    """Một đoạn. `info` là phần tử tương ứng của `analysis.relate_sections_to_keys`: khi có thì
    `roman` tính theo giọng của vùng chứa đoạn và có thêm `key_index`, `transposed_from`, `semitones`."""
    out = {
        "start": round(float(sec.start), 3),
        "end": round(float(sec.end), 3),
        "label": sec.label,
        "progression": [harte_to_symbol(label, key) for label in sec.progression],
        "roman": list(sec.roman) if key is not None else None,
    }
    if info is not None:
        out["roman"] = list(info["roman"]) if info["key_index"] is not None else None
        out["key_index"] = info["key_index"]
        if info["transposed_from"] is not None:
            out["transposed_from"] = info["transposed_from"]
            out["semitones"] = info["semitones"]
    return out


def key_region_to_json(region: KeyRegion) -> dict:
    k = region.key
    return {
        "start": round(float(region.start), 3),
        "end": round(float(region.end), 3),
        "tonic": k.tonic,
        "mode": k.mode,
        "confidence": _confidence(k.confidence),
        "white_key_transpose": transpose.white_key_offset(k),
    }


def modulations_to_json(regions: list[KeyRegion]) -> list[dict]:
    """Chỗ đổi giọng giữa hai vùng liền nhau. `semitones`: khoảng dịch nốt chủ trong [-6, +5]
    (C trưởng -> D trưởng = +2, C trưởng -> A thứ = -3); `transpose_from/to`: giá trị transpose
    phím trắng trước và sau chỗ đổi."""
    out = []
    for a, b in zip(regions, regions[1:]):
        out.append({
            "time": round(float(b.start), 3),
            "from": analysis.key_to_mir_eval(a.key),
            "to": analysis.key_to_mir_eval(b.key),
            "semitones": transpose.wrap_semitones(analysis.key_tonic_pc(b.key) - analysis.key_tonic_pc(a.key)),
            "transpose_from": transpose.white_key_offset(a.key),
            "transpose_to": transpose.white_key_offset(b.key),
        })
    return out


def main_key(whole: KeyEstimate | None, regions: list[KeyRegion]) -> KeyEstimate | None:
    """Giọng chính (`key` trong JSON): `estimate_key` trên cả bài (đúng hơn vùng dài nhất trên
    Isophonics, 0.870 so với 0.860). Nhưng nếu bài có nhiều vùng mà giọng đó không thuộc vùng nào
    (giọng "dung hòa", ví dụ G khi bài đổi từ C sang D) thì lấy giọng có tổng thời lượng lớn nhất."""
    if whole is None or len(regions) < 2:
        return whole
    pc = analysis.key_tonic_pc(whole)
    if any(analysis.key_tonic_pc(r.key) == pc and r.key.mode == whole.mode for r in regions):
        return whole
    return analysis.main_region_key(regions)


def _relabel_transposed(sections_json: list[dict]) -> None:
    """Đoạn nâng tông lấy nhãn của đoạn gốc (ví dụ chorus cuối nâng +2 vẫn là "chorus").

    Các đoạn khác mang cùng nhãn cũ với đoạn nâng tông (cùng cụm A/B/C) cũng đổi theo, miễn là
    không đoạn nào mang nhãn cũ đó nằm ở vùng giọng của đoạn gốc."""
    renames: dict[str, str] = {}
    for s in sections_json:
        src = s.get("transposed_from")
        if src is None:
            continue
        old, new = s["label"], sections_json[src]["label"]
        s["label"] = new
        src_key = sections_json[src].get("key_index")
        if all(o.get("key_index") != src_key for o in sections_json if o["label"] == old):
            renames[old] = new
    for s in sections_json:
        if s["label"] in renames:
            s["label"] = renames[s["label"]]


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
    regions: list[KeyRegion] | None = None,
) -> dict:
    """Dựng JSON kết quả. `regions` (vùng giọng) None hoặc rỗng thì mọi hợp âm và đoạn dùng `key`."""
    regions = list(regions or [])
    chord_idx = [analysis.region_index(regions, s.start, s.end) for s in segments]
    chords = [chord_to_json(s, regions[i].key if i is not None else key, i) for s, i in zip(segments, chord_idx)]
    if regions:
        infos = analysis.relate_sections_to_keys(sections, regions)
        secs = [section_to_json(s, regions[inf["key_index"]].key if inf["key_index"] is not None else key, inf)
                for s, inf in zip(sections, infos)]
        _relabel_transposed(secs)
    else:
        secs = [section_to_json(s, key) | {"key_index": None} for s in sections]
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
        "keys": [key_region_to_json(r) for r in regions],
        "modulations": modulations_to_json(regions),
        "chords": chords,
        "sections": secs,
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

