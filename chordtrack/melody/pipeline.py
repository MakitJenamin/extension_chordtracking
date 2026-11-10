"""Dò giai điệu chính của một bài cho server: nguồn audio -> `<Melody>` (hợp đồng ở `docs/melody_api.md`).

    fetch (dùng lại CACHE_DIR/<song_id>/mix.wav nếu có) -> YourMT3+ (tiến trình riêng) -> MuScriptor small
    -> ghép theo cổng giọng (`build_line`) -> CACHE_DIR/<song_id>/melody.json

Cách ghép `yourmt3>muscriptor_small` thắng xếp hạng chung trong `eval/melody/score.py` (PLAN.md mục 11): dòng
YourMT3+ (`source_viterbi`: ưu tiên giọng + Viterbi) trong các khoảng có nốt `voice` của chính YourMT3+ nới
`GATE_PAD` giây, ngoài cổng dùng dòng MuScriptor small (`salience`, không tính độ dài nốt). Cấu hình là hằng số ở
đây, chép từ kết quả CV (`data/eval/results/melody/cv_choices.json`, khóa `yourmt3>muscriptor_small|0`), không
đọc file eval lúc chạy. Test `tests/test_melody_pipeline.py` kiểm tra `build_line` ra đúng dòng mà
`eval/melody/score.py` chấm.

YourMT3+ cần transformers 4.45.1 riêng (`third_party_src/YourMT3_pydeps`), không nạp chung tiến trình với
huggingface_hub 2.x, nên chạy bằng `python -m chordtrack.melody.yourmt3 <wav> <out.npz>`: tiến trình con tự lấy
`gpu_lock()` và thoát khi xong, nên không giữ VRAM. MuScriptor chạy trong tiến trình server: model giữ trên CPU,
chỉ lên GPU trong `gpu_lock()` (xem `chordtrack/melody/muscriptor.py`).

Đầu ra thô của hai model cache ở `CACHE_DIR/<song_id>/melody_raw/<model>.npz` (kèm hash mã wrapper), nên khi chỉ
đổi cách ghép thì chạy lại mất khoảng một giây. `force=True` chạy lại cả hai model.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

from chordtrack import audio as audio_mod
from chordtrack import paths
from chordtrack.audio import read_json, write_json
from chordtrack.melody import line as L
from chordtrack.types import MelodyModelOutput, MelodyNote

log = logging.getLogger(__name__)

MODEL = "yourmt3>muscriptor_small"
RESULT_FILE = "melody.json"
RESULT_VERSION = 1
STEPS = ("fetch", "melody_voice", "melody_fill", "melody_line")
MUSCRIPTOR_SIZE = "small"

# Cấu hình chọn bằng CV (cả 3 phần chọn giống nhau): a=source_viterbi b=salience_nodur gate=a pad=0.6.
VOICE_PARAMS = L.LineParams(strategy="viterbi", source_mode="strict")
FILL_PARAMS = L.LineParams(strategy="salience", source_mode="bonus", w_conf=1.0, w_dur=0.0, w_pitch=0.5)
GATE_PAD = 0.6
# Sửa nhảy quãng tám (`line.fix_octave_jumps`); None = tắt. Chọn bằng CV (họ `nn_gated_oct` trong
# eval/melody/score.py, 2/3 phần chọn): trung bình cân bằng 47.51 -> 47.55, Kajiura 3.82 -> 1.42 bước nhảy
# > 1 quãng tám mỗi phút (YourMT3+ một mình 1.65).
OCTAVE_FIX: dict | None = {"max_run": 0.6, "min_jump": 12}
# Nốt có confidence nhỏ hơn ngưỡng này bị đánh dấu `uncertain` (hiện mờ). Confidence của dòng = confidence của nốt
# gốc (YourMT3+ không có nên lấy 0.8; MuScriptor là xác suất token cao độ) nhân hệ số phần nốt gốc được giữ
# (`line.notes_to_line`), nên khoảng 85% nốt có 0.8 trở lên. Ở ngưỡng 0.6 trên dòng ghép (dữ liệu eval, 2026-11-11):
# MIR-ST500 mờ 9.4% nốt, nốt mờ đúng cao độ 7.3% so với 53.1% ở nốt rõ; ADC2004 mờ 4.6% (17% / 49%); MIREX05 5.4%
# (9% / 42%); Kajiura mờ 4.6%. Ngưỡng 0.5 chỉ làm mờ khoảng 1%, ngưỡng 0.8 làm mờ 12-18% kể cả nhiều nốt đúng.
UNCERTAIN_THRESHOLD = 0.6

YOURMT3_TIMEOUT = 3600.0   # giây; gồm cả thời gian chờ gpu_lock (tối đa 30 phút) của tiến trình con


def _file_hash(*files: Path) -> str:
    h = hashlib.sha1()
    for f in files:
        h.update(Path(f).read_bytes())
    return h.hexdigest()[:12]


@functools.lru_cache(maxsize=1)
def code_revision() -> str:
    """Hash mã các bước dò giai điệu. Mã đổi thì melody.json cũ không dùng lại (đầu ra thô của model thì vẫn
    dùng lại nếu mã wrapper của model đó không đổi)."""
    from chordtrack.melody import muscriptor, yourmt3

    return _file_hash(Path(__file__), Path(L.__file__), Path(yourmt3.__file__), Path(muscriptor.__file__))


@functools.lru_cache(maxsize=None)
def _wrapper_revision(model: str) -> str:
    from chordtrack.melody import muscriptor, yourmt3

    mod = yourmt3 if model == "yourmt3" else muscriptor
    return _file_hash(Path(mod.__file__))


# --------------------------------------------------------------------------- ghép dòng


def build_line(voice: MelodyModelOutput, fill: MelodyModelOutput) -> list[MelodyNote]:
    """Dòng giai điệu đơn âm từ đầu ra YourMT3+ (`voice`) và MuScriptor small (`fill`)."""
    line_a = L.notes_to_line(voice.notes, VOICE_PARAMS)
    line_b = L.notes_to_line(fill.notes, FILL_PARAMS)
    gate = L.voice_intervals(voice.notes, pad=GATE_PAD)
    line = L.combine_gated(line_a, line_b, gate)
    if OCTAVE_FIX:
        line = L.fix_octave_jumps(line, **OCTAVE_FIX)
    return line


def note_to_json(n: MelodyNote) -> dict:
    conf = 0.0 if n.confidence is None else float(np.clip(n.confidence, 0.0, 1.0))
    return {"start": round(float(n.start), 3), "end": round(float(n.end), 3), "pitch": int(n.pitch),
            "confidence": round(conf, 3), "uncertain": conf < UNCERTAIN_THRESHOLD, "source": n.source}


def build_result(song_id: str, duration: float, notes: list[MelodyNote], timing: dict | None = None,
                 peak_vram_mb: dict | None = None) -> dict:
    return {
        "version": RESULT_VERSION,
        "song_id": song_id,
        "model": MODEL,
        "code_revision": code_revision(),
        "duration": round(float(duration), 3),
        "uncertain_threshold": UNCERTAIN_THRESHOLD,
        "notes": [note_to_json(n) for n in notes],
        "timing": {k: round(v, 2) for k, v in (timing or {}).items()},
        "peak_vram_mb": peak_vram_mb or {},
    }


# --------------------------------------------------------------------------- cache


def result_path(song_id: str) -> Path:
    return paths.CACHE_DIR / song_id / RESULT_FILE


def cached_melody(song_id: str) -> dict | None:
    """`<Melody>` trong cache nếu đúng phiên bản và `code_revision`, ngược lại None."""
    data = read_json(result_path(song_id))
    if data is None or data.get("version") != RESULT_VERSION or data.get("code_revision") != code_revision():
        return None
    return data


def stale_melody(song_id: str) -> dict | None:
    """`<Melody>` trong cache bất kể `code_revision` (để extension hiện tạm kết quả của mã cũ), miễn đúng phiên bản
    định dạng."""
    data = read_json(result_path(song_id))
    return data if data is not None and data.get("version") == RESULT_VERSION else None


def _raw_path(cache_dir: Path, model: str) -> Path:
    return cache_dir / "melody_raw" / f"{model}.npz"


def _load_raw(path: Path, model: str) -> MelodyModelOutput | None:
    from chordtrack.melody.io import load_output

    if not path.exists():
        return None
    try:
        out = load_output(path)
    except Exception as e:  # file hỏng: chạy lại model
        log.warning("Không đọc được %s: %s", path, e)
        return None
    rev = out.extras.get("wrapper_rev")
    return out if rev is not None and str(rev) == _wrapper_revision(model) else None


def _save_raw(out: MelodyModelOutput, path: Path, model: str) -> None:
    from chordtrack.melody.io import save_output

    out.extras["wrapper_rev"] = np.array(_wrapper_revision(model))
    save_output(out, path)


# --------------------------------------------------------------------------- chạy model


def run_yourmt3(audio_path: Path, out_path: Path) -> MelodyModelOutput:
    """Chạy YourMT3+ ở tiến trình con (`python -m chordtrack.melody.yourmt3`), đọc kết quả từ `out_path`."""
    from chordtrack.melody.io import load_output

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.stem + ".part.npz")
    tmp.unlink(missing_ok=True)
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env["PYTHONPATH"] = os.pathsep.join([str(root)] + [p for p in [env.get("PYTHONPATH")] if p])
    # --exit-with-parent: tiến trình con đọc stdin và tự thoát khi pipe đóng, tức khi server (tiến trình cha) tắt
    # giữa chừng, để không còn tiến trình mồ côi giữ gpu_lock.
    cmd = [sys.executable, "-m", "chordtrack.melody.yourmt3", str(audio_path), str(tmp), "--exit-with-parent"]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    import tempfile

    with tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
        proc = subprocess.Popen(cmd, cwd=str(root), env=env, stdin=subprocess.PIPE, stdout=out_f, stderr=err_f,
                                creationflags=flags)
        try:
            returncode = proc.wait(timeout=YOURMT3_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise
        finally:
            if proc.stdin is not None:
                proc.stdin.close()
        out_f.seek(0)
        err_f.seek(0)
        stdout = out_f.read().decode("utf-8", "replace")
        stderr = err_f.read().decode("utf-8", "replace")
    if returncode != 0 or not tmp.exists():
        tail = " ".join((stderr or stdout).strip().splitlines()[-6:])
        raise RuntimeError(f"YourMT3+ lỗi (mã thoát {returncode}): {tail[-800:]}")
    log.info("YourMT3+: %s", stdout.strip())
    out = load_output(tmp)
    tmp.unlink(missing_ok=True)
    return out


def run_muscriptor(audio_path: Path) -> MelodyModelOutput:
    """MuScriptor small trong tiến trình hiện tại (hàm tự lấy `gpu_lock()`, model trả về CPU khi xong)."""
    from chordtrack.melody import muscriptor

    return muscriptor.transcribe(audio_path, MUSCRIPTOR_SIZE)


def _prepare_audio(source: str) -> tuple[str, Path]:
    """(song_id, mix.wav). YouTube: dùng lại `CACHE_DIR/<id>/mix.wav` (do phân tích hợp âm tạo) nếu có, khỏi gọi
    mạng; chưa có thì tải và chuẩn hóa như `pipeline._prepare_stems`."""
    from chordtrack import separation

    local = Path(source).expanduser()
    video_id = None if local.is_file() else audio_mod.parse_video_id(source)
    if video_id is not None:
        mix = paths.CACHE_DIR / video_id / "mix.wav"
        if mix.exists():
            return video_id, mix
    src = audio_mod.fetch(source)
    mix = paths.CACHE_DIR / src.song_id / "mix.wav"
    if not mix.exists():
        mix.parent.mkdir(parents=True, exist_ok=True)
        separation.normalize_audio(src.path, mix)
    return src.song_id, mix


def analyze_melody(source: str, *, force: bool = False,
                   progress: Callable[[str, float], None] | None = None) -> dict:
    """Dò giai điệu một bài (videoId, URL YouTube hoặc file), trả `<Melody>` (dict JSON) và ghi cache.

    `progress(step, fraction)` được gọi ở đầu (0.0) và cuối (1.0) mỗi bước trong `STEPS`. Có cache thì không gọi.
    """
    report = progress or (lambda step, fraction: None)
    if not force and not Path(source).expanduser().is_file():
        vid = audio_mod.parse_video_id(source)
        cached = cached_melody(vid) if vid else None
        if cached is not None:
            return cached

    timing: dict[str, float] = {}
    vram: dict[str, float | None] = {}
    t_start = time.perf_counter()

    report("fetch", 0.0)
    t0 = time.perf_counter()
    song_id, mix = _prepare_audio(source)
    timing["fetch"] = time.perf_counter() - t0
    report("fetch", 1.0)
    if not force:
        cached = cached_melody(song_id)
        if cached is not None:
            return cached
    cache_dir = paths.CACHE_DIR / song_id

    # Bước dùng lại đầu ra thô có sẵn chỉ báo xong (không báo bắt đầu), để thời gian gần 0 của nó không được ghi
    # vào thống kê ước tính thời gian của job.
    t0 = time.perf_counter()
    raw_a = _raw_path(cache_dir, "yourmt3")
    voice = None if force else _load_raw(raw_a, "yourmt3")
    if voice is None:
        report("melody_voice", 0.0)
        voice = run_yourmt3(mix, raw_a)
        _save_raw(voice, raw_a, "yourmt3")
    timing["melody_voice"] = time.perf_counter() - t0
    vram["yourmt3"] = _num(voice.extras.get("peak_vram_mib"))
    report("melody_voice", 1.0)

    t0 = time.perf_counter()
    raw_b = _raw_path(cache_dir, "muscriptor_small")
    fill = None if force else _load_raw(raw_b, "muscriptor_small")
    if fill is None:
        report("melody_fill", 0.0)
        fill = run_muscriptor(mix)
        _save_raw(fill, raw_b, "muscriptor_small")
    timing["melody_fill"] = time.perf_counter() - t0
    vram["muscriptor_small"] = _num(fill.extras.get("peak_vram_mb"))
    report("melody_fill", 1.0)

    report("melody_line", 0.0)
    t0 = time.perf_counter()
    import soundfile as sf

    duration = float(sf.info(str(mix)).duration)
    notes = build_line(voice, fill)
    timing["melody_line"] = time.perf_counter() - t0
    timing["total"] = time.perf_counter() - t_start
    result = build_result(song_id, duration, notes, timing, vram)
    write_json(result_path(song_id), result)
    report("melody_line", 1.0)
    return result


def _num(v) -> float | None:
    try:
        f = float(np.asarray(v))
    except (TypeError, ValueError):
        return None
    return round(f, 1) if np.isfinite(f) else None


def melody_ready() -> dict[str, bool]:
    """Đã cài đủ những gì dò giai điệu cần chưa (chỉ kiểm tra file, không nạp model)."""
    import importlib.util

    from chordtrack.melody import muscriptor, yourmt3

    w = muscriptor.weights_dir(MUSCRIPTOR_SIZE)
    return {
        "yourmt3_code": (yourmt3.SRC_DIR / "model_helper.py").exists(),
        "yourmt3_pydeps": (yourmt3.PYDEPS_DIR / "transformers").is_dir(),
        "yourmt3_checkpoint": yourmt3.CHECKPOINT.exists(),
        "muscriptor_package": importlib.util.find_spec("muscriptor") is not None,
        "muscriptor_weights": all((w / f).exists() for f in ("model.safetensors", "config.json")),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m chordtrack.melody.pipeline",
                                 description="Dò giai điệu một bài, in vài nốt đầu và ghi melody.json")
    ap.add_argument("source", help="videoId, URL YouTube hoặc đường dẫn file audio")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    res = analyze_melody(args.source, force=args.force,
                         progress=lambda s, f: print(f"[{s}] {'bắt đầu' if f == 0 else 'xong'}", flush=True))
    print(json.dumps({**res, "notes": res["notes"][:10]}, ensure_ascii=False, indent=1))
    print(f"{len(res['notes'])} nốt, ghi ở {result_path(res['song_id'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
