"""Cache đặc trưng cho bộ đánh giá, để thử nghiệm fusion không phải chạy lại GPU.

    python -m eval.cache_features [--only NAME ...] [--force]

Với mỗi bài trong `isophonics_youtube.csv` có audio (kể cả bài bị loại khi căn, để đổi quy tắc
loại bài không phải chạy lại):
1. `separation.separate` -> stem ở `CACHE_DIR/eval_<tên>/` (có cache sẵn trong hàm).
2. `beats.track_beats` trên mix.
3. `run_chordmini` (BTC-PL) và `run_cnnlstm` trên **mix** và **instrumental**.

Kết quả ghi ra `EVAL_DIR/features/<tên>.pkl` (pickle một dict, xem `load_features`). Mỗi bước GPU
tự lấy `gpu_lock()` riêng bên trong hàm của module, nên agent khác vẫn dùng GPU xen kẽ được.
Ma trận xác suất lưu ở float32 cho nhẹ.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf

from chordtrack.paths import CACHE_DIR, EVAL_DIR
from chordtrack.types import ChordModelOutput
from eval.datasets import AUDIO_DIR, read_mapping
from eval.evaluate import _utf8_console

FEATURES_DIR = EVAL_DIR / "features"
FEATURES_VERSION = 1
SOURCES = ("mix", "instrumental")


def stem_dir(name: str) -> Path:
    return CACHE_DIR / f"eval_{name}"


def features_path(name: str) -> Path:
    return FEATURES_DIR / f"{name}.pkl"


def _compact(out: ChordModelOutput) -> ChordModelOutput:
    """Đổi các mảng float64 sang float32 để file pickle nhỏ lại."""
    def f32(a):
        return None if a is None else np.asarray(a, dtype=np.float32)

    return ChordModelOutput(
        model_name=out.model_name,
        segments=out.segments,
        frame_times=None if out.frame_times is None else np.asarray(out.frame_times, dtype=np.float64),
        frame_probs=f32(out.frame_probs),
        vocabulary=out.vocabulary,
        extras={k: f32(v) for k, v in out.extras.items()},
    )


def extract(name: str, audio: Path, stems_out: Path | None = None) -> dict:
    """stems_out: thư mục stem (mặc định `stem_dir(name)`); bộ Kajiura dùng `CACHE_DIR/kajiura_<tên>`."""
    from chordtrack.beats import track_beats
    from chordtrack.chords.chordmini import run_chordmini
    from chordtrack.chords.cnnlstm import run_cnnlstm
    from chordtrack.separation import separate

    out_dir = Path(stems_out) if stems_out is not None else stem_dir(name)
    timings: dict[str, float] = {}
    t = time.perf_counter()
    stems = separate(audio, out_dir)
    timings["separate"] = time.perf_counter() - t

    t = time.perf_counter()
    beats = track_beats(stems.mix)
    timings["beats"] = time.perf_counter() - t

    models: dict[str, ChordModelOutput] = {}
    for src in SOURCES:
        path = getattr(stems, src)
        t = time.perf_counter()
        models[f"chordmini_{src}"] = _compact(run_chordmini(path))
        timings[f"chordmini_{src}"] = time.perf_counter() - t
        t = time.perf_counter()
        models[f"cnnlstm_{src}"] = _compact(run_cnnlstm(path))
        timings[f"cnnlstm_{src}"] = time.perf_counter() - t

    info = sf.info(str(stems.mix))
    return {
        "version": FEATURES_VERSION,
        "name": name,
        "duration": info.frames / info.samplerate,
        "stems_dir": str(out_dir),
        "beats": beats,
        "models": models,
        "timings": timings,
    }


def load_features(name: str) -> dict:
    """Đọc dict đã cache: version, name, duration, stems_dir, beats (BeatGrid),
    models ({"chordmini_mix", "chordmini_instrumental", "cnnlstm_mix", "cnnlstm_instrumental"}), timings."""
    with open(features_path(name), "rb") as f:
        return pickle.load(f)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Cache đặc trưng (stem, phách, xác suất hợp âm) cho bộ đánh giá")
    ap.add_argument("--only", nargs="*", default=None, help="chỉ chạy các bài này (tên trong CSV)")
    ap.add_argument("--force", action="store_true", help="chạy lại cả bài đã có file đặc trưng")
    args = ap.parse_args(argv)
    _utf8_console()

    import logging
    logging.basicConfig(level=logging.WARNING)

    rows = read_mapping()
    if args.only:
        rows = [r for r in rows if r["name"] in set(args.only)]
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    t_all = time.perf_counter()
    for i, row in enumerate(rows, 1):
        name = row["name"]
        audio = AUDIO_DIR / f"{name}.wav"
        dst = features_path(name)
        if dst.exists() and not args.force:
            print(f"[{i}/{len(rows)}] {name}: đã có, bỏ qua", flush=True)
            continue
        if not audio.exists():
            print(f"[{i}/{len(rows)}] {name}: thiếu audio {audio}", flush=True)
            failed.append(name)
            continue
        try:
            feats = extract(name, audio)
        except Exception:
            traceback.print_exc()
            failed.append(name)
            continue
        tmp = dst.with_suffix(".part")
        with open(tmp, "wb") as f:
            pickle.dump(feats, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(dst)
        tm = feats["timings"]
        print(f"[{i}/{len(rows)}] {name}: {feats['duration']:.0f}s audio, "
              + ", ".join(f"{k} {v:.1f}s" for k, v in tm.items()), flush=True)
    print(f"Xong trong {(time.perf_counter() - t_all) / 60:.1f} phút. Lỗi: {failed or 'không'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
