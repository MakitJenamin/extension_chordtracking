"""Chấm hậu xử lý phách mạnh (chordtrack.beats.postprocess) trên The Beatles (Isophonics).

    python -m eval.beats_meter --build          # tạo lại cache đầu ra beat_this (GPU, một lần)
    python -m eval.beats_meter [--penalties 2 4 6 8 12]

Đầu vào:
- `EVAL_DIR/beats/B_<album>_<bài>_<tên>.npz` (audio `isophonics/audio/<album>_<bài>_<tên>.wav`) và `K_<tên>.npz`
  (audio `kajiura/audio/<tên>.wav`): đầu ra beat_this (beats, downbeats "minimal", downbeat_frames = xác suất phách
  mạnh từng khung, 50 khung/giây). `--build` tạo bằng `chordtrack.beats.run_model`.
- `EVAL_DIR/isophonics/beat/<album>/<bài>_-_<tên>.txt`: đáp án "thời điểm  vị trí trong ô nhịp" (1 = phách mạnh).
- `EVAL_DIR/isophonics/alignment.json`: độ lệch giữa nhãn và audio YouTube (`offset` cộng vào thời gian nhãn).

So sánh: phách mạnh của model ("minimal"), và `postprocess` với từng mức phạt đổi nhịp. Chỉ số: F-measure phách và
phách mạnh của mir_eval (cửa sổ ±70 ms và ±150 ms; đáp án lệch audio YouTube khoảng 0,1 s nên 70 ms cho điểm thấp
cho mọi biến thể), tỉ lệ ô nhịp có đúng số phách chung. Bài Kajiura (không có đáp án) chỉ báo tỉ lệ ô nhịp đúng số
phách chung, để thấy độ ổn định.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mir_eval
import numpy as np

from chordtrack import beats as B
from chordtrack.paths import EVAL_DIR

NPZ_DIR = EVAL_DIR / "beats"
ISO = EVAL_DIR / "isophonics"
KAJ_AUDIO = EVAL_DIR / "kajiura" / "audio"
WINDOWS = (0.07, 0.15)


def load_alignment() -> dict[str, dict]:
    return {a["name"]: a for a in json.loads((ISO / "alignment.json").read_text(encoding="utf8"))}


def reference(name: str, alignment: dict[str, dict]) -> tuple[np.ndarray, np.ndarray] | None:
    """(beats, downbeats) đáp án của bài `B_<album>_<bài>_...` theo thời gian audio, cắt về vùng chung."""
    key = name[2:]
    a = alignment.get(key)
    if a is None or a["status"] != "ok":
        return None
    album, track = key.split("_")[:2]
    adir = next((d for d in (ISO / "beat").iterdir() if d.name.split("_")[0] == album), None)
    f = next((p for p in adir.iterdir() if p.name.startswith(f"{track}_-_")), None) if adir else None
    if f is None:
        return None
    t, pos = [], []
    for line in f.read_text(encoding="utf8", errors="replace").split("\n"):
        parts = line.split()
        if len(parts) >= 2:
            try:
                t.append(float(parts[0]) + a["offset"])
                pos.append(float(parts[1]))
            except ValueError:
                pass
    t, pos = np.array(t), np.array(pos)
    ok = (t >= 0) & (t <= a["common_end"])
    return t[ok], t[ok & (pos == 1)]


def bar_share(beats: np.ndarray, downbeats: np.ndarray) -> float:
    """Tỉ lệ ô nhịp (giữa hai phách mạnh liên tiếp) có đúng số phách phổ biến nhất."""
    if len(downbeats) < 2:
        return float("nan")
    c = np.diff(np.searchsorted(beats, downbeats - 1e-3))
    values, freq = np.unique(c, return_counts=True)
    return float(np.mean(c == values[np.argmax(freq)]))


def build(force: bool = False) -> None:
    """Chạy beat_this trên audio eval, lưu `NPZ_DIR/<B_|K_><tên>.npz`. Bài đã có thì bỏ qua (trừ `force`)."""
    NPZ_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [("B_" + p.stem, p) for p in sorted((ISO / "audio").glob("*.wav"))]
    jobs += [("K_" + p.stem, p) for p in sorted(KAJ_AUDIO.glob("*.wav"))]
    for name, audio in jobs:
        out = NPZ_DIR / f"{name}.npz"
        if out.exists() and not force:
            continue
        signal, sr = B._load_audio(audio, None)
        beats, down, frames = B.run_model(signal, sr)
        np.savez_compressed(out, beats=beats, downbeats=down, downbeat_frames=frames)
        print(f"{name}: {len(beats)} phách")


def evaluate(penalties: list[float]) -> None:
    files = sorted(NPZ_DIR.glob("*.npz"))
    if not files:
        print(f"Không có dữ liệu trong {NPZ_DIR}. Chạy `python -m eval.beats_meter --build` trước.")
        return
    alignment = load_alignment()
    pm = penalties[len(penalties) // 2]
    variants = ["model"] + [f"lock{p:g}" for p in penalties] + [f"lock{pm:g}-notempo"]
    rows: dict[str, list[list[float]]] = {v: [] for v in variants}
    kaj: dict[str, list[float]] = {v: [] for v in variants}
    for f in files:
        z = np.load(f)
        beats = np.sort(z["beats"].astype(float))
        raw = B._snap_downbeats(beats, z["downbeats"].astype(float))
        out = {"model": (beats, raw)}
        for p in penalties:
            out[f"lock{p:g}"] = B.postprocess(beats, z["downbeat_frames"], raw, switch_penalty=p)
        out[f"lock{pm:g}-notempo"] = B.postprocess(beats, z["downbeat_frames"], raw, switch_penalty=pm,
                                                   fix_tempo=False)
        ref = reference(f.stem, alignment) if f.stem.startswith("B_") else None
        for v, (eb, ed) in out.items():
            if ref is None:
                if f.stem.startswith("K_"):
                    kaj[v].append(bar_share(eb, ed))
                continue
            rb, rd = ref
            row = [mir_eval.beat.f_measure(rb, eb, w) for w in WINDOWS]
            row += [mir_eval.beat.f_measure(rd, ed, w) for w in WINDOWS]
            rows[v].append(row + [bar_share(eb, ed)])
    if rows["model"]:
        print(f"The Beatles: {len(rows['model'])} bài có đáp án. Trung bình theo bài.")
        print(f"{'biến thể':<16}{'F phách 70/150 ms':>20}{'F phách mạnh 70/150 ms':>25}{'ô đúng nhịp':>13}")
        for v in variants:
            a = np.array(rows[v]).mean(axis=0)
            print(f"{v:<16}{a[0]:>11.3f} /{a[1]:.3f}{a[2]:>16.3f} /{a[3]:.3f}{a[4]:13.3f}")
        base = np.array(rows["model"])[:, 3]
        best = max(variants[1:], key=lambda v: np.array(rows[v])[:, 3].mean())
        d = np.array(rows[best])[:, 3] - base
        print(f"\n{best} so với model (F phách mạnh 150 ms): tốt hơn ở {int((d > 0.01).sum())} bài, kém hơn ở "
              f"{int((d < -0.01).sum())} bài (lệch > 0.01).")
    else:
        print("Không có bài The Beatles nào có đáp án (thiếu B_*.npz hoặc alignment không 'ok').")
    if kaj["model"]:
        print(f"\nKajiura: {len(kaj['model'])} bài, không có đáp án; tỉ lệ ô nhịp đúng số phách chung:")
        for v in variants:
            print(f"  {v:<16}{np.nanmean(kaj[v]):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--penalties", type=float, nargs="+", default=[2, 4, 6, 8, 12])
    ap.add_argument("--build", action="store_true", help="tạo cache đầu ra beat_this (cần GPU)")
    ap.add_argument("--force", action="store_true", help="với --build: tạo lại cả bài đã có")
    args = ap.parse_args()
    if args.build:
        build(args.force)
    evaluate(args.penalties)


if __name__ == "__main__":
    main()
