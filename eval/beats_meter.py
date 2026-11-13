"""Chấm hậu xử lý phách mạnh (chordtrack.beats.postprocess) trên The Beatles (Isophonics).

    python -m eval.beats_meter [--penalties 2 4 6 8 12]

Đầu vào có sẵn, không cần GPU:
- `EVAL_DIR/beats/B_<album>_<bài>_<tên>.npz`: đầu ra beat_this trên audio YouTube (beats, downbeats "minimal",
  downbeat_frames = xác suất phách mạnh từng khung, 50 khung/giây).
- `EVAL_DIR/isophonics/beat/<album>/<bài>_-_<tên>.txt`: đáp án "thời điểm  vị trí trong ô nhịp" (1 = phách mạnh).
- `EVAL_DIR/isophonics/alignment.json`: độ lệch giữa nhãn và audio YouTube (`offset` cộng vào thời gian nhãn).

So sánh: phách mạnh của model ("minimal"), và `postprocess` với từng mức phạt đổi nhịp. Chỉ số: F-measure phách và
phách mạnh của mir_eval (cửa sổ ±70 ms), tỉ lệ ô nhịp đúng `meter` phách. Bài Kajiura (K_*, không có đáp án) chỉ báo
tỉ lệ ô nhịp đúng `meter` phách, để thấy độ ổn định.
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


def reference(name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """(beats, downbeats) đáp án của bài `B_<album>_<bài>_...` theo thời gian audio, cắt về vùng chung."""
    al = {a["name"]: a for a in json.loads((ISO / "alignment.json").read_text(encoding="utf8"))}
    key = name[2:]
    a = al.get(key)
    if a is None or a["status"] != "ok":
        return None
    album, track = key.split("_")[:2]
    adir = next((d for d in (ISO / "beat").iterdir() if d.name.split("_")[0] == album), None)
    if adir is None:
        return None
    f = next((p for p in adir.iterdir() if p.name.startswith(f"{track}_-_")), None)
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


def bar_share(beats: np.ndarray, downbeats: np.ndarray, meter: int) -> float:
    """Tỉ lệ ô nhịp (giữa hai phách mạnh liên tiếp) có đúng `meter` phách."""
    if len(downbeats) < 2:
        return float("nan")
    c = np.diff(np.searchsorted(beats, downbeats - 1e-3))
    return float(np.mean(c == meter))


def evaluate(penalties: list[float]) -> None:
    files = sorted(NPZ_DIR.glob("*.npz"))
    variants = ["model"] + [f"lock{p:g}" for p in penalties] + [f"lock{penalties[len(penalties) // 2]:g}-notempo"]
    rows: dict[str, list[tuple[float, float, float]]] = {v: [] for v in variants}
    kaj: dict[str, list[float]] = {v: [] for v in variants}
    for f in files:
        z = np.load(f)
        beats = np.sort(z["beats"].astype(float))
        raw = B._snap_downbeats(beats, z["downbeats"].astype(float))
        meter = B.meter_from_downbeats(beats, raw)
        out = {"model": (beats, raw)}
        for p in penalties:
            out[f"lock{p:g}"] = B.postprocess(beats, z["downbeat_frames"], meter, switch_penalty=p)
        pm = penalties[len(penalties) // 2]
        out[f"lock{pm:g}-notempo"] = B.postprocess(beats, z["downbeat_frames"], meter, switch_penalty=pm,
                                                   fix_tempo=False)
        ref = reference(f.stem) if f.stem.startswith("B_") else None
        for v, (eb, ed) in out.items():
            share = bar_share(eb, ed, meter)
            if ref is None:
                if f.stem.startswith("K_"):
                    kaj[v].append(share)
                continue
            rb, rd = ref
            rows[v].append((mir_eval.beat.f_measure(rb, eb), mir_eval.beat.f_measure(rd, ed), share))
    n = len(rows["model"])
    print(f"The Beatles: {n} bài có đáp án. Trung bình theo bài.")
    print(f"{'biến thể':<16}{'F phách':>9}{'F phách mạnh':>14}{'ô đúng nhịp':>13}")
    for v in variants:
        a = np.array(rows[v])
        print(f"{v:<16}{a[:, 0].mean():9.3f}{a[:, 1].mean():14.3f}{a[:, 2].mean():13.3f}")
    base = np.array(rows["model"])[:, 1]
    best = max(variants[1:], key=lambda v: np.array(rows[v])[:, 1].mean())
    d = np.array(rows[best])[:, 1] - base
    print(f"\n{best} so với model: phách mạnh tốt hơn ở {int((d > 0.01).sum())} bài, kém hơn ở "
          f"{int((d < -0.01).sum())} bài (lệch > 0.01).")
    if kaj["model"]:
        print(f"\nKajiura: {len(kaj['model'])} bài, không có đáp án; tỉ lệ ô nhịp đúng số phách chung:")
        for v in variants:
            print(f"  {v:<16}{np.nanmean(kaj[v]):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--penalties", type=float, nargs="+", default=[2, 4, 6, 8, 12])
    evaluate(ap.parse_args().penalties)


if __name__ == "__main__":
    main()
