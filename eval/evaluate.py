"""Chấm điểm nhận diện hợp âm bằng mir_eval.

Dùng để so sánh mọi biến thể pipeline, nên cách tính phải cố định:

- WCSR (weighted chord symbol recall) theo các mức trong `METRICS`, tính đúng như
  `mir_eval.chord.evaluate`: kéo khoảng thời gian của nhãn đoán về đúng khoảng của nhãn chuẩn
  (phần thiếu điền "N"), ghép hai dãy rồi tính tỉ lệ thời lượng đoán đúng.
- Khoảng thời gian mà nhãn chuẩn không thuộc từ vựng của mức đó (mir_eval trả về -1, ví dụ
  nhãn "X" hoặc hợp âm 4 nốt ở mức majmin) không tính vào mẫu số.
- Độ phân mảnh: `underseg`, `overseg`, `seg = min(underseg, overseg)` (càng gần 1 càng tốt).

Trung bình nhiều bài là **trung bình gia quyền theo thời lượng**: cộng thời lượng đoán đúng của
mọi bài rồi chia cho tổng thời lượng hợp lệ, cho từng mức riêng. Độ phân mảnh lấy trung bình
gia quyền theo độ dài bài.

CLI:
    python -m eval.evaluate --ref-dir <dir .lab> --est-dir <dir .lab> [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence, Union

import mir_eval
import numpy as np

from chordtrack.types import ChordSegment

METRICS: tuple[str, ...] = (
    "root",
    "majmin",
    "majmin_inv",
    "thirds",
    "triads",
    "sevenths",
    "sevenths_inv",
    "tetrads",
    "mirex",
)
SEG_METRICS: tuple[str, ...] = ("underseg", "overseg", "seg")

# Các cột in ra bảng tổng (đủ rộng cho terminal).
TABLE_METRICS: tuple[str, ...] = ("root", "majmin", "majmin_inv", "sevenths", "sevenths_inv", "tetrads", "mirex", "seg")

LabInput = Union[str, Path, Sequence[ChordSegment], tuple]


def _load(lab: LabInput) -> tuple[np.ndarray, list[str]]:
    """Nhận đường dẫn .lab, list[ChordSegment] hoặc (intervals, labels)."""
    if isinstance(lab, (str, Path)):
        intervals, labels = mir_eval.io.load_labeled_intervals(str(lab))
        return np.asarray(intervals, dtype=float).reshape(-1, 2), list(labels)
    if isinstance(lab, tuple) and len(lab) == 2 and not isinstance(lab[0], ChordSegment):
        intervals, labels = lab
        return np.asarray(intervals, dtype=float).reshape(-1, 2), list(labels)
    segs = list(lab)
    intervals = np.array([[s.start, s.end] for s in segs], dtype=float).reshape(-1, 2)
    return intervals, [s.label for s in segs]


def _clean(intervals: np.ndarray, labels: list[str]) -> tuple[np.ndarray, list[str]]:
    """Cắt phần thời gian âm, bỏ đoạn có độ dài <= 0 (một số file .lab có), sắp theo thời gian bắt đầu."""
    intervals = np.clip(intervals, 0.0, None)
    keep = intervals[:, 1] > intervals[:, 0]
    intervals = intervals[keep]
    labels = [l for l, k in zip(labels, keep) if k]
    order = np.argsort(intervals[:, 0], kind="stable")
    return intervals[order], [labels[i] for i in order]


def evaluate_pair(ref_lab: LabInput, est_lab: LabInput) -> dict:
    """Chấm một bài. Trả về dict gồm điểm từng mức trong khoảng [0, 1] và các thông tin phụ.

    Khóa trả về:
      - mỗi tên trong `METRICS` và `SEG_METRICS`: điểm (float). Nếu mức đó không có khoảng
        thời gian hợp lệ nào thì giá trị là NaN.
      - `duration`: độ dài nhãn chuẩn (giây).
      - `weights`: {metric: thời lượng hợp lệ (giây)}, `correct`: {metric: thời lượng đúng}.
        Dùng để tính trung bình gia quyền nhiều bài.
    """
    ref_int, ref_lab_ = _clean(*_load(ref_lab))
    est_int, est_lab_ = _clean(*_load(est_lab))
    if len(ref_int) == 0:
        raise ValueError("Nhãn chuẩn rỗng")
    if len(est_int) == 0:
        est_int, est_lab_ = ref_int[[0]].copy(), ["N"]

    t0, t1 = float(ref_int[0, 0]), float(ref_int[-1, 1])
    # Giống mir_eval.chord.evaluate: kéo nhãn đoán về khoảng của nhãn chuẩn.
    est_int, est_lab_ = mir_eval.util.adjust_intervals(est_int, est_lab_, t0, t1, mir_eval.chord.NO_CHORD, mir_eval.chord.NO_CHORD)
    ref_int, ref_lab_ = mir_eval.util.adjust_intervals(ref_int, ref_lab_, t0, t1, mir_eval.chord.NO_CHORD, mir_eval.chord.NO_CHORD)
    merged, ref_m, est_m = mir_eval.util.merge_labeled_intervals(ref_int, ref_lab_, est_int, est_lab_)
    durations = mir_eval.util.intervals_to_durations(merged)

    out: dict = {"duration": t1 - t0, "weights": {}, "correct": {}}
    for name in METRICS:
        comp = np.asarray(getattr(mir_eval.chord, name)(ref_m, est_m), dtype=float)
        valid = comp >= 0
        w = float(durations[valid].sum())
        c = float((durations[valid] * comp[valid]).sum())
        out["weights"][name] = w
        out["correct"][name] = c
        out[name] = c / w if w > 0 else float("nan")

    under = float(mir_eval.chord.underseg(ref_int, est_int))
    over = float(mir_eval.chord.overseg(ref_int, est_int))
    out["underseg"], out["overseg"], out["seg"] = under, over, min(under, over)
    return out


def aggregate(results: dict[str, dict]) -> dict:
    """Trung bình gia quyền theo thời lượng trên nhiều bài."""
    agg: dict = {}
    for name in METRICS:
        w = sum(r["weights"][name] for r in results.values())
        c = sum(r["correct"][name] for r in results.values())
        agg[name] = c / w if w > 0 else float("nan")
    total = sum(r["duration"] for r in results.values())
    for name in SEG_METRICS:
        agg[name] = sum(r[name] * r["duration"] for r in results.values()) / total if total > 0 else float("nan")
    agg["duration"] = total
    agg["n_files"] = len(results)
    return agg


def evaluate_dirs(ref_dir: Path, est_dir: Path) -> tuple[dict[str, dict], dict, list[str]]:
    """Ghép file theo tên (stem). Trả về (kết quả từng bài, trung bình, danh sách bài thiếu nhãn đoán)."""
    refs = {p.stem: p for p in sorted(Path(ref_dir).glob("*.lab"))}
    ests = {p.stem: p for p in sorted(Path(est_dir).glob("*.lab"))}
    missing = sorted(set(refs) - set(ests))
    results = {name: evaluate_pair(refs[name], ests[name]) for name in sorted(set(refs) & set(ests))}
    return results, aggregate(results), missing


def format_table(results: dict[str, dict], agg: dict, metrics: Sequence[str] = TABLE_METRICS) -> str:
    width = max([len(n) for n in results] + [len("TRUNG BÌNH (gia quyền)")])
    head = f"{'bài':<{width}} {'giây':>6} " + " ".join(f"{m:>12}" for m in metrics)
    lines = [head, "-" * len(head)]

    def row(name: str, r: dict) -> str:
        return f"{name:<{width}} {r['duration']:>6.0f} " + " ".join(f"{100 * r[m]:>12.2f}" for m in metrics)

    for name, r in results.items():
        lines.append(row(name, r))
    lines.append("-" * len(head))
    lines.append(row("TRUNG BÌNH (gia quyền)", agg))
    return "\n".join(lines)


def _nan_to_none(x):
    """JSON chuẩn không có NaN: đổi thành null."""
    if isinstance(x, dict):
        return {k: _nan_to_none(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_nan_to_none(v) for v in x]
    if isinstance(x, float) and np.isnan(x):
        return None
    return x


def _utf8_console() -> None:
    """Console Windows mặc định là cp1252, không in được tiếng Việt."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Chấm điểm nhận diện hợp âm (WCSR) bằng mir_eval")
    ap.add_argument("--ref-dir", type=Path, required=True, help="thư mục nhãn chuẩn .lab")
    ap.add_argument("--est-dir", type=Path, required=True, help="thư mục nhãn đoán .lab, cùng tên file")
    ap.add_argument("--out", type=Path, default=None, help="ghi kết quả ra file JSON")
    args = ap.parse_args(argv)
    _utf8_console()

    results, agg, missing = evaluate_dirs(args.ref_dir, args.est_dir)
    if not results:
        print("Không có cặp file nào trùng tên giữa hai thư mục.", file=sys.stderr)
        return 1
    print(format_table(results, agg))
    if missing:
        print(f"\nThiếu nhãn đoán cho {len(missing)} bài: {', '.join(missing)}", file=sys.stderr)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ref_dir": str(args.ref_dir),
            "est_dir": str(args.est_dir),
            "mean": agg,
            "missing": missing,
            "files": {k: {m: v[m] for m in (*METRICS, *SEG_METRICS, "duration")} for k, v in results.items()},
        }
        args.out.write_text(json.dumps(_nan_to_none(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
