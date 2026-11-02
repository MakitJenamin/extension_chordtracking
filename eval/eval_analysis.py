"""Chấm phần analysis (giọng và chia đoạn) trên bộ Isophonics The Beatles.

Đầu vào mặc định là **nhãn hợp âm chuẩn** (chordlab) để đo riêng analysis, không bị lỗi nhận
hợp âm làm nhiễu. Nếu có kết quả model ở `EVAL_DIR/runs/<tên>/lab/*.lab` hoặc
`EVAL_DIR/results/<tên>/lab/*.lab` (tên file theo `eval.datasets.song_name`) thì chấm thêm trên đó;
khi đó nhãn giọng và nhãn đoạn được dịch theo độ lệch trong `alignment.json` cho khớp audio.

Cách tính:
- Giọng: giọng chuẩn của mỗi bài là giọng chiếm thời lượng lớn nhất trong keylab (bài đổi giọng
  chỉ lấy giọng chính). Mode lạ quy về major/minor: mixolydian, lydian, ionian -> major;
  dorian, aeolian, phrygian -> minor; "modal" -> bỏ bài. Điểm: `mir_eval.key.weighted_score`
  trung bình (đúng 1, quãng 5 0.5, song song/tương đối 0.3...) và tỉ lệ đúng tuyệt đối.
- Đoạn: `mir_eval.segment.detection` (F ở cửa sổ 0.5 s và 3 s, trim=True: không tính ranh giới
  đầu và cuối bài), `pairwise` F và `nce` (S_F) cho nhãn. Nhãn chuẩn và nhãn đoán được kéo về
  cùng khoảng [0, độ dài] bằng `mir_eval.util.adjust_intervals`.
- Kèm baseline "chia đều 15 s, mỗi đoạn một nhãn" để có mốc so sánh.

CLI:
    python -m eval.eval_analysis [--out results.json] [--no-runs] [--verbose]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mir_eval
import numpy as np

from chordtrack.analysis import analyze_structure, estimate_key, key_to_mir_eval
from chordtrack.paths import EVAL_DIR
from chordtrack.types import ChordSegment, lab_to_segments
from eval import datasets as D
from eval.evaluate import _utf8_console

RUN_ROOTS = (EVAL_DIR / "runs", EVAL_DIR / "results")
OUT_DEFAULT = EVAL_DIR / "results" / "analysis" / "results.json"

_MAJOR_MODES = {"", "major", "ionian", "mixolydian", "lydian"}
_MINOR_MODES = {"minor", "aeolian", "dorian", "phrygian"}


# ---------------------------------------------------------------- đọc nhãn

def read_keylab(path: Path) -> list[tuple[float, float, str]]:
    """Dòng 'start end Key C' hoặc 'start end Key A:minor'. Trả về (start, end, 'C major')."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4 or parts[2] != "Key":
                continue
            tonic, _, mode = parts[3].partition(":")
            mode = mode.lower()
            if mode in _MAJOR_MODES:
                key = f"{tonic} major"
            elif mode in _MINOR_MODES:
                key = f"{tonic} minor"
            else:
                key = "X"
            out.append((float(parts[0]), float(parts[1]), key))
    return out


def main_key(keylab: list[tuple[float, float, str]]) -> str | None:
    """Giọng chiếm thời lượng lớn nhất. None nếu không xác định (modal hoặc không có nhãn)."""
    dur: dict[str, float] = {}
    for a, b, k in keylab:
        dur[k] = dur.get(k, 0.0) + (b - a)
    if not dur:
        return None
    best = max(dur, key=dur.get)
    return None if best == "X" else best


def read_seglab(path: Path) -> tuple[np.ndarray, list[str]]:
    """Dòng 'start end <trống> nhãn' (có bài dùng dấu cách). Trả về (intervals, labels)."""
    iv, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            a, b = float(parts[0]), float(parts[1])
            if b <= a:
                continue
            iv.append([a, b])
            labels.append(parts[-1].lower())
    return np.array(iv, dtype=float).reshape(-1, 2), labels


# ---------------------------------------------------------------- chấm

def score_key(ref: str, est: str) -> dict:
    w = float(mir_eval.key.weighted_score(ref, est))
    return {"weighted": w, "correct": float(w == 1.0)}   # so theo cao độ: C# major = Db major


def score_segments(ref_iv: np.ndarray, ref_lab: list[str], est_iv: np.ndarray, est_lab: list[str]) -> dict:
    t_min, t_max = 0.0, float(ref_iv[-1, 1])
    r_iv, r_lab = mir_eval.util.adjust_intervals(ref_iv, ref_lab, t_min=t_min, t_max=t_max)
    e_iv, e_lab = mir_eval.util.adjust_intervals(est_iv, est_lab, t_min=t_min, t_max=t_max)
    _, _, f05 = mir_eval.segment.detection(r_iv, e_iv, window=0.5, trim=True)
    p3, r3, f3 = mir_eval.segment.detection(r_iv, e_iv, window=3.0, trim=True)
    pp, pr, pf = mir_eval.segment.pairwise(r_iv, r_lab, e_iv, e_lab)
    so, su, sf = mir_eval.segment.nce(r_iv, r_lab, e_iv, e_lab)
    return {"F@0.5": f05, "P@3": p3, "R@3": r3, "F@3": f3, "pair_P": pp, "pair_R": pr, "pair_F": pf,
            "S_o": so, "S_u": su, "S_F": sf, "n_est": len(est_iv), "n_ref": len(ref_iv)}


def uniform_baseline(duration: float, step: float = 15.0) -> tuple[np.ndarray, list[str]]:
    edges = np.append(np.arange(0.0, duration, step), duration)
    if len(edges) > 2 and edges[-1] - edges[-2] < step / 2:
        edges = np.delete(edges, -2)
    iv = np.stack([edges[:-1], edges[1:]], axis=1)
    return iv, [f"u{i}" for i in range(len(iv))]


def shift_intervals(iv: np.ndarray, offset: float) -> np.ndarray:
    return np.clip(iv + offset, 0.0, None)


def evaluate_song(segs: list[ChordSegment], ref_key: str | None, seg_ref: tuple[np.ndarray, list[str]] | None,
                  duration: float) -> dict:
    t0 = time.perf_counter()
    key = estimate_key(segs)
    sections = analyze_structure(segs, None, duration, key)
    elapsed = time.perf_counter() - t0
    res: dict = {"est_key": key_to_mir_eval(key), "time": elapsed}
    if ref_key is not None:
        res["ref_key"] = ref_key
        res.update({f"key_{k}": v for k, v in score_key(ref_key, key_to_mir_eval(key)).items()})
    if seg_ref is not None and len(seg_ref[0]):
        est_iv = np.array([[s.start, s.end] for s in sections], dtype=float).reshape(-1, 2)
        est_lab = [s.label for s in sections]
        res["seg"] = score_segments(seg_ref[0], seg_ref[1], est_iv, est_lab)
        res["seg_uniform"] = score_segments(seg_ref[0], seg_ref[1], *uniform_baseline(float(seg_ref[0][-1, 1])))
    return res


def aggregate(results: dict[str, dict]) -> dict:
    keys = [r for r in results.values() if "key_weighted" in r]
    segs = [r for r in results.values() if "seg" in r]
    out: dict = {"n_songs": len(results), "n_key": len(keys), "n_seg": len(segs)}
    if keys:
        out["key_weighted"] = float(np.mean([r["key_weighted"] for r in keys]))
        out["key_correct"] = float(np.mean([r["key_correct"] for r in keys]))
        cats = {"same": 1.0, "fifth": 0.5, "relative": 0.3, "parallel": 0.2}
        out["key_breakdown"] = {name: int(sum(1 for r in keys if abs(r["key_weighted"] - v) < 1e-9))
                                for name, v in cats.items()}
        out["key_breakdown"]["other"] = len(keys) - sum(out["key_breakdown"].values())
    for name in ("seg", "seg_uniform"):
        if segs:
            out[name] = {m: float(np.mean([r[name][m] for r in segs])) for m in segs[0][name]}
    out["time_total"] = float(sum(r["time"] for r in results.values()))
    return out


def format_summary(title: str, agg: dict) -> str:
    lines = [f"== {title}: {agg['n_songs']} bài (giọng {agg['n_key']}, đoạn {agg['n_seg']}), "
             f"thời gian analysis {agg['time_total']:.1f} s"]
    if "key_weighted" in agg:
        b = agg["key_breakdown"]
        lines.append(f"Giọng: weighted {agg['key_weighted']:.3f}, đúng tuyệt đối {agg['key_correct']:.3f} "
                     f"(đúng {b['same']}, quãng 5 {b['fifth']}, tương đối {b['relative']}, "
                     f"song song {b['parallel']}, khác {b['other']})")
    if "seg" in agg:
        cols = ("F@0.5", "P@3", "R@3", "F@3", "pair_P", "pair_R", "pair_F", "S_o", "S_u", "S_F", "n_est", "n_ref")
        lines.append(f"{'':12s}" + "".join(f"{c:>8s}" for c in cols))
        for name, label in (("seg", "analysis"), ("seg_uniform", "đều 15 s")):
            lines.append(f"{label:12s}" + "".join(f"{agg[name][c]:8.3f}" if c[0] != "n" else f"{agg[name][c]:8.1f}"
                                                  for c in cols))
    return "\n".join(lines)


# ---------------------------------------------------------------- dữ liệu

def ground_truth_songs() -> list[tuple[str, Path, Path | None, Path | None]]:
    """(tên bài, chordlab, keylab, seglab) cho mọi bài có chordlab."""
    out = []
    for lab in sorted(D.CHORDLAB_DIR.rglob("*.lab")):
        rel = lab.relative_to(D.CHORDLAB_DIR)
        key = D.KEYLAB_DIR / rel
        seg = D.SEGLAB_DIR / rel
        out.append((D.song_name(rel.as_posix()), lab, key if key.exists() else None, seg if seg.exists() else None))
    return out


def model_runs() -> dict[str, Path]:
    runs = {}
    for root in RUN_ROOTS:
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if (d / "lab").is_dir() and any((d / "lab").glob("*.lab")):
                runs.setdefault(d.name, d / "lab")
    return runs


def load_offsets() -> dict[str, float]:
    if not D.REPORT_PATH.exists():
        return {}
    rows = json.loads(D.REPORT_PATH.read_text(encoding="utf-8"))
    return {r["name"]: float(r["offset"]) for r in rows if r.get("status") == "ok"}


def run(include_runs: bool = True, verbose: bool = False) -> dict:
    D.download_labels()
    D.download_key_seg_labels()
    songs = ground_truth_songs()
    gt: dict[str, tuple[str | None, tuple | None]] = {}
    results: dict[str, dict] = {}
    for name, lab, keylab, seglab in songs:
        segs = lab_to_segments(lab)
        if not segs:
            continue
        ref_key = main_key(read_keylab(keylab)) if keylab else None
        seg_ref = read_seglab(seglab) if seglab else None
        gt[name] = (ref_key, seg_ref)
        duration = float(seg_ref[0][-1, 1]) if seg_ref is not None and len(seg_ref[0]) else segs[-1].end
        results[name] = evaluate_song(segs, ref_key, seg_ref, duration)
        if verbose:
            r = results[name]
            print(f"{name:45s} key {r.get('ref_key', '-'):>10s} -> {r['est_key']:>10s}  "
                  f"F@3 {r['seg']['F@3'] if 'seg' in r else float('nan'):.2f}  "
                  f"pairF {r['seg']['pair_F'] if 'seg' in r else float('nan'):.2f}")
    report = {"chordlab": {"mean": aggregate(results), "files": results}}

    if include_runs:
        offsets = load_offsets()
        for run_name, lab_dir in model_runs().items():
            if run_name == "analysis":
                continue
            rres: dict[str, dict] = {}
            for lab in sorted(lab_dir.glob("*.lab")):
                name = lab.stem
                if name not in gt:
                    continue
                ref_key, seg_ref = gt[name]
                off = offsets.get(name, 0.0)
                if seg_ref is not None:
                    seg_ref = (shift_intervals(seg_ref[0], off), seg_ref[1])
                segs = lab_to_segments(lab)
                if not segs:
                    continue
                duration = max(segs[-1].end, float(seg_ref[0][-1, 1]) if seg_ref is not None else 0.0)
                rres[name] = evaluate_song(segs, ref_key, seg_ref, duration)
                # Cùng các bài đó nhưng dùng chordlab: để so model với nhãn chuẩn trên cùng tập bài
            if rres:
                same = {n: results[n] for n in rres}
                report[f"run:{run_name}"] = {"mean": aggregate(rres), "chordlab_same_songs": aggregate(same),
                                            "files": rres}
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Chấm giọng và chia đoạn trên Isophonics The Beatles")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--no-runs", action="store_true", help="chỉ chấm trên chordlab")
    ap.add_argument("--verbose", action="store_true", help="in điểm từng bài")
    args = ap.parse_args(argv)
    _utf8_console()
    report = run(include_runs=not args.no_runs, verbose=args.verbose)
    for name, rep in report.items():
        print(format_summary(name, rep["mean"]))
        if "chordlab_same_songs" in rep:
            print(format_summary(f"{name} (chordlab cùng bài)", rep["chordlab_same_songs"]))
        print()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Đã ghi {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
