"""Đánh giá phách mạnh và số chỉ nhịp (`beats.refine_downbeats`) trên Beatles và Kajiura.

    python -m eval.eval_beats download          # nhãn beat Isophonics -> EVAL_DIR/isophonics/beat/
    python -m eval.eval_beats cache [--force]   # beat_this (kèm xác suất theo khung) -> EVAL_DIR/beats/
    python -m eval.eval_beats run [--grid]      # chấm các biến thể; --grid: dò tham số MeterParams

**Beatles** (Isophonics có nhãn beat: "thời điểm  số phách trong ô"): nhãn dịch theo `offset` trong
`isophonics/alignment.json` (như nhãn hợp âm), cắt về `common_end`. Chấm phách mạnh bằng
`mir_eval.beat.f_measure` (cửa sổ ± 70 ms, bỏ 5 s đầu như mặc định của mir_eval). Số chỉ nhịp chuẩn là
số phách mỗi ô phổ biến nhất trong nhãn.

**Kajiura** không có nhãn beat. Chấm gián tiếp: tỉ lệ chỗ đổi hợp âm của bản hợp âm ChordWiki (đã căn,
`kajiura/ref/*.lab`, chỉ bài căn tốt) rơi vào phách mạnh (phách gần nhất của chỗ đổi là phách mạnh), và tỉ
lệ ô nhịp có độ dài khác số chỉ nhịp ("ô lệch").

Chỗ đổi hợp âm đưa vào `refine_downbeats` lấy từ `fusion.fuse_probs` với `DEFAULT_PARAMS` trên đặc trưng
đã cache (`eval.cache_features`, `eval.ablation_kajiura cache`), giống pipeline.
"""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
import tarfile
import tempfile
import time
import urllib.request
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from chordtrack import beats as B
from chordtrack import fusion as F
from chordtrack.paths import EVAL_DIR
from chordtrack.types import BeatGrid
from eval import datasets as D
from eval import kajiura as K
from eval.evaluate import _utf8_console

BEAT_LABEL_DIR = D.ISO_DIR / "beat"
BEATS_DIR = EVAL_DIR / "beats"
RESULTS_PATH = EVAL_DIR / "results" / "eval_beats.json"


# ---------------------------------------------------------------- nhãn beat Isophonics

def download_beat_labels(force: bool = False) -> int:
    """Tải tarball Isophonics, giữ các file beat/<Album>/<Track>.txt. Trả về số file."""
    have = len(list(BEAT_LABEL_DIR.rglob("*.txt"))) if BEAT_LABEL_DIR.exists() else 0
    if have and not force:
        return have
    if BEAT_LABEL_DIR.exists():
        shutil.rmtree(BEAT_LABEL_DIR)
    BEAT_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    with tempfile.TemporaryDirectory() as tmp:
        tgz = Path(tmp) / "beatles.tar.gz"
        print(f"Tải {D.ISO_URL}")
        urllib.request.urlretrieve(D.ISO_URL, tgz)
        with tarfile.open(tgz) as tar:
            for m in tar.getmembers():
                parts = Path(m.name).parts
                # beat/The Beatles/<Album>/<Track>.txt
                if not m.isfile() or not m.name.endswith(".txt") or not parts or parts[0] != "beat":
                    continue
                rel = parts[2:]
                if len(rel) != 2 or any(p in ("..", "") for p in rel):
                    continue
                dst = BEAT_LABEL_DIR.joinpath(*rel)
                dst.parent.mkdir(parents=True, exist_ok=True)
                src = tar.extractfile(m)
                assert src is not None
                dst.write_bytes(src.read())
                n += 1
    return n


def parse_beat_file(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Nội dung file beat Isophonics -> (thời điểm, số phách trong ô). Dòng hỏng bị bỏ."""
    times, nums = [], []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            t, k = float(parts[0]), int(float(parts[1]))
        except ValueError:
            continue
        times.append(t)
        nums.append(k)
    order = np.argsort(times, kind="stable")
    return np.asarray(times, dtype=np.float64)[order], np.asarray(nums, dtype=np.int64)[order]


def reference_meter(nums: np.ndarray) -> int | None:
    """Số phách mỗi ô phổ biến nhất (giá trị lớn nhất của số phách trước mỗi phách 1)."""
    ones = np.nonzero(nums == 1)[0]
    if len(ones) < 2:
        return None
    lengths = np.diff(ones)
    values, counts = np.unique(lengths, return_counts=True)
    return int(values[np.argmax(counts)])


def beatles_reference(name: str, lab_file: str, align: dict) -> dict | None:
    path = BEAT_LABEL_DIR / Path(lab_file).with_suffix(".txt")
    if not path.exists():
        return None
    t, k = parse_beat_file(path.read_text(encoding="utf-8", errors="replace"))
    t = t + float(align.get("offset", 0.0))
    end = float(align.get("common_end") or np.inf)
    keep = (t >= 0) & (t <= end)
    t, k = t[keep], k[keep]
    return {"beats": t, "downbeats": t[k == 1], "meter": reference_meter(k)}


# ---------------------------------------------------------------- cache beat_this

def _cache_path(ds: str, name: str) -> Path:
    return BEATS_DIR / f"{ds}_{name}.npz"


def _songs() -> list[tuple[str, str, Path, Path]]:
    """(bộ, tên, file đặc trưng hợp âm, audio mix) của mọi bài có đặc trưng."""
    from eval.ablation_kajiura import features_path as k_features
    from eval.cache_features import features_path as b_features, load_features

    out = []
    for ref in sorted(D.REF_DIR.glob("*.lab")):
        fp = b_features(ref.stem)
        if fp.exists():
            out.append(("B", ref.stem, fp, Path(load_features(ref.stem)["stems_dir"]) / "mix.wav"))
    import pickle

    for row in K.read_mapping():
        fp = k_features(row["name"])
        if fp.exists():
            with open(fp, "rb") as f:
                stems = Path(pickle.load(f)["stems_dir"])
            out.append(("K", row["name"], fp, stems / "mix.wav"))
    return out


def cache(force: bool = False) -> int:
    BEATS_DIR.mkdir(parents=True, exist_ok=True)
    failed = 0
    for ds, name, _fp, mix in _songs():
        dst = _cache_path(ds, name)
        if dst.exists() and not force:
            continue
        if not mix.exists():
            print(f"{ds}:{name}: thiếu {mix}")
            failed += 1
            continue
        t0 = time.perf_counter()
        tr = B.track_beats_full(mix)
        tmp = dst.with_name(dst.stem + ".part.npz")
        np.savez_compressed(tmp, beats=tr.grid.beats, downbeats=tr.grid.downbeats,
                            downbeat_frames=tr.downbeat_frames.astype(np.float32))
        tmp.replace(dst)
        print(f"{ds}:{name}: {len(tr.grid.beats)} phách, {time.perf_counter() - t0:.1f}s", flush=True)
    return 1 if failed else 0


# ---------------------------------------------------------------- dữ liệu chấm

def _change_times(ds: str, name: str) -> list[float]:
    """Chỗ đổi hợp âm của `fuse_probs` mặc định (bỏ chỗ đổi từ/sang N)."""
    from eval.ablation import Song, predict
    from eval.ablation_kajiura import KajiuraSong
    from eval.refit_fusion import fusion_cfg

    song = Song(name) if ds == "B" else KajiuraSong(name)
    segs = predict(song, fusion_cfg(**asdict(F.DEFAULT_PARAMS)))
    return chord_changes(segs)


def chord_changes(segs) -> list[float]:
    return B.chord_change_times(segs)


def load_data() -> list[dict]:
    """Mỗi bài: beats, xác suất phách mạnh theo phách, downbeats gốc, chỗ đổi hợp âm và nhãn chuẩn."""
    mapping = {r["name"]: r for r in D.read_mapping()}
    b_align = {r["name"]: r for r in json.loads(D.REPORT_PATH.read_text(encoding="utf-8"))}
    k_report = K.read_report()
    changes_path = BEATS_DIR / "chord_changes.json"
    changes = json.loads(changes_path.read_text(encoding="utf-8")) if changes_path.exists() else {}
    dirty = False
    out = []
    for ds, name, _fp, _mix in _songs():
        path = _cache_path(ds, name)
        if not path.exists():
            continue
        z = np.load(path)
        item = {"ds": ds, "name": name, "beats": z["beats"], "downbeats": z["downbeats"],
                "probs": B.beat_downbeat_probs(z["beats"], z["downbeat_frames"])}
        item["clean"] = B.clean_beats(z["beats"])
        item["clean_probs"] = B.beat_downbeat_probs(item["clean"], z["downbeat_frames"])
        key = f"{ds}:{name}"
        if key not in changes:
            changes[key] = _change_times(ds, name)
            dirty = True
        item["changes"] = np.asarray(changes[key], dtype=np.float64)
        if ds == "B":
            ref = beatles_reference(name, mapping[name]["lab_file"], b_align.get(name, {}))
            if ref is None or len(ref["downbeats"]) < 4:
                continue
            item["ref"] = ref
        else:
            rep = k_report.get(name)
            lab = K.REF_DIR / f"{name}.lab"
            if not rep or not rep.get("ok") or not lab.exists():
                item["ref"] = None
            else:
                from chordtrack.types import lab_to_segments

                item["ref"] = {"changes": np.asarray(chord_changes(lab_to_segments(lab)))}
        out.append(item)
    if dirty:
        changes_path.write_text(json.dumps(changes), encoding="utf-8")
    return out


# ---------------------------------------------------------------- chấm

def bar_stats(beats: np.ndarray, downbeats: np.ndarray, meter: int) -> float:
    """Tỉ lệ ô nhịp có số phách khác `meter` (bỏ trước phách mạnh đầu và sau phách mạnh cuối)."""
    idx = np.searchsorted(beats, np.asarray(downbeats) - 1e-3)
    lengths = np.diff(idx)
    return float(np.mean(lengths != meter)) if len(lengths) else 0.0


def changes_on_downbeats(beats: np.ndarray, downbeats: np.ndarray, changes: np.ndarray) -> float:
    """Tỉ lệ chỗ đổi hợp âm (nhãn chuẩn) có phách gần nhất là phách mạnh."""
    if len(changes) == 0 or len(beats) == 0:
        return float("nan")
    near = beats[np.abs(beats[None, :] - changes[:, None]).argmin(axis=1)]
    return float(np.isin(near, downbeats).mean())


VARIANTS = ("beat_this", "refine_act", "refine_chords", "refine", "clean+refine")


def estimate(item: dict, variant: str, params: B.MeterParams) -> tuple[np.ndarray, np.ndarray, int]:
    """(phách, phách mạnh, số chỉ nhịp) của một biến thể."""
    from chordtrack.pipeline import estimate_meter

    beats = item["beats"]
    if variant == "beat_this":
        return beats, item["downbeats"], estimate_meter(BeatGrid(beats, item["downbeats"], 0.0))
    if variant == "refine_act":
        return (beats, *B.refine_downbeats(beats, item["probs"], None, params))
    if variant == "refine_chords":
        return (beats, *B.refine_downbeats(beats, None, item["changes"], params))
    if variant == "refine":
        return (beats, *B.refine_downbeats(beats, item["probs"], item["changes"], params))
    if variant == "clean+refine":
        return (item["clean"], *B.refine_downbeats(item["clean"], item["clean_probs"], item["changes"], params))
    raise ValueError(variant)


def local_align(ref_beats: np.ndarray, est_beats: np.ndarray, half_window: int = 8) -> np.ndarray:
    """Dịch từng phách chuẩn theo độ lệch cục bộ tới phách đoán gần nhất (trung vị trượt ± `half_window`
    phách, chỉ dùng cặp lệch dưới 40% khoảng phách). Bản YouTube trôi so với nhãn tới vài trăm ms ở vài bài;
    phép căn này chỉ dùng phách (không dùng phách mạnh) nên công bằng giữa các cách chọn phách mạnh."""
    r = np.asarray(ref_beats, dtype=np.float64)
    e = np.asarray(est_beats, dtype=np.float64)
    if len(r) < 2 or len(e) == 0:
        return r
    d = e[np.abs(e[None, :] - r[:, None]).argmin(axis=1)] - r
    ibi = np.median(np.diff(r))
    valid = np.abs(d) < 0.4 * ibi
    out = r.copy()
    for j in range(len(r)):
        lo, hi = max(0, j - half_window), min(len(r), j + half_window + 1)
        w = d[lo:hi][valid[lo:hi]]
        if len(w):
            out[j] = r[j] + float(np.median(w))
    return out


def score(data: list[dict], variant: str, params: B.MeterParams) -> dict:
    import mir_eval

    fs, bfs, meters, irregular, k_on, k_irr = [], [], [], [], [], []
    for item in data:
        beats, downs, meter = estimate(item, variant, params)
        if item["ds"] == "B":
            ref = item["ref"]
            if "aligned_downbeats" not in ref:
                aligned = local_align(ref["beats"], item["beats"])
                ref["aligned_beats"] = aligned
                ref["aligned_downbeats"] = aligned[np.isin(ref["beats"], ref["downbeats"])]
            bfs.append(mir_eval.beat.f_measure(mir_eval.beat.trim_beats(ref["aligned_beats"]),
                                               mir_eval.beat.trim_beats(np.asarray(beats))))
            fs.append(mir_eval.beat.f_measure(mir_eval.beat.trim_beats(ref["aligned_downbeats"]),
                                              mir_eval.beat.trim_beats(np.asarray(downs))))
            if ref["meter"] is not None:
                meters.append(meter == ref["meter"])
            irregular.append(bar_stats(beats, downs, meter))
        else:
            k_irr.append(bar_stats(beats, downs, meter))
            if item["ref"] is not None:
                k_on.append(changes_on_downbeats(beats, downs, item["ref"]["changes"]))
    return {
        "beatles_downbeat_f": float(np.mean(fs)) if fs else float("nan"),
        "beatles_beat_f": float(np.mean(bfs)) if bfs else float("nan"),
        "beatles_meter_acc": float(np.mean(meters)) if meters else float("nan"),
        "beatles_irregular_bars": float(np.mean(irregular)) if irregular else float("nan"),
        "kajiura_changes_on_downbeat": float(np.nanmean(k_on)) if k_on else float("nan"),
        "kajiura_irregular_bars": float(np.mean(k_irr)) if k_irr else float("nan"),
        "n_beatles": len(fs), "n_kajiura": len(k_irr), "n_kajiura_ref": len(k_on),
    }


def param_grid() -> list[B.MeterParams]:
    out = []
    for meters, cw, half, reset, mp, fp in itertools.product(((3, 4), (2, 3, 4)), (0.0, 0.5, 1.0, 2.0), (0.0, 0.5),
                                                             (1.0, 2.0, 4.0, 8.0, 16.0), (5.0, 20.0), (0.0, 0.5)):
        out.append(B.MeterParams(meters=meters, change_weight=cw, half_bar_change=half, reset_penalty=reset,
                                 meter_penalty=mp, four_prior=fp))
    return out


def _fmt(r: dict) -> str:
    return (f"B beatF {100 * r['beatles_beat_f']:.2f}  B dbF {100 * r['beatles_downbeat_f']:.2f}  B meter {100 * r['beatles_meter_acc']:.1f}%  "
            f"B lệch {100 * r['beatles_irregular_bars']:.1f}%  K đổi@1 {100 * r['kajiura_changes_on_downbeat']:.1f}%  "
            f"K lệch {100 * r['kajiura_irregular_bars']:.1f}%")


def run(do_grid: bool) -> int:
    data = load_data()
    nb = sum(1 for d in data if d["ds"] == "B")
    print(f"{nb} bài Beatles có nhãn beat, {len(data) - nb} bài Kajiura")
    rows = {}
    p0 = B.DEFAULT_METER_PARAMS
    for variant in VARIANTS:
        rows[variant] = score(data, variant, p0)
        print(f"{variant:14s} {_fmt(rows[variant])}")
    grid_rows = []
    if do_grid:
        for p in param_grid():
            r = score(data, "clean+refine", p)
            grid_rows.append({"params": asdict(p), **r})
        grid_rows.sort(key=lambda r: -r["beatles_downbeat_f"])
        print("\nTop 15 theo downbeat F của Beatles:")
        for r in grid_rows[:15]:
            p = r["params"]
            print(f"  meters {p['meters']} mp {p['meter_penalty']} cw {p['change_weight']} half {p['half_bar_change']} reset {p['reset_penalty']} "
                  f"four {p['four_prior']}: {_fmt(r)}")
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({"default": asdict(p0), "variants": rows, "grid": grid_rows},
                                       indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"Đã ghi {RESULTS_PATH}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Đánh giá phách mạnh và số chỉ nhịp")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--force", action="store_true")
    c = sub.add_parser("cache")
    c.add_argument("--force", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--grid", action="store_true")
    args = ap.parse_args(argv)
    _utf8_console()
    if args.cmd == "download":
        print(f"{download_beat_labels(args.force)} file nhãn beat")
        return 0
    if args.cmd == "cache":
        return cache(args.force)
    return run(args.grid)


if __name__ == "__main__":
    raise SystemExit(main())
