"""Chọn lại tham số `FusionParams` trên cả hai bộ (Beatles + Kajiura), chỉ dùng đặc trưng của mix.

    python -m eval.refit_fusion [--folds 3] [--jobs 6] [--quick] [--out EVAL_DIR/results/refit_fusion.json]

Cần đặc trưng đã cache của cả hai bộ: `python -m eval.cache_features` (Beatles) và
`python -m eval.ablation_kajiura cache` + `python -m eval.kajiura align` (Kajiura).

Không tách nguồn (mặc định mới của pipeline), nên mọi cấu hình trong lưới dùng `bass_source="cnn"`.

**Hàm mục tiêu** (ghi rõ để so được giữa các lần chạy):

    objective = 0.6 * mean4(Kajiura theo thời gian) + 0.4 * mean4(Beatles)
    mean4     = trung bình (majmin, majmin_inv, sevenths, sevenths_inv), WCSR gia quyền thời lượng

Kajiura theo thời gian chỉ gồm các bài căn tốt (`alignment.json` ok). F1 theo chuỗi của Kajiura (mọi bài)
chỉ báo thêm, không nằm trong hàm mục tiêu.

**Cross-validation:** chia bài thành `--folds` phần, riêng từng bộ theo thứ tự tên (bài thứ i vào phần
i mod k), nên phần nào cũng có cả bài Beatles lẫn Kajiura. Với mỗi phần: chọn cấu hình tốt nhất trên các
phần còn lại, chấm phần giữ lại. Điểm CV là điểm của các phần giữ lại ghép lại. Lưới được chấm một lần trên
mọi bài (nhanh: khoảng 30 ms một bài một cấu hình), việc chọn theo phần chỉ là cộng lại.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from chordtrack import fusion as F
from chordtrack.paths import EVAL_DIR
from eval import kajiura as K
from eval.ablation import Song, fusion_cfg as _ablation_fusion_cfg, predict
from eval.ablation_kajiura import OLD_DEFAULTS, KajiuraSong, features_path as kajiura_features, score_song
from eval.cache_features import features_path as beatles_features
from eval.datasets import REF_DIR
from eval.evaluate import METRICS, _nan_to_none, _utf8_console, evaluate_pair

OBJECTIVE = ("majmin", "majmin_inv", "sevenths", "sevenths_inv")
KAJIURA_WEIGHT = 0.6
BEATLES_WEIGHT = 0.4
SEQ_METRICS = ("seq_majmin", "seq_sevenths")

def fusion_cfg(**params) -> dict:
    return _ablation_fusion_cfg("mix", **params)


def grid(quick: bool = False) -> list[dict]:
    """Lưới tham số (toàn tích Descartes), mọi cấu hình bass_source="cnn", làm mượt HMM theo khung."""
    if quick:
        weights, quals, sps, mds, ths = (0.35, 0.7, 1.0), (0.0, 0.7), (0.8, 0.9), (0.25,), (0.4,)
    else:
        weights = (0.35, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        quals = (0.0, 0.35, 0.5, 0.7, 1.0)          # trọng số CNN-LSTM cho loại hợp âm trong họ ("family")
        sps = (0.7, 0.8, 0.9, 0.95)
        mds = (0.25, 0.5, 1.0)
        ths = (0.3, 0.4, 0.5, 0.6)
    levels = [("full", 0.0)] + [("family", q) for q in quals]
    out = []
    for w, (lv, q), sp, md, th in itertools.product(weights, levels, sps, mds, ths):
        out.append(fusion_cfg(cnn_weight=w, mix_level=lv, quality_weight=q, smoothing="hmm", self_prob=sp,
                              min_duration=md, bass_source="cnn", bass_threshold=th))
    return out


# ---------------------------------------------------------------- chấm (tiến trình con)

def _load(song_id: str):
    ds, name = song_id.split(":", 1)
    return Song(name) if ds == "B" else KajiuraSong(name)


def _score_song(args: tuple[str, list[dict]]) -> tuple[str, np.ndarray, np.ndarray]:
    """Trả về (song_id, correct (C, M), weights (C, M)) theo `METRICS` cho từng cấu hình."""
    song_id, cfgs = args
    song = _load(song_id)
    corr = np.zeros((len(cfgs), len(METRICS)))
    wts = np.zeros((len(cfgs), len(METRICS)))
    for i, cfg in enumerate(cfgs):
        r = evaluate_pair(song.ref, predict(song, cfg))
        corr[i] = [r["correct"][m] for m in METRICS]
        wts[i] = [r["weights"][m] for m in METRICS]
    return song_id, corr, wts


class Scores:
    """correct/weights (cấu hình, bài, mức) trên mọi bài chấm theo thời gian."""

    def __init__(self, cfgs: list[dict], songs: list[str], jobs: int):
        self.cfgs, self.songs = cfgs, songs
        self.index = {s: i for i, s in enumerate(songs)}
        self.corr = np.zeros((len(cfgs), len(songs), len(METRICS)))
        self.wts = np.zeros_like(self.corr)
        tasks = [(s, cfgs) for s in songs]
        t0 = time.perf_counter()
        with Pool(jobs) as pool:
            for k, (sid, c, w) in enumerate(pool.imap_unordered(_score_song, tasks), 1):
                i = self.index[sid]
                self.corr[:, i], self.wts[:, i] = c, w
                if k % 10 == 0 or k == len(songs):
                    print(f"  chấm {k}/{len(songs)} bài ({time.perf_counter() - t0:.0f}s)", flush=True)

    def per_song(self, ci: int, songs: list[str]) -> dict[str, dict]:
        """Kết quả theo bài của cấu hình thứ `ci`, dạng dùng cho `evaluate_rows`."""
        return {s: {"correct": self.corr[ci, self.index[s]].tolist(), "weights": self.wts[ci, self.index[s]].tolist()}
                for s in songs}

    def agg(self, songs: list[str]) -> np.ndarray:
        """(cấu hình, mức) WCSR gia quyền thời lượng trên `songs`."""
        if not songs:
            raise ValueError("không có bài nào để chấm (phần huấn luyện rỗng?)")
        idx = [self.index[s] for s in songs]
        c, w = self.corr[:, idx].sum(axis=1), self.wts[:, idx].sum(axis=1)
        return np.where(w > 0, c / np.maximum(w, 1e-12), np.nan)


OBJ_IDX = [METRICS.index(m) for m in OBJECTIVE]


def mean4(agg: np.ndarray) -> np.ndarray:
    return agg[..., OBJ_IDX].mean(axis=-1)


def objective(scores: Scores, beatles: list[str], kajiura: list[str]) -> np.ndarray:
    return KAJIURA_WEIGHT * mean4(scores.agg(kajiura)) + BEATLES_WEIGHT * mean4(scores.agg(beatles))


# ---------------------------------------------------------------- báo cáo

def _metrics(corr: np.ndarray, wts: np.ndarray) -> dict:
    c, w = corr.sum(axis=0), wts.sum(axis=0)
    out = {m: float(c[j] / w[j]) if w[j] > 0 else float("nan") for j, m in enumerate(METRICS)}
    out["mean4"] = float(np.mean([out[m] for m in OBJECTIVE]))
    return out


def evaluate_rows(per_song: dict[str, dict], beatles: list[str], kajiura_ok: list[str]) -> dict:
    """per_song: song_id -> {"correct": [...], "weights": [...]} -> điểm tổng của từng bộ."""
    def part(ids):
        return _metrics(np.array([per_song[s]["correct"] for s in ids]), np.array([per_song[s]["weights"] for s in ids]))
    b, k = part(beatles), part(kajiura_ok)
    return {"beatles": b, "kajiura": k, "objective": KAJIURA_WEIGHT * k["mean4"] + BEATLES_WEIGHT * b["mean4"]}


def seq_f1(cfg_for_song, names: list[str], report: dict) -> dict:
    """F1 theo chuỗi Kajiura (trung bình theo bài) với cấu hình có thể khác nhau theo bài (CV)."""
    vals = {m: [] for m in SEQ_METRICS}
    per = {}
    for n in names:
        song = KajiuraSong(n)
        s = score_song(song, K.load_sheet(n), report[n], predict(song, cfg_for_song(n)))
        per[n] = {m: s[m] for m in SEQ_METRICS}
        for m in SEQ_METRICS:
            vals[m].append(s[m])
    return {"mean": {m: float(np.mean(v)) for m, v in vals.items()}, "per_song": per}


def describe(cfg: dict) -> str:
    if cfg["kind"] != "fusion":
        return cfg["kind"]
    p = cfg["params"]
    keys = ("cnn_weight", "mix_level", "quality_weight", "self_prob", "min_duration", "bass_source", "bass_threshold")
    return ", ".join(f"{k}={p[k]}" for k in keys)


def print_table(rows: list[tuple[str, dict]]) -> None:
    head = ("| phương án | K majmin | K majmin_inv | K sevenths | K sevenths_inv | K mean4 | K F1 chuỗi majmin "
            "| K F1 chuỗi sevenths | B majmin | B majmin_inv | B sevenths | B sevenths_inv | B mean4 | mục tiêu |")
    print(head)
    print("|---" * (head.count("|") - 1) + "|")
    for name, r in rows:
        k, b, s = r["kajiura"], r["beatles"], r["seq"]
        cells = [k[m] for m in OBJECTIVE] + [k["mean4"]] + [s["seq_majmin"], s["seq_sevenths"]] + \
                [b[m] for m in OBJECTIVE] + [b["mean4"], r["objective"]]
        print(f"| {name} | " + " | ".join(f"{100 * v:.2f}" for v in cells) + " |")


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Chọn lại FusionParams trên Beatles + Kajiura (mix, không tách nguồn)")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--quick", action="store_true", help="lưới nhỏ để thử")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    _utf8_console()
    out_path = args.out or EVAL_DIR / "results" / "refit_fusion.json"

    beatles = [f"B:{r.stem}" for r in sorted(REF_DIR.glob("*.lab")) if beatles_features(r.stem).exists()]
    report = K.read_report()
    k_names = sorted(n for n in (r["name"] for r in K.read_mapping())
                     if kajiura_features(n).exists() and n in report and report[n]["end"] > report[n]["start"])
    k_ok = [n for n in k_names if report[n]["ok"]]
    kajiura = [f"K:{n}" for n in k_ok]
    if not beatles or not kajiura:
        print("Thiếu đặc trưng của Beatles hoặc Kajiura", file=sys.stderr)
        return 1
    folds_b = [[s for i, s in enumerate(beatles) if i % args.folds == f] for f in range(args.folds)]
    folds_kn = [[n for i, n in enumerate(k_names) if i % args.folds == f] for f in range(args.folds)]
    folds_k = [[f"K:{n}" for n in fk if n in set(k_ok)] for fk in folds_kn]
    print(f"Beatles {len(beatles)} bài, Kajiura {len(k_names)} bài ({len(k_ok)} căn tốt), {args.folds} phần: "
          f"Beatles {[len(f) for f in folds_b]}, Kajiura {[len(f) for f in folds_k]} căn tốt / "
          f"{[len(f) for f in folds_kn]} cả bộ", flush=True)

    cfgs = grid(args.quick)
    print(f"Lưới {len(cfgs)} cấu hình x {len(beatles) + len(kajiura)} bài", flush=True)
    t0 = time.perf_counter()
    scores = Scores(cfgs, beatles + kajiura, args.jobs)

    # Chọn trên cả bộ
    obj = objective(scores, beatles, kajiura)
    order = np.argsort(-obj)
    best_full = cfgs[int(order[0])]

    # Cross-validation
    fold_cfgs = []
    held: dict[str, dict] = {}
    cfg_of_k: dict[str, dict] = {}
    for f in range(args.folds):
        test_b, test_k = set(folds_b[f]), set(folds_k[f])
        tb = [s for s in beatles if s not in test_b]
        tk = [s for s in kajiura if s not in test_k]
        ci = int(np.argmax(objective(scores, tb, tk)))
        c = cfgs[ci]
        fold_cfgs.append(c)
        held.update(scores.per_song(ci, folds_b[f] + folds_k[f]))
        for n in folds_kn[f]:
            cfg_of_k[n] = c
        print(f"phần {f}: {describe(c)}", flush=True)
    print(f"cả bộ: {describe(best_full)}", flush=True)

    # Bảng so sánh
    baselines = [
        ("CNN-LSTM một mình", {"kind": "cnnlstm", "src": "mix"}),
        ("ChordMini một mình", {"kind": "chordmini", "src": "mix"}),
        ("fuse cũ (mặc định cũ, có stem bass)", fusion_cfg(**OLD_DEFAULTS)),
        ("fuse cũ, bass CNN-LSTM (không tách nguồn)", fusion_cfg(**{**OLD_DEFAULTS, "bass_source": "cnn"})),
    ]
    base_scores = Scores([c for _, c in baselines], beatles + kajiura, args.jobs)
    rows = []
    per_song_rows = {}
    for bi, (name, cfg) in enumerate(baselines):
        ps = base_scores.per_song(bi, beatles + kajiura)
        r = evaluate_rows(ps, beatles, kajiura)
        r["seq"] = seq_f1(lambda n, c=cfg: c, k_names, report)["mean"]
        rows.append((name, r))
        per_song_rows[name] = ps
    cv = evaluate_rows(held, beatles, kajiura)
    cv["seq"] = seq_f1(lambda n: cfg_of_k[n], k_names, report)["mean"]
    rows.append((f"fuse mới (CV {args.folds} phần)", cv))
    per_song_rows["cv"] = held
    full = scores.per_song(int(order[0]), beatles + kajiura)
    fr = evaluate_rows(full, beatles, kajiura)
    fr["seq"] = seq_f1(lambda n: best_full, k_names, report)["mean"]
    rows.append(("fuse mới (chọn trên cả bộ)", fr))
    per_song_rows["full"] = full

    print(f"\nMục tiêu = {KAJIURA_WEIGHT} x mean4(Kajiura theo thời gian, {len(kajiura)} bài) + "
          f"{BEATLES_WEIGHT} x mean4(Beatles, {len(beatles)} bài); F1 chuỗi trên {len(k_names)} bài Kajiura.\n")
    print_table(rows)

    # Chênh lệch theo bài (Kajiura, theo thời gian): fuse mới (CV) so với CNN-LSTM một mình
    from eval.ablation_kajiura import time_weighted_diff

    rng = np.random.default_rng(0)
    cn = per_song_rows["CNN-LSTM một mình"]
    def as_res(ps):
        return {s: {"correct": dict(zip(METRICS, v["correct"])), "weights": dict(zip(METRICS, v["weights"]))}
                for s, v in ps.items()}
    diffs = {}
    for label, ps in (("CV - CNN-LSTM", held), ("cả bộ - CNN-LSTM", full)):
        for m in OBJECTIVE:
            diffs[f"{label} {m}"] = time_weighted_diff(as_res(ps), as_res(cn), kajiura, m, 2000, rng)
    print("\nKajiura theo thời gian, chênh lệch so với CNN-LSTM một mình [CI bootstrap 95%], thắng/thua:")
    for k, d in diffs.items():
        print(f"  {k}: {100 * d['mean']:+.2f} [{100 * d['lo']:+.2f}, {100 * d['hi']:+.2f}] {d['wins']}/{d['losses']}")

    print("\nTop 10 cấu hình trên cả bộ (mục tiêu, K mean4, B mean4):")
    kb, bb = mean4(scores.agg(kajiura)), mean4(scores.agg(beatles))
    for i in order[:10]:
        print(f"  {100 * obj[i]:.2f}  K {100 * kb[i]:.2f}  B {100 * bb[i]:.2f}  {describe(cfgs[i])}")
    print(f"\nMặc định mới đề xuất: {describe(best_full)}")
    print(f"{len(cfgs)} cấu hình, {time.perf_counter() - t0:.0f}s")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_nan_to_none({
        "objective": f"{KAJIURA_WEIGHT} * mean4(kajiura) + {BEATLES_WEIGHT} * mean4(beatles)",
        "beatles": beatles, "kajiura_time": kajiura, "kajiura_seq": k_names,
        "folds": {"beatles": folds_b, "kajiura": folds_kn},
        "fold_configs": fold_cfgs, "best_full": best_full,
        "table": {n: r for n, r in rows}, "diffs_vs_cnnlstm": diffs,
        "top": [{"objective": float(obj[i]), "kajiura": float(kb[i]), "beatles": float(bb[i]), "config": cfgs[i]}
                for i in order[:50]],
    }), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Đã ghi {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
