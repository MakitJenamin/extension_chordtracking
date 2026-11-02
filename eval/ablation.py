"""So sánh các cách kết hợp model và hậu xử lý hợp âm, chọn tham số bằng cross-validation.

    python -m eval.ablation [--folds 3] [--jobs 6] [--out EVAL_DIR/results/ablation.json]

Cần chạy `python -m eval.cache_features` trước (đặc trưng của mọi bài trong `ref/`).

Các biến thể (mỗi biến thể một dòng trong bảng):
 1  ChordMini trên mix (baseline)
 2  ChordMini trên instrumental
 3  CNN-LSTM trên mix / instrumental
 4  ChordMini + bass từ `bass_probs` của CNN-LSTM
 5  ChordMini + bass từ chroma của stem bass
 6  ChordMini + bass trộn hai nguồn
 7  Trộn xác suất hai model hợp âm (trung bình log), giải mã HMM: a) trên cả 170 nhãn, b) theo họ hợp âm
 8  Làm mượt: bỏ phiếu theo phách, HMM theo khung, HMM theo phách (chỉ ChordMini, không bass)
 9  Bộ tốt nhất: tìm theo hai tầng (tầng hợp âm rồi tầng bass)

Biến thể có tham số được chọn bằng cross-validation: chia bài thành `--folds` phần theo thứ tự tên
(bài thứ i vào phần i mod k), chọn tham số tốt nhất trên các phần còn lại rồi chấm phần giữ lại. Điểm
CV là trung bình gia quyền thời lượng trên các phần giữ lại ghép lại. Hàm mục tiêu là trung bình của
majmin, majmin_inv, sevenths, sevenths_inv. Bảng thứ hai là điểm khi chọn tham số trên cả bộ (lạc quan
hơn, chỉ để tham khảo). Bộ này từng là mặc định của `FusionParams` (đợt 2, chỉ Beatles); mặc định hiện tại
chọn trên Beatles + Kajiura bằng `eval/refit_fusion.py`.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import asdict, replace
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from chordtrack import fusion as F
from chordtrack.paths import EVAL_DIR
from chordtrack.types import ChordSegment
from eval.cache_features import FEATURES_DIR, features_path, load_features
from eval.datasets import REF_DIR
from eval.evaluate import METRICS, SEG_METRICS, TABLE_METRICS, _nan_to_none, _utf8_console, aggregate, evaluate_pair

OBJECTIVE = ("majmin", "majmin_inv", "sevenths", "sevenths_inv")
SOURCES = {"mix": "mix", "inst": "instrumental"}

# ---------------------------------------------------------------- dữ liệu một bài (trong tiến trình con)


class Song:
    def __init__(self, name: str):
        self.name = name
        self.ref = REF_DIR / f"{name}.lab"
        f = load_features(name)
        self.duration = float(f["duration"])
        self.beats = f["beats"]
        self.models = f["models"]
        self.stems_dir = Path(f["stems_dir"])
        self.times = F.frame_grid(self.duration)
        self._cache: dict = {}

    def model(self, kind: str, src: str):
        return self.models[f"{kind}_{SOURCES[src]}"]

    def cm_probs(self, src: str) -> np.ndarray:
        key = ("cm", src)
        if key not in self._cache:
            m = self.model("chordmini", src)
            self._cache[key] = F._normalize(F.resample_probs(m.frame_times, m.frame_probs, self.times))
        return self._cache[key]

    def cn_probs(self, src: str) -> np.ndarray:
        key = ("cn", src)
        if key not in self._cache:
            self._cache[key] = F.cnnlstm_on_chordmini(self.model("cnnlstm", src), self.model("chordmini", src),
                                                       self.times, self.cm_probs(src))
        return self._cache[key]

    def chord(self, src: str, w: float, level: str, quality_weight: float = 0.0) -> np.ndarray:
        """Giống `fusion.chord_probs` (dùng chung các hàm của fusion), nhớ phần đã tính để chạy nhanh."""
        if w <= 0 and (level != "family" or quality_weight <= 0):
            return self.cm_probs(src)
        abstain = F.cnnlstm_abstain(list(self.model("cnnlstm", src).vocabulary), self.vocab(src))
        return F.mix_models(self.cm_probs(src), self.cn_probs(src), self.vocab(src), w, level, quality_weight, abstain)

    def cnn_bass(self, src: str) -> np.ndarray:
        key = ("bass", src)
        if key not in self._cache:
            self._cache[key] = F.cnnlstm_bass(self.model("cnnlstm", src), self.times)
        return self._cache[key]

    def stem_chroma(self) -> np.ndarray:
        if "stem" not in self._cache:
            path = FEATURES_DIR / f"{self.name}.bass_chroma.npz"
            if path.exists():
                d = np.load(path)
                t, c = d["times"], d["chroma"]
            else:
                t, c = F.bass_chroma(self.stems_dir / "bass.wav")
                c = c.astype(np.float32)  # dùng đúng giá trị như khi đọc lại từ cache
                tmp = path.with_name(path.stem + ".part.npz")
                np.savez_compressed(tmp, times=t, chroma=c)
                tmp.replace(path)
            self._cache["stem"] = F.resample_probs(t, c, self.times)
        return self._cache["stem"]

    def vocab(self, src: str) -> list[str]:
        return list(self.model("chordmini", src).vocabulary)


@lru_cache(maxsize=4)
def _song(name: str) -> Song:
    return Song(name)


def _segments_with_bass(song: Song, src: str, base: list[ChordSegment], p: F.FusionParams) -> list[ChordSegment]:
    """Gán bass cho đoạn sẵn có của ChordMini (biến thể 4–6)."""
    vocab = song.vocab(src)
    index = {lab: i for i, lab in enumerate(vocab)}
    segs = [(s.start, s.end, index.get(s.label, index["N"])) for s in base]
    bass = F.bass_distribution(song.cnn_bass(src) if p.bass_source in ("cnn", "both") else None,
                               song.stem_chroma() if p.bass_source in ("stem", "both") else None, p)
    if bass is None:
        return base
    intervals = F._beat_intervals(song.beats, song.duration) if p.bass_per_beat else None
    out = F.assign_bass(segs, vocab, bass, song.times, p.bass_threshold, intervals)
    return [ChordSegment(a, b, lab) for a, b, _i, lab in out]


def predict(song: Song, cfg: dict) -> list[ChordSegment]:
    """cfg: {"kind": ..., "src": "mix"|"inst", "params": dict FusionParams}."""
    kind, src = cfg["kind"], cfg["src"]
    if kind == "chordmini":
        return F.fill_gaps(song.model("chordmini", src).segments, song.duration)
    if kind == "cnnlstm":
        return F.fill_gaps(song.model("cnnlstm", src).segments, song.duration)
    p = F.FusionParams(**cfg["params"])
    if kind == "cm_bass":
        base = F.fill_gaps(song.model("chordmini", src).segments, song.duration)
        return _segments_with_bass(song, src, base, p)
    if kind == "fusion":
        cnn_bass = song.cnn_bass(src) if p.bass_source in ("cnn", "both") else None
        stem = song.stem_chroma() if p.bass_source in ("stem", "both") else None
        return F.fuse_probs(song.chord(src, p.cnn_weight, p.mix_level, p.quality_weight), song.vocab(src), song.times, song.beats, song.duration, p,
                            cnn_bass=cnn_bass, stem_chroma=stem)
    raise ValueError(kind)


def cfg_key(cfg: dict) -> str:
    return json.dumps(cfg, sort_keys=True)


def _run_song(args: tuple[str, list[dict]]) -> tuple[str, dict[str, dict]]:
    name, cfgs = args
    song = _song(name)
    out = {}
    for cfg in cfgs:
        r = evaluate_pair(song.ref, predict(song, cfg))
        out[cfg_key(cfg)] = r
    return name, out


class Runner:
    """Chấm (bài, cấu hình) song song, nhớ kết quả để các tầng tìm kiếm dùng lại."""

    def __init__(self, names: list[str], jobs: int):
        self.names = names
        self.results: dict[str, dict[str, dict]] = {}   # cfg_key -> {bài: kết quả}
        self.pool = Pool(jobs) if jobs > 1 else None

    def run(self, cfgs: list[dict]) -> None:
        todo = [c for c in cfgs if cfg_key(c) not in self.results]
        if not todo:
            return
        tasks = [(n, todo) for n in self.names]
        it = self.pool.imap_unordered(_run_song, tasks) if self.pool else map(_run_song, tasks)
        for name, res in it:
            for k, r in res.items():
                self.results.setdefault(k, {})[name] = r

    def agg(self, cfg: dict, names: list[str]) -> dict:
        r = self.results[cfg_key(cfg)]
        return aggregate({n: r[n] for n in names})

    def close(self) -> None:
        if self.pool:
            self.pool.close()
            self.pool.join()


def objective(agg: dict) -> float:
    return float(np.mean([agg[m] for m in OBJECTIVE]))


def best_cfg(runner: Runner, cfgs: list[dict], names: list[str]) -> dict:
    runner.run(cfgs)
    return max(cfgs, key=lambda c: objective(runner.agg(c, names)))


# ---------------------------------------------------------------- không gian tìm kiếm

def fusion_cfg(src: str, **params) -> dict:
    return {"kind": "fusion", "src": src, "params": asdict(F.FusionParams(**params))}


BASS_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def bass_grid(source: str) -> list[dict]:
    weights = (0.3, 0.5, 0.7) if source == "both" else (0.5,)
    return [dict(bass_source=source, stem_weight=w, bass_threshold=t, bass_per_beat=pb)
            for w, t, pb in itertools.product(weights, BASS_THRESHOLDS, (False, True))]


def chord_grid(srcs=("mix", "inst"), weights=(0.0, 0.2, 0.35, 0.5), levels=("full", "family"),
               smoothings=("beat", "hmm", "beat_hmm"), self_probs=(0.8, 0.9, 0.95)) -> list[dict]:
    out = []
    for src, w, sm in itertools.product(srcs, weights, smoothings):
        sps = (0.9,) if sm == "beat" else self_probs
        for sp, lv in itertools.product(sps, levels if w > 0 else ("family",)):
            # quality_weight=0: cách trộn "family" của đợt 2 (giữ loại hợp âm 7 của ChordMini).
            out.append(fusion_cfg(src, cnn_weight=w, mix_level=lv, quality_weight=0.0, smoothing=sm, self_prob=sp,
                                  bass_source="none"))
    return out


def with_params(cfg: dict, **params) -> dict:
    p = F.FusionParams(**cfg["params"])
    return {**cfg, "params": asdict(replace(p, **params))}


# Mỗi biến thể có tham số: hàm (runner, các bài huấn luyện) -> cấu hình chọn được.
def search_cm_bass(source: str):
    def search(runner: Runner, train: list[str]) -> dict:
        src = best_cfg(runner, [{"kind": "chordmini", "src": s} for s in SOURCES], train)["src"]
        cfgs = [{"kind": "cm_bass", "src": src, "params": asdict(F.FusionParams(**b))} for b in bass_grid(source)]
        return best_cfg(runner, cfgs, train)
    return search


def search_mix(level: str):
    def search(runner: Runner, train: list[str]) -> dict:
        return best_cfg(runner, chord_grid(weights=(0.2, 0.35, 0.5, 0.65), levels=(level,), smoothings=("hmm",),
                                           self_probs=(0.9,)), train)
    return search


def search_smoothing(runner: Runner, train: list[str]) -> dict:
    return best_cfg(runner, chord_grid(weights=(0.0,)), train)


def search_full(runner: Runner, train: list[str]) -> dict:
    chord = best_cfg(runner, chord_grid(), train)
    cfgs = [with_params(chord, **b) for s in ("cnn", "stem", "both") for b in bass_grid(s)]
    best = best_cfg(runner, cfgs, train)
    # Tầng 3: thử lại độ dài tối thiểu với bộ đã chọn.
    return best_cfg(runner, [with_params(best, min_duration=d) for d in (0.25, 0.5, 1.0)], train)


VARIANTS: list[tuple[str, object]] = [
    ("1 ChordMini mix (baseline)", {"kind": "chordmini", "src": "mix"}),
    ("2 ChordMini instrumental", {"kind": "chordmini", "src": "inst"}),
    ("3a CNN-LSTM mix", {"kind": "cnnlstm", "src": "mix"}),
    ("3b CNN-LSTM instrumental", {"kind": "cnnlstm", "src": "inst"}),
    ("4 ChordMini + bass CNN-LSTM", search_cm_bass("cnn")),
    ("5 ChordMini + bass stem", search_cm_bass("stem")),
    ("6 ChordMini + bass cả hai", search_cm_bass("both")),
    ("7a Trộn 2 model (cả 170 nhãn), HMM", search_mix("full")),
    ("7b Trộn 2 model (theo họ hợp âm), HMM", search_mix("family")),
    ("8 Làm mượt (phách/HMM), ChordMini", search_smoothing),
    ("9 Bộ tốt nhất", search_full),
]


def describe(cfg: dict) -> str:
    if cfg["kind"] in ("chordmini", "cnnlstm"):
        return f"{cfg['kind']} {cfg['src']}"
    p = cfg["params"]
    d = F.FusionParams()
    diff = {k: v for k, v in p.items() if v != getattr(d, k)}
    return f"{cfg['kind']} {cfg['src']} " + ", ".join(f"{k}={v}" for k, v in sorted(diff.items()))


def format_rows(rows: list[tuple[str, dict]]) -> str:
    width = max(len(n) for n, _ in rows)
    head = f"{'biến thể':<{width}} " + " ".join(f"{m:>12}" for m in TABLE_METRICS)
    lines = [head, "-" * len(head)]
    for name, agg in rows:
        lines.append(f"{name:<{width}} " + " ".join(f"{100 * agg[m]:>12.2f}" for m in TABLE_METRICS))
    return "\n".join(lines)


def markdown_rows(rows: list[tuple[str, dict]]) -> str:
    cols = ("root", *TABLE_METRICS[1:])
    out = ["| biến thể | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, agg in rows:
        out.append(f"| {name} | " + " | ".join(f"{100 * agg[m]:.2f}" for m in cols) + " |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ablation fusion hợp âm, chọn tham số bằng cross-validation")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--only", nargs="*", default=None, help="chỉ chạy các biến thể có số này, ví dụ 1 9")
    ap.add_argument("--out", type=Path, default=EVAL_DIR / "results" / "ablation.json")
    args = ap.parse_args(argv)
    _utf8_console()

    refs = sorted(REF_DIR.glob("*.lab"))
    names = [r.stem for r in refs if features_path(r.stem).exists()]
    missing = [r.stem for r in refs if not features_path(r.stem).exists()]
    if missing:
        print(f"Thiếu đặc trưng cho {len(missing)} bài (chạy eval.cache_features): {', '.join(missing)}", file=sys.stderr)
    if not names:
        return 1
    folds = [[n for i, n in enumerate(names) if i % args.folds == k] for k in range(args.folds)]
    print(f"{len(names)} bài, {args.folds} phần: {[len(f) for f in folds]}", flush=True)

    t0 = time.perf_counter()
    runner = Runner(names, args.jobs)
    cv_rows, full_rows, report = [], [], {}
    try:
        for vname, spec in VARIANTS:
            if args.only and vname.split()[0].rstrip("ab") not in args.only:
                continue
            if isinstance(spec, dict):
                runner.run([spec])
                agg = runner.agg(spec, names)
                cv_rows.append((vname, agg))
                full_rows.append((vname, agg))
                report[vname] = {"cv": agg, "full": agg, "config": spec}
                print(f"{vname}: {describe(spec)} ({time.perf_counter() - t0:.0f}s)", flush=True)
                continue
            # Cross-validation: chọn trên các phần còn lại, chấm phần giữ lại.
            held: dict[str, dict] = {}
            chosen = []
            for k, test in enumerate(folds):
                train = [n for n in names if n not in set(test)]
                cfg = spec(runner, train)
                chosen.append(cfg)
                runner.run([cfg])
                held.update({n: runner.results[cfg_key(cfg)][n] for n in test})
            cv = aggregate(held)
            full_cfg = spec(runner, names)
            full = runner.agg(full_cfg, names)
            cv_rows.append((vname, cv))
            full_rows.append((vname, full))
            report[vname] = {"cv": cv, "full": full, "config": full_cfg, "fold_configs": chosen}
            print(f"{vname}: cả bộ -> {describe(full_cfg)}; theo phần -> "
                  + " | ".join(describe(c) for c in chosen) + f" ({time.perf_counter() - t0:.0f}s)", flush=True)
    finally:
        runner.close()

    print(f"\n=== Điểm cross-validation ({args.folds} phần, chấm trên phần giữ lại) ===")
    print(format_rows(cv_rows))
    print("\n=== Chọn tham số trên cả bộ (lạc quan, để tham khảo) ===")
    print(format_rows(full_rows))
    print("\nMarkdown (CV):\n" + markdown_rows(cv_rows))
    print("\nMarkdown (cả bộ):\n" + markdown_rows(full_rows))
    print(f"\n{len(names)} bài, {sum(len(v) for v in runner.results.values())} lần chấm, "
          f"{time.perf_counter() - t0:.0f}s")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_nan_to_none({"songs": names, "folds": folds, "variants": report}),
                                   indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
