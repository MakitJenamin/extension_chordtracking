"""Đánh giá ChordFormer và consonance-ACE, một mình và trong bộ trộn với ChordMini, CNN-LSTM.

    python -m eval.new_models_eval [--jobs 6] [--folds 3] [--quick]

Cần: đặc trưng gốc (`eval.cache_features`, `eval.ablation_kajiura cache`, `eval.kajiura align`) và
`python -m eval.cache_new_models`. Mọi model chạy trên **mix**, không tách nguồn.

Bốn model, ký hiệu: cm = ChordMini BTC-PL, cn = Chord-CNN-LSTM, cf = ChordFormer, cs = consonance-ACE.
Cách trộn tổng quát hóa `fusion.mix_models(level="family")`:
- **Họ hợp âm** (gốc + maj/min/dim/aug/sus): trung bình log có trọng số của cả bốn model (`w`, tổng 1),
  mỗi model đã đưa về từ vựng 170 nhãn của ChordMini (cn, cf qua `cnnlstm_on_chordmini`).
- **Loại hợp âm trong họ** (C, C7, Cmaj7): trộn ChordMini với một model `qsrc` (cn, cf hoặc cs) theo
  trọng số `q`, giống `quality_weight`.
- **Bass**: `bass_probs` của cn, cf, cs hoặc trung bình (`bass`), ngưỡng `th`.
- Làm mượt HMM theo khung (`sp`), gộp đoạn < `md` giây, như `fusion.fuse_probs`.

Hàm mục tiêu, cross-validation và cách chấm giống `eval.refit_fusion` (0.6 Kajiura + 0.4 Beatles, mean4).
⚠️ consonance-ACE và ChordFormer đều học trên Isophonics (Beatles), nên điểm Beatles của chúng lạc quan.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
import time
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from chordtrack import fusion as F
from chordtrack.paths import EVAL_DIR
from chordtrack.types import ChordSegment
from eval import kajiura as K
from eval.ablation import Song
from eval.ablation_kajiura import KajiuraSong, score_song
from eval.ablation_kajiura import features_path as kajiura_features
from eval.cache_features import features_path as beatles_features
from eval.cache_new_models import new_models_path
from eval.datasets import REF_DIR
from eval.evaluate import METRICS, _nan_to_none, _utf8_console, evaluate_pair
from eval.refit_fusion import BEATLES_WEIGHT, KAJIURA_WEIGHT, OBJECTIVE, SEQ_METRICS, evaluate_rows, mean4

MODELS = ("cm", "cn", "cf", "cs")
KEYS = {"cm": "chordmini_mix", "cn": "cnnlstm_mix", "cf": "chordformer_mix", "cs": "consonance_mix"}


# ---------------------------------------------------------------- bài hát

class _Probs:
    """Xác suất của bốn model trên lưới khung chung và từ vựng ChordMini, nhớ lại để chấm nhanh."""

    def _init_new(self, feats_pkl: Path):
        with open(new_models_path(feats_pkl), "rb") as f:
            self.models.update(pickle.load(f)["models"])
        self._p: dict = {}

    def vocab_cm(self) -> list[str]:
        return list(self.models["chordmini_mix"].vocabulary)

    def p(self, x: str) -> np.ndarray:
        if x not in self._p:
            cm = self.models["chordmini_mix"]
            if x == "cm":
                v = F._normalize(F.resample_probs(cm.frame_times, cm.frame_probs, self.times))
            elif x in ("cn", "cf"):
                v = F.cnnlstm_on_chordmini(self.models[KEYS[x]], cm, self.times, self.p("cm"))
            else:
                m = self.models[KEYS[x]]
                assert list(m.vocabulary) == self.vocab_cm()
                v = F._normalize(F.resample_probs(m.frame_times, m.frame_probs, self.times))
            self._p[x] = v
        return self._p[x]

    def abstain(self, x: str) -> np.ndarray | None:
        if x in ("cn", "cf"):
            return F.cnnlstm_abstain(list(self.models[KEYS[x]].vocabulary), self.vocab_cm())
        return None

    def bass(self, src: str) -> np.ndarray:
        key = ("bass", src)
        if key not in self._p:
            if src == "mean":
                v = np.mean([self.bass("cn"), self.bass("cf"), self.bass("cs")], axis=0)
            elif src == "cnf":
                v = np.mean([self.bass("cn"), self.bass("cf")], axis=0)
            else:
                v = F.cnnlstm_bass(self.models[KEYS[src]], self.times)
            self._p[key] = v
        return self._p[key]


class BSong(_Probs, Song):
    def __init__(self, name: str):
        Song.__init__(self, name)
        self._init_new(beatles_features(name))


class KSong(_Probs, KajiuraSong):
    def __init__(self, name: str):
        KajiuraSong.__init__(self, name)
        self._init_new(kajiura_features(name))


@lru_cache(maxsize=2)
def load(song_id: str):
    ds, name = song_id.split(":", 1)
    return BSong(name) if ds == "B" else KSong(name)


# ---------------------------------------------------------------- dự đoán

def mix(song: _Probs, w: dict, qsrc: str, q: float) -> np.ndarray:
    vocab = song.vocab_cm()
    a = F._family_matrix(tuple(vocab))
    logfam = 0.0
    for x, wx in w.items():
        if wx > 0:
            logfam = logfam + wx * np.log(song.p(x) @ a + F.EPS)
    logfam = logfam - logfam.max(axis=1, keepdims=True)
    fam = F._normalize(np.exp(logfam))
    p_cm = song.p("cm")
    fam_cm = p_cm @ a
    within = p_cm / np.maximum(fam_cm @ a.T, 1e-12)
    if q > 0 and qsrc != "cm":
        p_q = song.p(qsrc)
        ab = song.abstain(qsrc)
        if ab is None:
            within_q = p_q / np.maximum((p_q @ a) @ a.T, 1e-12)
        else:
            cov = ~ab
            cov_mass = ((p_q * cov) @ a) @ a.T
            free = 1.0 - ((within * ~cov) @ a) @ a.T
            within_q = np.where(cov, p_q / np.maximum(cov_mass, 1e-12) * free, within)
        within = np.exp((1 - q) * np.log(within + F.EPS) + q * np.log(within_q + F.EPS))
        within = within / np.maximum((within @ a) @ a.T, 1e-300)
    return F._normalize(within * (fam @ a.T))


def predict(song, cfg: dict) -> list[ChordSegment]:
    kind = cfg["kind"]
    if kind == "raw":
        return F.fill_gaps(song.models[KEYS[cfg["model"]]].segments, song.duration)
    probs = mix(song, cfg["w"], cfg["qsrc"], cfg["q"])
    p = F.FusionParams(smoothing="hmm", self_prob=cfg["sp"], min_duration=cfg["md"],
                       bass_source="cnn" if cfg["bass"] != "none" else "none", bass_threshold=cfg["th"])
    bass = song.bass(cfg["bass"]) if cfg["bass"] != "none" else None
    return F.fuse_probs(probs, song.vocab_cm(), song.times, song.beats, song.duration, p, cnn_bass=bass)


def cfg(w: dict, qsrc="cn", q=1.0, bass="cn", th=0.5, sp=0.7, md=0.25) -> dict:
    w = {x: round(float(w.get(x, 0.0)), 4) for x in MODELS}
    return {"kind": "multi", "w": w, "qsrc": qsrc, "q": q, "bass": bass, "th": th, "sp": sp, "md": md}


def describe(c: dict) -> str:
    if c["kind"] == "raw":
        return f"{c['model']} (đầu ra gốc)"
    w = " ".join(f"{x}={c['w'][x]:g}" for x in MODELS if c["w"][x] > 0)
    return f"w[{w}] loại={c['qsrc']}:{c['q']:g} bass={c['bass']}@{c['th']:g} sp={c['sp']:g} md={c['md']:g}"


# ---------------------------------------------------------------- chấm

def _score(args):
    sid, cfgs = args
    song = load(sid)
    corr = np.zeros((len(cfgs), len(METRICS)))
    wts = np.zeros_like(corr)
    for i, c in enumerate(cfgs):
        r = evaluate_pair(song.ref, predict(song, c))
        corr[i] = [r["correct"][m] for m in METRICS]
        wts[i] = [r["weights"][m] for m in METRICS]
    return sid, corr, wts


class Scores:
    def __init__(self, cfgs, songs, jobs):
        self.cfgs, self.songs = cfgs, songs
        self.index = {s: i for i, s in enumerate(songs)}
        self.corr = np.zeros((len(cfgs), len(songs), len(METRICS)))
        self.wts = np.zeros_like(self.corr)
        t0 = time.perf_counter()
        with Pool(jobs) as pool:
            for k, (sid, c, w) in enumerate(pool.imap_unordered(_score, [(s, cfgs) for s in songs]), 1):
                i = self.index[sid]
                self.corr[:, i], self.wts[:, i] = c, w
                if k % 15 == 0 or k == len(songs):
                    print(f"  chấm {k}/{len(songs)} bài, {len(cfgs)} cấu hình ({time.perf_counter() - t0:.0f}s)", flush=True)

    def agg(self, songs):
        idx = [self.index[s] for s in songs]
        c, w = self.corr[:, idx].sum(axis=1), self.wts[:, idx].sum(axis=1)
        return np.where(w > 0, c / np.maximum(w, 1e-12), np.nan)

    def per_song(self, ci, songs):
        return {s: {"correct": self.corr[ci, self.index[s]].tolist(), "weights": self.wts[ci, self.index[s]].tolist()}
                for s in songs}


def objective(scores, beatles, kajiura):
    return KAJIURA_WEIGHT * mean4(scores.agg(kajiura)) + BEATLES_WEIGHT * mean4(scores.agg(beatles))


def seq_f1(cfg_for, names, report) -> dict:
    vals = {m: [] for m in SEQ_METRICS}
    for n in names:
        song = load(f"K:{n}")
        s = score_song(song, K.load_sheet(n), report[n], predict(song, cfg_for(n)))
        for m in SEQ_METRICS:
            vals[m].append(s[m])
    return {m: float(np.mean(v)) for m, v in vals.items()}


# ---------------------------------------------------------------- lưới

def simplex(step: float, models=MODELS):
    n = int(round(1 / step))
    for parts in itertools.product(range(n + 1), repeat=len(models)):
        if sum(parts) == n:
            yield {x: k / n for x, k in zip(models, parts)}


def grid_chords(quick: bool) -> list[dict]:
    step = 0.25 if quick else 0.1
    out = []
    for w in simplex(step):
        for qsrc, q in (("cn", 1.0), ("cf", 1.0), ("cf", 0.5), ("cs", 0.5), ("cm", 0.0)):
            out.append(cfg(w, qsrc=qsrc, q=q, bass="cf"))
    return out


def grid_refine(best: list[dict]) -> list[dict]:
    out = []
    for c in best:
        for bass, th, sp, md in itertools.product(("cn", "cf", "cs", "cnf", "mean"), (0.4, 0.5, 0.6),
                                                  (0.6, 0.7, 0.8, 0.9), (0.25, 0.5)):
            out.append(cfg(c["w"], c["qsrc"], c["q"], bass, th, sp, md))
    uniq = {json.dumps(c, sort_keys=True): c for c in out}
    return list(uniq.values())


# ---------------------------------------------------------------- main

def table(rows) -> None:
    head = ("| phương án | K majmin | K majmin_inv | K sevenths | K sevenths_inv | **K mean4** | K F1 chuỗi majmin "
            "| K F1 chuỗi sevenths | B majmin | B majmin_inv | B sevenths | B sevenths_inv | **B mean4** | mục tiêu |")
    print(head)
    print("|---" * (head.count("|") - 1) + "|")
    for name, r in rows:
        k, b, s = r["kajiura"], r["beatles"], r["seq"]
        cells = [k[m] for m in OBJECTIVE] + [k["mean4"]] + [s["seq_majmin"], s["seq_sevenths"]] + \
                [b[m] for m in OBJECTIVE] + [b["mean4"], r["objective"]]
        print(f"| {name} | " + " | ".join(f"{100 * v:.2f}" for v in cells) + " |")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--top", type=int, default=8, help="số bộ trộn tốt nhất đem chỉnh bass/làm mượt")
    args = ap.parse_args(argv)
    _utf8_console()
    t_all = time.perf_counter()

    beatles = [f"B:{r.stem}" for r in sorted(REF_DIR.glob("*.lab"))
               if beatles_features(r.stem).exists() and new_models_path(beatles_features(r.stem)).exists()]
    report = K.read_report()
    k_names = sorted(n for n in (r["name"] for r in K.read_mapping())
                     if kajiura_features(n).exists() and new_models_path(kajiura_features(n)).exists()
                     and n in report and report[n]["end"] > report[n]["start"])
    k_ok = [n for n in k_names if report[n]["ok"]]
    kajiura = [f"K:{n}" for n in k_ok]
    songs = beatles + kajiura
    print(f"Beatles {len(beatles)}, Kajiura {len(k_names)} ({len(k_ok)} căn tốt)", flush=True)

    # 1) Từng model một mình và các mốc so sánh
    refs = [
        ("ChordMini một mình (gốc)", {"kind": "raw", "model": "cm"}),
        ("CNN-LSTM một mình (gốc)", {"kind": "raw", "model": "cn"}),
        ("ChordFormer một mình (gốc)", {"kind": "raw", "model": "cf"}),
        ("consonance-ACE một mình (gốc, argmax)", {"kind": "raw", "model": "cs"}),
        ("consonance-ACE + HMM, bass riêng", cfg({"cs": 1}, qsrc="cm", q=0.0, bass="cs")),
        ("ChordFormer + HMM của ta, bass riêng", cfg({"cf": 1}, qsrc="cf", q=1.0, bass="cf")),
        ("**Mặc định hiện tại** (cm+cn 0.6, loại cn)", cfg({"cm": 0.4, "cn": 0.6}, qsrc="cn", q=1.0, bass="cn")),
        ("Thay CNN-LSTM bằng ChordFormer (cùng tham số)", cfg({"cm": 0.4, "cf": 0.6}, qsrc="cf", q=1.0, bass="cf")),
    ]
    ref_scores = Scores([c for _, c in refs], songs, args.jobs)

    # 2) Lưới bộ trộn (bass cf), rồi chỉnh bass/làm mượt cho các bộ tốt nhất
    g1 = grid_chords(args.quick)
    print(f"Lưới bộ trộn: {len(g1)} cấu hình", flush=True)
    s1 = Scores(g1, songs, args.jobs)
    o1 = objective(s1, beatles, kajiura)
    # chọn top theo cả bộ và theo từng phần CV để phần chỉnh không bỏ sót bộ mà CV sẽ chọn
    folds_b = [[s for i, s in enumerate(beatles) if i % args.folds == f] for f in range(args.folds)]
    folds_kn = [[n for i, n in enumerate(k_names) if i % args.folds == f] for f in range(args.folds)]
    folds_k = [[f"K:{n}" for n in fk if n in set(k_ok)] for fk in folds_kn]
    top_idx = set(np.argsort(-o1)[: args.top].tolist())
    for f in range(args.folds):
        tb = [s for s in beatles if s not in set(folds_b[f])]
        tk = [s for s in kajiura if s not in set(folds_k[f])]
        top_idx |= set(np.argsort(-objective(s1, tb, tk))[:3].tolist())
    g2 = grid_refine([g1[i] for i in sorted(top_idx)])
    print(f"Lưới chỉnh bass/làm mượt: {len(g2)} cấu hình", flush=True)
    s2 = Scores(g2, songs, args.jobs)

    # Ghép hai lưới để chọn (CV công bằng: chọn trên phần huấn luyện trong cả g1 lẫn g2)
    all_cfgs = g1 + g2
    corr = np.concatenate([s1.corr, s2.corr])
    wts = np.concatenate([s1.wts, s2.wts])
    s_all = Scores.__new__(Scores)
    s_all.cfgs, s_all.songs, s_all.index, s_all.corr, s_all.wts = all_cfgs, songs, s1.index, corr, wts
    obj = objective(s_all, beatles, kajiura)
    best_full = all_cfgs[int(np.argmax(obj))]

    held, cfg_of_k, fold_cfgs = {}, {}, []
    for f in range(args.folds):
        tb = [s for s in beatles if s not in set(folds_b[f])]
        tk = [s for s in kajiura if s not in set(folds_k[f])]
        ci = int(np.argmax(objective(s_all, tb, tk)))
        fold_cfgs.append(all_cfgs[ci])
        held.update(s_all.per_song(ci, folds_b[f] + folds_k[f]))
        for n in folds_kn[f]:
            cfg_of_k[n] = all_cfgs[ci]
        print(f"phần {f}: {describe(all_cfgs[ci])}", flush=True)
    print(f"cả bộ: {describe(best_full)}", flush=True)

    rows = []
    per = {}
    for i, (name, c) in enumerate(refs):
        ps = ref_scores.per_song(i, songs)
        r = evaluate_rows(ps, beatles, kajiura)
        r["seq"] = seq_f1(lambda n, c=c: c, k_names, report)
        rows.append((name, r))
        per[name] = ps
    cv = evaluate_rows(held, beatles, kajiura)
    cv["seq"] = seq_f1(lambda n: cfg_of_k[n], k_names, report)
    rows.append((f"**Bộ trộn mới (CV {args.folds} phần)**", cv))
    per["cv"] = held
    bi = int(np.argmax(obj))
    full = s_all.per_song(bi, songs)
    fr = evaluate_rows(full, beatles, kajiura)
    fr["seq"] = seq_f1(lambda n: best_full, k_names, report)
    rows.append(("Bộ trộn mới (chọn trên cả bộ, lạc quan)", fr))
    per["full"] = full

    print()
    table(rows)

    # Tốt nhất theo từng tổ hợp model (chọn trên cả bộ) để thấy đóng góp của từng model
    print("\nTốt nhất theo tổ hợp model (chọn trên cả bộ; mục tiêu / K mean4 / B mean4):")
    kb, bb = mean4(s_all.agg(kajiura)), mean4(s_all.agg(beatles))
    combos = {}
    for i, c in enumerate(all_cfgs):
        used = tuple(x for x in MODELS if c["w"][x] > 0) + (f"loại:{c['qsrc']}",)
        if used not in combos or obj[i] > obj[combos[used]]:
            combos[used] = i
    for used, i in sorted(combos.items(), key=lambda kv: -obj[kv[1]])[:25]:
        print(f"  {100 * obj[i]:.2f}  K {100 * kb[i]:.2f}  B {100 * bb[i]:.2f}  {describe(all_cfgs[i])}")

    # Chênh lệch theo bài so với mặc định hiện tại (Kajiura theo thời gian)
    from eval.ablation_kajiura import time_weighted_diff

    def as_res(ps):
        return {s: {"correct": dict(zip(METRICS, v["correct"])), "weights": dict(zip(METRICS, v["weights"]))}
                for s, v in ps.items()}
    rng = np.random.default_rng(0)
    base = per["**Mặc định hiện tại** (cm+cn 0.6, loại cn)"]
    diffs = {}
    for label, ps in (("CV − mặc định", held), ("cả bộ − mặc định", full),
                      ("ChordFormer gốc − CNN-LSTM gốc", per["ChordFormer một mình (gốc)"])):
        ref_ps = per["CNN-LSTM một mình (gốc)"] if "gốc −" in label else base
        for ds_name, ids in (("Kajiura", kajiura), ("Beatles", beatles)):
            for m in OBJECTIVE:
                diffs[f"{label} {ds_name} {m}"] = time_weighted_diff(as_res(ps), as_res(ref_ps), ids, m, 2000, rng)
    print("\nChênh lệch theo bài [CI bootstrap 95%], thắng/thua:")
    for k, d in diffs.items():
        print(f"  {k}: {100 * d['mean']:+.2f} [{100 * d['lo']:+.2f}, {100 * d['hi']:+.2f}] {d['wins']}/{d['losses']}")

    out = EVAL_DIR / "results" / "new_models_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_nan_to_none({
        "beatles": beatles, "kajiura_time": kajiura, "kajiura_seq": k_names,
        "fold_configs": fold_cfgs, "best_full": best_full, "table": {n: r for n, r in rows},
        "diffs": diffs,
        "combos": [{"models": list(u), "objective": float(obj[i]), "kajiura": float(kb[i]), "beatles": float(bb[i]),
                    "config": all_cfgs[i]} for u, i in sorted(combos.items(), key=lambda kv: -obj[kv[1]])],
    }), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nĐã ghi {out}. Tổng {(time.perf_counter() - t_all) / 60:.1f} phút")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
