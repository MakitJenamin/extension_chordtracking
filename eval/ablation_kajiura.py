"""So sánh các phương án nhận hợp âm trên bộ Kajiura, và có nên giữ bước tách nguồn hay không.

    python -m eval.ablation_kajiura cache [--only NAME ...]   # tách nguồn + chạy model (GPU), cache đặc trưng
    python -m eval.ablation_kajiura run [--beatles] [--boot 2000]

`cache` dùng `eval.cache_features.extract` (mỗi bước GPU tự lấy `gpu_lock()`), stem ở
`CACHE_DIR/kajiura_<tên>/`, đặc trưng ở `EVAL_DIR/kajiura/features/<tên>.pkl`.

Mọi phương án dùng tham số **mặc định** của `FusionParams` (chọn trên Beatles), không chỉnh lại trên
Kajiura, nên đây là phép thử chuyển giao:
 1  ChordMini trên mix
 2  ChordMini trên instrumental
 3a CNN-LSTM trên mix,  3b CNN-LSTM trên instrumental
 4  fuse() trên mix, bass chỉ từ CNN-LSTM (`bass_source="cnn"`)   <- không cần tách nguồn
 4b fuse() trên mix, không gán bass                               <- không cần tách nguồn
 5  fuse() trên mix, bass CNN-LSTM + stem bass (mặc định hiện tại)
 6  fuse() chạy model trên instrumental, bass cả hai nguồn
 7  fuse() trộn đặc trưng mix và instrumental (trung bình log xác suất hợp âm), bass cả hai nguồn
 8  thăm dò: như 4 nhưng trọng số CNN-LSTM 0.65, trộn trên cả 170 nhãn (CNN-LSTM một mình tốt hơn fuse
    mặc định trên Kajiura). Chỉ để tham khảo; chọn tham số trên chính bộ này thì lạc quan.

Chấm điểm (xem eval/kajiura.py):
- Theo chuỗi (mọi bài có audio): F1 của LCS và F1 cặp chuyển hợp âm ở mức majmin và sevenths. Độ dịch
  giọng của đáp án lấy từ bước căn bằng audio (không phụ thuộc phương án). Chuỗi dự đoán chỉ lấy trong
  vùng thời gian mà bản hợp âm phủ. Trung bình không gia quyền trên các bài.
- Theo thời gian (chỉ bài căn tốt, `alignment.json` ok): `evaluate_pair` với nhãn đã căn, gia quyền thời lượng.
- So sánh cặp phương án theo từng bài: bootstrap 95% của chênh lệch trung bình, kiểm định dấu hạng Wilcoxon.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from chordtrack import fusion as F
from chordtrack.types import ChordSegment
from eval import kajiura as K
from eval.ablation import Song, predict
from eval.evaluate import TABLE_METRICS, _nan_to_none, _utf8_console, aggregate, evaluate_pair

FEATURES_DIR = K.KAJIURA_DIR / "features"
RESULTS_PATH = K.EVAL_DIR / "results" / "ablation_kajiura.json"
SEQ_LEVELS = ("majmin", "sevenths")


def features_path(name: str) -> Path:
    return FEATURES_DIR / f"{name}.pkl"


# ---------------------------------------------------------------- cache đặc trưng

def cache(only: list[str] | None, force: bool = False) -> int:
    from eval.cache_features import extract

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    rows = K.read_mapping()
    if only:
        rows = [r for r in rows if r["name"] in set(only)]
    for i, row in enumerate(rows, 1):
        name = row["name"]
        audio = K.AUDIO_DIR / f"{name}.wav"
        dst = features_path(name)
        if dst.exists() and not force:
            continue
        if not audio.exists():
            print(f"[{i}/{len(rows)}] {name}: thiếu audio", flush=True)
            continue
        try:
            feats = extract(name, audio, K.stems_dir(name))
        except Exception:
            traceback.print_exc()
            failed.append(name)
            continue
        tmp = dst.with_suffix(".part")
        with open(tmp, "wb") as f:
            pickle.dump(feats, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(dst)
        print(f"[{i}/{len(rows)}] {name}: {feats['duration']:.0f}s audio, "
              + ", ".join(f"{k} {v:.1f}s" for k, v in feats["timings"].items()), flush=True)
    print(f"Lỗi: {failed or 'không'}")
    return 1 if failed else 0


# ---------------------------------------------------------------- dữ liệu một bài

class KajiuraSong(Song):
    """Như `eval.ablation.Song` nhưng đọc đặc trưng và chroma stem bass trong thư mục Kajiura."""

    def __init__(self, name: str):  # noqa: D401 - không gọi Song.__init__ (đường dẫn khác)
        with open(features_path(name), "rb") as f:
            feats = pickle.load(f)
        self.name = name
        self.ref = K.REF_DIR / f"{name}.lab"
        self.duration = float(feats["duration"])
        self.beats = feats["beats"]
        self.models = feats["models"]
        self.timings = feats["timings"]
        self.stems_dir = Path(feats["stems_dir"])
        self.times = F.frame_grid(self.duration)
        self._cache: dict = {}

    def stem_chroma(self) -> np.ndarray:
        if "stem" not in self._cache:
            path = FEATURES_DIR / f"{self.name}.bass_chroma.npz"
            if path.exists():
                d = np.load(path)
                t, c = d["times"], d["chroma"]
            else:
                t, c = F.bass_chroma(self.stems_dir / "bass.wav")
                c = c.astype(np.float32)
                tmp = path.with_name(path.stem + ".part.npz")   # ghi tạm rồi đổi tên, tránh file hỏng
                np.savez_compressed(tmp, times=t, chroma=c)
                tmp.replace(path)
            self._cache["stem"] = F.resample_probs(t, c, self.times)
        return self._cache["stem"]


def _fusion_cfg(src: str, **params) -> dict:
    from dataclasses import asdict

    return {"kind": "fusion", "src": src, "params": asdict(F.FusionParams(**params))}


VARIANTS: list[tuple[str, dict]] = [
    ("1 ChordMini mix", {"kind": "chordmini", "src": "mix"}),
    ("2 ChordMini instrumental", {"kind": "chordmini", "src": "inst"}),
    ("3a CNN-LSTM mix", {"kind": "cnnlstm", "src": "mix"}),
    ("3b CNN-LSTM instrumental", {"kind": "cnnlstm", "src": "inst"}),
    ("4 fuse mix, bass CNN-LSTM (không tách nguồn)", _fusion_cfg("mix", bass_source="cnn")),
    ("4b fuse mix, không bass (không tách nguồn)", _fusion_cfg("mix", bass_source="none")),
    ("5 fuse mix, bass CNN-LSTM + stem (mặc định)", _fusion_cfg("mix")),
    ("6 fuse instrumental, bass cả hai", _fusion_cfg("inst")),
    ("7 fuse trộn mix + instrumental", {"kind": "fusion_both", "src": "mix", "params": _fusion_cfg("mix")["params"]}),
    # Thăm dò (không phải mặc định): tăng trọng số CNN-LSTM, trộn trên cả 170 nhãn, không tách nguồn.
    ("8 fuse mix, CNN-LSTM 0.65 cả 170 nhãn, bass CNN-LSTM (thăm dò)",
     _fusion_cfg("mix", cnn_weight=0.65, mix_level="full", bass_source="cnn")),
]


def predict_variant(song: Song, cfg: dict) -> list[ChordSegment]:
    if cfg["kind"] != "fusion_both":
        return predict(song, cfg)
    p = F.FusionParams(**cfg["params"])
    a = song.chord("mix", p.cnn_weight, p.mix_level)
    b = song.chord("inst", p.cnn_weight, p.mix_level)
    probs = F.mix_log(a, b, 0.5)
    stem = song.stem_chroma() if p.bass_source in ("stem", "both") else None
    cnn_bass = song.cnn_bass("mix") if p.bass_source in ("cnn", "both") else None
    return F.fuse_probs(probs, song.vocab("mix"), song.times, song.beats, song.duration, p,
                        cnn_bass=cnn_bass, stem_chroma=stem)


# ---------------------------------------------------------------- chấm điểm

def score_song(song: KajiuraSong, sheet: K.Sheet, align: dict, est: list[ChordSegment]) -> dict:
    out: dict = {}
    start, end = align["start"], align["end"]
    est_labels = K.segments_in_window(est, start, end)
    for level in SEQ_LEVELS:
        ref = K.transpose_sequence(K.chord_sequence(sheet.chords, level), align["seq_shift"])
        e = K.chord_sequence(est_labels, level)
        fixed = K.score_sequence(ref, e)
        free = K.score_sequence(ref, e, shifts=range(12))
        out[f"seq_{level}"] = fixed["f1"]
        out[f"seq_{level}_p"] = fixed["precision"]
        out[f"seq_{level}_r"] = fixed["recall"]
        out[f"bigram_{level}"] = fixed["bigram_f1"]
        out[f"free_shift_{level}"] = (free["shift"] + align["seq_shift"]) % 12   # độ dịch tuyệt đối so với sheet
        out[f"seq_{level}_free"] = free["f1"]
    if align["ok"]:
        out["time"] = evaluate_pair(song.ref, est)
    return out


def bootstrap_diff(a: np.ndarray, b: np.ndarray, n: int, rng: np.random.Generator) -> dict:
    """Chênh lệch trung bình theo bài (a - b), khoảng tin cậy bootstrap 95%, p Wilcoxon."""
    d = np.asarray(a) - np.asarray(b)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boots = d[idx].mean(axis=1)
    try:
        from scipy.stats import wilcoxon

        p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except Exception:  # pragma: no cover - scipy thiếu hoặc mọi chênh lệch bằng 0
        p = float("nan")
    return {"mean": float(d.mean()), "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)),
            "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "p_wilcoxon": p}


def time_weighted_diff(ra: dict, rb: dict, names: list[str], metric: str, n: int, rng) -> dict:
    """Bootstrap theo bài cho điểm gia quyền thời lượng (lấy mẫu lại bài, tính lại tỉ số)."""
    ca = np.array([ra[x]["correct"][metric] for x in names])
    cb = np.array([rb[x]["correct"][metric] for x in names])
    w = np.array([ra[x]["weights"][metric] for x in names])
    idx = rng.integers(0, len(names), size=(n, len(names)))
    boots = (ca[idx].sum(1) - cb[idx].sum(1)) / w[idx].sum(1)
    per_song = np.where(w > 0, (ca - cb) / np.maximum(w, 1e-9), 0.0)
    return {"mean": float((ca.sum() - cb.sum()) / w.sum()), "lo": float(np.percentile(boots, 2.5)),
            "hi": float(np.percentile(boots, 97.5)), "wins": int((per_song > 1e-9).sum()),
            "losses": int((per_song < -1e-9).sum())}


# ---------------------------------------------------------------- thời gian xử lý

def cold_fuse_seconds(name: str, cfg: dict) -> float:
    """Thời gian `fuse` của một bài khi chưa có gì trong cache: tính lại xác suất trên lưới khung và
    chroma stem bass từ file WAV (không đọc npz đã cache), giống pipeline thật."""
    p = F.FusionParams(**cfg["params"])
    song = KajiuraSong(name)
    t = time.perf_counter()
    if p.bass_source in ("stem", "both"):
        song._cache["stem"] = F.resample_probs(*F.bass_chroma(song.stems_dir / "bass.wav"), song.times)
    predict_variant(song, cfg)
    return time.perf_counter() - t


def timing_summary(songs: list[KajiuraSong], fuse_seconds: dict[str, list[float]]) -> dict:
    """Thời gian mỗi bài (giây) của từng bước, đọc từ đặc trưng và separation_stats.json."""
    rows = []
    for s in songs:
        stats = json.loads((s.stems_dir / "separation_stats.json").read_text(encoding="utf-8"))
        rows.append({
            "duration": s.duration,
            "normalize": (stats.get("normalize") or {}).get("seconds", np.nan),
            "roformer": (stats.get("roformer") or {}).get("seconds", np.nan),
            "demucs": (stats.get("demucs") or {}).get("seconds", np.nan),
            "roformer_vram": (stats.get("roformer") or {}).get("peak_vram_mb", np.nan),
            "demucs_vram": (stats.get("demucs") or {}).get("peak_vram_mb", np.nan),
            "beats": s.timings.get("beats", np.nan),
            "chordmini": s.timings.get("chordmini_mix", np.nan),
            "cnnlstm": s.timings.get("cnnlstm_mix", np.nan),
        })
    mean = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
    mean["fuse_no_stem"] = float(np.mean(fuse_seconds["4"]))
    mean["fuse_with_stem"] = float(np.mean(fuse_seconds["5"]))
    mean["total_no_separation"] = mean["beats"] + mean["chordmini"] + mean["cnnlstm"] + mean["fuse_no_stem"]
    mean["total_with_separation"] = (mean["normalize"] + mean["roformer"] + mean["demucs"] + mean["beats"]
                                     + mean["chordmini"] + mean["cnnlstm"] + mean["fuse_with_stem"])
    return mean


# ---------------------------------------------------------------- Beatles (cùng phương án, để đặt cạnh nhau)

def run_beatles(jobs: int) -> dict[str, dict]:
    from eval.ablation import Runner, cfg_key
    from eval.cache_features import features_path as beatles_features
    from eval.datasets import REF_DIR

    names = [r.stem for r in sorted(REF_DIR.glob("*.lab")) if beatles_features(r.stem).exists()]
    out = {}
    runner = Runner(names, jobs)
    try:
        cfgs = [c for _, c in VARIANTS if c["kind"] != "fusion_both"]
        runner.run(cfgs)
        for vname, cfg in VARIANTS:
            if cfg["kind"] == "fusion_both":
                res = {}
                for n in names:
                    song = Song(n)
                    res[n] = evaluate_pair(song.ref, predict_variant(song, cfg))
                out[vname] = aggregate(res)
            else:
                out[vname] = aggregate(runner.results[cfg_key(cfg)])
    finally:
        runner.close()
    out["_n_songs"] = len(names)
    return out


# ---------------------------------------------------------------- main

def run(n_boot: int, beatles: bool, jobs: int) -> int:
    report = K.read_report()
    names = [r["name"] for r in K.read_mapping() if features_path(r["name"]).exists() and r["name"] in report]
    if not names:
        print("Chưa có đặc trưng hoặc chưa căn (chạy `cache` và `python -m eval.kajiura align`)", file=sys.stderr)
        return 1
    skipped = [n for n in names if report[n]["end"] <= report[n]["start"]]
    if skipped:   # bản hợp âm quá ngắn hoặc không căn được: không có vùng để chấm
        print(f"Bỏ {len(skipped)} bài không có vùng căn: {', '.join(skipped)}", flush=True)
        names = [n for n in names if n not in skipped]
    ok_names = [n for n in names if report[n]["ok"]]
    print(f"{len(names)} bài có đặc trưng, {len(ok_names)} bài căn tốt (dùng cho chấm theo thời gian)", flush=True)

    songs = {n: KajiuraSong(n) for n in names}
    sheets = {n: K.load_sheet(n) for n in names}
    scores: dict[str, dict[str, dict]] = {}
    for vname, cfg in VARIANTS:
        key = vname.split()[0]
        scores[key] = {n: score_song(songs[n], sheets[n], report[n], predict_variant(songs[n], cfg)) for n in names}
        print(f"{vname}: xong", flush=True)
    cfgs = dict(VARIANTS)
    fuse_seconds = {key: [cold_fuse_seconds(n, cfgs[v]) for n in names]
                    for key, v in (("4", VARIANTS[4][0]), ("5", VARIANTS[6][0]))}

    # Bảng tổng
    keys = [v.split()[0] for v, _ in VARIANTS]
    table = {}
    for (vname, _), key in zip(VARIANTS, keys):
        row = {m: float(np.mean([scores[key][n][m] for n in names]))
               for m in ("seq_majmin", "bigram_majmin", "seq_sevenths", "bigram_sevenths",
                         "seq_majmin_free", "seq_majmin_p", "seq_majmin_r")}
        row["time"] = aggregate({n: scores[key][n]["time"] for n in ok_names}) if ok_names else {}
        table[vname] = row

    rng = np.random.default_rng(0)
    pairs = [("4", "5"), ("4b", "4"), ("1", "2"), ("3a", "3b"), ("5", "6"), ("7", "5"), ("4", "1"), ("5", "1"),
             ("3a", "4"), ("8", "4")]
    tests = {}
    for a, b in pairs:
        t = {}
        for m in ("seq_majmin", "seq_sevenths", "bigram_majmin"):
            t[m] = bootstrap_diff(np.array([scores[a][n][m] for n in names]),
                                  np.array([scores[b][n][m] for n in names]), n_boot, rng)
        if ok_names:
            for m in ("majmin", "majmin_inv", "sevenths", "sevenths_inv"):
                t[f"time_{m}"] = time_weighted_diff({n: scores[a][n]["time"] for n in ok_names},
                                                    {n: scores[b][n]["time"] for n in ok_names}, ok_names, m, n_boot, rng)
        tests[f"{a} - {b}"] = t

    timing = timing_summary(list(songs.values()), fuse_seconds)
    beatles_rows = run_beatles(jobs) if beatles else None

    _print(table, tests, timing, beatles_rows, names, ok_names)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(_nan_to_none({
        "songs": names, "aligned_songs": ok_names, "table": table, "tests": tests, "timing": timing,
        "beatles": beatles_rows, "per_song": scores, "alignment": {n: report[n] for n in names},
    }), indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\nĐã ghi {RESULTS_PATH}")
    return 0


def _print(table, tests, timing, beatles_rows, names, ok_names) -> None:
    print(f"\n=== Kajiura theo chuỗi ({len(names)} bài, trung bình theo bài, %) ===")
    print("| phương án | F1 majmin | F1 cặp majmin | F1 sevenths | F1 cặp sevenths | F1 majmin (tự dịch) |")
    print("|---|---|---|---|---|---|")
    for v, r in table.items():
        print(f"| {v} | {100 * r['seq_majmin']:.1f} | {100 * r['bigram_majmin']:.1f} | {100 * r['seq_sevenths']:.1f} "
              f"| {100 * r['bigram_sevenths']:.1f} | {100 * r['seq_majmin_free']:.1f} |")
    if ok_names:
        print(f"\n=== Kajiura theo thời gian ({len(ok_names)} bài căn tốt, WCSR gia quyền thời lượng, %) ===")
        print("| phương án | " + " | ".join(TABLE_METRICS) + " |")
        print("|---|" + "---|" * len(TABLE_METRICS))
        for v, r in table.items():
            print(f"| {v} | " + " | ".join(f"{100 * r['time'][m]:.2f}" for m in TABLE_METRICS) + " |")
    if beatles_rows:
        print(f"\n=== Beatles, cùng phương án ({beatles_rows['_n_songs']} bài, WCSR %) ===")
        print("| phương án | " + " | ".join(TABLE_METRICS) + " |")
        print("|---|" + "---|" * len(TABLE_METRICS))
        for v, r in beatles_rows.items():
            if not v.startswith("_"):
                print(f"| {v} | " + " | ".join(f"{100 * r[m]:.2f}" for m in TABLE_METRICS) + " |")
    print("\n=== So sánh theo từng bài: chênh lệch a - b (điểm %), [CI bootstrap 95%], thắng/thua, p Wilcoxon ===")
    for pair, t in tests.items():
        parts = []
        for m, d in t.items():
            s = f"{m} {100 * d['mean']:+.2f} [{100 * d['lo']:+.2f}, {100 * d['hi']:+.2f}] {d['wins']}/{d['losses']}"
            if "p_wilcoxon" in d:
                s += f" p={d['p_wilcoxon']:.3f}"
            parts.append(s)
        print(f"{pair}:\n    " + "\n    ".join(parts))
    print("\n=== Thời gian trung bình mỗi bài (giây; không tính chờ khóa GPU ở bước tách nguồn) ===")
    for k, v in timing.items():
        print(f"  {k:24s} {v:8.2f}")


def main(argv: list[str] | None = None) -> int:
    _utf8_console()
    ap = argparse.ArgumentParser(description="Ablation trên bộ Kajiura")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--only", nargs="*")
    c.add_argument("--force", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--boot", type=int, default=2000)
    r.add_argument("--beatles", action="store_true", help="chạy cùng phương án trên bộ Beatles để đặt cạnh nhau")
    r.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args(argv)
    import logging
    logging.basicConfig(level=logging.WARNING)
    if args.cmd == "cache":
        return cache(args.only, args.force)
    return run(args.boot, args.beatles, args.jobs)


if __name__ == "__main__":
    raise SystemExit(main())
