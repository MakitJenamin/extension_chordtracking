"""Cache xác suất của ChordFormer và consonance-ACE trên **mix** cho cả hai bộ đánh giá.

    python -m eval.cache_new_models [--only NAME ...] [--force]

Ghi ra `<thư mục đặc trưng>/<tên>.new_models.pkl` (dict "version", "models": {"chordformer_mix",
"consonance_mix"}, "timings"), cạnh file `<tên>.pkl` của `eval.cache_features` / `eval.ablation_kajiura
cache`. `eval.ablation.Song` tự đọc file này nếu có. Cần đặc trưng gốc (để biết thư mục stem có mix.wav).
"""
from __future__ import annotations

import argparse
import pickle
import time
import traceback
from pathlib import Path

from eval.cache_features import _compact

NEW_VERSION = 1


def new_models_path(features_pkl: Path) -> Path:
    return features_pkl.with_name(features_pkl.stem + ".new_models.pkl")


def _songs() -> list[tuple[str, Path, Path]]:
    """(tên, file đặc trưng gốc, mix.wav) của mọi bài Beatles rồi Kajiura."""
    from eval import kajiura as K
    from eval.ablation_kajiura import features_path as kajiura_features
    from eval.cache_features import features_path as beatles_features
    from eval.cache_features import stem_dir
    from eval.datasets import read_mapping

    out = []
    for r in read_mapping():
        out.append((r["name"], beatles_features(r["name"]), stem_dir(r["name"]) / "mix.wav"))
    for r in K.read_mapping():
        out.append((r["name"], kajiura_features(r["name"]), K.stems_dir(r["name"]) / "mix.wav"))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    from chordtrack.chords.chordformer import run_chordformer
    from chordtrack.chords.consonance import run_consonance
    from eval.evaluate import _utf8_console

    _utf8_console()
    songs = _songs()
    if args.only:
        songs = [s for s in songs if s[0] in set(args.only)]
    failed = []
    t_all = time.perf_counter()
    for i, (name, feats, mix) in enumerate(songs, 1):
        dst = new_models_path(feats)
        if dst.exists() and not args.force:
            continue
        if not feats.exists() or not mix.exists():
            print(f"[{i}/{len(songs)}] {name}: thiếu đặc trưng hoặc mix.wav", flush=True)
            failed.append(name)
            continue
        try:
            timings, models = {}, {}
            t = time.perf_counter()
            models["chordformer_mix"] = _compact(run_chordformer(mix))
            timings["chordformer_mix"] = time.perf_counter() - t
            t = time.perf_counter()
            models["consonance_mix"] = _compact(run_consonance(mix))
            timings["consonance_mix"] = time.perf_counter() - t
        except Exception:
            traceback.print_exc()
            failed.append(name)
            continue
        tmp = dst.with_suffix(".part")
        with open(tmp, "wb") as f:
            pickle.dump({"version": NEW_VERSION, "models": models, "timings": timings}, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(dst)
        print(f"[{i}/{len(songs)}] {name}: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()), flush=True)
    print(f"Xong trong {(time.perf_counter() - t_all) / 60:.1f} phút. Lỗi: {failed or 'không'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
