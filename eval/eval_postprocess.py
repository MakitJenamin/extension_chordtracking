"""Đo ảnh hưởng của hai bước hậu xử lý trong `fusion.fuse_probs` lên WCSR (Beatles + Kajiura):

- thay N.C. giữa bài kém tin cậy bằng hợp âm tốt thứ hai (`nc_min_prob`, `nc_min_duration`);
- gộp đoạn ngắn và kém tin cậy (`weak_max_beats`, `weak_max_conf`).

    python -m eval.eval_postprocess [--jobs 6]

Mọi cấu hình dùng `DEFAULT_PARAMS` và chỉ đổi các tham số trên. Bài và cách chấm giống `eval.refit_fusion`
(Kajiura theo thời gian chỉ gồm bài căn tốt). Kèm số đoạn N.C. giữa bài và số đoạn trung bình mỗi bài.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace

import numpy as np

from chordtrack import fusion as F
from chordtrack.paths import EVAL_DIR
from eval import kajiura as K
from eval.ablation_kajiura import features_path as kajiura_features
from eval.cache_features import features_path as beatles_features
from eval.datasets import REF_DIR
from eval.evaluate import _utf8_console
from eval.refit_fusion import OBJECTIVE, Scores, _load, evaluate_rows, fusion_cfg, predict

OFF = dict(nc_min_prob=0.0, weak_max_beats=0.0)


def variants() -> list[tuple[str, dict]]:
    base = asdict(F.DEFAULT_PARAMS)
    rows = [("tắt cả hai (trước)", {**base, **OFF})]
    for prob, dur in ((0.5, 2.0), (0.6, 2.0), (0.6, 4.0), (0.7, 2.0)):
        rows.append((f"N.C. ≥ {prob}, ≥ {dur}s", {**base, **OFF, "nc_min_prob": prob, "nc_min_duration": dur}))
    for beats_, conf in ((1.0, 0.25), (1.5, 0.25), (2.5, 0.25), (2.5, 0.2)):
        rows.append((f"gộp < {beats_} phách, conf < {conf}",
                     {**base, **OFF, "weak_max_beats": beats_, "weak_max_conf": conf}))
    rows.append(("mặc định mới", base))
    return [(n, fusion_cfg(**p)) for n, p in rows]


def segment_stats(cfg: dict, songs: list[str]) -> dict:
    inner_nc, n_segs = 0, 0
    for sid in songs:
        song = _load(sid)
        segs = predict(song, cfg)
        n_segs += len(segs)
        inner_nc += sum(1 for s in segs[1:-1] if s.label == "N")
    return {"inner_nc_per_song": inner_nc / len(songs), "segments_per_song": n_segs / len(songs)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="WCSR khi bật/tắt hậu xử lý N.C. và đoạn ngắn kém tin cậy")
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args(argv)
    _utf8_console()
    beatles = [f"B:{r.stem}" for r in sorted(REF_DIR.glob("*.lab")) if beatles_features(r.stem).exists()]
    report = K.read_report()
    kajiura = [f"K:{n}" for n in sorted(r["name"] for r in K.read_mapping())
               if kajiura_features(n).exists() and n in report and report[n]["ok"]]
    rows = variants()
    scores = Scores([c for _, c in rows], beatles + kajiura, args.jobs)
    out = {}
    print(f"\nBeatles {len(beatles)} bài, Kajiura {len(kajiura)} bài căn tốt. WCSR (%), N.C. giữa bài / số đoạn mỗi bài")
    print("| phương án | " + " | ".join(f"K {m}" for m in OBJECTIVE) + " | " + " | ".join(f"B {m}" for m in OBJECTIVE)
          + " | N.C. giữa bài | số đoạn |")
    print("|---" * (2 + 2 * len(OBJECTIVE) + 1) + "|")
    for ci, (name, cfg) in enumerate(rows):
        r = evaluate_rows(scores.per_song(ci, beatles + kajiura), beatles, kajiura)
        st = segment_stats(cfg, beatles + kajiura)
        out[name] = {**r, **st, "params": cfg["params"]}
        cells = [r["kajiura"][m] for m in OBJECTIVE] + [r["beatles"][m] for m in OBJECTIVE]
        print(f"| {name} | " + " | ".join(f"{100 * v:.2f}" for v in cells)
              + f" | {st['inner_nc_per_song']:.2f} | {st['segments_per_song']:.1f} |", flush=True)
    path = EVAL_DIR / "results" / "eval_postprocess.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Đã ghi {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
