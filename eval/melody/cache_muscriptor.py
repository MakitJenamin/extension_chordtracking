"""Chạy MuScriptor trên mọi bài trong các manifest giai điệu và cache kết quả `.npz`.

    python -m eval.melody.cache_muscriptor --size small [--dataset NAME ...] [--force] [--limit N]
    python -m eval.melody.cache_muscriptor --size small --audio a.wav b.wav   # thử nhanh, dataset "adhoc"

Đọc `data/eval/melody/<dataset>/manifest.jsonl` (xem `eval/melody/README.md`), ghi
`cache_path("muscriptor_<size>", dataset, track_id)`, bỏ qua bài đã có cache. Mỗi bài giữ khóa GPU riêng.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from chordtrack import paths
from chordtrack.melody.io import cache_path, save_output

MANIFEST_ROOT = paths.EVAL_DIR / "melody"


def read_manifests(datasets: list[str] | None = None) -> list[dict]:
    """Mọi dòng của các manifest (theo tên dataset nếu có), theo thứ tự tên dataset."""
    if datasets:
        files = [MANIFEST_ROOT / d / "manifest.jsonl" for d in datasets]
        missing = [str(f) for f in files if not f.exists()]
        if missing:
            raise FileNotFoundError(f"không có manifest: {', '.join(missing)}")
    else:
        files = sorted(MANIFEST_ROOT.glob("*/manifest.jsonl"))
    rows = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row.setdefault("dataset", f.parent.name)
                rows.append(row)
    return rows


def adhoc_rows(audio: list[str]) -> list[dict]:
    return [{"track_id": Path(a).stem, "dataset": "adhoc", "audio": str(Path(a).resolve())} for a in audio]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--size", required=True, choices=("small", "medium", "large"))
    ap.add_argument("--dataset", nargs="*", default=None, help="tên dataset; mặc định mọi manifest")
    ap.add_argument("--audio", nargs="*", default=None, help="chạy trên các file này (dataset 'adhoc')")
    ap.add_argument("--dtype", default=None, help="float32/float16/bfloat16; mặc định theo cỡ")
    ap.add_argument("--force", action="store_true", help="chạy lại cả bài đã có cache")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    from chordtrack.melody import muscriptor as M

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    rows = adhoc_rows(args.audio) if args.audio else read_manifests(args.dataset)
    name = M.model_name(args.size)
    todo = [r for r in rows if args.force or not cache_path(name, r["dataset"], r["track_id"]).exists()]
    skipped = len(rows) - len(todo)
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"{name}: {len(rows)} bài, bỏ qua {skipped} đã có cache, chạy {len(todo)}", flush=True)
    failed = []
    t_all = time.perf_counter()
    for i, r in enumerate(todo, 1):
        dst = cache_path(name, r["dataset"], r["track_id"])
        try:
            out = M.transcribe(r["audio"], args.size, dtype=args.dtype)
        except M.MuscriptorAuthError as e:
            print(f"DỪNG: {e}", flush=True)
            return 2
        except Exception:
            traceback.print_exc()
            failed.append(f"{r['dataset']}/{r['track_id']}")
            continue
        save_output(out, dst)
        x = out.extras
        voice = sum(n.source == M.VOICE_SOURCE for n in out.notes)
        print(f"[{i}/{len(todo)}] {r['dataset']}/{r['track_id']}: {len(out.notes)} nốt ({voice} voice), "
              f"{float(x['duration_s']):.0f}s audio, nạp {float(x['load_s']):.1f}s, "
              f"chạy {float(x['infer_s']):.1f}s, VRAM đỉnh {float(x['peak_vram_mb']):.0f} MB", flush=True)
    print(f"xong {len(todo) - len(failed)}/{len(todo)} bài trong {time.perf_counter() - t_all:.0f}s", flush=True)
    if failed:
        print("lỗi: " + ", ".join(failed), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
