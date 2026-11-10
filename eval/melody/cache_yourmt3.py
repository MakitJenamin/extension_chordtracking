"""Chạy YourMT3+ trên mọi bài của các manifest melody và lưu cache `.npz` (bỏ qua bài đã có cache).

    python -m eval.melody.cache_yourmt3                       # mọi bộ trong data/eval/melody/*/manifest.jsonl
    python -m eval.melody.cache_yourmt3 --dataset orchset --dataset adc2004

Khóa GPU giữ theo từng bài; giữa hai bài model được chuyển về CPU để trả VRAM cho agent khác.
Mỗi file cache có thêm `extras`: `elapsed_s` (thời gian suy luận), `peak_vram_mib` (VRAM reserved đỉnh).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from chordtrack import gpu, paths
from chordtrack.melody import io as mio
from chordtrack.melody import yourmt3

MANIFEST_ROOT = paths.EVAL_DIR / "melody"


def read_manifest(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def available_datasets() -> list[str]:
    return sorted(p.parent.name for p in MANIFEST_ROOT.glob("*/manifest.jsonl"))


def run_track(audio: Path, batch_size: int, dtype: str):
    import torch

    with gpu.gpu_lock():
        try:
            cuda = torch.cuda.is_available()
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            out = yourmt3.transcribe(audio, batch_size=batch_size, dtype=dtype)
            elapsed = time.perf_counter() - t0
            peak = torch.cuda.max_memory_reserved() / 2**20 if cuda else 0.0
        finally:
            yourmt3.offload()
    out.extras["elapsed_s"] = np.array(elapsed)
    out.extras["peak_vram_mib"] = np.array(peak)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="append", help="tên bộ (lặp lại được); mặc định mọi bộ có manifest")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--limit", type=int, default=None, help="chỉ chạy N bài đầu mỗi bộ (để thử)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):   # Windows: stdout chuyển hướng ra file mặc định cp1252
        sys.stdout.reconfigure(encoding="utf-8")

    datasets = args.dataset or available_datasets()
    n_done = n_skip = n_fail = 0
    for ds in datasets:
        manifest = MANIFEST_ROOT / ds / "manifest.jsonl"
        if not manifest.exists():
            print(f"[{ds}] không có manifest {manifest}, bỏ qua", flush=True)
            continue
        rows = read_manifest(manifest)[: args.limit]
        for i, row in enumerate(rows, 1):
            dst = mio.cache_path(yourmt3.MODEL_NAME, ds, row["track_id"])
            if dst.exists():
                n_skip += 1
                continue
            try:
                out = run_track(Path(row["audio"]), args.batch_size, args.dtype)
            except Exception as e:  # noqa: BLE001 — ghi lỗi rồi chạy bài tiếp
                n_fail += 1
                print(f"[{ds}] {i}/{len(rows)} {row['track_id']}: LỖI {type(e).__name__}: {e}", flush=True)
                continue
            dur = float(out.extras["audio_duration_s"])
            mio.save_output(out, dst)
            n_done += 1
            voice = sum(n.source == "voice" for n in out.notes)
            print(f"[{ds}] {i}/{len(rows)} {row['track_id']}: {len(out.notes)} nốt ({voice} voice), "
                  f"{dur:.0f} s audio, {float(out.extras['elapsed_s']):.1f} s, "
                  f"VRAM {float(out.extras['peak_vram_mib']):.0f} MiB", flush=True)
    print(f"Xong: {n_done} bài mới, {n_skip} đã có cache, {n_fail} lỗi", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
