"""Vòng lặp cache chung cho các script `eval.melody.cache_<model>`.

Đọc mọi `data/eval/melody/<dataset>/manifest.jsonl` (hoặc chỉ các bộ trong `--dataset`), chạy
`transcribe(audio)` cho từng bài chưa có cache, ghi bằng `chordtrack.melody.io`. Thời gian cả lệnh
`transcribe` (giây, gồm đọc audio và chờ khóa GPU) lưu ở `extras["runtime_sec"]`; thời gian riêng model do
wrapper tự ghi ở `extras["infer_sec"]`, VRAM đỉnh (MB, nếu có dùng CUDA) ở `extras["vram_peak_mb"]`.
`--audio <file>` chạy thử một file lẻ vào bộ "adhoc" (track_id = tên file, trùng tên thì dùng `--force`).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np

from chordtrack import paths
from chordtrack.melody import io
from chordtrack.types import MelodyModelOutput

MELODY_EVAL_DIR = paths.EVAL_DIR / "melody"


def manifests(datasets: list[str] | None = None) -> dict[str, Path]:
    found = {p.parent.name: p for p in sorted(MELODY_EVAL_DIR.glob("*/manifest.jsonl"))}
    if datasets:
        missing = [d for d in datasets if d not in found]
        if missing:
            raise SystemExit(f"Không thấy manifest cho: {', '.join(missing)} (có: {', '.join(found) or 'không có'})")
        found = {d: found[d] for d in datasets}
    return found


def read_manifest(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _cuda_reset() -> None:
    """Đặt lại số đo VRAM đỉnh, chỉ khi CUDA đã khởi tạo (không tự khởi tạo CUDA ngoài khóa GPU)."""
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.reset_peak_memory_stats()


def _cuda_peak_mb() -> float | None:
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_initialized():
        return None
    return torch.cuda.max_memory_allocated() / 2**20


def run_one(model_name: str, transcribe: Callable[[Path], MelodyModelOutput], dataset: str, track_id: str,
            audio: Path, force: bool = False) -> str:
    """Chạy và ghi cache một bài. Trả về "skip", "ok" hoặc "fail"."""
    out_path = io.cache_path(model_name, dataset, track_id)
    if out_path.exists() and not force:
        return "skip"
    try:
        _cuda_reset()
        t0 = time.perf_counter()
        out = transcribe(audio)
        dt = time.perf_counter() - t0
        out.extras["runtime_sec"] = np.float64(dt)
        peak = _cuda_peak_mb()
        if peak is not None:
            out.extras["vram_peak_mb"] = np.float64(peak)
        io.save_output(out, out_path)
    except Exception:
        traceback.print_exc()
        return "fail"
    infer = f" (model {float(out.extras['infer_sec']):.1f} s)" if "infer_sec" in out.extras else ""
    vram = f", VRAM đỉnh {out.extras['vram_peak_mb']:.0f} MB" if "vram_peak_mb" in out.extras else ""
    pt = out.pitch_track
    size = f"{len(pt.times)} khung" if pt is not None else f"{len(out.notes)} nốt"
    print(f"  {dataset}/{track_id}: {dt:.1f} s{infer}{vram}, {size}", flush=True)
    return "ok"


def main(model_name: str, transcribe: Callable[[Path], MelodyModelOutput], argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"Cache kết quả {model_name} cho các bộ đánh giá giai điệu")
    ap.add_argument("--dataset", nargs="*", help="chỉ chạy các bộ này (mặc định: mọi manifest)")
    ap.add_argument("--audio", type=Path, help="chạy thử một file audio lẻ (ghi vào bộ 'adhoc')")
    ap.add_argument("--force", action="store_true", help="chạy lại cả bài đã có cache")
    args = ap.parse_args(argv)

    if args.audio:
        jobs = {"adhoc": [{"track_id": args.audio.stem, "audio": str(args.audio)}]}
    else:
        jobs = {d: read_manifest(p) for d, p in manifests(args.dataset).items()}
        if not jobs:
            print(f"Chưa có manifest nào trong {MELODY_EVAL_DIR}")
            return 0

    n_fail = 0
    for dataset, rows in jobs.items():
        counts = {"ok": 0, "skip": 0, "fail": 0}
        print(f"[{model_name}] {dataset}: {len(rows)} bài", flush=True)
        for r in rows:
            counts[run_one(model_name, transcribe, dataset, r["track_id"], Path(r["audio"]), args.force)] += 1
        print(f"[{model_name}] {dataset}: chạy {counts['ok']}, bỏ qua {counts['skip']} (đã có), lỗi {counts['fail']}",
              flush=True)
        n_fail += counts["fail"]
    return 1 if n_fail else 0
