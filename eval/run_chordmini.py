"""Baseline: chạy script gốc của ChordMini trên bộ đánh giá rồi chấm điểm.

    python -m eval.run_chordmini [--out-dir DIR] [--checkpoint btc_model_best.pth]

Chạy `third_party_src/ChordMini/src/evaluation/test.py` bằng subprocess với cờ đã chốt ở
PLAN.md mục 11. Khóa GPU được giữ trong suốt subprocess. Chỉ chạy trên các bài đã căn đạt
(`EVAL_DIR/isophonics/ref/*.lab`): audio được hard-link sang một thư mục tạm.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from chordtrack.gpu import gpu_lock
from chordtrack.paths import CHORDMINI_DIR, EVAL_DIR
from eval.datasets import AUDIO_DIR, REF_DIR, _utf8_console
from eval.evaluate import aggregate, evaluate_pair, format_table

CHORDMINI_FLAGS = [
    "--model_type", "BTC",
    "--config", "config/ChordMini.yaml",
    "--smooth_logits", "--use_overlap", "--use_gaussian",
    "--kernel_size", "9",
    "--vote_aggregation", "logit",
    "--min_segment_duration", "0.5",
    "--smooth_predictions",
]


class VramMonitor:
    """Đọc VRAM đã dùng (MiB) bằng nvidia-smi mỗi 0.5 giây. Là số của cả GPU, không riêng tiến trình."""

    def __init__(self) -> None:
        self.peak = 0
        self.baseline = self._read()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _read() -> int:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout
            return int(out.split()[0])
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
            return 0

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self._read())
            self._stop.wait(0.5)

    def __enter__(self) -> "VramMonitor":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()


def run_chordmini(audio_dir: Path, save_dir: Path, checkpoint: str) -> dict:
    py = sys.executable
    cmd = [py, "src/evaluation/test.py", "--audio_dir", str(audio_dir), "--save_dir", str(save_dir),
           "--checkpoint", f"checkpoints/{checkpoint}", *CHORDMINI_FLAGS]
    t_wait = time.time()
    with gpu_lock():
        t0 = time.time()
        with VramMonitor() as vram:
            res = subprocess.run(cmd, cwd=CHORDMINI_DIR, capture_output=True, text=True, encoding="utf-8", errors="replace")
        elapsed = time.time() - t0
    if res.returncode != 0:
        raise RuntimeError(f"ChordMini lỗi (mã {res.returncode}):\n{res.stderr[-3000:]}")
    errors = [l for l in (res.stdout + res.stderr).splitlines() if "Error processing" in l]
    return {"seconds": elapsed, "lock_wait_seconds": t0 - t_wait, "vram_peak_mib": vram.peak,
            "vram_before_mib": vram.baseline, "errors": errors}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Baseline ChordMini trên bộ đánh giá Isophonics")
    ap.add_argument("--out-dir", type=Path, default=EVAL_DIR / "results" / "chordmini_btc_pl")
    ap.add_argument("--checkpoint", default="btc_model_best.pth")
    args = ap.parse_args(argv)
    _utf8_console()

    refs = sorted(REF_DIR.glob("*.lab"))
    if not refs:
        print(f"Chưa có nhãn đã căn trong {REF_DIR}. Chạy `python -m eval.datasets all` trước.", file=sys.stderr)
        return 1
    est_dir = args.out_dir / "lab"
    if est_dir.exists():
        shutil.rmtree(est_dir)
    est_dir.mkdir(parents=True)

    with tempfile.TemporaryDirectory(dir=EVAL_DIR) as tmp:
        for ref in refs:
            src = AUDIO_DIR / f"{ref.stem}.wav"
            try:
                os.link(src, Path(tmp) / src.name)
            except OSError:
                shutil.copy2(src, Path(tmp) / src.name)
        stats = run_chordmini(Path(tmp), est_dir, args.checkpoint)

    results = {r.stem: evaluate_pair(r, est_dir / r.name) for r in refs if (est_dir / r.name).exists()}
    agg = aggregate(results)
    print(format_table(results, agg))
    audio_seconds = agg["duration"]
    print(f"\n{len(results)}/{len(refs)} bài, {audio_seconds / 60:.1f} phút audio. "
          f"Thời gian chạy {stats['seconds']:.1f}s ({stats['seconds'] / max(1, len(results)):.1f}s/bài), "
          f"VRAM đỉnh {stats['vram_peak_mib']} MiB (trước khi chạy {stats['vram_before_mib']} MiB)")
    for e in stats["errors"]:
        print(e, file=sys.stderr)
    (args.out_dir / "results.json").write_text(json.dumps({
        "checkpoint": args.checkpoint, "flags": CHORDMINI_FLAGS, "run": stats, "mean": agg,
        "files": {k: {m: v[m] for m in v if m not in ("weights", "correct")} for k, v in results.items()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if len(results) == len(refs) else 2


if __name__ == "__main__":
    raise SystemExit(main())
