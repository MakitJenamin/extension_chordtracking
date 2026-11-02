"""Khóa GPU dùng chung giữa các tiến trình.

Máy chỉ có 8 GB VRAM, nên mỗi lúc chỉ được một model chạy trên GPU. Mọi đoạn code dùng
GPU (chạy model, kể cả test) phải nằm trong `with gpu_lock():`. Khóa là file trong %TEMP%
nên có tác dụng cho mọi worktree và mọi agent trên máy.
"""
from __future__ import annotations

import contextlib
import gc
import os
import tempfile
from pathlib import Path

from filelock import FileLock

LOCK_PATH = Path(tempfile.gettempdir()) / "chordtrack_gpu.lock"


@contextlib.contextmanager
def gpu_lock(timeout: float = 1800):
    lock = FileLock(str(LOCK_PATH), timeout=timeout)
    with lock:
        try:
            yield
        finally:
            free_vram()


def free_vram() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def device() -> str:
    if os.environ.get("CHORDTRACK_CPU") == "1":
        return "cpu"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
