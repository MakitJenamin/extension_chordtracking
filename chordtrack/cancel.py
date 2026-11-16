"""Hủy job đang chạy theo kiểu hợp tác (cooperative).

Worker bọc lúc chạy job trong `watching(is_cancelled)`; các đoạn chạy lâu trên cùng thread gọi `check()` (ném
`Cancelled` khi người dùng đã bấm hủy) hoặc `requested()` (vd để kill tiến trình con YourMT3+). Ngoài `watching`
thì hai hàm này không làm gì, nên gọi từ script/eval vẫn an toàn.

`is_cancelled` (thường đọc SQLite) chỉ được gọi tối đa mỗi `interval` giây, vì `check()` có thể nằm trong vòng lặp
sinh token của model.
"""
from __future__ import annotations

import contextlib
import threading
import time
from typing import Callable, Iterator


class Cancelled(Exception):
    """Người dùng đã hủy job."""


_local = threading.local()


@contextlib.contextmanager
def watching(is_cancelled: Callable[[], bool], interval: float = 0.5) -> Iterator[None]:
    prev = getattr(_local, "state", None)
    _local.state = {"fn": is_cancelled, "interval": interval, "at": 0.0, "value": False}
    try:
        yield
    finally:
        _local.state = prev


def requested(fresh: bool = False) -> bool:
    """Người dùng đã hủy job chưa. `fresh`: đọc lại cờ ngay, bỏ qua `interval` (dùng ở ranh giới bước)."""
    st = getattr(_local, "state", None)
    if st is None:
        return False
    if st["value"]:
        return True
    now = time.monotonic()
    if fresh or now - st["at"] >= st["interval"]:
        st["at"] = now
        try:
            st["value"] = bool(st["fn"]())
        except Exception:     # đọc cờ hủy lỗi thì coi như chưa hủy, không làm hỏng job
            st["value"] = False
    return st["value"]


def check(fresh: bool = False) -> None:
    if requested(fresh):
        raise Cancelled("Đã hủy")
