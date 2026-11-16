"""Hủy job giữa bước model: tiến trình con YourMT3+ bị kill, MuScriptor dừng ở lần sinh token kế tiếp (không cần GPU)."""
import subprocess
import sys
import time

import pytest

from chordtrack import cancel
from chordtrack.melody import pipeline as mp


def test_run_yourmt3_kills_child_on_cancel(tmp_path, monkeypatch):
    procs = []
    real_popen = subprocess.Popen

    def fake_popen(cmd, **kw):
        # Thay YourMT3+ bằng tiến trình ngủ lâu.
        p = real_popen([sys.executable, "-c", "import time; time.sleep(60)"], stdout=kw.get("stdout"),
                       stderr=kw.get("stderr"), stdin=subprocess.DEVNULL)
        procs.append(p)
        return p

    monkeypatch.setattr(mp.subprocess, "Popen", fake_popen)
    t0 = time.monotonic()
    with cancel.watching(lambda: time.monotonic() - t0 > 0.3, interval=0.05):
        with pytest.raises(cancel.Cancelled):
            mp.run_yourmt3(tmp_path / "mix.wav", tmp_path / "out.npz")
    assert time.monotonic() - t0 < 10
    assert procs and procs[0].poll() is not None      # đã bị kill


def test_muscriptor_cancellable_hooks_token_loop():
    lm = pytest.importorskip("muscriptor.models.lm")
    with mp._muscriptor_cancellable():
        pass
    with mp._muscriptor_cancellable():      # gọi lần hai không bọc chồng
        pass
    fn = lm.LMModel._compute_logits
    assert fn.__wrapped__ is not None and not hasattr(fn.__wrapped__, "__wrapped__")
    with cancel.watching(lambda: True, interval=0):
        with pytest.raises(cancel.Cancelled):
            fn(object(), None, None, None, True)


def test_gpu_lock_wait_is_cancellable(monkeypatch, tmp_path):
    from filelock import FileLock

    from chordtrack import gpu

    monkeypatch.setattr(gpu, "LOCK_PATH", tmp_path / "gpu.lock")
    import threading

    held = threading.Event()
    release = threading.Event()

    def holder():      # tiến trình khác giữ khóa GPU (thread khác để FileLock không coi là cùng chủ)
        with FileLock(str(tmp_path / "gpu.lock"), is_singleton=False):
            held.set()
            release.wait(10)

    t = threading.Thread(target=holder)
    t.start()
    assert held.wait(5)
    t0 = time.monotonic()
    try:
        with cancel.watching(lambda: time.monotonic() - t0 > 0.3, interval=0.05):
            with pytest.raises(cancel.Cancelled):
                with gpu.gpu_lock():
                    pass
        assert time.monotonic() - t0 < 5
    finally:
        release.set()
        t.join(5)
