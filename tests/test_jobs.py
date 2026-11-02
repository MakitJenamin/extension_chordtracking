"""Test hàng đợi job SQLite (không cần GPU)."""
import threading
import time

import pytest

from chordtrack import jobs


@pytest.fixture
def store(tmp_path):
    s = jobs.JobStore(tmp_path / "jobs.sqlite")
    yield s
    s.close()


def test_submit_dedupes_active_jobs(store):
    a, created_a = store.submit("abcdefghijk", "abcdefghijk")
    b, created_b = store.submit("abcdefghijk", "abcdefghijk")
    assert created_a and not created_b and a.id == b.id
    other, created = store.submit("zzzzzzzzzzz", "zzzzzzzzzzz")
    assert created and other.id != a.id
    assert store.queue_length() == 2
    # Job xong rồi thì gửi lại tạo job mới.
    store.claim_next()
    store.finish(a.id)
    c, created_c = store.submit("abcdefghijk", "abcdefghijk")
    assert created_c and c.id != a.id


def test_force_upgrades_queued_job(store):
    a, _ = store.submit("abcdefghijk", "abcdefghijk")
    b, created = store.submit("abcdefghijk", "abcdefghijk", force=True)
    assert not created and b.id == a.id and b.force


def test_recover_running_jobs(tmp_path):
    path = tmp_path / "jobs.sqlite"
    s = jobs.JobStore(path)
    job, _ = s.submit("abcdefghijk", "abcdefghijk")
    s.claim_next()
    s.step_started(job.id, "fetch")
    s.close()

    s2 = jobs.JobStore(path)
    assert s2.get(job.id).status == "running"
    assert s2.recover() == 1
    j = s2.get(job.id)
    assert j.status == "queued" and j.step is None and j.steps_done == []
    s2.close()


def test_step_times_and_progress(store):
    job, _ = store.submit("abcdefghijk", "abcdefghijk")
    store.claim_next()
    for step in jobs.STEPS:
        store.step_started(job.id, step)
        store.step_finished(job.id, step)
    store.finish(job.id)
    avg = store.step_averages()
    assert all(avg[s] < 1.0 for s in jobs.STEPS)   # đã thay giá trị mặc định bằng số đo thật

    j2, _ = store.submit("zzzzzzzzzzz", "zzzzzzzzzzz")
    store.claim_next()
    store.step_started(j2.id, "fetch")
    store.step_finished(j2.id, "fetch")
    store.step_started(j2.id, "separation")
    st = jobs.job_status(store, store.get(j2.id))
    assert st["status"] == "running" and st["step"] == "separation"
    assert 0.0 < st["progress"] < 1.0 and st["eta_seconds"] >= 0.0
    assert st["message"]


def test_failed_job_times_not_recorded(store):
    job, _ = store.submit("abcdefghijk", "abcdefghijk")
    store.claim_next()
    store.step_started(job.id, "fetch")
    store.step_finished(job.id, "fetch")
    store.finish(job.id, error="hỏng")
    assert store.step_averages()["fetch"] == jobs.DEFAULT_STEP_SECONDS["fetch"]
    st = jobs.job_status(store, store.get(job.id))
    assert st["status"] == "error" and st["error"] == "hỏng"


def test_queued_eta_counts_jobs_ahead(store):
    a, _ = store.submit("aaaaaaaaaaa", "aaaaaaaaaaa")
    b, _ = store.submit("bbbbbbbbbbb", "bbbbbbbbbbb")
    total = sum(jobs.DEFAULT_STEP_SECONDS.values())
    assert jobs.job_status(store, a)["eta_seconds"] == pytest.approx(total)
    st_b = jobs.job_status(store, b)
    assert st_b["eta_seconds"] == pytest.approx(2 * total)
    assert "1 bài phía trước" in st_b["message"]


@pytest.mark.parametrize("message, expect", [
    ("ERROR: [youtube] abc: Sign in to confirm you’re not a bot. Use --cookies", "CHORDTRACK_YT_COOKIES"),
    ("ERROR: [youtube] abc: Video unavailable", "không tồn tại"),
    ("ERROR: [youtube] abc: Private video. Sign in if you've been granted access", "riêng tư"),
    ("CUDA out of memory. Tried to allocate 2.00 GiB", "VRAM"),
    ("No supported JavaScript runtime could be found", "Deno"),
    ("ERROR: Unable to download webpage: <urlopen error [Errno 11001] getaddrinfo failed>", "kết nối"),
    ("HTTP Error 429: Too Many Requests", "429"),
    ("Không tìm thấy ffmpeg trong PATH (yt-dlp cần để đổi sang WAV)", "ffmpeg"),
    ("cái gì đó lạ", "Lỗi khi phân tích (bước chords)"),
])
def test_friendly_error(message, expect):
    assert expect in jobs.friendly_error(RuntimeError(message), "chords")


def test_worker_survives_errors_and_runs_in_order(store):
    order = []
    done = threading.Event()

    def analyze(source, *, force=False, progress=None):
        order.append(source)
        progress("fetch", 0.0)
        if source == "bad":
            raise RuntimeError("ERROR: Video unavailable")
        progress("fetch", 1.0)
        if source == "last":
            done.set()
        return {}

    ids = [store.submit(s, s)[0].id for s in ("bad", "good", "last")]
    w = jobs.JobWorker(store, analyze, poll_interval=0.05)
    w.start()
    try:
        assert done.wait(5)
        deadline = time.time() + 5
        while store.get(ids[2]).status != "done" and time.time() < deadline:
            time.sleep(0.02)
    finally:
        w.stop()
    assert order == ["bad", "good", "last"]
    bad = store.get(ids[0])
    assert bad.status == "error" and "không tồn tại" in bad.error and bad.step == "fetch"
    assert [store.get(i).status for i in ids[1:]] == ["done", "done"]
