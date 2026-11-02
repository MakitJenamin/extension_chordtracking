"""Test API bằng fastapi.testclient, pipeline giả (không cần GPU)."""
import threading
import time

import pytest
from fastapi.testclient import TestClient

from chordtrack import jobs, paths, pipeline, server
from tests.test_pipeline import fake_env  # noqa: F401  (fixture pipeline giả)

VID = "abcdefghijk"
VID2 = "zyxwvutsrqp"


class FakePipeline:
    """`analyze` giả: báo tiến độ qua đủ các bước; có thể chặn ở bước "chords" hoặc ném lỗi."""

    def __init__(self):
        self.results: dict[str, dict] = {}
        self.calls: list[tuple[str, bool]] = []
        self.gate = threading.Event()
        self.gate.set()
        self.in_chords = threading.Event()
        self.errors: dict[str, Exception] = {}

    def cached(self, song_id):
        return self.results.get(song_id)

    def analyze(self, source, *, force=False, progress=None):
        self.calls.append((source, force))
        for step in jobs.STEPS:
            progress(step, 0.0)
            if step in ("fetch",) and source in self.errors:
                raise self.errors[source]
            if step == "chords":
                self.in_chords.set()
                assert self.gate.wait(10)
            progress(step, 1.0)
        result = {"song_id": source, "source": {"title": f"Bài {source}"}, "duration": 10.0,
                  "key": {"tonic": "C", "mode": "major", "confidence": 0.9}, "chords": []}
        self.results[source] = result
        return result


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "CACHE_DIR", tmp_path / "cache")
    return FakePipeline()


def make_client(fake, tmp_path, **kw):
    app = server.create_app(db_path=tmp_path / "jobs.sqlite", analyze=fake.analyze, cached_result=fake.cached,
                            poll_interval=0.05, **kw)
    return TestClient(app, base_url="http://127.0.0.1")


def wait_status(client, job_id, status, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = client.get(f"/jobs/{job_id}").json()
        if st["status"] == status:
            return st
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} không tới trạng thái {status}: {st}")


def test_health(fake, tmp_path):
    with make_client(fake, tmp_path) as c:
        h = c.get("/health").json()
    assert h["status"] == "ok" and h["queue"] == 0 and h["worker_alive"] is True
    assert {"version", "gpu", "deno", "ffmpeg", "cookies"} <= set(h)


def test_cache_hit_returns_done(fake, tmp_path):
    fake.results[VID] = {"song_id": VID, "chords": []}
    with make_client(fake, tmp_path) as c:
        r = c.post("/analyze", json={"source": f"https://www.youtube.com/watch?v={VID}&t=10"})
    assert r.status_code == 200
    assert r.json() == {"status": "done", "song_id": VID, "result": {"song_id": VID, "chords": []}}
    assert fake.calls == []


def test_job_created_progress_and_result(fake, tmp_path):
    fake.gate.clear()
    with make_client(fake, tmp_path) as c:
        r = c.post("/analyze", json={"source": VID}).json()
        assert r["status"] == "queued" and r["song_id"] == VID
        job_id = r["job_id"]
        assert fake.in_chords.wait(5)
        st = c.get(f"/jobs/{job_id}").json()
        assert st["status"] == "running" and st["step"] == "chords"
        assert 0.0 < st["progress"] < 1.0 and st["eta_seconds"] is not None and st["error"] is None
        assert c.get(f"/result/{VID}").status_code == 404
        fake.gate.set()
        st = wait_status(c, job_id, "done")
        assert st["progress"] == 1.0 and st["eta_seconds"] == 0.0
        assert c.get(f"/result/{VID}").json()["song_id"] == VID
        # Lần sau có cache: trả kết quả ngay.
        assert c.post("/analyze", json={"source": VID}).json()["status"] == "done"
    assert fake.calls == [(VID, False)]


def test_duplicate_jobs_merged(fake, tmp_path):
    fake.gate.clear()
    with make_client(fake, tmp_path) as c:
        a = c.post("/analyze", json={"source": VID}).json()
        assert fake.in_chords.wait(5)
        b = c.post("/analyze", json={"source": f"https://youtu.be/{VID}"}).json()
        assert b["job_id"] == a["job_id"] and b["status"] == "running"
        # Bài khác xếp hàng phía sau, gửi trùng cũng gộp.
        x = c.post("/analyze", json={"source": VID2}).json()
        y = c.post("/analyze", json={"source": VID2}).json()
        assert x["status"] == "queued" and y["job_id"] == x["job_id"]
        assert c.get("/health").json()["queue"] == 1
        st = c.get(f"/jobs/{x['job_id']}").json()
        assert st["status"] == "queued" and st["eta_seconds"] > 0
        fake.gate.set()
        wait_status(c, x["job_id"], "done")
    assert fake.calls == [(VID, False), (VID2, False)]


def test_force_reanalyzes(fake, tmp_path):
    fake.results[VID] = {"song_id": VID}
    with make_client(fake, tmp_path) as c:
        r = c.post("/analyze", json={"source": VID, "force": True}).json()
        assert r["status"] == "queued"
        wait_status(c, r["job_id"], "done")
    assert fake.calls == [(VID, True)]


def test_job_error_is_friendly_and_worker_continues(fake, tmp_path):
    fake.errors[VID] = RuntimeError("ERROR: [youtube] abcdefghijk: Sign in to confirm you're not a bot")
    with make_client(fake, tmp_path) as c:
        bad = c.post("/analyze", json={"source": VID}).json()
        good = c.post("/analyze", json={"source": VID2}).json()
        st = wait_status(c, bad["job_id"], "error")
        assert "CHORDTRACK_YT_COOKIES" in st["error"] and st["step"] == "fetch"
        wait_status(c, good["job_id"], "done")
        assert c.get("/health").json()["worker_alive"] is True


def test_restart_requeues_running_job(fake, tmp_path):
    # Lần chạy trước: job đang chạy dở thì server tắt.
    store = jobs.JobStore(tmp_path / "jobs.sqlite")
    job, _ = store.submit(VID, VID)
    store.claim_next()
    store.step_started(job.id, "beats")
    store.close()
    with make_client(fake, tmp_path) as c:
        st = wait_status(c, job.id, "done")
        assert st["progress"] == 1.0
    assert fake.calls == [(VID, False)]


def test_bad_requests(fake, tmp_path):
    with make_client(fake, tmp_path, start_worker=False) as c:
        assert c.post("/analyze", json={"source": "https://example.com/x"}).status_code == 400
        assert c.post("/analyze", content=f'{{"source": "{VID}"}}',
                      headers={"Content-Type": "text/plain"}).status_code in (415, 422)
        assert c.get("/jobs/khongco").status_code == 404
        assert c.get("/result/..%2F..%2Fetc").status_code in (400, 404)
        assert c.get("/result/a.b").status_code == 400
        assert c.get("/health").json()["worker_alive"] is False
    assert fake.calls == []


def test_cors_and_host(fake, tmp_path):
    with make_client(fake, tmp_path, start_worker=False) as c:
        ext = "chrome-extension://" + "a" * 32
        r = c.options("/analyze", headers={"Origin": ext, "Access-Control-Request-Method": "POST",
                                           "Access-Control-Request-Headers": "content-type",
                                           "Access-Control-Request-Private-Network": "true"})
        assert r.status_code == 200 and r.headers["access-control-allow-origin"] == ext
        assert r.headers["access-control-allow-private-network"] == "true"
        r = c.get("/health", headers={"Origin": "https://www.youtube.com"})
        assert r.headers["access-control-allow-origin"] == "https://www.youtube.com"
        r = c.get("/health", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in r.headers
        r = c.options("/analyze", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
        assert r.status_code == 400
        # Host lạ (DNS rebinding) bị chặn.
        assert c.get("/health", headers={"Host": "evil.example"}).status_code == 400


def test_songs_and_stale_result(fake, tmp_path, monkeypatch):
    # Dùng hàm tra cache thật của pipeline trên CACHE_DIR tạm.
    def write(song_id, revision, title):
        d = paths.CACHE_DIR / song_id
        d.mkdir(parents=True, exist_ok=True)
        from chordtrack.audio import write_json
        write_json(d / "result.json", {
            "version": pipeline.RESULT_VERSION, "song_id": song_id, "source": {"title": title},
            "duration": 243.1, "key": {"tonic": "C", "mode": "major", "confidence": 0.9},
            "pipeline": {"code_revision": revision}})

    write(VID, pipeline.code_revision(), "Let It Be")
    write(VID2, "macu", "Bài cũ")
    app = server.create_app(db_path=tmp_path / "jobs.sqlite", analyze=fake.analyze, start_worker=False)
    with TestClient(app, base_url="http://127.0.0.1") as c:
        r = c.post("/analyze", json={"source": VID}).json()
        assert r["status"] == "done" and r["result"]["source"]["title"] == "Let It Be"
        # Kết quả của mã cũ: /analyze tạo job, /result vẫn trả kèm header báo cũ.
        assert c.post("/analyze", json={"source": VID2}).json()["status"] == "queued"
        stale = c.get(f"/result/{VID2}")
        assert stale.status_code == 200 and stale.headers["x-chordtrack-stale"] == "1"
        assert "x-chordtrack-stale" not in c.get(f"/result/{VID}").headers
        songs = c.get("/songs").json()["songs"]
    assert {s["song_id"] for s in songs} == {VID, VID2}
    s = next(s for s in songs if s["song_id"] == VID)
    assert s["title"] == "Let It Be" and s["duration"] == 243.1 and s["key"] == "C major" and s["analyzed_at"]


def test_cookies_option(tmp_path, monkeypatch):
    from chordtrack import audio

    monkeypatch.delenv(audio.COOKIES_ENV, raising=False)
    assert "cookiefile" not in audio._ydl_opts()
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n")
    monkeypatch.setenv(audio.COOKIES_ENV, str(cookies))
    assert audio._ydl_opts()["cookiefile"] == str(cookies)
    monkeypatch.setenv(audio.COOKIES_ENV, str(tmp_path / "khongco.txt"))
    with pytest.raises(FileNotFoundError, match=audio.COOKIES_ENV):
        audio._ydl_opts()


def test_pipeline_progress_callback(fake_env):
    """`pipeline.analyze(progress=...)` gọi đầu và cuối mọi bước, theo đúng thứ tự."""
    events = []
    pipeline.analyze(VID, progress=lambda step, f: events.append((step, f)))
    assert events == [(s, f) for s in jobs.STEPS for f in (0.0, 1.0)]
    events.clear()
    pipeline.analyze(VID, progress=lambda step, f: events.append((step, f)))
    assert events == []   # có cache: không chạy bước nào
