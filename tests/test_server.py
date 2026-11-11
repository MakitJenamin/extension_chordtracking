"""Test API bằng fastapi.testclient, pipeline giả (không cần GPU)."""
import threading
import time

import pytest
from fastapi.testclient import TestClient

from chordtrack import audio, jobs, paths, pipeline, server
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
        assert source.startswith("https://www.youtube.com/watch?v=")   # luôn nhận URL chuẩn
        source = audio.parse_video_id(source)
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
        assert 0.0 < st["progress"] < 1.0 and st["eta_seconds"] is not None and st["error"] is None, st
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
    job, _ = store.submit(VID, server.youtube_url(VID))
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
                      headers={"Content-Type": "text/plain"}).status_code == 422
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


def test_no_stale_done_while_force_job_active(fake, tmp_path):
    fake.results[VID] = {"song_id": VID, "old": True}
    fake.gate.clear()
    with make_client(fake, tmp_path) as c:
        f = c.post("/analyze", json={"source": VID, "force": True}).json()
        assert fake.in_chords.wait(5)
        r = c.post("/analyze", json={"source": VID}).json()
        assert r == {"status": "running", "job_id": f["job_id"], "song_id": VID}
        fake.gate.set()
        wait_status(c, f["job_id"], "done")
        assert "old" not in c.post("/analyze", json={"source": VID}).json()["result"]



# --------------------------------------------------------------------------- dò giai điệu


class FakeMelody:
    """`analyze_melody` giả: báo tiến độ qua các bước melody, có thể chặn ở `melody_fill`."""

    def __init__(self):
        self.results: dict[str, dict] = {}
        self.stale: dict[str, dict] = {}
        self.calls: list[tuple[str, bool]] = []
        self.gate = threading.Event()
        self.gate.set()
        self.in_fill = threading.Event()
        self.ready = True

    def cached(self, song_id, variant="combo"):
        return self.results.get((song_id, variant))

    def stale_of(self, song_id, variant="combo"):
        return self.results.get((song_id, variant)) or self.stale.get((song_id, variant))

    def run(self, source, *, force=False, progress=None, variant="combo"):
        assert source.startswith("https://www.youtube.com/watch?v=")
        vid = audio.parse_video_id(source)
        self.calls.append((vid, force) if variant == "combo" else (vid, force, variant))
        for step in jobs.KINDS[jobs.MELODY_KIND_OF[variant]].steps:
            progress(step, 0.0)
            if step == "melody_fill":
                self.in_fill.set()
                assert self.gate.wait(10)
            progress(step, 1.0)
        model = {"combo": "yourmt3>muscriptor_small", "yourmt3": "yourmt3", "yourmt3_vocals": "yourmt3[vocals]",
                 "combo_vocals": "yourmt3[vocals]>muscriptor_small"}[variant]
        res = {"song_id": vid, "model": model, "code_revision": "x", "duration": 10.0,
               "notes": [{"start": 1.0, "end": 1.5, "pitch": 60 if variant == "combo" else 62, "confidence": 0.8,
                          "uncertain": False, "source": "voice"}]}
        self.results[(vid, variant)] = res
        return res


def make_melody_client(fake, mel, tmp_path, **kw):
    app = server.create_app(db_path=tmp_path / "jobs.sqlite", analyze=fake.analyze, cached_result=fake.cached,
                            melody=mel.run, cached_melody=mel.cached, stale_melody=mel.stale_of,
                            melody_ready=lambda: {"combo": mel.ready, "yourmt3": True}, poll_interval=0.05, **kw)
    return TestClient(app, base_url="http://127.0.0.1")


def test_health_melody_flag(fake, tmp_path):
    mel = FakeMelody()
    with make_melody_client(fake, mel, tmp_path, start_worker=False) as c:
        h = c.get("/health").json()
        assert h["melody"] is True and h["melody_models"] == {"combo": True, "yourmt3": True}
        mel.ready = False
        h = c.get("/health").json()
        assert h["melody"] is False and h["melody_models"] == {"combo": False, "yourmt3": True}


def test_melody_job_progress_and_result(fake, tmp_path):
    mel = FakeMelody()
    mel.gate.clear()
    with make_melody_client(fake, mel, tmp_path) as c:
        assert c.get(f"/melody/{VID}").status_code == 404
        r = c.post("/melody", json={"source": f"https://www.youtube.com/watch?v={VID}"}).json()
        assert r["status"] == "queued" and r["song_id"] == VID
        assert mel.in_fill.wait(5)
        st = c.get(f"/jobs/{r['job_id']}").json()
        assert st["kind"] == "melody" and st["status"] == "running" and st["step"] == "melody_fill"
        assert st["message"] == jobs.MELODY_STEP_MESSAGES["melody_fill"]
        assert 0.0 < st["progress"] < 1.0 and st["eta_seconds"] > 0
        # Gửi trùng thì gộp; job hợp âm cùng bài là việc khác, xếp hàng sau.
        assert c.post("/melody", json={"source": VID}).json()["job_id"] == r["job_id"]
        a = c.post("/analyze", json={"source": VID}).json()
        assert a["status"] == "queued" and a["job_id"] != r["job_id"]
        assert c.get(f"/jobs/{a['job_id']}").json()["kind"] == "analyze"
        mel.gate.set()
        st = wait_status(c, r["job_id"], "done")
        assert st["progress"] == 1.0
        wait_status(c, a["job_id"], "done")
        got = c.get(f"/melody/{VID}")
        assert got.status_code == 200 and got.json()["notes"][0]["pitch"] == 60
        assert "x-chordtrack-stale" not in got.headers
        done = c.post("/melody", json={"source": VID}).json()
        assert done["status"] == "done" and done["melody"]["song_id"] == VID and "result" not in done
    assert mel.calls == [(VID, False)] and fake.calls == [(VID, False)]


def test_melody_stale_force_and_bad_requests(fake, tmp_path):
    mel = FakeMelody()
    mel.stale[(VID, "combo")] = {"song_id": VID, "code_revision": "cu", "notes": []}
    with make_melody_client(fake, mel, tmp_path, start_worker=False) as c:
        r = c.get(f"/melody/{VID}")
        assert r.status_code == 200 and r.headers["x-chordtrack-stale"] == "1"
        assert c.get("/melody/a.b").status_code == 400
        assert c.post("/melody", json={"source": "https://example.com/x"}).status_code == 400
        assert c.post("/melody", content=f'{{"source": "{VID}"}}',
                      headers={"Content-Type": "text/plain"}).status_code == 422
        # Kết quả cũ không tính là xong: tạo job.
        assert c.post("/melody", json={"source": VID}).json()["status"] == "queued"
    mel.results[(VID2, "combo")] = {"song_id": VID2, "notes": []}
    with make_melody_client(fake, mel, tmp_path) as c:
        r = c.post("/melody", json={"source": VID2, "force": True}).json()
        assert r["status"] == "queued"
        wait_status(c, r["job_id"], "done")
    assert (VID2, True) in mel.calls


def test_melody_yourmt3_variant(fake, tmp_path):
    """`model: "yourmt3"`: job riêng (không chặn job ghép cùng bài), cache riêng, GET theo `?model=`."""
    mel = FakeMelody()
    mel.gate.clear()
    with make_melody_client(fake, mel, tmp_path) as c:
        assert c.get(f"/melody/{VID}?model=yourmt3").status_code == 404
        assert c.get(f"/melody/{VID}?model=khac").status_code == 422
        assert c.post("/melody", json={"source": VID, "model": "khac"}).status_code == 422
        combo = c.post("/melody", json={"source": VID}).json()          # không có model = ghép
        assert mel.in_fill.wait(5)
        y3 = c.post("/melody", json={"source": VID, "model": "yourmt3"}).json()
        assert y3["status"] == "queued" and y3["job_id"] != combo["job_id"]
        assert c.post("/melody", json={"source": VID, "model": "yourmt3"}).json()["job_id"] == y3["job_id"]
        assert c.post("/melody", json={"source": VID, "model": "combo"}).json()["job_id"] == combo["job_id"]
        st = c.get(f"/jobs/{y3['job_id']}").json()
        assert st["kind"] == "melody_yourmt3" and st["model"] == "yourmt3" and st["status"] == "queued"
        mel.gate.set()
        wait_status(c, y3["job_id"], "done")
        wait_status(c, combo["job_id"], "done")
        a = c.get(f"/melody/{VID}?model=yourmt3").json()
        b = c.get(f"/melody/{VID}").json()
        assert a["model"] == "yourmt3" and a["notes"][0]["pitch"] == 62
        assert b["model"] == "yourmt3>muscriptor_small" and c.get(f"/melody/{VID}?model=combo").json() == b
        done = c.post("/melody", json={"source": VID, "model": "yourmt3"}).json()
        assert done["status"] == "done" and done["melody"]["model"] == "yourmt3"
        # Chỉ có bản YourMT3+: biến thể ghép vẫn chưa có.
        assert c.get(f"/melody/{VID2}?model=yourmt3").status_code == 404
        r = c.post("/melody", json={"source": VID2, "model": "yourmt3", "force": True}).json()
        wait_status(c, r["job_id"], "done")
        assert c.get(f"/melody/{VID2}").status_code == 404
        assert c.get(f"/melody/{VID2}?model=yourmt3").status_code == 200
    assert mel.calls == [(VID, False), (VID, False, "yourmt3"), (VID2, True, "yourmt3")]


def test_second_server_on_same_db_refuses_jobs(fake, tmp_path):
    mel = FakeMelody()
    with make_melody_client(fake, mel, tmp_path) as c1, make_melody_client(fake, mel, tmp_path) as c2:
        assert c1.get("/health").json()["worker_alive"] is True
        assert c2.get("/health").json()["worker_alive"] is False
        assert c2.post("/melody", json={"source": VID}).status_code == 503
        assert c2.post("/analyze", json={"source": VID}).status_code == 503


def test_melody_vocals_flag(fake, tmp_path):
    """`vocals: true` (tách giọng): biến thể riêng cho mỗi model, job và cache riêng, GET theo `&vocals=1`."""
    mel = FakeMelody()
    with make_melody_client(fake, mel, tmp_path) as c:
        assert c.get(f"/melody/{VID}?model=yourmt3&vocals=1").status_code == 404
        r = c.post("/melody", json={"source": VID, "model": "yourmt3", "vocals": True}).json()
        st = wait_status(c, r["job_id"], "done")
        assert st["kind"] == "melody_yourmt3_vocals" and st["model"] == "yourmt3" and st["vocals"] is True
        assert c.get(f"/melody/{VID}?model=yourmt3&vocals=1").json()["model"] == "yourmt3[vocals]"
        assert c.get(f"/melody/{VID}?model=yourmt3").status_code == 404          # bản không tách giọng chưa có
        assert c.get(f"/melody/{VID}?model=combo&vocals=true").status_code == 404
        r = c.post("/melody", json={"source": VID, "vocals": True}).json()        # không có model = ghép
        st = wait_status(c, r["job_id"], "done")
        assert st["kind"] == "melody_combo_vocals" and st["model"] == "combo" and st["vocals"] is True
        assert st["step"] is None or st["progress"] == 1.0
        got = c.get(f"/melody/{VID}?model=combo&vocals=1").json()
        assert got["model"] == "yourmt3[vocals]>muscriptor_small"
        done = c.post("/melody", json={"source": VID, "model": "yourmt3", "vocals": True}).json()
        assert done["status"] == "done" and done["melody"]["model"] == "yourmt3[vocals]"
        st = c.get(f"/jobs/{r['job_id']}").json()
        assert jobs.KINDS[st["kind"]].steps == ("fetch", "melody_separate", "melody_voice", "melody_fill", "melody_line")
    assert mel.calls == [(VID, False, "yourmt3_vocals"), (VID, False, "combo_vocals")]
