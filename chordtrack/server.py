"""Backend HTTP local cho extension: `python -m chordtrack.server [--port 8765]`.

Chỉ lắng nghe 127.0.0.1, không mở ra mạng ngoài. Endpoint:

- `GET  /health`              trạng thái server, GPU, Deno, số job đang chờ
- `POST /analyze`             `{"source": "<videoId hoặc URL YouTube>", "force": false}`
                              -> `{"status": "done", "result": {...}}` nếu có cache, ngược lại
                              `{"status": "queued"|"running", "job_id", "song_id"}`
- `GET  /jobs/{job_id}`       tiến độ job (xem `jobs.job_status`)
- `GET  /result/{song_id}`    JSON kết quả hoặc 404
- `GET  /songs`               danh sách bài đã phân tích (cho popup của extension)
- `POST /melody`              dò giai điệu, cùng cách dùng với `/analyze` (hợp đồng: `docs/melody_api.md`)
- `GET  /melody/{song_id}`    `<Melody>` hoặc 404

Hàng đợi job lưu ở `data/cache/jobs.sqlite`, một worker thread duy nhất cho cả job hợp âm (`kind` = `analyze`) và
job dò giai điệu (`kind` = `melody`), vì GPU chỉ có 8 GB (xem `chordtrack/jobs.py`).
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import inspect
import logging
import os
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from chordtrack import audio as audio_mod
from chordtrack import jobs, paths

log = logging.getLogger(__name__)

HOST = "127.0.0.1"          # cố định: không cho bind ra ngoài máy
DEFAULT_PORT = 8765
SONG_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# CORS: chỉ hai nơi được đọc phản hồi của server:
# - `chrome-extension://<id>`: service worker và popup của extension (MV3) gọi API với origin này.
#   ID extension là 32 chữ cái a–p; khi chưa đóng gói, ID đổi theo thư mục nên không ghim cứng.
# - `https://www.youtube.com`: fetch từ content script chạy trong trang YouTube mang origin của trang.
# Trang web khác không đọc được kết quả, và vì POST bắt buộc `Content-Type: application/json`
# (xem `analyze`), trình duyệt luôn gửi preflight nên trang lạ cũng không tạo được job.
CORS_ORIGIN_REGEX = r"^(chrome-extension://[a-p]{32}|https://www\.youtube\.com)$"
# Chống DNS rebinding: chỉ nhận Host là localhost.
ALLOWED_HOSTS = ["127.0.0.1", "localhost"]


class UTF8JSONResponse(JSONResponse):
    # Ghi rõ charset: Invoke-RestMethod của PowerShell 5.1 đọc JSON không có charset theo ISO-8859-1.
    media_type = "application/json; charset=utf-8"


class AnalyzeRequest(BaseModel):
    source: str
    force: bool = False


def default_cached_result(song_id: str) -> dict | None:
    """Kết quả trong cache của cấu hình mặc định của `pipeline.analyze`, đúng `code_revision`."""
    from chordtrack import pipeline

    return pipeline._cached_result(song_id, _default_config())


@functools.lru_cache(maxsize=1)
def _default_config():
    """Cấu hình mặc định lấy từ chữ ký `pipeline.analyze`, để server luôn tra đúng file cache mà
    `analyze(source)` sẽ ghi (kể cả khi mặc định trong pipeline đổi)."""
    from chordtrack import pipeline

    params = inspect.signature(pipeline.analyze).parameters
    fields = inspect.signature(pipeline.PipelineConfig).parameters
    kwargs = {name: params[name].default for name in fields if name in params}
    return pipeline.PipelineConfig(**kwargs).effective()


def default_analyze(source: str, *, force: bool = False, progress=None) -> dict:
    from chordtrack import pipeline

    return pipeline.analyze(source, force=force, progress=progress)


def default_melody(source: str, *, force: bool = False, progress=None) -> dict:
    from chordtrack.melody import pipeline as melody_pipeline

    return melody_pipeline.analyze_melody(source, force=force, progress=progress)


def default_cached_melody(song_id: str) -> dict | None:
    from chordtrack.melody import pipeline as melody_pipeline

    return melody_pipeline.cached_melody(song_id)


def default_stale_melody(song_id: str) -> dict | None:
    from chordtrack.melody import pipeline as melody_pipeline

    return melody_pipeline.stale_melody(song_id)


def default_melody_ready() -> bool:
    from chordtrack.melody import pipeline as melody_pipeline

    try:
        return all(melody_pipeline.melody_ready().values())
    except Exception:
        log.exception("Không kiểm tra được model dò giai điệu")
        return False


@functools.lru_cache(maxsize=1)
def gpu_name() -> str | None:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        return None


def app_version() -> str:
    try:
        from importlib.metadata import version

        return version("chordtrack")
    except Exception:
        return "0.0.1"


class SongIndex:
    """Danh sách bài đã phân tích, đọc từ `CACHE_DIR/*/result.json`. Nhớ theo mtime để khỏi đọc lại."""

    def __init__(self):
        self._lock = threading.Lock()
        self._memo: dict[Path, tuple[float, dict | None]] = {}

    def list(self, result_filename: str = "result.json") -> list[dict]:
        songs = []
        cache_dir = paths.CACHE_DIR
        if not cache_dir.is_dir():
            return songs
        with self._lock:
            for path in cache_dir.glob(f"*/{result_filename}"):
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    continue
                memo = self._memo.get(path)
                if memo is None or memo[0] != mtime:
                    memo = (mtime, _song_summary(path, mtime))
                    self._memo[path] = memo
                if memo[1] is not None:
                    songs.append(memo[1])
        songs.sort(key=lambda s: s["analyzed_at"], reverse=True)
        return songs


def _song_summary(path: Path, mtime: float) -> dict | None:
    data = audio_mod.read_json(path)
    if not data or "song_id" not in data:
        return None
    key = data.get("key") or None
    return {
        "song_id": data["song_id"],
        "title": (data.get("source") or {}).get("title"),
        "duration": data.get("duration"),
        "key": f"{key['tonic']} {key['mode']}" if key else None,
        "analyzed_at": datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(timespec="seconds"),
    }


def create_app(
    *,
    db_path: Path | str | None = None,
    analyze: Callable[..., dict] = default_analyze,
    cached_result: Callable[[str], dict | None] = default_cached_result,
    start_worker: bool = True,
    poll_interval: float = 2.0,
    melody: Callable[..., dict] = default_melody,
    cached_melody: Callable[[str], dict | None] = default_cached_melody,
    stale_melody: Callable[[str], dict | None] = default_stale_melody,
    melody_ready: Callable[[], bool] = default_melody_ready,
) -> FastAPI:
    """Tạo app. Test truyền `analyze`/`cached_result`/`melody`/`cached_melody`... giả và `db_path` tạm."""
    store_holder: dict[str, Any] = {}

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        store = jobs.JobStore(db_path or (paths.CACHE_DIR / "jobs.sqlite"))
        worker = jobs.JobWorker(store, analyze, poll_interval=poll_interval, melody=melody)
        store_holder.update(store=store, worker=worker)
        if start_worker:
            worker.start()
        try:
            yield
        finally:
            worker.stop()
            # Job đang chạy không ngắt được: để worker (daemon) còn ghi được thì không đóng DB;
            # tiến trình thoát thì job đó được khôi phục ở lần khởi động sau.
            if not worker.alive:
                store.close()

    app = FastAPI(title="chordtrack", version=app_version(), lifespan=lifespan,
                  default_response_class=UTF8JSONResponse)
    # allow_private_network: Chrome (Private Network Access) hỏi trước khi trang công khai như
    # youtube.com gọi vào 127.0.0.1, và chỉ cho gọi khi preflight trả
    # `Access-Control-Allow-Private-Network: true`.
    app.add_middleware(CORSMiddleware, allow_origin_regex=CORS_ORIGIN_REGEX, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"], allow_private_network=True, max_age=600)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)

    songs = SongIndex()

    def store() -> jobs.JobStore:
        return store_holder["store"]

    @app.get("/health")
    def health() -> dict:
        worker = store_holder.get("worker")
        return {
            "status": "ok",
            "version": app_version(),
            "gpu": gpu_name(),
            "queue": store().queue_length(),
            "worker_alive": bool(worker and worker.alive),
            "deno": jobs.deno_path() is not None,
            "ffmpeg": shutil.which("ffmpeg") is not None,
            "cookies": _cookies_ok(),
            "melody": bool(melody_ready()),
        }

    def submit(body: AnalyzeRequest, kind: str, cached: Callable[[str], dict | None], key: str) -> dict:
        # FastAPI chỉ đọc body khi Content-Type là JSON (khác thì 422), nên trình duyệt luôn gửi preflight CORS.
        video_id = audio_mod.parse_video_id(body.source)
        if video_id is None:
            raise HTTPException(400, f"Không nhận ra videoId hoặc URL YouTube: {body.source!r}")
        active = store().active(video_id, kind)
        if not body.force and (active is None or not active.force):
            # Đang chạy lại (force) thì không trả kết quả cũ như thể đã xong.
            result = cached(video_id)
            if result is not None:
                return {"status": "done", "song_id": video_id, key: result}
        # Truyền URL chuẩn, không truyền videoId trần: pipeline coi chuỗi trùng tên file trong cwd là file local.
        job, created = store().submit(video_id, youtube_url(video_id), force=body.force, kind=kind)
        if created:
            store_holder["worker"].notify()
        return {"status": job.status, "job_id": job.id, "song_id": job.song_id}

    @app.post("/analyze")
    def analyze_endpoint(body: AnalyzeRequest) -> dict:
        return submit(body, "analyze", cached_result, "result")

    @app.post("/melody")
    def melody_endpoint(body: AnalyzeRequest) -> dict:
        return submit(body, "melody", cached_melody, "melody")

    @app.get("/melody/{song_id}")
    def melody_result_endpoint(song_id: str):
        if not SONG_ID_RE.match(song_id):
            raise HTTPException(400, "song_id không hợp lệ")
        result = cached_melody(song_id)
        if result is not None:
            return result
        stale = stale_melody(song_id)
        if stale is None:
            raise HTTPException(404, "Chưa dò giai điệu bài này")
        return UTF8JSONResponse(stale, headers={"X-Chordtrack-Stale": "1"})

    @app.get("/jobs/{job_id}")
    def job_endpoint(job_id: str) -> dict:
        job = store().get(job_id)
        if job is None:
            raise HTTPException(404, "Không có job này")
        return jobs.job_status(store(), job)

    @app.get("/result/{song_id}")
    def result_endpoint(song_id: str):
        if not SONG_ID_RE.match(song_id):
            raise HTTPException(400, "song_id không hợp lệ")
        result = cached_result(song_id)
        if result is not None:
            return result
        # Kết quả của mã cũ (code_revision khác) vẫn trả về để extension hiện tạm, kèm header báo cũ.
        stale = audio_mod.read_json(paths.CACHE_DIR / song_id / _result_filename())
        if stale is None:
            raise HTTPException(404, "Chưa có kết quả cho bài này")
        return UTF8JSONResponse(stale, headers={"X-Chordtrack-Stale": "1"})

    @app.get("/songs")
    def songs_endpoint() -> dict:
        return {"songs": songs.list(_result_filename())}

    return app


def youtube_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _cookies_ok() -> bool:
    try:
        return audio_mod.cookies_file() is not None
    except FileNotFoundError:
        return False


def _result_filename() -> str:
    try:
        return _default_config().result_filename()
    except Exception:
        return "result.json"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m chordtrack.server", description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    import uvicorn

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log.info("chordtrack server: http://%s:%d (Deno: %s, cookie: %s)", HOST, args.port,
             jobs.deno_path() or "không có", "có" if os.environ.get(audio_mod.COOKIES_ENV) else "không")
    uvicorn.run(create_app(), host=HOST, port=args.port, workers=1, log_level=args.log_level)


if __name__ == "__main__":
    main()
