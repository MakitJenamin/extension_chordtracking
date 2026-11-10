"""Hàng đợi job phân tích, lưu bằng SQLite (`data/cache/jobs.sqlite`), và một worker thread duy nhất.

- Hai loại job (`kind`): `"analyze"` (hợp âm, `pipeline.analyze`) và `"melody"` (dò giai điệu,
  `chordtrack.melody.pipeline.analyze_melody`). Mỗi loại có danh sách bước, câu thông báo và thời gian mặc định
  riêng (`KINDS`), chung một hàng đợi theo thứ tự gửi.

- Chỉ **một** worker xử lý lần lượt từng job vì GPU chỉ có 8 GB (mỗi bước model còn tự lấy
  `gpu_lock()` bên trong pipeline). Model được các wrapper cache trong tiến trình, nên các job
  sau không phải nạp lại.
- Job bền qua restart: job đang `running` khi server tắt sẽ được đưa về `queued` lúc khởi động lại.
- Mỗi (`song_id`, `kind`) chỉ có tối đa một job đang chờ hoặc đang chạy; gửi trùng thì trả lại job đó.
- Tiến độ: pipeline gọi `progress(step, 0.0|1.0)` ở đầu và cuối mỗi bước. Thời gian từng bước
  của các job đã xong được lưu vào bảng `step_times` để ước tính `progress` và `eta_seconds`.
- Lỗi trong job được đổi thành câu tiếng Việt dễ hiểu (`friendly_error`) và lưu vào `error`;
  worker không chết.
"""
from __future__ import annotations

import logging
import shutil
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

from chordtrack.melody.pipeline import STEPS as MELODY_STEPS
from chordtrack.pipeline import TIMING_KEYS

STEPS = tuple(k for k in TIMING_KEYS if k != "total")   # các bước pipeline báo qua `progress`
STEP_MESSAGES = {
    "fetch": "Đang tải audio từ YouTube",
    "separation": "Đang chuẩn bị audio (tách nguồn nếu bật)",
    "beats": "Đang bắt nhịp",
    "chords": "Đang nhận diện hợp âm",
    "fusion": "Đang kết hợp kết quả hai model",
    "analysis": "Đang tìm giọng, đoạn và vòng hợp âm",
}
# Thời gian mặc định (giây) mỗi bước khi chưa có lịch sử, đo trên bài khoảng 4 phút, không tách nguồn.
DEFAULT_STEP_SECONDS = {"fetch": 10.0, "separation": 3.0, "beats": 5.0, "chords": 15.0, "fusion": 2.0,
                        "analysis": 1.0}

MELODY_STEP_MESSAGES = {
    "fetch": "Đang lấy audio (dùng lại bản đã tải nếu có)",
    "melody_voice": "Đang dò giọng hát (YourMT3+)",
    "melody_fill": "Đang dò giai điệu nhạc cụ (MuScriptor)",
    "melody_line": "Đang ghép dòng giai điệu",
}
# Đo trên "Let It Be" (243 s, RTX 3060 Ti), audio đã tải sẵn; xem PLAN.md mục 6.
MELODY_DEFAULT_STEP_SECONDS = {"fetch": 1.0, "melody_voice": 50.0, "melody_fill": 120.0, "melody_line": 1.0}


@dataclass(frozen=True)
class Kind:
    steps: tuple[str, ...]
    messages: dict[str, str]
    default_seconds: dict[str, float]
    error_prefix: str           # đầu câu báo lỗi chung


KINDS = {
    "analyze": Kind(STEPS, STEP_MESSAGES, DEFAULT_STEP_SECONDS, "Lỗi khi phân tích"),
    "melody": Kind(MELODY_STEPS, MELODY_STEP_MESSAGES, MELODY_DEFAULT_STEP_SECONDS, "Lỗi khi dò giai điệu"),
}


def kind_name(kind: str | None) -> str:
    """Loại job hợp lệ (loại lạ, vd từ DB của bản mới hơn, coi như `analyze` khi hiển thị)."""
    return kind if kind in KINDS else "analyze"
HISTORY = 20            # số lần chạy gần nhất của mỗi bước dùng để lấy trung bình
ACTIVE = ("queued", "running")

AnalyzeFn = Callable[..., dict]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL DEFAULT 'analyze',
    song_id TEXT NOT NULL,
    source TEXT NOT NULL,
    force INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    step TEXT,
    step_started_at REAL,
    steps_done TEXT NOT NULL DEFAULT '',
    error TEXT,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL
);
CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs (status, created_at);
CREATE TABLE IF NOT EXISTS step_times (
    job_id TEXT NOT NULL,
    step TEXT NOT NULL,
    seconds REAL NOT NULL,
    finished_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS step_times_step ON step_times (step, finished_at);
"""


@dataclass
class Job:
    id: str
    song_id: str
    source: str
    force: bool
    status: str
    step: str | None
    step_started_at: float | None
    steps_done: list[str]
    error: str | None
    created_at: float
    started_at: float | None
    finished_at: float | None
    kind: str = "analyze"

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Job":
        return cls(
            id=row["id"], song_id=row["song_id"], source=row["source"], force=bool(row["force"]),
            status=row["status"], step=row["step"], step_started_at=row["step_started_at"],
            steps_done=[s for s in (row["steps_done"] or "").split(",") if s], error=row["error"],
            created_at=row["created_at"], started_at=row["started_at"], finished_at=row["finished_at"],
            kind=row["kind"] or "analyze",
        )


class JobStore:
    """Bảng job trong SQLite. Một kết nối dùng chung giữa các thread, có khóa."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._averages: dict[str, dict[str, float]] = {}
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(_SCHEMA)
            cols = {r["name"] for r in self._db.execute("PRAGMA table_info(jobs)")}
            if "kind" not in cols:   # DB của bản cũ (chỉ có job hợp âm)
                self._db.execute("ALTER TABLE jobs ADD COLUMN kind TEXT NOT NULL DEFAULT 'analyze'")
            self._db.execute("DROP INDEX IF EXISTS jobs_song_status")
            self._db.execute("CREATE INDEX IF NOT EXISTS jobs_song_kind_status ON jobs (song_id, kind, status)")

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # --- tạo và đọc job ---

    def submit(self, song_id: str, source: str, force: bool = False, kind: str = "analyze") -> tuple[Job, bool]:
        """Tạo job mới, hoặc trả lại job đang chờ/chạy cùng `song_id` và `kind`. Trả về (job, có tạo mới không).
        Gửi `force` cho job đang chờ thì job đó chuyển sang `force`."""
        if kind not in KINDS:
            raise ValueError(f"kind phải là một trong {tuple(KINDS)}, nhận {kind!r}")
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM jobs WHERE song_id = ? AND kind = ? AND status IN ('queued', 'running') "
                "ORDER BY created_at LIMIT 1", (song_id, kind)).fetchone()
            if row is not None:
                if force and row["status"] == "queued" and not row["force"]:
                    self._db.execute("UPDATE jobs SET force = 1 WHERE id = ?", (row["id"],))
                    row = self._row(row["id"])
                return Job.from_row(row), False
            job_id = uuid.uuid4().hex[:16]
            self._db.execute(
                "INSERT INTO jobs (id, kind, song_id, source, force, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, 'queued', ?)",
                (job_id, kind, song_id, source, int(force), time.time()))
            return Job.from_row(self._row(job_id)), True

    def active(self, song_id: str, kind: str = "analyze") -> Job | None:
        """Job đang chờ hoặc đang chạy của `song_id` (loại `kind`), nếu có."""
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM jobs WHERE song_id = ? AND kind = ? AND status IN ('queued', 'running') "
                "ORDER BY created_at LIMIT 1", (song_id, kind)).fetchone()
        return None if row is None else Job.from_row(row)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            row = self._row(job_id)
        return None if row is None else Job.from_row(row)

    def _row(self, job_id: str) -> sqlite3.Row | None:
        return self._db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()

    def queued(self) -> list[Job]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at, rowid").fetchall()
        return [Job.from_row(r) for r in rows]

    def running(self) -> Job | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM jobs WHERE status = 'running' LIMIT 1").fetchone()
        return None if row is None else Job.from_row(row)

    def queue_length(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM jobs WHERE status = 'queued'").fetchone()[0])

    # --- chuyển trạng thái (chỉ worker gọi) ---

    def recover(self) -> int:
        """Job còn `running` từ lần chạy trước (server tắt giữa chừng) quay về `queued`."""
        with self._lock:
            cur = self._db.execute(
                "UPDATE jobs SET status = 'queued', step = NULL, step_started_at = NULL, steps_done = '', "
                "started_at = NULL WHERE status = 'running'")
            return cur.rowcount

    def claim_next(self) -> Job | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at, rowid LIMIT 1").fetchone()
            if row is None:
                return None
            self._db.execute("UPDATE jobs SET status = 'running', started_at = ? WHERE id = ?",
                             (time.time(), row["id"]))
            return Job.from_row(self._row(row["id"]))

    def step_started(self, job_id: str, step: str) -> None:
        with self._lock:
            self._db.execute("UPDATE jobs SET step = ?, step_started_at = ? WHERE id = ?", (step, time.time(), job_id))

    def step_finished(self, job_id: str, step: str) -> None:
        with self._lock:
            row = self._row(job_id)
            if row is None:
                return
            done = [s for s in (row["steps_done"] or "").split(",") if s]
            if step not in done:
                done.append(step)
            started = row["step_started_at"] if row["step"] == step else None
            self._db.execute("UPDATE jobs SET steps_done = ? WHERE id = ?", (",".join(done), job_id))
            if started is not None:
                self._db.execute("INSERT INTO step_times (job_id, step, seconds, finished_at) VALUES (?, ?, ?, ?)",
                                 (job_id, step, max(0.0, time.time() - started), time.time()))

    def finish(self, job_id: str, error: str | None = None) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE jobs SET status = ?, error = ?, finished_at = ?, step_started_at = NULL WHERE id = ?",
                ("error" if error else "done", error, time.time(), job_id))
            if error:
                # Thời gian các bước của job lỗi không đại diện, bỏ khỏi thống kê.
                self._db.execute("DELETE FROM step_times WHERE job_id = ?", (job_id,))
            # Chỉ giữ HISTORY lần gần nhất mỗi bước để bảng không phình mãi.
            self._db.execute(
                "DELETE FROM step_times WHERE rowid IN (SELECT rid FROM (SELECT t.rowid AS rid, ROW_NUMBER() "
                "OVER (PARTITION BY j.kind, t.step ORDER BY t.finished_at DESC) AS n FROM step_times t "
                "LEFT JOIN jobs j ON j.id = t.job_id) WHERE n > ?)", (HISTORY,))
            self._averages = {}

    # --- thống kê thời gian ---

    def step_averages(self, kind: str = "analyze") -> dict[str, float]:
        """Thời gian trung bình (giây) mỗi bước của loại job `kind` trên `HISTORY` lần chạy gần nhất; chưa có thì
        mặc định. Nhớ lại giữa các lần gọi (mỗi lần poll `/jobs`), chỉ tính lại khi có job kết thúc."""
        spec = KINDS[kind]
        with self._lock:
            if kind in self._averages:
                return dict(self._averages[kind])
            out = dict(spec.default_seconds)
            for step in spec.steps:
                rows = self._db.execute(
                    "SELECT t.seconds FROM step_times t JOIN jobs j ON j.id = t.job_id "
                    "WHERE t.step = ? AND j.kind = ? AND j.status = 'done' ORDER BY t.finished_at DESC LIMIT ?",
                    (step, kind, HISTORY)).fetchall()
                if rows:
                    out[step] = sum(r[0] for r in rows) / len(rows)
            self._averages[kind] = out
            return dict(out)


def job_status(store: JobStore, job: Job, now: float | None = None) -> dict[str, Any]:
    """JSON trả cho `GET /jobs/{id}`: trạng thái, tiến độ 0–1, bước, thông báo, ETA, lỗi."""
    now = time.time() if now is None else now
    kind = kind_name(job.kind)
    spec = KINDS[kind]
    avg = store.step_averages(kind)
    total = sum(avg[s] for s in spec.steps) or 1.0
    out: dict[str, Any] = {"job_id": job.id, "song_id": job.song_id, "kind": job.kind, "status": job.status,
                           "step": None, "progress": 0.0, "message": None, "eta_seconds": None, "error": None}
    if job.status == "done":
        out.update(progress=1.0, message="Xong", eta_seconds=0.0)
    elif job.status == "error":
        out.update(step=job.step, message="Lỗi", error=job.error)
    elif job.status == "running":
        remaining, done_time = _running_remaining(job, avg, now, spec.steps)
        out.update(step=job.step, message=spec.messages.get(job.step or "", "Đang bắt đầu"),
                   progress=round(min(0.99, done_time / total), 3), eta_seconds=round(remaining, 1))
    else:  # queued
        before = _jobs_ahead(store, job)
        ahead = len(before)
        running = store.running()
        wait = 0.0
        if running is not None:
            rk = kind_name(running.kind)
            wait = _running_remaining(running, store.step_averages(rk), now, KINDS[rk].steps)[0]
        eta = wait + sum(_kind_total(store, j.kind) for j in before) + total
        msg = "Đang chờ" + (f" ({ahead + (running is not None)} bài phía trước)" if ahead or running else "")
        out.update(message=msg, eta_seconds=round(eta, 1))
    return out


def _jobs_ahead(store: JobStore, job: Job) -> list[Job]:
    """Các job đang chờ xếp trước `job` trong hàng đợi."""
    queued = store.queued()
    for i, j in enumerate(queued):
        if j.id == job.id:
            return queued[:i]
    return []


def _kind_total(store: JobStore, kind: str) -> float:
    kind = kind_name(kind)
    avg = store.step_averages(kind)
    return sum(avg[s] for s in KINDS[kind].steps)


def _running_remaining(job: Job, avg: dict[str, float], now: float,
                       steps: tuple[str, ...] = STEPS) -> tuple[float, float]:
    """(giây còn lại, giây "đã làm" quy đổi theo trung bình) của một job đang chạy."""
    done_time = sum(avg[s] for s in job.steps_done if s in avg)
    current = job.step if job.step not in job.steps_done else None
    current_done = 0.0
    if current is not None and job.step_started_at is not None:
        # Callback chỉ báo đầu và cuối bước, nên trong bước thì nội suy theo thời gian đã trôi,
        # chặn ở 95% thời gian trung bình của bước để không báo xong khi chưa xong.
        current_done = min(max(0.0, now - job.step_started_at), 0.95 * avg.get(current, 0.0))
    remaining = sum(avg[s] for s in steps if s not in job.steps_done and s != current)
    if current is not None:
        remaining += max(0.0, avg.get(current, 0.0) - current_done)
    return remaining, done_time + current_done


def friendly_error(exc: BaseException, step: str | None = None, kind: str = "analyze") -> str:
    """Đổi lỗi thành câu tiếng Việt dễ hiểu cho extension hiển thị."""
    text = f"{type(exc).__name__}: {exc}"
    low = text.lower()
    where = f" (bước {step})" if step else ""
    prefix = KINDS[kind_name(kind)].error_prefix
    # Các dấu hiệu lỗi mạng chỉ là lỗi YouTube khi đang tải audio (job hợp âm tải trong bước fetch; bước model của
    # job melody có thể hết giờ hoặc lỗi kết nối Hugging Face, không liên quan YouTube).
    youtube_step = kind_name(kind) == "analyze" or step in (None, "fetch")
    if type(exc).__name__ == "TimeoutExpired":
        return f"{prefix}{where}: model chạy quá lâu nên bị dừng (có thể GPU đang bận). Thử lại sau."
    if type(exc).__name__ == "MuscriptorAuthError":
        return ("Chưa tải được weights MuScriptor small: repo trên Hugging Face cần đăng nhập và đồng ý license "
                "CC BY-NC 4.0. Chạy `hf auth login`, bấm đồng ý ở https://huggingface.co/MuScriptor/muscriptor-small "
                "rồi chạy `python scripts/fetch_models.py --melody` (xem README).")
    if "thiếu checkpoint yourmt3" in low or ("yourmt3" in low and "no module named" in low):
        return ("Chưa cài đủ YourMT3+ để dò giai điệu. Chạy `python scripts/fetch_models.py --melody` "
                f"(xem README, mục dò giai điệu). Chi tiết: {_short(exc)}")
    if "sign in to confirm" in low or "not a bot" in low:
        return ("YouTube đòi xác minh \"không phải bot\". Xuất cookie YouTube (đã đăng nhập) ra file cookies.txt "
                "định dạng Netscape, đặt biến môi trường CHORDTRACK_YT_COOKIES=<đường dẫn file> rồi khởi động "
                "lại server.")
    if "confirm your age" in low or "age-restricted" in low or "inappropriate for some users" in low:
        return ("Video giới hạn độ tuổi, cần đăng nhập. Đặt biến môi trường CHORDTRACK_YT_COOKIES=<file "
                "cookies.txt của tài khoản đã đăng nhập> rồi khởi động lại server.")
    if "private video" in low:
        return "Video ở chế độ riêng tư, không tải được."
    if any(k in low for k in ("video unavailable", "this video is not available", "has been removed",
                              "does not exist", "not available in your country", "this video has been")):
        return "Video không tồn tại hoặc không xem được (đã bị gỡ, bị chặn theo vùng hoặc sai videoId)."
    if "javascript runtime" in low or "deno" in low or "n challenge" in low or "nsig" in low:
        return ("yt-dlp cần Deno để giải mã YouTube nhưng không chạy được. Cài Deno (winget install DenoLand.Deno), "
                "đảm bảo `deno` có trong PATH rồi khởi động lại server.")
    if "ffmpeg" in low and ("không tìm thấy" in low or "not found" in low):
        return "Không tìm thấy ffmpeg trong PATH. Cài ffmpeg rồi khởi động lại server (dùng scripts/start_server.ps1)."
    if "chordtrack_yt_cookies" in low:
        return f"Cấu hình cookie sai: {exc}"
    if "out of memory" in low or type(exc).__name__ == "OutOfMemoryError":
        return (f"GPU hết bộ nhớ (VRAM){where}. Có thể tiến trình khác đang dùng GPU; đóng bớt rồi gửi lại bài.")
    if type(exc).__name__ == "Timeout" and "lock" in low:
        return "Chờ khóa GPU quá lâu: tiến trình khác giữ GPU quá 30 phút. Thử lại sau."
    if "http error 429" in low or "too many requests" in low:
        return "YouTube đang giới hạn số lần tải (HTTP 429). Đợi một lúc rồi thử lại, hoặc đặt CHORDTRACK_YT_COOKIES."
    if youtube_step and any(k in low for k in ("unable to download", "getaddrinfo", "failed to resolve", "connection",
                                               "timed out", "http error 5")):
        return f"Không kết nối được YouTube{where}. Kiểm tra mạng rồi thử lại. Chi tiết: {_short(exc)}"
    return f"{prefix}{where}: {type(exc).__name__}: {_short(exc)}"


def _short(exc: BaseException, limit: int = 300) -> str:
    s = " ".join(str(exc).split())
    return s if len(s) <= limit else s[:limit] + "…"


class JobWorker:
    """Một worker thread lấy job theo thứ tự tạo và gọi hàm của loại job đó: `analyze(source, force=...,
    progress=...)` với job `"analyze"`, `melody(...)` (cùng chữ ký) với job `"melody"`."""

    def __init__(self, store: JobStore, analyze: AnalyzeFn, poll_interval: float = 2.0, *,
                 melody: AnalyzeFn | None = None):
        self.store = store
        self.lock_conflict = False     # server khác đã chạy worker trên cùng DB
        self.runners: dict[str, AnalyzeFn | None] = {"analyze": analyze, "melody": melody}
        self.poll_interval = poll_interval
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Bắt đầu worker. Mỗi file DB chỉ có một worker (khóa `<db>.worker.lock`): server thứ hai trỏ cùng DB
        (vd chạy thêm một bản ở cổng khác) không chạy worker, nếu không nó sẽ `recover()` rồi chạy lại cả job đang
        chạy của server kia. Server bản cũ (trước khi có khóa) vẫn có thể làm vậy, nên đừng chạy song song."""
        if str(self.store.path) != ":memory:":
            from filelock import FileLock, Timeout

            lock = FileLock(str(self.store.path) + ".worker.lock")
            try:
                lock.acquire(timeout=0)
            except Timeout:
                log.error("Đã có server khác chạy worker trên %s; server này không xử lý job", self.store.path)
                self.lock_conflict = True
                return
            self._db_lock = lock
        n = self.store.recover()
        if n:
            log.info("Đưa %d job đang chạy dở về hàng đợi", n)
        self._thread = threading.Thread(target=self._loop, name="chordtrack-worker", daemon=True)
        self._thread.start()

    def notify(self) -> None:
        self._wake.set()

    def stop(self, timeout: float | None = 5.0) -> None:
        """Dừng nhận job mới. Job đang chạy dở (không ngắt được) sẽ được khôi phục ở lần khởi động sau."""
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
        lock = getattr(self, "_db_lock", None)
        if lock is not None and not self.alive:   # job còn chạy thì giữ khóa tới khi tiến trình thoát
            lock.release()
            self._db_lock = None

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _loop(self) -> None:
        while not self._stop.is_set():
            # Xóa cờ trước khi lấy job, để notify() đến trong lúc này không bị mất.
            self._wake.clear()
            try:
                job = self.store.claim_next()
                if job is not None:
                    self.run_job(job)
                    continue
            except Exception:
                # Worker là thread duy nhất: lỗi bất ngờ (SQLite...) chỉ được log, không làm chết vòng lặp.
                log.exception("Lỗi trong worker")
            self._wake.wait(self.poll_interval)

    def run_job(self, job: Job) -> None:
        log.info("Bắt đầu job %s %s (%s)", job.kind, job.id, job.song_id)
        current: list[str | None] = [None]

        def progress(step: str, fraction: float) -> None:
            try:
                if fraction <= 0.0:
                    current[0] = step
                    self.store.step_started(job.id, step)
                elif fraction >= 1.0:
                    self.store.step_finished(job.id, step)
            except Exception:  # báo tiến độ hỏng không được làm hỏng job
                log.exception("Không ghi được tiến độ job %s", job.id)

        prefix = KINDS[kind_name(job.kind)].error_prefix
        runner = self.runners.get(job.kind)
        if runner is None:
            self.store.finish(job.id, error=f"{prefix}: server không hỗ trợ loại job {job.kind!r}")
            return
        t0 = time.perf_counter()
        try:
            runner(job.source, force=job.force, progress=progress)
        except Exception as e:
            log.exception("Job %s (%s) lỗi ở bước %s", job.id, job.song_id, current[0])
            _free_vram()
            try:
                message = friendly_error(e, current[0], job.kind)
            except Exception:
                message = f"{prefix}: {type(e).__name__}"
            self.store.finish(job.id, error=message)
        except BaseException as e:  # SystemExit... từ thư viện: ghi lỗi cho job rồi để worker tiếp tục
            log.exception("Job %s (%s) dừng bất thường", job.id, job.song_id)
            self.store.finish(job.id, error=f"{prefix}: {type(e).__name__}")
        else:
            self.store.finish(job.id)
            log.info("Xong job %s (%s) sau %.1f s", job.id, job.song_id, time.perf_counter() - t0)


def _free_vram() -> None:
    try:
        from chordtrack.gpu import free_vram

        free_vram()
    except Exception:
        pass


def deno_path() -> str | None:
    return shutil.which("deno")
