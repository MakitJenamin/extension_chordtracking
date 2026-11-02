"""Server tĩnh cho harness kiểm thử extension (không cần cài extension, không sửa backend).

    <venv python> extension/dev/serve.py            # http://127.0.0.1:8770/watch?v=QDYfEBY9NM4
    <venv python> extension/dev/serve.py --port 8771 --backend http://127.0.0.1:8765

- `/watch?v=<id>`  trả về harness.html (URL giống YouTube để content.js nhận videoId).
- `/src/...`, `/dev/...`  file của thư mục extension/.
- `/media/<id>`  audio của bài (hỗ trợ HTTP Range để tua). Chỉ đọc data/audio và data/eval/kajiura/audio.
- `/api/...`  chuyển tiếp tới backend (CORS của backend chỉ cho chrome-extension:// và youtube.com,
  nên trang harness gọi qua proxy cùng origin). `/api-down/...` luôn lỗi kết nối, để thử trạng thái
  "server chưa chạy".
- videoId giả (11 ký tự) để thử các trạng thái mà không cần tải YouTube:
    devmagia_01  kết quả thật của "Magia" (Kajiura, đổi giọng nhiều), audio data/eval/kajiura/audio/magia.wav
    devslow_001  job giả chạy ~15 giây (thanh tiến độ, các bước, ETA), xong thì trả kết quả Let It Be
    devblock_01  job giả lỗi "YouTube đòi xác minh không phải bot" (hướng dẫn cookie)
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
import time
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

EXT_DIR = Path(__file__).resolve().parents[1]
REPO = EXT_DIR.parent
sys.path.insert(0, str(REPO))
from chordtrack import paths  # noqa: E402

AUDIO = {
    "QDYfEBY9NM4": paths.AUDIO_DIR / "QDYfEBY9NM4.wav",
    "devmagia_01": paths.EVAL_DIR / "kajiura" / "audio" / "magia.wav",
    "devslow_001": paths.AUDIO_DIR / "QDYfEBY9NM4.wav",
    "devblock_01": paths.AUDIO_DIR / "QDYfEBY9NM4.wav",
}
RESULT_ALIAS = {"devmagia_01": "magia-f18f50a2"}
SLOW_SECONDS = 15.0
SLOW_STEPS = ["fetch", "beats", "chords", "fusion", "analysis"]
BLOCK_MSG = ("YouTube đòi xác minh \"không phải bot\". Xuất cookie YouTube (đã đăng nhập) ra file cookies.txt "
             "định dạng Netscape, đặt biến môi trường CHORDTRACK_YT_COOKIES=<đường dẫn file> rồi khởi động lại server.")
fake_jobs: dict[str, dict] = {}


class Handler(SimpleHTTPRequestHandler):
    backend = "http://127.0.0.1:8765"

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(EXT_DIR), **kw)

    def log_message(self, fmt, *args):  # gọn log
        if "/media/" not in self.path:
            sys.stderr.write("[dev] " + (fmt % args) + "\n")

    # ---------------- định tuyến ----------------
    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/watch" or path == "/":
            return self._file(EXT_DIR / "dev" / "harness.html", "text/html; charset=utf-8")
        if path.startswith("/media/"):
            return self._media(path[len("/media/"):])
        if path.startswith("/api-down/"):
            return self._drop()
        if path.startswith("/api/"):
            return self._api("GET", path[4:])
        return super().do_GET()

    def do_POST(self):
        path = urlsplit(self.path).path
        if path.startswith("/api-down/"):
            return self._drop()
        if path.startswith("/api/"):
            return self._api("POST", path[4:])
        self.send_error(404)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    # ---------------- tiện ích ----------------
    def _drop(self):
        """Đóng kết nối không trả lời: fetch() phía trình duyệt lỗi giống hệt khi server không chạy."""
        self.close_connection = True
        try:
            self.connection.shutdown(2)
        except OSError:
            pass

    def _json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, p: Path, ctype: str):
        data = p.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _media(self, vid: str):
        p = AUDIO.get(vid)
        if p is None:
            cand = paths.AUDIO_DIR / f"{vid}.wav"
            p = cand if re.fullmatch(r"[A-Za-z0-9_-]{11}", vid) and cand.exists() else None
        if p is None or not p.exists():
            return self.send_error(404, "không có audio cho videoId này")
        size = p.stat().st_size
        start, end = 0, size - 1
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:
                start = size - int(m.group(2))
            end = min(end, size - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Type", mimetypes.guess_type(p.name)[0] or "audio/wav")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        try:
            with p.open("rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(1 << 16, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _backend(self, method: str, path: str, body: bytes | None = None):
        req = urllib.request.Request(self.backend + path, data=body, method=method,
                                     headers={"Content-Type": "application/json"} if body else {})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.read(), dict(r.headers)
        except urllib.error.HTTPError as e:
            return e.code, e.read(), dict(e.headers)

    def _api(self, method: str, path: str):
        try:
            if method == "POST" and path == "/analyze":
                raw = self._body()
                src = (json.loads(raw or b"{}").get("source") or "")
                if src in RESULT_ALIAS:
                    st, data, _ = self._backend("GET", f"/result/{RESULT_ALIAS[src]}")
                    if st != 200:
                        return self._json({"detail": f"backend không có kết quả {RESULT_ALIAS[src]}"}, st)
                    return self._json({"status": "done", "song_id": src, "result": json.loads(data)})
                if src in ("devslow_001", "devblock_01"):
                    jid = f"{src}-{int(time.time())}"
                    fake_jobs[jid] = {"src": src, "t0": time.time()}
                    return self._json({"status": "queued", "job_id": jid, "song_id": src})
                return self._relay(*self._backend("POST", path, raw))
            m = re.fullmatch(r"/jobs/(dev\w+-\d+)", path)
            if m and m.group(1) in fake_jobs:
                return self._json(self._fake_job(m.group(1)))
            if path == "/result/devslow_001":
                return self._relay(*self._backend("GET", "/result/QDYfEBY9NM4"))
            return self._relay(*self._backend(method, path, self._body() if method == "POST" else None))
        except urllib.error.URLError as e:
            sys.stderr.write(f"[dev] không kết nối được backend {self.backend}: {e.reason}\n")
            self._drop()

    def _relay(self, status, data, headers):
        self.send_response(status)
        self.send_header("Content-Type", headers.get("content-type") or headers.get("Content-Type") or "application/json")
        if headers.get("x-chordtrack-stale") or headers.get("X-Chordtrack-Stale"):
            self.send_header("X-Chordtrack-Stale", "1")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _fake_job(self, jid: str) -> dict:
        j = fake_jobs[jid]
        el = time.time() - j["t0"]
        base = {"job_id": jid, "song_id": j["src"], "error": None}
        if el < 2:
            return base | {"status": "queued", "progress": 0.0, "step": None, "message": "Đang chờ (1 bài phía trước)",
                           "eta_seconds": SLOW_SECONDS + 2 - el}
        if j["src"] == "devblock_01":
            return base | {"status": "error", "progress": 0.0, "step": "fetch", "message": "Lỗi", "error": BLOCK_MSG}
        frac = min(1.0, (el - 2) / SLOW_SECONDS)
        if frac >= 1:
            return base | {"status": "done", "progress": 1.0, "step": None, "message": "Xong", "eta_seconds": 0}
        step = SLOW_STEPS[min(len(SLOW_STEPS) - 1, int(frac * len(SLOW_STEPS)))]
        return base | {"status": "running", "progress": round(frac, 3), "step": step, "message": step,
                       "eta_seconds": round(SLOW_SECONDS * (1 - frac), 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1",
                    help="địa chỉ lắng nghe; chỉ đổi khi trình duyệt kiểm thử chạy ở máy khác (lộ proxy ra mạng LAN)")
    ap.add_argument("--backend", default="http://127.0.0.1:8765")
    a = ap.parse_args()
    Handler.backend = a.backend.rstrip("/")
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"Harness: http://{a.host}:{a.port}/watch?v=QDYfEBY9NM4   (backend {Handler.backend})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
