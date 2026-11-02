"""Bước 0: lấy audio từ videoId, URL YouTube hoặc file local.

YouTube: tải bằng `yt_dlp` (Python API) về `paths.AUDIO_DIR/<videoId>.wav`, kèm file
`<videoId>.info.json` chứa metadata (tiêu đề, độ dài...). Có rồi thì dùng lại, không gọi mạng.
File local: `song_id` là tên file (đã lọc ký tự lạ) kèm 8 ký tự đầu của SHA-1 nội dung file.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from chordtrack import paths

log = logging.getLogger(__name__)

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YOUTUBE_HOSTS = ("youtube.com", "youtube-nocookie.com", "youtu.be")
INFO_KEYS = ("id", "title", "duration", "uploader", "channel", "upload_date", "webpage_url")


@dataclass
class AudioSource:
    song_id: str
    path: Path
    type: str                                  # "youtube" hoặc "file"
    video_id: str | None = None
    url: str | None = None
    title: str | None = None
    duration: float | None = None
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        out: dict = {"type": self.type}
        if self.type == "youtube":
            out["video_id"] = self.video_id
        else:
            out["path"] = str(self.path)
        out["title"] = self.title
        out["url"] = self.url
        return out


def parse_video_id(source: str) -> str | None:
    """Lấy videoId từ chuỗi videoId hoặc URL YouTube. Không phải YouTube thì trả None."""
    s = source.strip()
    if VIDEO_ID_RE.match(s):
        return s
    try:
        u = urlparse(s if "://" in s else "https://" + s)
    except ValueError:
        return None
    host = (u.hostname or "").lower()
    if not any(host == h or host.endswith("." + h) for h in _YOUTUBE_HOSTS):
        return None
    candidates: list[str] = []
    if host.endswith("youtu.be"):
        candidates.append(u.path.strip("/").split("/")[0])
    candidates += parse_qs(u.query).get("v", [])
    parts = [p for p in u.path.split("/") if p]
    if len(parts) >= 2 and parts[0] in ("shorts", "embed", "live", "v"):
        candidates.append(parts[1])
    for c in candidates:
        if VIDEO_ID_RE.match(c):
            return c
    return None


def fetch(source: str) -> AudioSource:
    """Như `fetch_audio` nhưng trả về cả metadata."""
    local = Path(source).expanduser()
    if local.is_file():
        return _local_source(local)
    video_id = parse_video_id(source)
    if video_id is None:
        raise ValueError(f"Không phải videoId, URL YouTube hay file có sẵn: {source!r}")
    return _youtube_source(video_id)


def fetch_audio(source: str) -> tuple[str, Path]:
    """Nhận videoId, URL YouTube hoặc đường dẫn file. Trả về (song_id, đường dẫn WAV/audio)."""
    src = fetch(source)
    return src.song_id, src.path


def _local_source(path: Path) -> AudioSource:
    path = path.resolve()
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_")[:40] or "audio"
    return AudioSource(
        song_id=f"{stem}-{h.hexdigest()[:8]}",
        path=path,
        type="file",
        title=path.stem,
        duration=_probe_duration(path),
    )


def _youtube_source(video_id: str, audio_dir: Path | None = None) -> AudioSource:
    audio_dir = Path(audio_dir or paths.AUDIO_DIR)
    wav = audio_dir / f"{video_id}.wav"
    info_path = audio_dir / f"{video_id}.info.json"
    info = read_json(info_path)
    if not wav.exists():
        info = download_youtube(video_id, audio_dir)
    elif info is None:
        # Audio có sẵn (vd. tải tay) nhưng chưa có metadata: thử lấy, không được thì bỏ qua.
        try:
            info = _extract_info(video_id)
        except Exception as e:  # mạng hỏng, video bị gỡ...
            log.warning("Không lấy được metadata của %s: %s", video_id, e)
            info = {"id": video_id}
        # Ghi cả khi lỗi để lần sau không gọi mạng lại (xóa file này để thử lại).
        write_json(info_path, info)
    duration = info.get("duration") if info else None
    return AudioSource(
        song_id=video_id,
        path=wav,
        type="youtube",
        video_id=video_id,
        url=(info or {}).get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}",
        title=(info or {}).get("title"),
        duration=float(duration) if duration else _probe_duration(wav),
        metadata=info or {},
    )


COOKIES_ENV = "CHORDTRACK_YT_COOKIES"


def cookies_file() -> Path | None:
    """File cookie YouTube (định dạng Netscape) từ biến môi trường `CHORDTRACK_YT_COOKIES`.

    Dùng khi YouTube đòi xác minh "Sign in to confirm you're not a bot". Chỉ truyền đường dẫn
    cho yt-dlp, không đọc hay log nội dung. Biến đặt mà file không tồn tại thì báo lỗi rõ ràng."""
    value = os.environ.get(COOKIES_ENV, "").strip().strip('"')
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{COOKIES_ENV} trỏ tới file không tồn tại: {path}")
    return path


def _ydl_opts(extra: dict | None = None) -> dict:
    opts = {"quiet": True, "no_warnings": True, "noprogress": True, "noplaylist": True}
    cookies = cookies_file()
    if cookies is not None:
        opts["cookiefile"] = str(cookies)
    opts.update(extra or {})
    return opts


def _extract_info(video_id: str) -> dict:
    import yt_dlp

    with yt_dlp.YoutubeDL(_ydl_opts()) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
    return {k: info.get(k) for k in INFO_KEYS}


def download_youtube(video_id: str, audio_dir: Path) -> dict:
    """Tải audio tốt nhất rồi đổi sang WAV bằng ffmpeg. Ghi `<id>.wav` và `<id>.info.json`."""
    import yt_dlp

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("Không tìm thấy ffmpeg trong PATH (yt-dlp cần để đổi sang WAV)")
    audio_dir.mkdir(parents=True, exist_ok=True)
    # Tải vào thư mục tạm rồi mới chuyển sang, để lần tải hỏng không để lại file dở.
    with tempfile.TemporaryDirectory(dir=audio_dir, prefix=f".{video_id}_", ignore_cleanup_errors=True) as tmp:
        opts = _ydl_opts({
            "format": "bestaudio/best",
            "outtmpl": str(Path(tmp) / f"{video_id}.%(ext)s"),
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        })
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
        out = Path(tmp) / f"{video_id}.wav"
        if not out.exists():
            raise RuntimeError(f"yt-dlp không tạo {out.name}; có: {sorted(p.name for p in Path(tmp).iterdir())}")
        meta = {k: info.get(k) for k in INFO_KEYS}
        write_json(audio_dir / f"{video_id}.info.json", meta)
        out.replace(audio_dir / f"{video_id}.wav")
    return meta


def _probe_duration(path: Path) -> float | None:
    try:
        import soundfile as sf

        return float(sf.info(str(path)).duration)
    except Exception:
        return None


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_json(path: Path, data: dict) -> None:
    """Ghi JSON nguyên tử: ghi ra file .part rồi đổi tên."""
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
