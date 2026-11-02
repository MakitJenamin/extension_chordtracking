"""Bước 4: bắt nhịp và phách mạnh bằng beat_this (CPJKU, ISMIR 2024).

Dùng checkpoint mặc định `final0` (huấn luyện trên mọi tập trừ GTZAN), hậu xử lý
"minimal" (lấy đỉnh của xác suất từng khung), không cần madmom.
Weights tải về `data/models/beat_this/` ở lần chạy đầu.
"""
from __future__ import annotations

import functools
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

from chordtrack.gpu import device, gpu_lock
from chordtrack.paths import MODELS_DIR
from chordtrack.types import BeatGrid

CHECKPOINT_NAME = "final0"
CHECKPOINT_DIR = MODELS_DIR / "beat_this"


def _checkpoint_path() -> Path:
    """Đường dẫn checkpoint local, tải về nếu chưa có."""
    path = CHECKPOINT_DIR / f"{CHECKPOINT_NAME}.ckpt"
    if not path.exists():
        import torch
        from beat_this.inference import CHECKPOINT_URL

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        torch.hub.download_url_to_file(f"{CHECKPOINT_URL}/{CHECKPOINT_NAME}.ckpt", str(tmp))
        tmp.replace(path)
    return path


_MODEL_LOCK = threading.Lock()


@functools.lru_cache(maxsize=1)
def _load_model():
    """Audio2Beats nạp một lần cho cả tiến trình (server không nạp lại mỗi bài), giữ trên CPU.
    `track_beats` chỉ chuyển lên GPU khi đang giữ `gpu_lock()`."""
    from beat_this.inference import Audio2Beats

    return Audio2Beats(checkpoint_path=str(_checkpoint_path()), device="cpu", float16=False, dbn=False)


def _move(model, dev) -> None:
    """Chuyển Audio2Beats sang thiết bị `dev`: model, bộ tính mel và thuộc tính `device` mà
    `signal2spect` dùng để tạo tensor đầu vào."""
    model.device = dev
    model.model.to(dev)
    model.spect.to(dev)


def clear_model_cache() -> None:
    _load_model.cache_clear()


def _load_audio(audio_path: Path, max_duration: float | None) -> tuple[np.ndarray, int]:
    """Đọc audio thành mono float32. Đọc bằng soundfile để khỏi phụ thuộc backend torchaudio."""
    with sf.SoundFile(str(audio_path)) as f:
        sr = f.samplerate
        frames = -1 if max_duration is None else int(round(max_duration * sr))
        signal = f.read(frames=frames, dtype="float32", always_2d=True)
    return signal.mean(axis=1), sr


def estimate_tempo(beats: np.ndarray) -> float:
    """BPM từ trung vị khoảng cách giữa các phách. Trả 0.0 nếu có ít hơn 2 phách."""
    if len(beats) < 2:
        return 0.0
    ibi = float(np.median(np.diff(beats)))
    return 60.0 / ibi if ibi > 0 else 0.0


def _snap_downbeats(beats: np.ndarray, downbeats: np.ndarray) -> np.ndarray:
    """Gán mỗi phách mạnh vào phách gần nhất để downbeats luôn là tập con của beats."""
    if len(beats) == 0 or len(downbeats) == 0:
        return np.zeros(0, dtype=np.float64)
    idx = np.abs(beats[None, :] - downbeats[:, None]).argmin(axis=1)
    return beats[np.unique(idx)]


def track_beats(audio_path: Path, max_duration: float | None = None) -> BeatGrid:
    """Bắt phách và phách mạnh của cả file (hoặc `max_duration` giây đầu)."""
    import torch

    signal, sr = _load_audio(Path(audio_path), max_duration)
    with _MODEL_LOCK:
        model = _load_model()
        with gpu_lock():
            try:
                _move(model, torch.device(device()))
                beats, downbeats = model(signal, sr)
            finally:
                _move(model, torch.device("cpu"))

    beats = np.sort(np.asarray(beats, dtype=np.float64))
    downbeats = _snap_downbeats(beats, np.asarray(downbeats, dtype=np.float64))
    return BeatGrid(beats=beats, downbeats=downbeats, tempo=estimate_tempo(beats))


def beat_synchronous_segments(grid: BeatGrid, duration: float) -> list[tuple[float, float]]:
    """Các khoảng giữa hai phách liên tiếp, phủ kín [0, duration].

    Khoảng đầu là [0, phách đầu] và khoảng cuối là [phách cuối, duration] nếu chúng dài
    hơn 0. Phách nằm ngoài (0, duration) bị bỏ. Không có phách thì trả về một khoảng duy nhất.
    """
    if duration <= 0:
        return []
    beats = np.asarray(grid.beats, dtype=np.float64)
    inner = np.unique(beats[(beats > 0) & (beats < duration)])
    bounds = np.concatenate([[0.0], inner, [float(duration)]])
    return [(float(a), float(b)) for a, b in zip(bounds[:-1], bounds[1:])]
