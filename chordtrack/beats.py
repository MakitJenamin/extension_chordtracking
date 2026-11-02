"""Bước 4: bắt nhịp và phách mạnh bằng beat_this (CPJKU, ISMIR 2024).

Dùng checkpoint mặc định `final0` (huấn luyện trên mọi tập trừ GTZAN), hậu xử lý
"minimal" (lấy đỉnh của xác suất từng khung), không cần madmom.
Weights tải về `data/models/beat_this/` ở lần chạy đầu.
"""
from __future__ import annotations

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
    signal, sr = _load_audio(Path(audio_path), max_duration)
    ckpt = str(_checkpoint_path())

    from beat_this.inference import Audio2Beats

    with gpu_lock():
        dev = device()
        model = Audio2Beats(checkpoint_path=ckpt, device=dev, float16=False, dbn=False)
        try:
            beats, downbeats = model(signal, sr)
        finally:
            del model

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
