"""Bước 4: bắt nhịp và phách mạnh bằng beat_this (CPJKU, ISMIR 2024).

Dùng checkpoint mặc định `final0` (huấn luyện trên mọi tập trừ GTZAN), hậu xử lý
"minimal" (lấy đỉnh của xác suất từng khung), không cần madmom.
Weights tải về `data/models/beat_this/` ở lần chạy đầu.

Hậu xử lý riêng (không cần madmom), vì "minimal" đoán từng phách mạnh riêng lẻ:
- `fix_double_time`: đoạn dài model bắt gấp đôi tốc độ (khoảng cách phách ~ nửa trung vị của bài) thì bỏ cách một
  phách, giữ dãy nối đều nhất với phần còn lại.
- `lock_meter`: Viterbi trên vị trí trong ô nhịp. Xác suất phách mạnh của model chấm điểm cho "phách 1"; ô nhịp khác
  `meter` phách (2, 3, 5…) bị phạt `switch_penalty`. Nhạc đổi hợp âm mỗi 2 phách làm model đặt phách mạnh cứ 2 phách
  một lần; khi hai cách đặt ngang điểm, Viterbi giữ nhịp cũ thay vì đổi.
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


FPS = 50                 # khung/giây của đầu ra beat_this
DOUBLE_RATIO = 0.7       # khoảng cách phách < 0.7 x trung vị: coi là đang bắt gấp đôi tốc độ
DOUBLE_MIN_RUN = 6       # số khoảng liên tiếp tối thiểu của một đoạn gấp đôi
SWITCH_PENALTY = 6.0     # phạt (log) mỗi ô nhịp khác `meter` phách; chọn bằng eval/beats_meter.py
EMISSION_WEIGHT = 1.0
MAX_EXTRA = 2            # ô nhịp dài nhất = meter + MAX_EXTRA phách


def beat_activation(beats: np.ndarray, frames: np.ndarray, fps: int = FPS, radius: int = 2) -> np.ndarray:
    """Xác suất phách mạnh tại mỗi phách: lớn nhất trong ±`radius` khung quanh phách."""
    frames = np.asarray(frames, dtype=np.float64)
    if len(frames) == 0:
        return np.zeros(len(beats))
    idx = np.clip(np.round(np.asarray(beats) * fps).astype(int), 0, len(frames) - 1)
    out = np.empty(len(idx))
    for k, i in enumerate(idx):
        out[k] = frames[max(0, i - radius): i + radius + 1].max()
    return out


def fix_double_time(beats: np.ndarray, ratio: float = DOUBLE_RATIO, min_run: int = DOUBLE_MIN_RUN) -> np.ndarray:
    """Bỏ cách một phách ở các đoạn model bắt gấp đôi tốc độ: ít nhất `min_run` khoảng liên tiếp ngắn hơn
    `ratio` x trung vị khoảng cách phách của cả bài. Giữ dãy (chẵn hay lẻ) mà khoảng cách sau khi bỏ, kể cả hai khoảng
    nối với phần ngoài đoạn, gần trung vị nhất."""
    beats = np.sort(np.asarray(beats, dtype=np.float64))
    if len(beats) < min_run + 2:
        return beats
    ibi = np.diff(beats)
    g = float(np.median(ibi))
    short = ibi < ratio * g
    keep = np.ones(len(beats), dtype=bool)
    i = 0
    while i < len(short):
        if not short[i]:
            i += 1
            continue
        j = i
        while j < len(short) and short[j]:
            j += 1
        # các khoảng i..j-1 ngắn: phách beats[i..j] thuộc đoạn gấp đôi
        if j - i >= min_run:
            inner = np.arange(i, j + 1)
            best, best_cost = 0, np.inf
            for par in (0, 1):
                pts = beats[inner[(inner - i) % 2 == par]]
                if i > 0:
                    pts = np.concatenate([[beats[i - 1]], pts])
                if j + 1 < len(beats):
                    pts = np.concatenate([pts, [beats[j + 1]]])
                cost = float(np.abs(np.diff(pts) - g).mean()) if len(pts) > 1 else np.inf
                if cost < best_cost:
                    best, best_cost = par, cost
            keep[inner[(inner - i) % 2 != best]] = False
        i = j
    return drop_extra_beats(beats[keep], g, ratio)


def drop_extra_beats(beats: np.ndarray, g: float, ratio: float = DOUBLE_RATIO, tol: float = 0.2) -> np.ndarray:
    """Phách thừa lẻ tẻ (đoạn gấp đôi quá ngắn cho `fix_double_time`): khoảng ngắn hơn `ratio` x `g` mà cộng với
    khoảng kề bên được xấp xỉ `g` (lệch dưới `tol`) thì bỏ phách ở giữa hai khoảng đó. Chọn khoảng kề (trước hoặc
    sau) cho tổng gần `g` hơn. Lặp tới khi không còn gì để bỏ."""
    b = list(np.asarray(beats, dtype=np.float64))
    i = 0
    while i < len(b) - 1:
        d = b[i + 1] - b[i]
        if d >= ratio * g:
            i += 1
            continue
        prev = b[i + 1] - b[i - 1] if i > 0 else np.inf          # bỏ b[i]
        nxt = b[i + 2] - b[i] if i + 2 < len(b) else np.inf      # bỏ b[i+1]
        ep, en = abs(prev - g), abs(nxt - g)
        if min(ep, en) <= tol * g:
            del b[i if ep < en else i + 1]
            i = max(0, i - 1)
        else:
            i += 1
    return np.asarray(b)


def lock_meter(act: np.ndarray, meter: int, switch_penalty: float = SWITCH_PENALTY,
               weight: float = EMISSION_WEIGHT, max_extra: int = MAX_EXTRA) -> np.ndarray:
    """Chỉ số các phách là phách mạnh, chọn bằng Viterbi. Trạng thái k = vị trí trong ô nhịp (0 = phách 1),
    k = 0..meter+max_extra-1. Đi tiếp k -> k+1 miễn phí khi k+1 < meter; về 0 sau đúng `meter` phách miễn phí;
    về 0 sớm hơn hoặc đi quá `meter` (ô nhịp dài) phạt `switch_penalty`. Điểm mỗi phách: weight x log(act) nếu
    là phách 1, weight x log(1 - act) nếu không. Phách đầu bài ở vị trí nào cũng được (nhịp lấy đà)."""
    act = np.clip(np.asarray(act, dtype=np.float64), 1e-4, 1 - 1e-4)
    n = len(act)
    if n == 0 or meter < 2:
        return np.zeros(0, dtype=int)
    K = meter + max_extra
    e_down, e_up = weight * np.log(act), weight * np.log1p(-act)
    score = np.full(K, e_up[0])
    score[0] = e_down[0]
    back = np.zeros((n, K), dtype=np.int16)
    for i in range(1, n):
        new = np.empty(K)
        arg = np.empty(K, dtype=np.int16)
        cand = score - switch_penalty           # vào phách 1 sớm/muộn hơn `meter` phách: phạt
        cand[meter - 1] = score[meter - 1]      # đúng `meter` phách: miễn phí
        arg[0] = int(np.argmax(cand))
        new[0] = cand[arg[0]] + e_down[i]
        for k in range(1, K):                   # đi tiếp trong ô nhịp; vượt quá `meter` phách thì phạt một lần
            new[k] = score[k - 1] - (switch_penalty if k == meter else 0.0) + e_up[i]
            arg[k] = k - 1
        score, back[i] = new, arg
    k = int(np.argmax(score))
    path = np.empty(n, dtype=int)
    for i in range(n - 1, -1, -1):
        path[i] = k
        if i:
            k = int(back[i, k])
    return np.flatnonzero(path == 0)


def meter_from_downbeats(beats: np.ndarray, downbeats: np.ndarray, default: int = 4) -> int:
    """Số phách mỗi ô nhịp phổ biến nhất (như `pipeline.estimate_meter`); hòa thì lấy số nhỏ hơn."""
    if len(downbeats) < 2 or len(beats) == 0:
        return default
    idx = np.searchsorted(np.sort(beats), np.sort(downbeats) - 1e-3)
    counts = np.diff(idx)
    counts = counts[counts > 1]
    if counts.size == 0:
        return default
    values, freq = np.unique(counts, return_counts=True)
    return int(values[np.argmax(freq)])


def postprocess(beats: np.ndarray, downbeat_frames: np.ndarray, meter: int,
                switch_penalty: float = SWITCH_PENALTY, fix_tempo: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Sửa đoạn gấp đôi tốc độ rồi đặt lại phách mạnh theo `meter` (số phách mỗi ô nhịp phổ biến nhất của model).
    `downbeat_frames`: xác suất phách mạnh từng khung (FPS). Trả về (beats, downbeats)."""
    beats = np.sort(np.asarray(beats, dtype=np.float64))
    if fix_tempo:
        beats = fix_double_time(beats)
    down = lock_meter(beat_activation(beats, downbeat_frames), meter, switch_penalty)
    return beats, beats[down]


def track_beats(audio_path: Path, max_duration: float | None = None) -> BeatGrid:
    """Bắt phách và phách mạnh của cả file (hoặc `max_duration` giây đầu), rồi `postprocess`."""
    import torch

    signal, sr = _load_audio(Path(audio_path), max_duration)
    with _MODEL_LOCK:
        model = _load_model()
        with gpu_lock():
            try:
                _move(model, torch.device(device()))
                beat_logits, down_logits = model.spect2frames(model.signal2spect(signal, sr))
                beats, downbeats = model.frames2beats(beat_logits, down_logits)
                down_frames = torch.sigmoid(down_logits).float().cpu().numpy()
            finally:
                _move(model, torch.device("cpu"))

    beats = np.sort(np.asarray(beats, dtype=np.float64))
    raw_down = _snap_downbeats(beats, np.asarray(downbeats, dtype=np.float64))
    if len(beats) >= 8 and len(raw_down) >= 2:
        beats, downbeats = postprocess(beats, down_frames, meter_from_downbeats(beats, raw_down))
    else:
        downbeats = raw_down
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
