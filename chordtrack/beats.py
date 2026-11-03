"""Bước 4: bắt nhịp và phách mạnh bằng beat_this (CPJKU, ISMIR 2024).

Dùng checkpoint mặc định `final0` (huấn luyện trên mọi tập trừ GTZAN), hậu xử lý
"minimal" (lấy đỉnh của xác suất từng khung), không cần madmom.
Weights tải về `data/models/beat_this/` ở lần chạy đầu.
"""
from __future__ import annotations

import functools
import threading
from dataclasses import dataclass
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


FPS = 50   # khung đầu ra của beat_this (giây = khung / FPS)


@dataclass
class BeatTracking:
    """Đầu ra đầy đủ của `track_beats_full`: lưới phách thô của beat_this và xác suất phách mạnh
    theo khung (sigmoid của logit, `FPS` khung mỗi giây) để `refine_downbeats` dùng lại."""

    grid: BeatGrid
    downbeat_frames: np.ndarray   # (T,) xác suất phách mạnh từng khung


def track_beats_full(audio_path: Path, max_duration: float | None = None) -> BeatTracking:
    """Bắt phách và phách mạnh của cả file (hoặc `max_duration` giây đầu), giữ cả xác suất theo khung."""
    import torch

    signal, sr = _load_audio(Path(audio_path), max_duration)
    with _MODEL_LOCK:
        model = _load_model()
        with gpu_lock():
            try:
                _move(model, torch.device(device()))
                spect = model.signal2spect(signal, sr)
                beat_logits, downbeat_logits = model.spect2frames(spect)
                beats, downbeats = model.frames2beats(beat_logits, downbeat_logits)
                db_frames = torch.sigmoid(downbeat_logits).cpu().numpy().astype(np.float64)
            finally:
                _move(model, torch.device("cpu"))

    beats = np.sort(np.asarray(beats, dtype=np.float64))
    downbeats = _snap_downbeats(beats, np.asarray(downbeats, dtype=np.float64))
    return BeatTracking(BeatGrid(beats=beats, downbeats=downbeats, tempo=estimate_tempo(beats)), db_frames)


def track_beats(audio_path: Path, max_duration: float | None = None) -> BeatGrid:
    """Bắt phách và phách mạnh của cả file (hoặc `max_duration` giây đầu), chưa hậu xử lý ô nhịp."""
    return track_beats_full(audio_path, max_duration).grid


# ---------------------------------------------------------------- làm sạch phách

def _local_ibi(beats: np.ndarray, i: int, half_window: int = 8) -> float:
    """Trung vị khoảng phách quanh phách i (± `half_window` khoảng)."""
    ibi = np.diff(beats)
    lo, hi = max(0, i - half_window), min(len(ibi), i + half_window)
    return float(np.median(ibi[lo:hi])) if hi > lo else float(np.median(ibi))


def clean_beats(beats: np.ndarray, short: float = 0.65, tol: float = 0.25) -> np.ndarray:
    """Sửa lỗi beat_this hay gặp ở đoạn có nhiều nốt móc đơn: chèn thêm phách lẻ giữa hai phách.

    Với L là khoảng phách cục bộ (trung vị), phách i tạo khoảng ngắn (< `short` × L) với phách trước:
    - nếu bỏ nó thì phách trước và sau cách nhau khoảng L (sai số `tol`): bỏ phách i (phách thừa);
    - nếu hai phách kề cách nhau khoảng 2L: dời phách i về giữa (phách bị đặt lệch).
    Không thêm phách vào chỗ trống, để không bịa phách ở đoạn nghỉ hoặc đoạn đổi tempo.
    """
    b = np.sort(np.asarray(beats, dtype=np.float64))
    if len(b) < 4:
        return b
    changed = True
    while changed:
        changed = False
        for i in range(1, len(b) - 1):
            L = _local_ibi(b, i)
            if L <= 0:
                continue
            prev_gap, next_gap = b[i] - b[i - 1], b[i + 1] - b[i]
            if min(prev_gap, next_gap) >= short * L:
                continue
            span = b[i + 1] - b[i - 1]
            if abs(span - L) <= tol * L:
                b = np.delete(b, i)
                changed = True
                break
            if abs(span - 2 * L) <= tol * L:
                b[i] = (b[i - 1] + b[i + 1]) / 2
                changed = True
                break
    return b


# ---------------------------------------------------------------- hậu xử lý ô nhịp

@dataclass(frozen=True)
class MeterParams:
    """Tham số `refine_downbeats`. Mặc định chọn bằng `python -m eval.eval_beats --grid` trên Beatles
    (PLAN.md, mục 11, "Phách mạnh và số chỉ nhịp")."""

    meters: tuple[int, ...] = (2, 3, 4)   # 2: nhãn Isophonics ghi 2/4 cho vài bài (Blackbird, Two of Us)
    activation_weight: float = 1.0   # trọng số log xác suất phách mạnh của beat_this
    change_weight: float = 2.0       # điểm cộng khi có chỗ đổi hợp âm ở phách 1
    half_bar_change: float = 0.5     # 4/4: chỗ đổi hợp âm ở phách 3 được tỉ lệ này của điểm phách 1
    reset_penalty: float = 4.0       # phạt (log) khi ô nhịp bị cắt ngắn để đổi pha
    meter_penalty: float = 20.0      # phạt (log) khi đổi số chỉ nhịp giữa bài
    four_prior: float = 0.0          # điểm cộng mỗi ô nhịp 4 phách (ưu tiên 4/4 khi không rõ)


DEFAULT_METER_PARAMS = MeterParams()


def beat_downbeat_probs(beats: np.ndarray, downbeat_frames: np.ndarray, fps: int = FPS,
                        radius: float = 0.07) -> np.ndarray:
    """Xác suất phách mạnh của từng phách: giá trị lớn nhất của `downbeat_frames` trong ± `radius` giây."""
    beats = np.asarray(beats, dtype=np.float64)
    act = np.asarray(downbeat_frames, dtype=np.float64)
    if len(act) == 0:
        return np.full(len(beats), 0.5)
    r = int(round(radius * fps))
    centers = np.clip(np.round(beats * fps).astype(int), 0, len(act) - 1)
    return np.array([act[max(0, c - r):c + r + 1].max() for c in centers])


def chord_change_strength(beats: np.ndarray, change_times, tolerance: float = 0.35) -> np.ndarray:
    """(N,) bằng 1 ở phách có chỗ đổi hợp âm trong ± `tolerance` × khoảng phách, ngược lại 0.
    Mỗi chỗ đổi chỉ gán cho phách gần nhất."""
    beats = np.asarray(beats, dtype=np.float64)
    out = np.zeros(len(beats))
    if len(beats) < 2 or change_times is None:
        return out
    ibi = np.diff(beats)
    local = np.concatenate([[ibi[0]], np.minimum(ibi[:-1], ibi[1:]), [ibi[-1]]])
    for t in np.asarray(change_times, dtype=np.float64):
        i = int(np.argmin(np.abs(beats - t)))
        if abs(beats[i] - t) <= tolerance * local[i]:
            out[i] = 1.0
    return out


def refine_downbeats(
    beats: np.ndarray,
    beat_probs: np.ndarray | None,
    change_times=None,
    params: MeterParams | None = None,
) -> tuple[np.ndarray, int]:
    """Chọn lại phách mạnh: Viterbi trên trạng thái (số chỉ nhịp M, vị trí trong ô 0..M-1).

    Vị trí tăng 1 mỗi phách, hết ô thì về 0 (cùng M; đổi M bị phạt `meter_penalty`). Ô nhịp được
    phép cắt ngắn (về 0 sớm) với phạt `reset_penalty`, nên khi beat_this lệch pha một đoạn thì chuỗi
    tự bắt lại, nhưng không nhảy pha theo từng phách mạnh lẻ. Quan sát ở mỗi phách: log xác suất phách
    mạnh của beat_this (`beat_probs`, None = không có thông tin) và chỗ đổi hợp âm (`change_times`:
    đổi ở phách 1 được cộng `change_weight`, ở phách 3 của 4/4 được `half_bar_change` phần).

    Trả về (phách mạnh, số chỉ nhịp chiếm nhiều phách nhất). Ít hơn 2 phách thì trả nguyên.
    """
    p = params or DEFAULT_METER_PARAMS
    beats = np.asarray(beats, dtype=np.float64)
    n = len(beats)
    if n < 2:
        return beats.copy(), max(p.meters)
    if beat_probs is None:
        act = np.full(n, 0.5)
    else:
        act = np.clip(np.asarray(beat_probs, dtype=np.float64), 1e-4, 1 - 1e-4)
    chg = chord_change_strength(beats, change_times)

    states = [(m, k) for m in p.meters for k in range(m)]
    index = {s: j for j, s in enumerate(states)}
    S = len(states)
    obs = np.zeros((n, S))
    for j, (m, k) in enumerate(states):
        if k == 0:
            obs[:, j] = p.activation_weight * np.log(act) + p.change_weight * chg + (p.four_prior if m == 4 else 0.0)
        else:
            obs[:, j] = p.activation_weight * np.log(1 - act)
            if m == 4 and k == 2:
                obs[:, j] += p.half_bar_change * p.change_weight * chg

    # Ma trận chuyển (log), hàng là trạng thái nguồn, cột là trạng thái đích.
    trans = np.full((S, S), -np.inf)
    for j, (m, k) in enumerate(states):
        full = k + 1 == m
        if not full:
            trans[j, index[(m, k + 1)]] = 0.0
        trans[j, index[(m, 0)]] = 0.0 if full else -p.reset_penalty
        for m2 in p.meters:
            if m2 != m:
                trans[j, index[(m2, 0)]] = -p.meter_penalty - (0.0 if full else p.reset_penalty)

    delta = obs[0].copy()
    back = np.zeros((n, S), dtype=np.int64)
    cols = np.arange(S)
    for i in range(1, n):
        cand = delta[:, None] + trans
        back[i] = np.argmax(cand, axis=0)
        delta = cand[back[i], cols] + obs[i]
    path = np.empty(n, dtype=np.int64)
    path[-1] = int(np.argmax(delta))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]

    pos = np.array([states[j][1] for j in path])
    meter_of = np.array([states[j][0] for j in path])
    values, counts = np.unique(meter_of, return_counts=True)
    return beats[pos == 0], int(values[np.argmax(counts)])


def chord_change_times(segments) -> list[float]:
    """Chỗ đổi hợp âm giữa hai đoạn (ChordSegment) liền nhau, không tính chỗ đổi từ/sang N/X."""
    return [b.start for a, b in zip(segments, segments[1:])
            if a.label not in ("N", "X") and b.label not in ("N", "X") and a.label != b.label]


def postprocess(tracking: BeatTracking, segments=None, params: MeterParams | None = None) -> tuple[BeatGrid, int]:
    """Lưới phách cuối cùng của pipeline: `clean_beats` rồi `refine_downbeats` với xác suất phách mạnh của
    beat_this và chỗ đổi hợp âm của `segments` (None = chỉ dùng beat_this). Trả về (lưới, số chỉ nhịp)."""
    beats = clean_beats(tracking.grid.beats)
    probs = beat_downbeat_probs(beats, tracking.downbeat_frames)
    changes = chord_change_times(segments) if segments else None
    downs, meter = refine_downbeats(beats, probs, changes, params)
    return BeatGrid(beats=beats, downbeats=downs, tempo=estimate_tempo(beats)), meter


def refine_grid(grid: BeatGrid, beat_probs: np.ndarray | None, change_times=None,
                params: MeterParams | None = None) -> tuple[BeatGrid, int]:
    """`refine_downbeats` trên một BeatGrid: trả về (lưới cùng phách với phách mạnh mới, số chỉ nhịp)."""
    downs, meter = refine_downbeats(grid.beats, beat_probs, change_times, params)
    return BeatGrid(beats=np.asarray(grid.beats, dtype=np.float64), downbeats=downs, tempo=grid.tempo), meter


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
