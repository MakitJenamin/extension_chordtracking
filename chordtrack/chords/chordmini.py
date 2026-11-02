"""Wrapper cho ChordMini (BTC-PL và ChordNet 2E1D), chạy trực tiếp trong Python.

Tái hiện đúng luồng của `src/evaluation/test.py` với các cờ
`--smooth_logits --use_overlap --use_gaussian --kernel_size 9 --vote_aggregation logit
--min_segment_duration 0.5 --smooth_predictions`, nhưng giữ lại logits theo khung để
trả về xác suất.

Luồng xử lý:
1. CQT log-biên độ (22050 Hz, hop 2048, 144 bin, 24 bin/quãng tám, từ C1).
2. Cửa sổ trượt dài `seq_len` (108 khung), chồng lấn 50%.
3. Mỗi cửa sổ: logits -> làm mượt theo thời gian (Gaussian, kernel 9) -> cộng dồn.
4. Nhãn mỗi khung = argmax tổng logits, rồi lọc đa số (kernel 9).
5. Gộp khung liên tiếp cùng nhãn thành đoạn, bỏ đoạn ngắn hơn 0.5 giây
   (giống script gốc nên giữa các đoạn có thể có khoảng trống).

`frame_probs` = softmax của trung bình logits đã làm mượt (bước 3). Argmax của nó trùng
với nhãn khung trước bước lọc đa số.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np

from chordtrack.gpu import device as _device_name
from chordtrack.gpu import gpu_lock
from chordtrack.paths import CHORDMINI_DIR
from chordtrack.types import ChordModelOutput, ChordSegment

CHECKPOINTS = {
    "btc": ("BTC", "btc_model_best.pth"),
    "2e1d": ("ChordNet", "2e1d_model_best.pth"),
}

# Tham số tương ứng với các cờ CLI đã chốt.
KERNEL_SIZE = 9
USE_GAUSSIAN = True
OVERLAP_RATIO = 0.5          # --use_overlap không kèm --overlap_ratio
MIN_SEGMENT_DURATION = 0.5
BATCH_SIZE = 16
N_CLASSES = 170


# Model đã nạp, giữ trên CPU suốt tiến trình (server không nạp lại mỗi bài). Chỉ chuyển lên
# GPU khi đang giữ `gpu_lock()` rồi trả về CPU, để không chiếm VRAM của agent/tiến trình khác.
_MODEL_CACHE: dict[str, tuple] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _load_model_cached(variant: str):
    """(model trên CPU, mean, std, seq_len của checkpoint hoặc None, config) của `variant`, nạp một lần."""
    with _MODEL_CACHE_LOCK:
        if variant not in _MODEL_CACHE:
            import torch
            from src.models import load_model
            from src.utils import HParams, extract_normalization_stats

            model_type, ckpt_name = CHECKPOINTS[variant]
            ckpt_path = CHORDMINI_DIR / "checkpoints" / ckpt_name
            config = HParams.load(str(CHORDMINI_DIR / "config" / "ChordMini.yaml"))
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            mean, std = extract_normalization_stats(checkpoint)
            ckpt_seq_len = checkpoint.get("timestep") if isinstance(checkpoint, dict) else None
            del checkpoint
            model, _, _ = load_model(str(ckpt_path), model_type, config, torch.device("cpu"))
            model.eval()
            _MODEL_CACHE[variant] = (model, mean, std, ckpt_seq_len, config)
        return _MODEL_CACHE[variant]


def clear_model_cache() -> None:
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()


def _ensure_import_path() -> None:
    root = str(CHORDMINI_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)


def to_harte(label: str) -> str:
    """Nhãn ChordMini -> Harte: hợp âm trưởng ghi tắt "C" thành "C:maj". "N", "X" giữ nguyên."""
    if label in ("N", "X") or ":" in label:
        return label
    return f"{label}:maj"


def vocabulary() -> list[str]:
    """170 nhãn Harte, đúng thứ tự cột logits của ChordMini."""
    _ensure_import_path()
    from src.utils import idx2voca_chord

    idx_to_chord = idx2voca_chord()
    return [to_harte(idx_to_chord[i]) for i in range(N_CLASSES)]


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def _sliding_window_logits(model, features: np.ndarray, mean: float, std: float, seq_len: int) -> np.ndarray:
    """Giống `predict_sliding_windows` (vote_aggregation='logit') nhưng trả về trung bình logits (T, 170).

    Hàm gốc lấy argmax của *tổng* logits; argmax của trung bình là như nhau vì mỗi hàng chỉ
    chia cho một số dương.
    """
    import torch

    num_frames = features.shape[0]
    remainder = num_frames % seq_len
    num_pad = 0 if remainder == 0 else seq_len - remainder
    padded = np.pad(features, ((0, num_pad), (0, 0)), mode="constant") if num_pad else features

    stride = max(1, int(seq_len * (1.0 - OVERLAP_RATIO)))
    padded_frames = padded.shape[0]
    num_instances = max(1, ((padded_frames - seq_len) // stride) + 1)
    starts = [stride * i for i in range(num_instances)]

    acc = np.zeros((num_frames, N_CLASSES), dtype=np.float32)
    counts = np.zeros(num_frames, dtype=np.int32)
    dev = next(model.parameters()).device
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=dev)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=dev)
    base = getattr(model, "module", model)

    with torch.no_grad():
        model.eval()
        for b in range(0, len(starts), BATCH_SIZE):
            windows, meta = [], []
            for s in starts[b:b + BATCH_SIZE]:
                if s + seq_len > padded_frames:
                    continue
                valid = min(seq_len, max(0, num_frames - s))
                if valid <= 0:
                    continue
                windows.append(padded[s:s + seq_len])
                meta.append((s, valid))
            if not windows:
                continue
            x = torch.from_numpy(np.stack(windows)).float().to(dev)
            x = (x - mean_t) / (std_t + 1e-8)
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            # BTC với --smooth_logits và ChordNet đều làm mượt logits trước khi bỏ phiếu.
            logits = base._apply_temporal_smoothing(logits, KERNEL_SIZE, USE_GAUSSIAN)
            arr = logits.detach().cpu().numpy()
            for k, (s, valid) in enumerate(meta):
                acc[s:s + valid] += arr[k, :valid]
                counts[s:s + valid] += 1

    out = np.zeros_like(acc)
    mask = counts > 0
    out[mask] = acc[mask] / counts[mask, None]
    return out


def _segments_from_predictions(
    preds: np.ndarray, frame_duration: float, probs: np.ndarray, vocab: list[str]
) -> list[ChordSegment]:
    """Giống `_prediction_segments` trong test.py, thêm confidence = trung bình xác suất nhãn trong đoạn."""
    segments: list[ChordSegment] = []
    if preds.size == 0:
        return segments
    start = 0
    n = len(preds)
    for i in range(1, n + 1):
        if i == n or preds[i] != preds[start]:
            t0, t1 = start * frame_duration, i * frame_duration
            if t1 - t0 >= MIN_SEGMENT_DURATION:
                idx = int(preds[start])
                conf = float(probs[start:i, idx].mean())
                segments.append(ChordSegment(t0, t1, vocab[idx], conf))
            start = i
    return segments


def run_chordmini(audio_path: str | Path, variant: str = "btc") -> ChordModelOutput:
    if variant not in CHECKPOINTS:
        raise ValueError(f"variant phải là một trong {sorted(CHECKPOINTS)}, nhận được {variant!r}")
    ckpt_path = CHORDMINI_DIR / "checkpoints" / CHECKPOINTS[variant][1]
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    _ensure_import_path()
    import librosa
    import torch
    from src.evaluation.utils.common import extract_song_features, majority_filter_indices
    from src.utils import get_config_value

    model, mean, std, ckpt_seq_len, config = _load_model_cached(variant)
    vocab = vocabulary()

    # Tiền xử lý chạy trên CPU, để ngoài khóa GPU.
    features, frame_duration = extract_song_features(str(audio_path), config)
    features = np.asarray(features, dtype=np.float32)
    try:
        song_duration = float(librosa.get_duration(path=str(audio_path)))
    except Exception:  # giống _song_duration_seconds của script gốc
        y, sr = librosa.load(str(audio_path), sr=get_config_value(config, "mp3", "song_hz", 22050))
        song_duration = len(y) / float(sr)

    # Thứ tự ưu tiên giống `_resolve_seq_len`: checkpoint > thuộc tính model > config > 108.
    seq_len = ckpt_seq_len
    if seq_len is None:
        seq_len = getattr(model, "timestep", None)
    if seq_len is None:
        seq_len = get_config_value(config, "model", "seq_len", get_config_value(config, "model", "timestep", 108))
    try:
        seq_len = int(seq_len)
    except (TypeError, ValueError):
        seq_len = 108
    if seq_len <= 0:
        seq_len = 108

    with gpu_lock():
        try:
            model.to(torch.device(_device_name()))
            mean_logits = _sliding_window_logits(model, features, mean, std, seq_len)
        finally:
            model.to("cpu")

    preds = np.argmax(mean_logits, axis=1).astype(np.int64)
    # Khung không có cửa sổ nào phủ (không xảy ra với stride hiện tại) để nhãn 0 như gốc.
    if preds.size >= KERNEL_SIZE:
        preds = majority_filter_indices(preds, KERNEL_SIZE)

    # Cắt theo độ dài thật của bài, như script gốc.
    max_frames = int(np.floor(song_duration / frame_duration)) if frame_duration > 0 else len(preds)
    if max_frames > 0:
        preds = preds[:max_frames]
        mean_logits = mean_logits[:max_frames]

    probs = _softmax(mean_logits.astype(np.float64))
    frame_times = (np.arange(len(preds)) + 0.5) * frame_duration
    segments = _segments_from_predictions(preds, frame_duration, probs, vocab)

    return ChordModelOutput(
        model_name=f"chordmini_{variant}",
        segments=segments,
        frame_times=frame_times,
        frame_probs=probs,
        vocabulary=vocab,
        extras={"frame_logits": mean_logits},
    )
