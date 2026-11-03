"""Wrapper cho consonance-ACE (Poltronieri, Serra, Rocamora, ISMIR 2025), model `conformer_decomposed`.

Repo gốc: `CONSONANCE_DIR` (github.com/andreamust/consonance-ACE, MIT). Checkpoint
`ACE/checkpoints/conformer_decomposed_smooth.ckpt` là checkpoint của Lightning; ở đây dựng lại
đúng phần `forward` bằng torch thuần (torchaudio Conformer), nên không cần cài lightning, gin,
harte-library...

Tái hiện `ACE/inference.py`:
1. Audio mono 22050 Hz, cắt từng đoạn 20 s (đoạn cuối đệm 0), mỗi đoạn chuẩn hóa về biên độ max 1.
2. `librosa.cqt` biên độ (hop 512, 144 bin, 24 bin/quãng tám, từ C1), không lấy log.
3. Model ra ba đầu: root (13, softmax; 0..11 = C..B, 12 = N), bass (13, như root) và
   onehot (12, sigmoid; nốt nào đang vang, tính theo cao độ tuyệt đối, C = 0).

Model không chấm điểm từng nhãn hợp âm, nên `frame_probs` trên từ vựng 170 nhãn của ChordMini được
dựng từ các đầu ra: log P(nhãn) = log P_root(gốc) + Σ_nốt log(q nếu nốt thuộc hợp âm, 1 − q nếu không),
với q là sigmoid của nốt. "N" lấy log P_root(N) + Σ_nốt log max(q, 1 − q) (không phạt theo nốt nào),
"X" bị loại. Sau đó chuẩn hóa theo khung. `segments` là argmax theo khung, gộp nhãn trùng (không HMM).
"""
from __future__ import annotations

import functools
import math
import threading
from pathlib import Path

import numpy as np

from chordtrack.gpu import device as _device_name
from chordtrack.gpu import gpu_lock
from chordtrack.paths import CONSONANCE_DIR
from chordtrack.types import ChordModelOutput, ChordSegment

SR = 22050
HOP_LENGTH = 512
CHUNK_SECONDS = 20.0
CHECKPOINT = CONSONANCE_DIR / "ACE" / "checkpoints" / "conformer_decomposed_smooth.ckpt"
EPS = 1e-6


def _build_model(hp: dict):
    import torch
    import torch.nn as nn
    from torchaudio.models import Conformer

    class PositionalEncoding(nn.Module):
        def __init__(self, hidden_size: int, max_len: int = 5000):
            super().__init__()
            pe = torch.zeros(max_len, hidden_size)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, : x.size(1)]

    class Decomposed(nn.Module):
        def __init__(self):
            super().__init__()
            d = hp["conformer_dim"]
            n_root, n_bass, n_chord = 13, 13, 12
            self.positional_encoding = hp.get("positional_encoding", True)
            self.positional_encodings = PositionalEncoding(hidden_size=d)
            self.input_projection = nn.Linear(hp["input_dim"], d)
            self.conformer = Conformer(
                input_dim=d, num_heads=hp["num_heads"], ffn_dim=hp["ffn_dim"], num_layers=hp["num_layers"],
                depthwise_conv_kernel_size=hp["depthwise_conv_kernel_size"], dropout=hp["dropout"],
                use_group_norm=hp["use_group_norm"], convolution_first=hp["convolution_first"],
            )
            self.output_chord = nn.Linear(d, n_chord)
            self.bass_fusion = nn.Sequential(nn.Linear(d + n_chord, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(0.1))
            self.output_bass = nn.Linear(d, n_bass)
            self.root_fusion = nn.Sequential(nn.Linear(d + n_chord, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(0.1))
            self.output_root = nn.Linear(d, n_root)

        def forward(self, x):  # x: (B, F, T)
            lengths = torch.full((x.shape[0],), x.shape[2], dtype=torch.long, device=x.device)
            x = self.input_projection(x.transpose(1, 2))
            if self.positional_encoding:
                x = self.positional_encodings(x)
            x, _ = self.conformer(x, lengths=lengths)
            chord = self.output_chord(x)
            h = torch.cat([x, chord], dim=-1)
            return self.output_root(self.root_fusion(h)), self.output_bass(self.bass_fusion(h)), chord

    return Decomposed()


_LOCK = threading.Lock()


@functools.lru_cache(maxsize=1)
def _load_model():
    """Nạp một lần mỗi tiến trình, giữ trên CPU; chỉ lên GPU trong `gpu_lock()`."""
    import torch

    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)
    ck = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=False)
    model = _build_model(ck["hyper_parameters"])
    state = {k: v for k, v in ck["state_dict"].items() if not k.startswith("loss.")}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def clear_model_cache() -> None:
    _load_model.cache_clear()


def _load_audio(audio_path: str | Path) -> np.ndarray:
    """Mono 22050 Hz như `AudioChunkProcessor`: đọc ở tần số gốc, đổi tần số bằng torchaudio, trộn kênh."""
    import librosa
    import torch
    from torchaudio.transforms import Resample

    y, sr = librosa.load(str(audio_path), sr=None, mono=False)
    t = torch.from_numpy(np.atleast_2d(y).astype(np.float32))
    if sr != SR:
        t = Resample(int(sr), SR, dtype=t.dtype)(t)
    return t.mean(dim=0).numpy()


def _chunk_features(y: np.ndarray) -> list[np.ndarray]:
    import librosa

    n = int(CHUNK_SECONDS * SR)
    feats = []
    for start in range(0, max(len(y), 1), n):
        seg = y[start:start + n]
        if len(seg) < n:
            seg = np.pad(seg, (0, n - len(seg)))
        peak = np.abs(seg).max()
        if peak > 0:
            seg = seg / peak
        cqt = np.abs(librosa.cqt(seg, sr=SR, hop_length=HOP_LENGTH, bins_per_octave=24, n_bins=144,
                                 fmin=librosa.note_to_hz("C1")))
        feats.append(cqt.astype(np.float32))
    return feats


def _run(feats: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Trả về root (T, 13), bass (T, 13), onehot (T, 12) xác suất, và số khung mỗi đoạn."""
    import torch

    dev = _device_name()
    roots, basses, chords = [], [], []
    with _LOCK:
        model = _load_model()
        with gpu_lock():
            try:
                model.to(dev)
                with torch.no_grad():
                    for f in feats:
                        x = torch.from_numpy(f).unsqueeze(0).to(dev)
                        r, b, c = model(x)
                        roots.append(torch.softmax(r[0], -1).cpu().numpy())
                        basses.append(torch.softmax(b[0], -1).cpu().numpy())
                        chords.append(torch.sigmoid(c[0]).cpu().numpy())
            finally:
                model.cpu()
    return np.concatenate(roots), np.concatenate(basses), np.concatenate(chords), feats[0].shape[1]


@functools.lru_cache(maxsize=4)
def _templates(vocab: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(gốc theo nhãn, mặt nạ 12 nốt tuyệt đối theo nhãn, là N, là X)."""
    import mir_eval

    roots = np.zeros(len(vocab), dtype=int)
    masks = np.zeros((len(vocab), 12))
    is_n = np.zeros(len(vocab), dtype=bool)
    is_x = np.zeros(len(vocab), dtype=bool)
    for i, lab in enumerate(vocab):
        if lab == "N":
            is_n[i] = True
            continue
        if lab == "X":
            is_x[i] = True
            continue
        root, bitmap, _bass = mir_eval.chord.encode(lab)
        roots[i] = root
        masks[i] = np.roll(bitmap, root)
    return roots, masks, is_n, is_x


def label_logprobs(root: np.ndarray, onehot: np.ndarray, vocab: list[str]) -> np.ndarray:
    """log P(nhãn) (chưa chuẩn hóa) (T, V) từ xác suất gốc (T, 13; 12 = N) và nốt (T, 12)."""
    roots, masks, is_n, is_x = _templates(tuple(vocab))
    q = np.clip(onehot.astype(np.float64), EPS, 1 - EPS)
    lq, l1q = np.log(q), np.log(1 - q)
    notes = lq @ masks.T + l1q @ (1 - masks).T                    # (T, V)
    lroot = np.log(np.clip(root.astype(np.float64), EPS, 1))
    out = lroot[:, roots] + notes
    out[:, is_n] = (lroot[:, 12] + np.maximum(lq, l1q).sum(axis=1))[:, None]
    out[:, is_x] = -np.inf
    return out


def _to_ours(p13: np.ndarray) -> np.ndarray:
    """(T, 13) với 0..11 = C..B, 12 = N  ->  cột 0 = N, cột 1 + pc (như `bass_probs` của CNN-LSTM)."""
    return np.concatenate([p13[:, 12:13], p13[:, :12]], axis=1)


def run_consonance(audio_path: str | Path, vocab: list[str] | None = None) -> ChordModelOutput:
    if vocab is None:
        from chordtrack.chords.chordmini import vocabulary

        vocab = vocabulary()
    y = _load_audio(audio_path)
    duration = len(y) / SR
    feats = _chunk_features(y)
    root, bass, onehot, frames_per_chunk = _run(feats)
    # Mỗi đoạn 20 s có `frames_per_chunk` khung; khung k của đoạn i bắt đầu ở i*20 + k*20/frames.
    step = CHUNK_SECONDS / frames_per_chunk
    n_chunks = len(feats)
    starts = (np.arange(n_chunks)[:, None] * CHUNK_SECONDS + np.arange(frames_per_chunk)[None, :] * step).ravel()
    keep = starts < duration
    root, bass, onehot, starts = root[keep], bass[keep], onehot[keep], starts[keep]

    logp = label_logprobs(root, onehot, vocab)
    logp -= logp.max(axis=1, keepdims=True)
    probs = np.exp(logp)
    probs /= probs.sum(axis=1, keepdims=True)
    frame_times = starts + step / 2

    idx = probs.argmax(axis=1)
    segments: list[ChordSegment] = []
    s = 0
    for i in range(len(idx)):
        if i + 1 == len(idx) or idx[i + 1] != idx[i]:
            end = duration if i + 1 == len(idx) else float(starts[i + 1])
            segments.append(ChordSegment(float(starts[s]), end, vocab[idx[i]], float(probs[s:i + 1, idx[i]].mean())))
            s = i + 1

    return ChordModelOutput(
        model_name="consonance_ace",
        segments=segments,
        frame_times=frame_times,
        frame_probs=probs,
        vocabulary=list(vocab),
        extras={
            "bass_probs": _to_ours(bass),   # (T, 13): cột 0 = không có bass, cột 1 + pc = nốt bass pc
            "root_probs": _to_ours(root),   # (T, 13): cột 0 = N, cột 1 + pc = nốt gốc pc
            "note_probs": onehot,           # (T, 12): xác suất từng nốt đang vang (C = 0)
        },
    )
