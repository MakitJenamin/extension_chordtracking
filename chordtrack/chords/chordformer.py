"""Wrapper cho ChordFormer (Akram, Dettori, Colla, Buttazzo; IEEE TASLP 2026), ensemble 5 model.

Repo gốc: `CHORDFORMER_DIR` (github.com/mwaseemrandhawa/ChordFormer). Repo không có license và tác
giả đã xóa thư mục checkpoint ở commit 77bec51; 5 file `cache_data/*.sdict` lấy lại từ commit
0d1071a, chỉ để trong máy, dùng cá nhân (người dùng quyết định 2026-11-04). Không commit weights.

ChordFormer phát triển từ mã của Chord-CNN-LSTM: cùng đầu vào, cùng 6 nhóm đầu ra (triad 73, bass 13,
7th 4, 9th 4, 11th 3, 13th 3), cùng từ điển hợp âm và bộ giải mã `XHMMDecoder`. Khác ở chỗ:
- CQT (`extractors/cqt.py` của ChordFormer) đổi sang dB với `ref=max` của cả bài;
- mạng là 4 lớp Conformer (256 chiều, 16 head, ffn 1024, kernel 31, GroupNorm, convolution trước),
  rồi một lớp tuyến tính ra 100 giá trị. Lớp `ConformerLayer` trong `confor_head16.py` là bản chép
  của torchaudio, nên ở đây dùng thẳng `torchaudio.models.conformer.ConformerLayer` và không import
  `confor_head16.py` (file đó gọi `set_specific_gpu(5)` ngay khi import, lỗi trên máy một GPU);
- HMM gốc của ChordFormer dùng `diff_trans_penalty=30`, `beat_trans_penalty=(15, 45, 100)`.

Đầu ra cùng dạng `run_cnnlstm` (frame_probs trên từ vựng HMM 301 nhãn, extras bass_probs...), nên
`fusion` dùng được ở chỗ của CNN-LSTM.
"""
from __future__ import annotations

import contextlib
import functools
import io
import threading
from pathlib import Path

import numpy as np

from chordtrack.chords.cnnlstm import CHORD_DICTS, N_BIAS, _import_modules, _root_probs, _segments
from chordtrack.gpu import device as _device_name
from chordtrack.gpu import gpu_lock
from chordtrack.paths import CHORDFORMER_DIR, CNNLSTM_DIR
from chordtrack.types import ChordModelOutput

SR = 22050
HOP_LENGTH = 512
MODEL_FILES = [f"chordformer_head16(1.0,1.0)_s{i}.best.sdict" for i in range(5)]
SPEC_START, SPEC_DIM = 6 * 3, 252          # SHIFT_HIGH * SHIFT_STEP, như `ChordNet.inference`
HEADS = (73, 13, 4, 4, 3, 3)               # triad, bass, 7th, 9th, 11th, 13th
HMM_KWARGS = {"diff_trans_penalty": 30.0, "beat_trans_penalty": (15.0, 45.0, 100.0)}


def _build_net():
    import torch.nn as nn
    from torchaudio.models.conformer import ConformerLayer

    class ChordConformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.conformer_layers = nn.ModuleList([
                ConformerLayer(256, 1024, 16, 31, dropout=0.1, use_group_norm=True, convolution_first=True)
                for _ in range(4)
            ])
            self.output_layer = nn.Linear(256, sum(HEADS))

        def forward(self, x):  # (B, T, 256)
            x = x.transpose(0, 1)
            for layer in self.conformer_layers:
                x = layer(x, None)
            return self.output_layer(x.transpose(0, 1))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.chordfor = ChordConformer()
            self.final_fc1 = nn.Linear(192, sum(HEADS))   # có trong checkpoint, không dùng khi chạy

        def forward(self, x):  # (B, T, 252)
            import torch.nn.functional as F

            x = F.pad(x, (0, 4))
            return self.chordfor(x)

    return Net()


_LOCK = threading.Lock()


@functools.lru_cache(maxsize=1)
def _load_ensemble() -> tuple:
    import torch

    nets = []
    for name in MODEL_FILES:
        path = CHORDFORMER_DIR / "cache_data" / name
        if not path.exists():
            raise FileNotFoundError(path)
        state = torch.load(str(path), map_location="cpu", weights_only=False)["net"]
        net = _build_net()
        net.load_state_dict(state, strict=True)
        net.eval()
        nets.append(net)
    return tuple(nets)


def clear_model_cache() -> None:
    _load_ensemble.cache_clear()


def _compute_cqt(audio_path: str | Path) -> np.ndarray:
    import librosa

    y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    cqt = librosa.hybrid_cqt(
        y, bins_per_octave=36, fmin=librosa.note_to_hz("F#0"), n_bins=288, tuning=None, hop_length=HOP_LENGTH
    ).T
    cqt = np.abs(cqt).astype(np.float32)
    return librosa.amplitude_to_db(cqt, ref=np.max).astype(np.float32)


def _run_ensemble(cqt: np.ndarray, chunk: int | None) -> list[np.ndarray]:
    """Trung bình xác suất 6 nhóm của 5 model. `chunk`: số khung mỗi lần chạy (None = cả bài,
    như bản gốc); chia đoạn có chồng lấn 1/4 mỗi bên, chỉ giữ phần giữa."""
    import torch

    dev = _device_name()
    x_all = cqt[:, SPEC_START:SPEC_START + SPEC_DIM]
    n = x_all.shape[0]
    sums = [np.zeros((n, k), dtype=np.float64) for k in HEADS]
    with _LOCK:
        nets = _load_ensemble()
        with gpu_lock():
            try:
                for net in nets:
                    net.to(dev)
                    with torch.no_grad():
                        logits = _forward(net, x_all, chunk, dev)
                    net.cpu()
                    start = 0
                    for i, k in enumerate(HEADS):
                        sums[i] += torch.softmax(torch.from_numpy(logits[:, start:start + k]), -1).numpy()
                        start += k
            finally:
                for net in nets:
                    net.cpu()
    return [s / len(nets) for s in sums]


def _forward(net, x_all: np.ndarray, chunk: int | None, dev: str) -> np.ndarray:
    import torch

    n = x_all.shape[0]
    if chunk is None or n <= chunk:
        x = torch.from_numpy(x_all).unsqueeze(0).to(dev)
        return net(x)[0].float().cpu().numpy()
    margin = chunk // 4
    core = chunk - 2 * margin
    out = np.zeros((n, sum(HEADS)), dtype=np.float32)
    for c0 in range(0, n, core):
        a, b = max(0, c0 - margin), min(n, c0 + core + margin)
        x = torch.from_numpy(x_all[a:b]).unsqueeze(0).to(dev)
        y = net(x)[0].float().cpu().numpy()
        e = min(n, c0 + core)
        out[c0:e] = y[c0 - a:e - a]
    return out


def run_chordformer(audio_path: str | Path, chord_dict: str = "submission", chunk: int | None = None) -> ChordModelOutput:
    if chord_dict not in CHORD_DICTS:
        raise ValueError(f"chord_dict phải là một trong {CHORD_DICTS}, nhận được {chord_dict!r}")
    template = CNNLSTM_DIR / "data" / f"{chord_dict}_chord_list.txt"
    _NetworkInterface, _ChordNet, XHMMDecoder = _import_modules()

    cqt = _compute_cqt(audio_path)
    probs = _run_ensemble(cqt, chunk)
    triad, bass, p7, p9, p11, p13 = probs
    n_frames = triad.shape[0]
    frame_duration = HOP_LENGTH / SR

    with contextlib.redirect_stdout(io.StringIO()):
        hmm = XHMMDecoder(template_file=str(template), **HMM_KWARGS)
        tags = hmm.decode(probs, np.ones((n_frames,), dtype=np.int8))
        unbiased = list(probs)
        unbiased[0] = triad.copy()
        unbiased[0][:, 0] /= N_BIAS
        vocab, logprob = hmm.get_chord_tag_obs(unbiased)

    logprob = logprob - logprob.max(axis=1, keepdims=True)
    frame_probs = np.exp(logprob.astype(np.float64))
    frame_probs /= frame_probs.sum(axis=1, keepdims=True)
    frame_times = (np.arange(n_frames) + 0.5) * frame_duration

    return ChordModelOutput(
        model_name=f"chordformer_{chord_dict}",
        segments=_segments(tags, frame_duration, frame_probs, list(vocab)),
        frame_times=frame_times,
        frame_probs=frame_probs,
        vocabulary=list(vocab),
        extras={
            "bass_probs": bass,
            "root_probs": _root_probs(triad),
            "triad_probs": triad,
            "seventh_probs": p7,
            "ninth_probs": p9,
            "eleventh_probs": p11,
            "thirteenth_probs": p13,
        },
    )
