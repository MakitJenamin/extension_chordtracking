"""Wrapper cho Chord-CNN-LSTM (ensemble 5 model), chạy trực tiếp trong Python.

Tái hiện `chord_recognition.py`:
1. Audio mono 22050 Hz, `librosa.hybrid_cqt` (hop 512, 288 bin, 36 bin/quãng tám, từ F#0).
2. Chạy 5 model, lấy trung bình từng nhóm xác suất: triad (T, 73), bass (T, 13),
   7th (T, 4), 9th (T, 4), 11th (T, 3), 13th (T, 3).
3. Giải mã HMM (`XHMMDecoder`, không dùng beat) trên từ điển hợp âm `chord_dict`.

`frame_probs`: từ điển của HMM (`vocabulary`) gồm "N" và mọi hợp âm trong file
`data/<chord_dict>_chord_list.txt` dịch sang 12 nốt gốc. Điểm quan sát của HMM cho mỗi hợp
âm là tích xác suất các thành phần (triad x bass x 7th x ...), nên `frame_probs` là tích đó
chuẩn hóa lại theo từng khung (softmax của log-xác suất). Bản gốc nhân xác suất "N" của
triad với 0.5 trước khi giải mã; `frame_probs` không có hệ số này để giữ đúng niềm tin của
model, còn phần giải mã thì giữ nguyên như gốc.
"""
from __future__ import annotations

import contextlib
import functools
import io
import os
import sys
from pathlib import Path

import numpy as np

from chordtrack.gpu import device as _device_name
from chordtrack.gpu import gpu_lock
from chordtrack.paths import CNNLSTM_DIR
from chordtrack.types import ChordModelOutput, ChordSegment

SR = 22050
HOP_LENGTH = 512
CHORD_DICTS = ("submission", "ismir2017", "full", "extended")
MODEL_FILES = [
    f"joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s{i}.best" for i in range(5)
]
# Bản gốc giảm xác suất "N" của triad đi một nửa trước khi giải mã HMM.
N_BIAS = 0.5


@functools.lru_cache(maxsize=1)
def _import_modules():
    """Import mã gốc một lần cho cả tiến trình.

    `train_eval_test_split` (import gián tiếp) đọc file theo đường dẫn tương đối ngay lúc
    import, nên phải chdir vào repo trong lúc import (chdir có tác dụng cho cả tiến trình,
    nhưng chỉ xảy ra một lần ở đây). Sau đó gỡ repo khỏi `sys.path` để các tên chung chung
    trong đó (`datasets`, `settings`, ...) không che gói thật về sau. Module đã import vẫn
    nằm trong `sys.modules`.
    """
    root = str(CNNLSTM_DIR)
    old_cwd = os.getcwd()
    sys.path.insert(0, root)
    os.chdir(root)
    # import đó còn gọi np.random.seed; giữ nguyên trạng thái RNG của người gọi.
    state = np.random.get_state()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            import mir.nn.train as mir_train
            from chordnet_ismir_naive import ChordNet
            from extractors.xhmm_ismir import XHMMDecoder
    finally:
        np.random.set_state(state)
        os.chdir(old_cwd)
        sys.path.remove(root)
    # NetworkInterface tìm weights theo WORKING_PATH (= cwd lúc import mir); đặt tuyệt đối.
    mir_train.WORKING_PATH = root
    return mir_train.NetworkInterface, ChordNet, XHMMDecoder


def _compute_cqt(audio_path: str | Path) -> np.ndarray:
    import librosa

    y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    # Giống extractors.cqt.CQTV2
    cqt = librosa.hybrid_cqt(
        y, bins_per_octave=36, fmin=librosa.note_to_hz("F#0"), n_bins=288, tuning=None, hop_length=HOP_LENGTH
    ).T
    return np.abs(cqt).astype(np.float32)


def _run_ensemble(cqt: np.ndarray, NetworkInterface, ChordNet) -> list[np.ndarray]:
    use_gpu = _device_name() == "cuda"
    all_probs = []
    with gpu_lock():
        for name in MODEL_FILES:
            weight_path = CNNLSTM_DIR / "cache_data" / f"{name}.sdict"
            if not weight_path.exists():
                # NetworkInterface lặng lẽ dùng weights ngẫu nhiên nếu không thấy file.
                raise FileNotFoundError(weight_path)
            net = ChordNet(None)
            net.use_gpu = use_gpu
            with contextlib.redirect_stdout(io.StringIO()):
                iface = NetworkInterface(net, name, load_checkpoint=False)
            if not iface.finalized:
                raise RuntimeError(f"Không nạp được weights {weight_path}")
            all_probs.append(iface.inference(cqt))
            del iface, net
    return [np.mean([p[i] for p in all_probs], axis=0) for i in range(len(all_probs[0]))]


def _root_probs(triad: np.ndarray) -> np.ndarray:
    """(T, 73) -> (T, 13): cột 0 là N, cột 1 + r là nốt gốc r (C=0), cộng mọi loại triad."""
    n_types = (triad.shape[1] - 1) // 12
    roots = triad[:, 1:].reshape(triad.shape[0], n_types, 12).sum(axis=1)
    return np.concatenate([triad[:, :1], roots], axis=1)


def _segments(tags: list[str], frame_duration: float, probs: np.ndarray, vocab: list[str]) -> list[ChordSegment]:
    """Giống `XHMMDecoder.decode_to_chordlab`, thêm confidence."""
    n = len(tags)
    if n == 0:
        return []
    col = {name: i for i, name in enumerate(vocab)}
    if all(t == "N" for t in tags):
        return [ChordSegment(0.0, n * frame_duration, "N", float(probs[:, col["N"]].mean()))]
    out: list[ChordSegment] = []
    start = 0
    for i in range(n):
        if i + 1 == n or tags[i + 1] != tags[i]:
            conf = float(probs[start:i + 1, col[tags[i]]].mean())
            out.append(ChordSegment(start * frame_duration, (i + 1) * frame_duration, tags[i], conf))
            start = i + 1
    return out


def run_cnnlstm(audio_path: str | Path, chord_dict: str = "submission") -> ChordModelOutput:
    if chord_dict not in CHORD_DICTS:
        raise ValueError(f"chord_dict phải là một trong {CHORD_DICTS}, nhận được {chord_dict!r}")
    template = CNNLSTM_DIR / "data" / f"{chord_dict}_chord_list.txt"
    NetworkInterface, ChordNet, XHMMDecoder = _import_modules()

    cqt = _compute_cqt(audio_path)
    probs = _run_ensemble(cqt, NetworkInterface, ChordNet)
    triad, bass, p7, p9, p11, p13 = probs
    n_frames = triad.shape[0]
    frame_duration = HOP_LENGTH / SR

    with contextlib.redirect_stdout(io.StringIO()):
        hmm = XHMMDecoder(template_file=str(template))
        # Giải mã đúng như gốc: decode_to_chordlab(..., use_layer_decode=False), không dùng beat.
        beat_arr = np.ones((n_frames,), dtype=np.int8)
        tags = hmm.decode(probs, beat_arr)
        # Quan sát không có hệ số giảm "N": nhân trước 1/N_BIAS để triệt tiêu hệ số bên trong.
        unbiased = list(probs)
        unbiased[0] = triad.copy()
        unbiased[0][:, 0] /= N_BIAS
        vocab, logprob = hmm.get_chord_tag_obs(unbiased)

    logprob = logprob - logprob.max(axis=1, keepdims=True)
    frame_probs = np.exp(logprob.astype(np.float64))
    frame_probs /= frame_probs.sum(axis=1, keepdims=True)
    frame_times = (np.arange(n_frames) + 0.5) * frame_duration

    return ChordModelOutput(
        model_name=f"cnnlstm_{chord_dict}",
        segments=_segments(tags, frame_duration, frame_probs, list(vocab)),
        frame_times=frame_times,
        frame_probs=frame_probs,
        vocabulary=list(vocab),
        extras={
            "bass_probs": bass,             # (T, 13): cột 0 = không có bass (N), cột 1 + pc = nốt bass pc (C=0)
            "root_probs": _root_probs(triad),  # (T, 13): cột 0 = N, cột 1 + pc = nốt gốc pc
            "triad_probs": triad,           # (T, 73): 0 = N, 1 + 12*(loại-1) + gốc; loại: maj, min, sus4, sus2, dim, aug
            "seventh_probs": p7,            # (T, 4): không, 7, b7, bb7
            "ninth_probs": p9,              # (T, 4): không, 9, #9, b9
            "eleventh_probs": p11,          # (T, 3): không, 11, #11
            "thirteenth_probs": p13,        # (T, 3): không, 13, b13
        },
    )
