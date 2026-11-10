"""RMVPE: dò f0 giai điệu chính trên bản mix (Wei và cộng sự, 2023).

Kiến trúc viết lại theo `infer/lib/rmvpe.py` của RVC (MIT). Weights `rmvpe.pt` lấy từ Hugging Face
`lj1995/VoiceConversionWebUI` (license MIT), đặt ở `paths.MODELS_DIR / "rmvpe" / "rmvpe.pt"`.

Đầu vào 16 kHz, mel 128 dải (30–8000 Hz), bước 160 mẫu = khung 10 ms. Model ra salience 360 bin
(mỗi bin 20 cent, từ ~31 Hz). f0 lấy trung bình có trọng số 9 bin quanh đỉnh, như RVC. Khung có đỉnh
salience ≤ ngưỡng (mặc định 0.03 của RVC) thì f0 = 0. `extras["salience_max"]` lưu đỉnh thô để bước sau
tự chọn ngưỡng; `extras["f0_raw_hz"]` là f0 chưa cắt theo ngưỡng,
`extras["infer_sec"]` là thời gian chạy model (không tính chờ khóa GPU).

Mel tính trên CPU; model chạy theo đoạn (`CHUNK_FRAMES` khung, kèm `CONTEXT_FRAMES` khung ngữ cảnh mỗi
bên) để VRAM không tăng theo độ dài bài.
"""
from __future__ import annotations

import contextlib
import time
from pathlib import Path

import numpy as np

from chordtrack import paths
from chordtrack.types import MelodyModelOutput, PitchTrack

MODEL_NAME = "rmvpe"
WEIGHTS = paths.MODELS_DIR / "rmvpe" / "rmvpe.pt"
WEIGHTS_SHA256 = "6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193"
SR = 16000
HOP = 160
FRAME_SEC = HOP / SR
DEFAULT_THRESHOLD = 0.03
CHUNK_FRAMES = 32 * 64      # ~20 s mỗi đoạn
CONTEXT_FRAMES = 128        # ~1.3 s ngữ cảnh mỗi bên

# cent của từng bin (so với 10 Hz), đệm 4 bin mỗi bên cho cửa sổ trung bình
_CENTS = np.pad(20 * np.arange(360) + 1997.3794084376191, (4, 4))

_model_cache: dict[str, object] = {}


def _build_model():
    import torch
    from torch import nn

    class BiGRU(nn.Module):
        def __init__(self, n_in, n_hidden, n_layers):
            super().__init__()
            self.gru = nn.GRU(n_in, n_hidden, num_layers=n_layers, batch_first=True, bidirectional=True)

        def forward(self, x):
            return self.gru(x)[0]

    class ConvBlockRes(nn.Module):
        def __init__(self, c_in, c_out, momentum=0.01):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, 1, 1, bias=False), nn.BatchNorm2d(c_out, momentum=momentum), nn.ReLU(),
                nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False), nn.BatchNorm2d(c_out, momentum=momentum), nn.ReLU(),
            )
            self.shortcut = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else None

        def forward(self, x):
            return self.conv(x) + (self.shortcut(x) if self.shortcut is not None else x)

    class ResEncoderBlock(nn.Module):
        def __init__(self, c_in, c_out, kernel_size, n_blocks=1, momentum=0.01):
            super().__init__()
            self.conv = nn.ModuleList([ConvBlockRes(c_in, c_out, momentum)]
                                      + [ConvBlockRes(c_out, c_out, momentum) for _ in range(n_blocks - 1)])
            self.pool = nn.AvgPool2d(kernel_size=kernel_size) if kernel_size is not None else None

        def forward(self, x):
            for c in self.conv:
                x = c(x)
            return (x, self.pool(x)) if self.pool is not None else x

    class Encoder(nn.Module):
        def __init__(self, c_in, in_size, n_encoders, kernel_size, n_blocks, c_out=16, momentum=0.01):
            super().__init__()
            self.bn = nn.BatchNorm2d(c_in, momentum=momentum)
            self.layers = nn.ModuleList()
            for _ in range(n_encoders):
                self.layers.append(ResEncoderBlock(c_in, c_out, kernel_size, n_blocks, momentum))
                c_in, c_out, in_size = c_out, c_out * 2, in_size // 2
            self.out_channel = c_out

        def forward(self, x):
            skips = []
            x = self.bn(x)
            for layer in self.layers:
                skip, x = layer(x)
                skips.append(skip)
            return x, skips

    class Intermediate(nn.Module):
        def __init__(self, c_in, c_out, n_inters, n_blocks, momentum=0.01):
            super().__init__()
            self.layers = nn.ModuleList([ResEncoderBlock(c_in, c_out, None, n_blocks, momentum)]
                                        + [ResEncoderBlock(c_out, c_out, None, n_blocks, momentum)
                                           for _ in range(n_inters - 1)])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    class ResDecoderBlock(nn.Module):
        def __init__(self, c_in, c_out, stride, n_blocks=1, momentum=0.01):
            super().__init__()
            out_padding = (0, 1) if stride == (1, 2) else (1, 1)
            self.conv1 = nn.Sequential(
                nn.ConvTranspose2d(c_in, c_out, 3, stride, 1, out_padding, bias=False),
                nn.BatchNorm2d(c_out, momentum=momentum), nn.ReLU(),
            )
            self.conv2 = nn.ModuleList([ConvBlockRes(c_out * 2, c_out, momentum)]
                                       + [ConvBlockRes(c_out, c_out, momentum) for _ in range(n_blocks - 1)])

        def forward(self, x, skip):
            x = torch.cat((self.conv1(x), skip), dim=1)
            for c in self.conv2:
                x = c(x)
            return x

    class Decoder(nn.Module):
        def __init__(self, c_in, n_decoders, stride, n_blocks, momentum=0.01):
            super().__init__()
            self.layers = nn.ModuleList()
            for _ in range(n_decoders):
                self.layers.append(ResDecoderBlock(c_in, c_in // 2, stride, n_blocks, momentum))
                c_in //= 2

        def forward(self, x, skips):
            for i, layer in enumerate(self.layers):
                x = layer(x, skips[-1 - i])
            return x

    class DeepUnet(nn.Module):
        def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4, c_in=1, en_out_channels=16):
            super().__init__()
            self.encoder = Encoder(c_in, 128, en_de_layers, kernel_size, n_blocks, en_out_channels)
            self.intermediate = Intermediate(self.encoder.out_channel // 2, self.encoder.out_channel,
                                             inter_layers, n_blocks)
            self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

        def forward(self, x):
            x, skips = self.encoder(x)
            return self.decoder(self.intermediate(x), skips)

    class E2E(nn.Module):
        def __init__(self, n_blocks, n_gru, kernel_size, en_out_channels=16):
            super().__init__()
            self.unet = DeepUnet(kernel_size, n_blocks, en_out_channels=en_out_channels)
            self.cnn = nn.Conv2d(en_out_channels, 3, 3, padding=1)
            self.fc = nn.Sequential(BiGRU(3 * 128, 256, n_gru), nn.Linear(512, 360), nn.Dropout(0.25), nn.Sigmoid())

        def forward(self, mel):  # mel (B, 128, T) -> salience (B, T, 360)
            x = mel.transpose(-1, -2).unsqueeze(1)
            x = self.cnn(self.unet(x)).transpose(1, 2).flatten(-2)
            return self.fc(x)

    return E2E(4, 1, (2, 2))


def _mel_basis():
    import librosa

    return librosa.filters.mel(sr=SR, n_fft=1024, n_mels=128, fmin=30, fmax=8000, htk=True)


def load_model():
    """Model (trên CPU), nhớ lại giữa các lần gọi. Chỉ chuyển lên GPU trong lúc chạy, để ngoài khóa GPU
    không giữ VRAM."""
    if "model" not in _model_cache:
        import torch

        if not WEIGHTS.exists():
            raise FileNotFoundError(
                f"Không thấy {WEIGHTS}. Tải bằng: huggingface_hub.hf_hub_download('lj1995/VoiceConversionWebUI', "
                f"'rmvpe.pt', local_dir=paths.MODELS_DIR / 'rmvpe')")
        model = _build_model()
        model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
        _model_cache["model"] = model.eval()
    return _model_cache["model"]


def unload() -> None:
    _model_cache.clear()


def log_mel(audio, mel_basis):
    """audio (N,) tensor 16 kHz -> log-mel (128, T), T = N // 160 + 1, khung i ở thời điểm i * 10 ms."""
    import torch

    window = torch.hann_window(1024, device=audio.device)
    spec = torch.stft(audio, 1024, HOP, 1024, window=window, center=True, return_complex=True).abs()
    return torch.log(torch.clamp(mel_basis @ spec, min=1e-5))


def _salience(audio: np.ndarray, device: str) -> np.ndarray:
    """audio mono 16 kHz -> salience (T, 360), chạy theo đoạn. Gọi với "cuda" thì phải đang giữ `gpu_lock()`
    (dùng `transcribe` nếu không chắc). Mel tính trên CPU, chỉ từng đoạn mel được đưa lên `device`."""
    import torch
    import torch.nn.functional as F

    model = load_model()
    with torch.inference_mode():
        mel = log_mel(torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)),
                      torch.from_numpy(_mel_basis()).float())
    n = mel.shape[-1]
    out = []
    model.to(device)
    try:
        with torch.inference_mode():
            for s in range(0, n, CHUNK_FRAMES):
                e = min(n, s + CHUNK_FRAMES)
                a, b = max(0, s - CONTEXT_FRAMES), min(n, e + CONTEXT_FRAMES)
                seg = mel[None, :, a:b].to(device)
                pad = -(b - a) % 32
                if pad:
                    seg = F.pad(seg, (0, pad), mode="reflect" if pad < b - a else "constant")
                hid = model(seg)[0, : b - a]
                out.append(hid[s - a: e - a].float().cpu().numpy())
    finally:
        model.to("cpu")
    return np.concatenate(out, axis=0)


def decode(sal: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """salience (T, 360) -> (f0 Hz có cắt ngưỡng, f0 Hz thô, đỉnh salience). f0 = 0 khi đỉnh ≤ ngưỡng."""
    center = np.argmax(sal, axis=1) + 4
    padded = np.pad(sal, ((0, 0), (4, 4)))
    idx = center[:, None] + np.arange(-4, 5)[None, :]
    w = np.take_along_axis(padded, idx, axis=1)
    cents = (w * _CENTS[idx]).sum(1) / np.maximum(w.sum(1), 1e-12)
    f0_raw = (10 * 2 ** (cents / 1200)).astype(np.float32)
    peak = sal.max(axis=1).astype(np.float32)
    f0 = np.where(peak > threshold, f0_raw, 0).astype(np.float32)
    return f0, f0_raw, peak


def load_audio(audio_path: Path) -> np.ndarray:
    import librosa

    y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    return y


def transcribe(audio_path: Path, threshold: float = DEFAULT_THRESHOLD, device: str | None = None) -> MelodyModelOutput:
    """Dò f0 trên bản mix. Giữ `gpu_lock()` trong lúc chạy model (nếu dùng GPU)."""
    from chordtrack import gpu

    y = load_audio(Path(audio_path))
    dev = device or gpu.device()
    load_model()   # đọc weights trước khi lấy khóa
    lock = contextlib.nullcontext() if dev == "cpu" else gpu.gpu_lock()
    with lock:
        t0 = time.perf_counter()   # không tính thời gian chờ khóa và đọc weights
        sal = _salience(y, dev)
        infer_sec = time.perf_counter() - t0
    f0, f0_raw, peak = decode(sal, threshold)
    times = np.arange(len(f0), dtype=np.float64) * FRAME_SEC
    return MelodyModelOutput(
        model_name=MODEL_NAME,
        pitch_track=PitchTrack(times, f0, np.clip(peak, 0, 1)),
        extras={"salience_max": peak, "f0_raw_hz": f0_raw, "threshold": np.float32(threshold),
                "infer_sec": np.float64(infer_sec)},
    )
