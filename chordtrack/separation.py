"""Bước 2 và 3 của pipeline: tách giọng (BS-Roformer) rồi tách bass (HTDemucs).

    mix.wav ──BS-Roformer-Viperx-1297──► vocals.wav, instrumental.wav
    instrumental.wav ──HTDemucs──► bass.wav, no_bass.wav (drums + other [+ guitar, piano])

Mỗi bước bỏ qua nếu đầu ra đã có (cache). Mỗi model chạy trong `gpu_lock()` riêng và giải
phóng VRAM khi xong. Thời gian và VRAM đỉnh của từng bước ghi vào `separation_stats.json`.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from chordtrack.gpu import device, gpu_lock
from chordtrack.paths import MODELS_DIR
from chordtrack.types import Stems

log = logging.getLogger(__name__)

SAMPLE_RATE = 44100
ROFORMER_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"  # BS-Roformer-Viperx-1297
DEMUCS_MODEL = "htdemucs_ft"
DEMUCS_SEGMENT = None  # None = độ dài mặc định của model (7.8 s), vừa 8 GB VRAM
STATS_FILE = "separation_stats.json"


def separate(audio_path: Path, out_dir: Path) -> Stems:
    audio_path = Path(audio_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stems = Stems(
        mix=out_dir / "mix.wav",
        vocals=out_dir / "vocals.wav",
        instrumental=out_dir / "instrumental.wav",
        bass=out_dir / "bass.wav",
        no_bass=out_dir / "no_bass.wav",
    )
    stats = _load_stats(out_dir)

    if not stems.mix.exists():
        t0 = time.perf_counter()
        normalize_audio(audio_path, stems.mix)
        stats["normalize"] = {"seconds": round(time.perf_counter() - t0, 2)}

    if not (stems.vocals.exists() and stems.instrumental.exists()):
        t_wait = time.perf_counter()
        with gpu_lock():
            t0 = time.perf_counter()
            _reset_peak()
            _run_roformer(stems.mix, stems.vocals, stems.instrumental)
            peak = _peak_mb()
            t1 = time.perf_counter()
        stats["roformer"] = {"seconds": round(t1 - t0, 2), "lock_wait_seconds": round(t0 - t_wait, 2),
                             "peak_vram_mb": peak,
                             "model": ROFORMER_MODEL}

    if not (stems.bass.exists() and stems.no_bass.exists()):
        t_wait = time.perf_counter()
        with gpu_lock():
            t0 = time.perf_counter()
            _reset_peak()
            _run_demucs(stems.instrumental, stems.bass, stems.no_bass)
            peak = _peak_mb()
            t1 = time.perf_counter()
        stats["demucs"] = {"seconds": round(t1 - t0, 2), "lock_wait_seconds": round(t0 - t_wait, 2),
                             "peak_vram_mb": peak,
                           "model": DEMUCS_MODEL}

    (out_dir / STATS_FILE).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    for name, s in stats.items():
        log.info("separation %s: %s", name, s)
    return stems


def normalize_audio(src: Path, dst: Path) -> None:
    """Đưa audio về WAV 44.1 kHz stereo 16-bit bằng ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("Không tìm thấy ffmpeg trong PATH")
    tmp = dst.with_name(dst.stem + ".part.wav")
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
           "-vn", "-ac", "2", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(tmp)]
    subprocess.run(cmd, check=True)
    tmp.replace(dst)


def _run_roformer(mix: Path, vocals: Path, instrumental: Path) -> None:
    from audio_separator.separator import Separator

    model_dir = MODELS_DIR / "audio-separator"
    model_dir.mkdir(parents=True, exist_ok=True)
    # Ghi ra thư mục tạm rồi mới chuyển sang, để lần chạy hỏng giữa chừng không để lại cache dở.
    with tempfile.TemporaryDirectory(dir=mix.parent, prefix="roformer_") as tmp:
        sep = Separator(
            log_level=logging.WARNING,
            model_file_dir=str(model_dir),
            output_dir=tmp,
            output_format="WAV",
            sample_rate=SAMPLE_RATE,
            use_soundfile=True,
            # 1.0: chỉ hạ mức khi bị clip, giữ đúng tương quan âm lượng với mix
            normalization_threshold=1.0,
            # Roformer chạy từng đoạn (dim_t của model, khoảng 8 s), batch không dùng tới.
            mdxc_params={"segment_size": 256, "override_model_segment_size": False,
                         "batch_size": 1, "overlap": 8, "pitch_shift": 0},
        )
        try:
            sep.load_model(ROFORMER_MODEL)
            # Bản 0.30.2 ghi theo đường dẫn tương đối với cwd, nên truyền đường dẫn tuyệt đối.
            tmp = Path(tmp)
            sep.separate(str(mix), custom_output_names={"Vocals": str(tmp / "vocals"),
                                                        "Instrumental": str(tmp / "instrumental")})
        finally:
            del sep
        for f in (vocals, instrumental):
            src = tmp / f.name
            if not src.exists():
                raise RuntimeError(f"audio-separator không tạo {f.name}; có: {sorted(p.name for p in tmp.iterdir())}")
        # instrumental chuyển sau cùng: điều kiện cache là cả hai file cùng tồn tại
        shutil.move(str(tmp / vocals.name), vocals)
        shutil.move(str(tmp / instrumental.name), instrumental)


def load_demucs(name: str = DEMUCS_MODEL):
    """Tải HTDemucs từ HuggingFace vào MODELS_DIR (không dùng cache mặc định trong ~)."""
    from demucs.apply import BagOfModels
    from demucs.hf import DEFAULT_NAMESPACE, hf_repo_name, load_safetensors_model
    from huggingface_hub import hf_hub_download
    import yaml

    repo_id = f"{DEFAULT_NAMESPACE}/{hf_repo_name(name)}"
    local = MODELS_DIR / "demucs" / name
    with open(hf_hub_download(repo_id, f"{name}.yaml", local_dir=local)) as f:
        bag = yaml.safe_load(f)
    models = [load_safetensors_model(hf_hub_download(repo_id, f"{sig}.safetensors", local_dir=local))
              for sig in bag["models"]]
    model = BagOfModels(models, bag.get("weights"), bag.get("segment"))
    model.eval()
    return model


def _run_demucs(instrumental: Path, bass: Path, no_bass: Path) -> None:
    import torch
    from demucs.apply import apply_model

    data, sr = sf.read(str(instrumental), dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{instrumental} có sample rate {sr}, cần {SAMPLE_RATE}")
    wav = torch.from_numpy(data.T.copy())
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)

    model = load_demucs()
    dev = device()
    try:
        # Chuẩn hóa như CLI của demucs, rồi trả lại mức ban đầu
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        with torch.no_grad():
            out = apply_model(model, ((wav - mean) / std)[None], device=dev, shifts=1, split=True,
                              overlap=0.25, segment=DEMUCS_SEGMENT, progress=False)[0]
        out = (out * std + mean).cpu().numpy()  # (S, C, T)
        sources = list(model.sources)  # htdemucs_ft: drums, bass, other, vocals
    finally:
        del model
    bass_wav = out[sources.index("bass")]
    # no_bass = mọi nhạc cụ trừ bass. Bỏ stem "vocals" vì đó là phần giọng còn sót sau Roformer.
    keep = [i for i, s in enumerate(sources) if s not in ("bass", "vocals")]
    no_bass_wav = out[keep].sum(0)
    _write_wav(bass, bass_wav)
    _write_wav(no_bass, no_bass_wav)


def _write_wav(path: Path, wav: np.ndarray) -> None:
    tmp = path.with_name(path.stem + ".part.wav")
    sf.write(str(tmp), wav.T, SAMPLE_RATE, subtype="FLOAT")
    tmp.replace(path)


def _load_stats(out_dir: Path) -> dict:
    p = out_dir / STATS_FILE
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _reset_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def _peak_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2**20, 1)
    except ImportError:
        pass
    return None
