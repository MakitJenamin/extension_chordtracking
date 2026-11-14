"""Tách giọng bằng BS-Roformer (dò giai điệu bản tách giọng, và pipeline hợp âm khi bật `use_separation`).

    mix.wav ──BS-Roformer-Viperx-1297──► vocals.wav, instrumental.wav

Mỗi bước bỏ qua nếu đầu ra đã có (cache). Model chạy trong `gpu_lock()`; tải weights và nạp model
làm trước khi lấy khóa. Thời gian và VRAM đỉnh ghi vào `separation_stats.json`.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import warnings
from pathlib import Path

from chordtrack.gpu import gpu_lock
from chordtrack.paths import MODELS_DIR
from chordtrack.types import Stems

log = logging.getLogger(__name__)

SAMPLE_RATE = 44100
ROFORMER_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"  # BS-Roformer-Viperx-1297
ROFORMER_FRIENDLY_NAME = "Roformer Model: BS-Roformer-Viperx-1297"
STATS_FILE = "separation_stats.json"
# Stem HTDemucs của bản cũ (đã bỏ): xóa khi gặp để cache không giữ mãi.
LEGACY_STEMS = ("bass.wav", "no_bass.wav")


def separate(audio_path: Path, out_dir: Path) -> Stems:
    audio_path = Path(audio_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stems = Stems(
        mix=out_dir / "mix.wav",
        vocals=out_dir / "vocals.wav",
        instrumental=out_dir / "instrumental.wav",
    )
    vocal_step = [stems.vocals, stems.instrumental]
    _remove([out_dir / n for n in LEGACY_STEMS])
    stats = _load_stats(out_dir)

    source = _source_id(audio_path)
    if source is not None and stats.get("source") not in (None, source):
        log.warning("%s đang chứa kết quả của file khác (%s), tách lại", out_dir, stats["source"])
        _remove([stems.mix, *vocal_step])
        stats = {}

    if not stems.mix.exists():
        _remove(vocal_step)
        t0 = time.perf_counter()
        normalize_audio(audio_path, stems.mix)
        stats = {"source": source, "normalize": {"seconds": round(time.perf_counter() - t0, 2)}}

    if not all(p.exists() for p in vocal_step):
        stats["roformer"] = _run_roformer(stems.mix, stems.vocals, stems.instrumental)

    (out_dir / STATS_FILE).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log.info("separation stats: %s", stats)
    return stems


def separate_vocals(out_dir: Path) -> dict:
    """Chỉ chạy bước BS-Roformer trên `out_dir/mix.wav` (đã chuẩn hóa), ghi `vocals.wav` và `instrumental.wav` như
    `separate()` để lần phân tích hợp âm sau dùng lại. Dùng cho dò giai điệu bản tách giọng. Thống kê thêm mục
    `roformer` vào `separation_stats.json` (giữ các mục khác, kể cả `source`). Trả thống kê của bước Roformer."""
    out_dir = Path(out_dir)
    mix = out_dir / "mix.wav"
    _remove([out_dir / n for n in LEGACY_STEMS])
    stats = _run_roformer(mix, out_dir / "vocals.wav", out_dir / "instrumental.wav")
    all_stats = _load_stats(out_dir)
    all_stats["roformer"] = stats
    (out_dir / STATS_FILE).write_text(json.dumps(all_stats, indent=2), encoding="utf-8")
    return stats


def normalize_audio(src: Path, dst: Path) -> None:
    """Đưa audio về WAV 44.1 kHz stereo 16-bit bằng ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("Không tìm thấy ffmpeg trong PATH")
    with _temp_path(dst) as tmp:
        cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
               "-vn", "-ac", "2", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", "-f", "wav", str(tmp)]
        subprocess.run(cmd, check=True)
        tmp.replace(dst)


def _run_roformer(mix: Path, vocals: Path, instrumental: Path) -> dict:
    from audio_separator.separator import Separator

    model_dir = MODELS_DIR / "audio-separator"
    model_dir.mkdir(parents=True, exist_ok=True)
    # Ghi ra thư mục tạm rồi mới chuyển sang, để lần chạy hỏng giữa chừng không để lại cache dở.
    # Separator gọi warnings.filterwarnings("ignore") cho cả tiến trình, nên bọc catch_warnings.
    with tempfile.TemporaryDirectory(dir=mix.parent, prefix="roformer_", ignore_cleanup_errors=True) as tmp_str, \
            warnings.catch_warnings():
        tmp = Path(tmp_str)
        sep = Separator(
            log_level=logging.WARNING,
            model_file_dir=str(model_dir),
            output_dir=tmp_str,
            output_format="WAV",
            sample_rate=SAMPLE_RATE,
            use_soundfile=True,
            # 1.0: chỉ hạ mức khi bị clip, giữ đúng tương quan âm lượng với mix
            normalization_threshold=1.0,
            # Roformer chạy từng đoạn (dim_t của model, khoảng 8 s), batch không dùng tới.
            mdxc_params={"segment_size": 256, "override_model_segment_size": False,
                         "batch_size": 1, "overlap": 8, "pitch_shift": 0},
        )
        use_local_model_files(sep)
        sep.download_model_files(ROFORMER_MODEL)  # tải weights (nếu chưa có) trước khi lấy khóa GPU

        def run():
            try:
                sep.load_model(ROFORMER_MODEL)
                # Bản 0.30.2 ghi theo đường dẫn tương đối với cwd, nên truyền đường dẫn tuyệt đối.
                sep.separate(str(mix), custom_output_names={"Vocals": str(tmp / "vocals"),
                                                            "Instrumental": str(tmp / "instrumental")})
            finally:
                sep.model_instance = None

        stats = _gpu_step(run)
        del sep
        for f in (vocals, instrumental):
            if not (tmp / f.name).exists():
                raise RuntimeError(f"audio-separator không tạo {f.name}; có: {sorted(p.name for p in tmp.iterdir())}")
        shutil.move(str(tmp / vocals.name), vocals)
        shutil.move(str(tmp / instrumental.name), instrumental)
    return {**stats, "model": ROFORMER_MODEL}


def use_local_model_files(sep) -> None:
    """Cho `Separator` nạp BS-Roformer thẳng từ file ckpt và yaml trong `model_file_dir`.

    Bản gốc `download_model_files` luôn đọc danh sách model (`download_checks.json`, tải từ
    GitHub nếu chưa có) để biết loại model, rồi mới kiểm tra file. Khi cả ckpt và yaml đã có
    thì hàm thay thế trả luôn thông tin model mà không đọc danh sách hay gọi mạng. Thiếu file
    thì quay về hàm gốc để tải.
    """
    original = sep.download_model_files

    def download_model_files(model_filename):
        if model_filename == ROFORMER_MODEL:
            ckpt = Path(sep.model_file_dir) / ROFORMER_MODEL
            yaml_name = Path(ROFORMER_MODEL).with_suffix(".yaml").name
            if ckpt.is_file() and (ckpt.parent / yaml_name).is_file():
                sep.model_is_uvr_vip = False
                sep.model_friendly_name = ROFORMER_FRIENDLY_NAME
                return ROFORMER_MODEL, "MDXC", ROFORMER_FRIENDLY_NAME, str(ckpt), yaml_name
        return original(model_filename)

    sep.download_model_files = download_model_files


def _gpu_step(fn) -> dict:
    """Chạy fn trong gpu_lock(), đo thời gian chạy, thời gian chờ khóa và VRAM đỉnh."""
    t_wait = time.perf_counter()
    with gpu_lock():
        t0 = time.perf_counter()
        _reset_peak()
        fn()
        peak = _peak_mb()
        t1 = time.perf_counter()
    return {"seconds": round(t1 - t0, 2), "lock_wait_seconds": round(t0 - t_wait, 2), "peak_vram_mb": peak}


@contextlib.contextmanager
def _temp_path(dst: Path):
    """Đường dẫn tạm duy nhất cạnh dst; tự xóa nếu chưa được đổi tên thành dst."""
    fd, name = tempfile.mkstemp(dir=dst.parent, prefix=dst.stem + ".", suffix=".part.wav")
    os.close(fd)
    tmp = Path(name)
    try:
        yield tmp
    finally:
        tmp.unlink(missing_ok=True)


def _remove(paths: list[Path]) -> None:
    for p in paths:
        p.unlink(missing_ok=True)


def _source_id(audio_path: Path) -> dict | None:
    try:
        return {"path": str(audio_path.resolve()), "size": audio_path.stat().st_size}
    except OSError:
        return None


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
