"""YourMT3+ (Chang và cộng sự, 2024, arXiv 2407.04822): chép nốt nhiều nhạc cụ từ bản mix.

Mã nguồn lấy từ Hugging Face Space `mimbres/YourMT3` (Apache-2.0), clone vào `third_party_src/YourMT3`, không sửa.
Checkpoint: `YPTF.MoE+Multi (noPS)` (mặc định của demo trên Space), file
`mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints/last.ckpt`, đặt ở
`data/models/yourmt3/YPTF.MoE+Multi_noPS_last.ckpt`.

Phụ thuộc: YourMT3 cần transformers 4.45.1, bản này đòi huggingface_hub < 1.0, trong khi venv đang có
huggingface_hub 2.x (demucs, muscriptor dùng). Vì vậy transformers 4.45.1, huggingface_hub 0.36.2, tokenizers và regex
được cài riêng vào `third_party_src/YourMT3_pydeps` (`pip install --no-deps --target`) và chỉ được thêm vào đầu
`sys.path` khi nạp model. Hệ quả: trong cùng một tiến trình không nạp được YourMT3 nếu huggingface_hub 2.x đã được
import trước (hàm `_load_model` báo lỗi rõ ràng). Chạy YourMT3 trong tiến trình riêng (`python -m eval.melody.cache_yourmt3`).

Quy ước `MelodyNote.source`:
- `"voice"`: giọng hát chính (program 100 của YourMT3, "Singing Voice").
- `"voice_chorus"`: bè hát / hợp xướng (program 101, "Singing Voice (chorus)").
- `"drums"`: trống (nốt trống có `pitch` là số nốt GM của bộ gõ, không phải cao độ).
- Còn lại: tên lớp nhạc cụ của YourMT3 (bộ `MT3_FULL_PLUS`, 34 lớp) viết thường, nối bằng `_`,
  ví dụ `"acoustic_piano"`, `"string_ensemble"`, `"choir_and_voice"`, `"electric_bass"`.

Checkpoint này không dự đoán velocity (mọi nốt có velocity 1), nên `confidence` để None. `extras` có
`note_program` (program MIDI của YourMT3, -1 với trống) và `note_is_drum`.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
import types
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np

from chordtrack import paths
from chordtrack.types import MelodyModelOutput, MelodyNote

MODEL_NAME = "yourmt3"
CHECKPOINT_LABEL = "YPTF.MoE+Multi (noPS)"
SRC_DIR = paths.THIRD_PARTY / "YourMT3"
PYDEPS_DIR = paths.THIRD_PARTY / "YourMT3_pydeps"
CHECKPOINT = paths.MODELS_DIR / "yourmt3" / "YPTF.MoE+Multi_noPS_last.ckpt"
CHECKPOINT_SHA256 = "ae38e415c79efd5592dcb9b658cdb99ddb11d4c4e1eaa364cab04a052473fc25"

# Tham số dòng lệnh của demo (app.py) cho "YPTF.MoE+Multi (noPS)"; ghi đè -w 0 để không ghi MIDI ra đĩa.
_ARGS = [
    "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt",
    "-p", "2024", "-tk", "mc13_full_plus_256", "-dec", "multi-t5", "-nl", "26", "-enc", "perceiver-tf",
    "-sqr", "1", "-ff", "moe", "-wf", "4", "-nmoe", "8", "-kmoe", "2", "-act", "silu", "-epe", "rope",
    "-rp", "1", "-ac", "spec", "-hop", "300", "-atc", "1", "-pr", "16", "-w", "0",
]

_MODEL = None
_MODEL_DEVICE: str | None = None


def _setup_import_path() -> None:
    hub = sys.modules.get("huggingface_hub")
    if hub is not None and not str(getattr(hub, "__file__", "")).startswith(str(PYDEPS_DIR)):
        raise RuntimeError(
            "huggingface_hub đã được import từ venv (bản 2.x) nên không nạp được transformers 4.45.1 của YourMT3. "
            "Chạy YourMT3 trong tiến trình riêng."
        )
    for name in ("model", "utils", "config"):   # tên package chung chung của amt/src
        mod = sys.modules.get(name)
        if mod is not None and not str(getattr(mod, "__file__", "")).startswith(str(SRC_DIR)):
            raise RuntimeError(f"Module '{name}' đã được import từ nơi khác ({mod.__file__}), trùng tên với YourMT3.")
    for p in (str(PYDEPS_DIR), str(SRC_DIR / "amt" / "src"), str(SRC_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)
    if "wandb" not in sys.modules:
        try:
            import wandb  # noqa: F401
        except ImportError:   # YourMT3 chỉ dùng wandb để ghi log lúc huấn luyện
            stub = types.ModuleType("wandb")
            stub.Table = lambda *a, **k: None
            sys.modules["wandb"] = stub


def _load_model(device: str):
    global _MODEL, _MODEL_DEVICE
    if _MODEL is not None:
        if _MODEL_DEVICE != device:
            _MODEL.to(device)
            _MODEL_DEVICE = device
        return _MODEL
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Thiếu checkpoint YourMT3: {CHECKPOINT}")
    _setup_import_path()
    import torch
    import model_helper  # third_party_src/YourMT3/model_helper.py
    from config.config import shared_cfg as default_shared_cfg

    # initialize_trainer gốc tìm checkpoint trong amt/logs và tạo thư mục log theo cwd; thay bằng bản chỉ trả đường dẫn.
    def initialize_trainer(args: argparse.Namespace, stage: str = "test"):
        if "@" in args.exp_id:
            args.exp_id = args.exp_id.split("@")[0]
        dir_info = dict(lightning_dir=str(CHECKPOINT.parent), last_ckpt_path=str(CHECKPOINT))
        return None, None, dir_info, deepcopy(default_shared_cfg)

    model_helper.initialize_trainer = initialize_trainer
    with contextlib.redirect_stdout(io.StringIO()):   # YourMT3 in toàn bộ cấu hình ra stdout
        m = model_helper.load_model_checkpoint(args=list(_ARGS), device="cpu")
    torch.set_float32_matmul_precision("high")
    m.to(device)
    _MODEL, _MODEL_DEVICE = m, device
    return m


def offload() -> None:
    """Chuyển model về CPU để trả VRAM (gọi trước khi nhả `gpu_lock()`; model vẫn giữ trong RAM)."""
    global _MODEL_DEVICE
    if _MODEL is not None and _MODEL_DEVICE != "cpu":
        _MODEL.to("cpu")
        _MODEL_DEVICE = "cpu"


def unload() -> None:
    """Bỏ hẳn model khỏi bộ nhớ."""
    global _MODEL, _MODEL_DEVICE
    _MODEL, _MODEL_DEVICE = None, None


def _instrument_names() -> dict[int, str]:
    from config.vocabulary import program_vocab_presets

    names: dict[int, str] = {}
    for cls, progs in program_vocab_presets["mt3_full_plus"].items():
        for p in progs:
            names[int(p)] = cls
    return names


def source_name(program: int, is_drum: bool, names: dict[int, str]) -> str:
    if is_drum:
        return "drums"
    if program == 100:
        return "voice"
    if program == 101:
        return "voice_chorus"
    cls = names.get(int(program), f"program_{program}")
    return cls.lower().replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")


def _load_audio(audio_path: Path, sr: int) -> np.ndarray:
    import librosa

    y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    return y.astype(np.float32)


def transcribe(audio_path: Path, *, device: str | None = None, batch_size: int = 8,
               dtype: str = "fp32") -> MelodyModelOutput:
    """Chép mọi nốt (mọi nhạc cụ) của bài. Chạy trên GPU nếu có; người gọi phải giữ `gpu_lock()` khi dùng GPU.

    `dtype`: "fp32" (mặc định), "fp16" hoặc "bf16" qua autocast (trọng số vẫn fp32). Demo trên Space truyền `-pr 16`
    nhưng tham số này không được dùng khi suy luận, tức demo thực chất chạy fp32. Đo trên "Let It Be" (243 s,
    RTX 3060 Ti): fp32 batch 8 mất 25,7 s, VRAM đỉnh 2,6 GB; fp16 batch 8 mất 34 s, 1,2 GB (autocast làm chậm bước
    giải mã tự hồi quy); số nốt và số lỗi giải mã gần như nhau.
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_model(device)
    from utils.audio import slice_padded_array
    from utils.event2note import merge_zipped_note_events_and_ties_to_notes
    from utils.note2event import mix_notes

    sr = model.audio_cfg["sample_rate"]
    seg_len = model.audio_cfg["input_frames"]
    audio = _load_audio(Path(audio_path), sr)[None, :]
    duration = audio.shape[1] / sr   # độ dài thật, trước khi đệm
    if audio.shape[1] < seg_len:
        audio = np.pad(audio, ((0, 0), (0, seg_len - audio.shape[1])))
    segments = slice_padded_array(audio, seg_len, seg_len)
    x = torch.from_numpy(segments.astype("float32")).unsqueeze(1).to(device)   # (n_seg, 1, seg_len)

    amp = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(dtype)
    with torch.inference_mode():
        if amp is not None and device.startswith("cuda"):
            with torch.autocast("cuda", dtype=amp):
                pred_tokens, _ = model.inference_file(bsz=batch_size, audio_segments=x)
        else:
            pred_tokens, _ = model.inference_file(bsz=batch_size, audio_segments=x)
    del x

    start_secs = [seg_len * i / sr for i in range(segments.shape[0])]
    per_channel = []
    err = Counter()
    for ch in range(model.task_manager.num_decoding_channels):
        arr_ch = [arr[:, ch, :] for arr in pred_tokens]
        zipped, _, _ = model.task_manager.detokenize_list_batches(arr_ch, start_secs, return_events=True)
        notes_ch, err_ch = merge_zipped_note_events_and_ties_to_notes(zipped)
        per_channel.append(notes_ch)
        err += err_ch
    raw = mix_notes(per_channel)

    names = _instrument_names()
    notes: list[MelodyNote] = []
    programs: list[int] = []
    drums: list[bool] = []
    for n in sorted(raw, key=lambda n: (n.onset, n.pitch)):
        if n.onset >= duration:
            continue
        end = min(float(n.offset), duration)
        if end <= n.onset:   # nốt trống không có offset thật: cho dài 10 ms
            end = float(n.onset) + 0.01
        notes.append(MelodyNote(float(n.onset), end, int(n.pitch), None, source_name(n.program, n.is_drum, names)))
        programs.append(-1 if n.is_drum else int(n.program))
        drums.append(bool(n.is_drum))
    extras = {
        "note_program": np.array(programs, dtype=np.int16),
        "note_is_drum": np.array(drums, dtype=bool),
        "decode_errors": np.array(sum(err.values()), dtype=np.int64),
        "audio_duration_s": np.array(duration),
    }
    return MelodyModelOutput(MODEL_NAME, notes, None, extras)


def transcribe_locked(audio_path: Path, *, batch_size: int = 8, dtype: str = "fp32") -> MelodyModelOutput:
    """Nạp model trên CPU (ngoài khóa), chạy `transcribe` trong `gpu_lock()` rồi trả model về CPU.

    `extras` có thêm `load_s` (nạp model), `elapsed_s` (suy luận), `peak_vram_mib` (VRAM reserved đỉnh)."""
    import time

    import torch

    from chordtrack import gpu

    t0 = time.perf_counter()
    _load_model("cpu")
    load_s = time.perf_counter() - t0
    device = gpu.device()
    with gpu.gpu_lock():
        try:
            cuda = device == "cuda"
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            t1 = time.perf_counter()
            out = transcribe(audio_path, device=device, batch_size=batch_size, dtype=dtype)
            elapsed = time.perf_counter() - t1
            peak = torch.cuda.max_memory_reserved() / 2**20 if cuda else 0.0
        finally:
            offload()
    out.extras.update(load_s=np.array(load_s), elapsed_s=np.array(elapsed), peak_vram_mib=np.array(peak))
    return out


def main(argv: list[str] | None = None) -> int:
    """`python -m chordtrack.melody.yourmt3 <audio> <out.npz>`: chép nốt một bài, lưu bằng `chordtrack.melody.io`.

    Dùng cho server: YourMT3 cần transformers cũ nên phải chạy ở tiến trình riêng (xem docstring module)."""
    from chordtrack.melody import io as mio

    ap = argparse.ArgumentParser(prog="python -m chordtrack.melody.yourmt3", description=main.__doc__.splitlines()[0])
    ap.add_argument("audio", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--exit-with-parent", action="store_true",
                    help="thoát ngay khi stdin đóng (tiến trình cha giữ pipe stdin mở trong lúc chờ)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.exit_with_parent:
        import os
        import threading

        def watch() -> None:
            sys.stdin.buffer.read()   # chặn tới khi tiến trình cha đóng pipe hoặc chết
            os._exit(3)

        threading.Thread(target=watch, daemon=True).start()
    out = transcribe_locked(args.audio, batch_size=args.batch_size, dtype=args.dtype)
    mio.save_output(out, args.out)
    voice = sum(n.source == "voice" for n in out.notes)
    print(f"{len(out.notes)} nốt ({voice} voice), nạp {float(out.extras['load_s']):.1f} s, "
          f"suy luận {float(out.extras['elapsed_s']):.1f} s, VRAM {float(out.extras['peak_vram_mib']):.0f} MiB",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
