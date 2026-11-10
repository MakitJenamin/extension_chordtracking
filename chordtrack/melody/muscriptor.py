"""Chép nốt nhiều nhạc cụ bằng MuScriptor (Kyutai + Mirelo, 2026, arXiv 2607.08168).

Code MIT (`pip install muscriptor`, xem `requirements/melody_muscriptor.txt`), weights trên Hugging Face
`MuScriptor/muscriptor-{small,medium,large}` (103M / 307M / 1.4B tham số), license **CC BY-NC 4.0**, repo gated:
phải `hf auth login` và bấm đồng ý điều khoản trên trang model. Weights tải về `MODELS_DIR/muscriptor/<size>/`.

Model là transformer decoder sinh token kiểu MT3 cho từng đoạn 5 s audio 16 kHz mono, nhãn nhạc cụ theo 36 nhóm
`MT3_FULL_PLUS` (34 nhóm có tên, xem `INSTRUMENT_GROUPS`). **Nhóm giọng hát tên là `"voice"`** (nhóm 17, gộp
GM program 52–54: choir aahs, voice oohs, synth voice), nên hợp xướng và giọng solo đều ra `"voice"`.

Cách chạy (cố định, để các lần cache so được với nhau):
- Giải mã tham lam (greedy), `cfg_coef = 1.0` tức **không dùng CFG** (mặc định của MuScriptor; CFG 2 tốn gấp đôi
  tính toán), có `prelude_forcing` (nốt ngân qua ranh giới đoạn được ép giữ đúng nhạc cụ), batch 1.
- Model không có velocity. `MelodyNote.confidence` là xác suất softmax của token cao độ lúc model sinh ra nốt đó
  (greedy nên là xác suất lớn nhất ở bước đó).
- Lệch thời gian hệ thống (≤ ~25 ms) được trừ giống `transcribe_and_postprocess` của MuScriptor: dò phách bằng
  beat_this trên CPU, đo độ trễ của các onset so với lưới phách. Giá trị đã trừ ghi ở `extras["onset_delay"]`.
- Trống không có cao độ nên **không** đưa vào `notes`; thời điểm và số nốt trống GM nằm ở `extras["drum_times"]`,
  `extras["drum_pitches"]`.
"""
from __future__ import annotations

import contextlib
import io
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from chordtrack import paths
from chordtrack.types import MelodyModelOutput, MelodyNote

SIZES = ("small", "medium", "large")
CFG_COEF = 1.0
VOICE_SOURCE = "voice"
DRUM_SOURCE = "drums"
MIN_NOTE_DURATION = 0.01   # giống MINIMUM_NOTE_DURATION_SEC của MuScriptor
# dtype của transformer trên CUDA. "float32" nghĩa là weights fp32 và MuScriptor tự bật autocast fp16 khi tính.
DEFAULT_DTYPE = {"small": "float32", "medium": "float32", "large": "float16"}
_WEIGHT_FILES = ("model.safetensors", "config.json")

# 34 nhóm có tên của MT3_FULL_PLUS (chép từ muscriptor.tokenizer.mt3 để dùng khi chưa cài muscriptor).
# Nhóm 34, 35 (GM 100, 101) không có tên; nốt rơi vào đó hiện là "program_<n>".
INSTRUMENT_GROUPS = (
    "acoustic_piano", "electric_piano", "chromatic_percussion", "organ", "acoustic_guitar",
    "clean_electric_guitar", "distorted_electric_guitar", "acoustic_bass", "electric_bass", "violin", "viola",
    "cello", "contrabass", "orchestral_harp", "timpani", "string_ensemble", "synth_strings", "voice",
    "orchestra_hit", "trumpet", "trombone", "tuba", "french_horn", "brass_section", "soprano_and_alto_sax",
    "tenor_sax", "baritone_sax", "oboe", "english_horn", "bassoon", "clarinet", "flutes", "synth_lead",
    "synth_pad", "drums",
)


class MuscriptorAuthError(RuntimeError):
    """Không tải được weights vì chưa đăng nhập Hugging Face hoặc chưa đồng ý license (401/403)."""


def model_name(size: str) -> str:
    return f"muscriptor_{size}"


def weights_dir(size: str) -> Path:
    return paths.MODELS_DIR / "muscriptor" / size


def ensure_weights(size: str) -> Path:
    """Tải (nếu chưa có) weights và config.json của một cỡ vào `weights_dir(size)`, trả về file safetensors."""
    if size not in SIZES:
        raise ValueError(f"size phải là một trong {SIZES}, nhận {size!r}")
    d = weights_dir(size)
    if all((d / f).exists() for f in _WEIGHT_FILES):
        return d / "model.safetensors"
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

    repo = f"MuScriptor/muscriptor-{size}"
    for f in _WEIGHT_FILES:
        try:
            hf_hub_download(repo, f, local_dir=d)
        except (GatedRepoError, RepositoryNotFoundError) as e:
            raise MuscriptorAuthError(_auth_msg(repo)) from e
        except HfHubHTTPError as e:
            if getattr(e.response, "status_code", None) in (401, 403):
                raise MuscriptorAuthError(_auth_msg(repo)) from e
            raise
    return d / "model.safetensors"


def _auth_msg(repo: str) -> str:
    return (f"Không tải được {repo}: repo gated. Chạy `hf auth login` (hoặc đặt HF_TOKEN) và bấm đồng ý "
            f"license CC BY-NC 4.0 ở https://huggingface.co/{repo}")


def events_to_notes(
    events: Iterable,
    confidence_of: Callable[[], float | None] = lambda: None,
) -> tuple[list[MelodyNote], np.ndarray, np.ndarray]:
    """Ghép luồng NoteStartEvent/NoteEndEvent của MuScriptor thành nốt.

    `confidence_of()` được gọi đúng lúc nhận NoteStartEvent (luồng sinh lười, batch 1, nên lúc đó bước sinh gần
    nhất chính là token cao độ của nốt). Trả về (nốt có cao độ sắp theo thời gian, thời điểm trống, số nốt trống).
    """
    from muscriptor.events import NoteEndEvent, NoteStartEvent

    open_: dict[int, tuple[NoteStartEvent, float | None]] = {}
    notes: list[MelodyNote] = []
    drum_t: list[float] = []
    drum_p: list[int] = []
    for ev in events:
        if isinstance(ev, NoteStartEvent):
            if ev.instrument == DRUM_SOURCE:
                drum_t.append(ev.start_time)
                drum_p.append(ev.pitch)
            else:
                open_[ev.index] = (ev, confidence_of())
        elif isinstance(ev, NoteEndEvent):
            item = open_.pop(ev.start_event_index, None)
            if item is None:
                continue  # nốt trống
            st, conf = item
            end = max(ev.end_time, st.start_time + MIN_NOTE_DURATION)
            notes.append(MelodyNote(st.start_time, end, st.pitch, conf, st.instrument))
    for st, conf in open_.values():  # MuScriptor bảo đảm mỗi start có end; phòng hờ
        notes.append(MelodyNote(st.start_time, st.start_time + MIN_NOTE_DURATION, st.pitch, conf, st.instrument))
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes, np.asarray(drum_t, dtype=np.float64), np.asarray(drum_p, dtype=np.int16)


def shift_notes(notes: list[MelodyNote], delay: float) -> list[MelodyNote]:
    """Trừ độ trễ `delay` giây khỏi mọi nốt, không để thời gian âm."""
    if not delay:
        return notes
    out = []
    for n in notes:
        s = max(0.0, n.start - delay)
        e = max(s + MIN_NOTE_DURATION, n.end - delay)
        out.append(MelodyNote(s, e, n.pitch, n.confidence, n.source))
    return out


class _ProbRecorder:
    """Bọc `LMModel._compute_logits` để nhớ xác suất của token vừa được chọn (greedy = argmax)."""

    def __init__(self, lm):
        self.last: float | None = None
        self._lm = lm
        self._orig = lm._compute_logits
        lm._compute_logits = self._wrapped

    def restore(self) -> None:
        """Bỏ lớp bọc (model được dùng lại cho bài sau)."""
        self._lm.__dict__.pop("_compute_logits", None)

    def _wrapped(self, *args, **kwargs):
        logits = self._orig(*args, **kwargs)
        # batch 1: một hàng. Lấy max softmax bằng logsumexp cho ổn định số học.
        row = logits[0]
        self.last = float((row.max() - row.logsumexp(-1)).exp())
        return logits

    def __call__(self) -> float | None:
        return self.last


def _detect_onset_delay(audio_path: Path, onsets: list[float]) -> float:
    """Độ trễ onset so với lưới phách beat_this (CPU), như `transcribe_and_postprocess`. 0 nếu không đo được."""
    from muscriptor.utils.audio import load_audio
    from muscriptor.utils.beats import BeatDetectionError, detect_grid

    try:
        grid = detect_grid(load_audio(audio_path, target_sr=16000), 16000, device="cpu")
    except BeatDetectionError:
        return 0.0
    return float(grid.with_onset_delay(onsets).onset_delay or 0.0)


_CPU_MODEL: dict[tuple[str, str], object] = {}   # giữ một model đã dựng trên CPU để dùng lại giữa các bài


def _load_on_cpu(size: str, dtype: str):
    """Dựng TranscriptionModel trên CPU với dtype mong muốn (nhớ lại cho lần gọi sau cùng cỡ, cùng dtype).

    Không dùng `TranscriptionModel.load_model(device="cuda")`: hàm đó dựng model fp32 trên GPU rồi nạp thêm
    state_dict fp32 lên GPU trước khi đổi dtype, nên bản large (5,2 GB fp32) lên tới 10,5 GB VRAM đỉnh.
    """
    key = (size, dtype)
    if key in _CPU_MODEL:
        return _CPU_MODEL[key]
    _CPU_MODEL.clear()
    import torch
    from muscriptor.tokenizer.mt3 import MT3Tokenizer
    from muscriptor.transcription_model import (
        TranscriptionModel,
        _build_model,
        _config_from_json,
        _remap_single_codebook_keys,
    )
    from safetensors.torch import load_file

    weights = ensure_weights(size)
    cpu = torch.device("cpu")
    model = _build_model(cpu, _config_from_json(weights.parent / "config.json"))
    model.eval()
    model.load_state_dict(_remap_single_codebook_keys(load_file(str(weights), device="cpu")))
    if dtype != "float32":
        model.to(getattr(torch, dtype))
        model.condition_provider.float()   # như load_model: phần mel/nhúng nhãn luôn fp32
    tm = TranscriptionModel(model, MT3Tokenizer(instrument_vocabulary="MT3_FULL_PLUS", max_shift_steps=1001), cpu)
    _CPU_MODEL[key] = tm
    return tm


def _to_device(tm, device) -> None:
    """Chuyển model sang `device`, sửa các thuộc tính `.device` mà conditioner của MuScriptor tự lưu."""
    import torch
    from muscriptor.models.lm import TorchAutocast

    device = torch.device(device)
    tm._model.to(device)
    for m in tm._model.modules():
        if "device" in vars(m):
            m.device = device
    tm._device = device
    # Giống _build_model: trên CUDA bật autocast fp16, nơi khác tắt.
    tm._model.autocast = (TorchAutocast(enabled=True, device_type="cuda", dtype=torch.float16)
                          if device.type == "cuda" else TorchAutocast(enabled=False))


def transcribe(
    audio_path: str | Path,
    size: str = "small",
    *,
    dtype: str | None = None,
    device: str | None = None,
    cfg_coef: float = CFG_COEF,
    correct_onset_delay: bool = True,
) -> MelodyModelOutput:
    """Chép mọi nốt có cao độ trong `audio_path` bằng MuScriptor cỡ `size`.

    Model dựng trên CPU (ngoài khóa, nhớ lại giữa các bài), chỉ chuyển lên GPU trong `gpu_lock()` rồi trả về CPU,
    để không giữ VRAM giữa các bài. `extras` có: `cfg_coef`, `dtype`, `onset_delay`, `load_s` (dựng trên CPU, 0 nếu
    dùng lại), `to_gpu_s`, `infer_s`, `peak_vram_mb` (CUDA), `duration_s`, `drum_times`, `drum_pitches`.
    """
    import torch

    from chordtrack.gpu import gpu_lock

    audio_path = Path(audio_path)
    dtype = dtype or DEFAULT_DTYPE[size]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cuda = device.startswith("cuda")

    t0 = time.perf_counter()
    tm = _load_on_cpu(size, dtype)
    load_s = time.perf_counter() - t0

    rec = None
    with gpu_lock():
        try:
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            t1 = time.perf_counter()
            _to_device(tm, device)
            t2 = time.perf_counter()
            rec = _ProbRecorder(tm._model)
            events = tm.transcribe(audio_path, cfg_coef=cfg_coef, use_sampling=False, beam_size=1,
                                   batch_size=1, prelude_forcing=True)
            with contextlib.redirect_stdout(io.StringIO()):   # MuScriptor in "mel-spec ..." cho mỗi đoạn 5 s
                notes, drum_t, drum_p = events_to_notes(events, rec)
            if cuda:
                torch.cuda.synchronize()
            t3 = time.perf_counter()
            peak = torch.cuda.max_memory_allocated() / 2**20 if cuda else float("nan")
        finally:
            if rec is not None:
                rec.restore()
            _to_device(tm, "cpu")

    delay = 0.0
    if correct_onset_delay:
        delay = _detect_onset_delay(audio_path, [n.start for n in notes] + drum_t.tolist())
        notes = shift_notes(notes, delay)
        drum_t = np.maximum(drum_t - delay, 0.0)

    import soundfile as sf

    extras = {
        "cfg_coef": np.array(cfg_coef),
        "dtype": np.array(dtype),
        "onset_delay": np.array(delay),
        "load_s": np.array(load_s),
        "to_gpu_s": np.array(t2 - t1),
        "infer_s": np.array(t3 - t2),
        "peak_vram_mb": np.array(peak),
        "duration_s": np.array(sf.info(str(audio_path)).duration),
        "drum_times": drum_t,
        "drum_pitches": drum_p,
    }
    return MelodyModelOutput(model_name(size), notes, None, extras)
