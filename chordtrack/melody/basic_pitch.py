"""Spotify Basic Pitch (Apache-2.0): chép nốt đa âm trên bản mix.

Chạy model ONNX đi kèm gói (`saved_models/icassp_2022/nmp.onnx`) bằng onnxruntime trên CPU, nên không
cần khóa GPU. Gói `basic-pitch` cài bằng `--no-deps` (xem `requirements/melody_basicpitch.txt`): cài
thường sẽ kéo tensorflow và hạ resampy.

`notes` là mọi nốt nghe được (kể cả đa âm), confidence = amplitude của Basic Pitch (trung bình độ kích
hoạt khung trong nốt, vốn nằm trong [0, 1]), cắt về [0, 1]. Việc chọn dòng giai điệu làm ở bước sau.
"""
from __future__ import annotations

import contextlib
import io
import logging
import time
from pathlib import Path

import numpy as np

from chordtrack.types import MelodyModelOutput, MelodyNote

MODEL_NAME = "basic_pitch"

# Tham số mặc định của basic_pitch.inference.predict
ONSET_THRESHOLD = 0.5
FRAME_THRESHOLD = 0.3
MIN_NOTE_MS = 127.70

_model = None


def _import():
    # basic_pitch/__init__.py ghi cảnh báo khi thiếu tensorflow/coreml/tflite: không cần, tắt đi
    root = logging.getLogger()
    level = root.level
    root.setLevel(logging.ERROR)
    try:
        import basic_pitch
        from basic_pitch import inference
    finally:
        root.setLevel(level)
    return basic_pitch, inference


def model_path() -> Path:
    basic_pitch, _ = _import()
    return Path(basic_pitch.__file__).parent / "saved_models" / "icassp_2022" / "nmp.onnx"


def load_model():
    global _model
    if _model is None:
        _, inference = _import()
        _model = inference.Model(model_path())
    return _model


def transcribe(audio_path: Path, onset_threshold: float = ONSET_THRESHOLD, frame_threshold: float = FRAME_THRESHOLD,
               min_note_ms: float = MIN_NOTE_MS) -> MelodyModelOutput:
    _, inference = _import()
    model = load_model()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):  # predict() in "Predicting MIDI for ..."
        _, _, events = inference.predict(
            str(audio_path), model, onset_threshold=onset_threshold, frame_threshold=frame_threshold,
            minimum_note_length=min_note_ms)
    events = sorted(events, key=lambda e: (e[0], e[2]))
    notes = [MelodyNote(float(s), float(e), int(p), float(np.clip(a, 0.0, 1.0)), None)
             for s, e, p, a, _ in events]
    return MelodyModelOutput(model_name=MODEL_NAME, notes=notes,
                             extras={"infer_sec": np.float64(time.perf_counter() - t0)})
