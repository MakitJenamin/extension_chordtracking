"""Kiểu dữ liệu dùng chung giữa các bước của pipeline.

Đây là "hợp đồng" giữa các module. Muốn đổi thì phải sửa ở đây trước, rồi cập nhật
mọi nơi đang dùng. Không tự định nghĩa lại các kiểu này trong module khác.

Quy ước:
- Thời gian luôn tính bằng giây (float), tính từ đầu file audio.
- Nhãn hợp âm dùng ký hiệu Harte giống mir_eval: "C:maj", "A:min7", "G:maj/3", "N" (không có hợp âm).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Stems:
    """Kết quả bước tách nguồn (separation.py). Mọi đường dẫn là file WAV."""

    mix: Path            # audio gốc đã chuẩn hóa
    vocals: Path
    instrumental: Path   # mix bỏ giọng hát (đầu ra của BS-Roformer)
    bass: Path           # chỉ có bass (HTDemucs chạy trên phần instrumental)
    no_bass: Path        # instrumental bỏ bass: hòa âm ở dải trung và cao


@dataclass
class BeatGrid:
    """Kết quả bước bắt nhịp (beats.py)."""

    beats: np.ndarray      # (N,) thời điểm từng phách, tăng dần
    downbeats: np.ndarray  # (M,) thời điểm phách mạnh (đầu ô nhịp), là tập con của beats
    tempo: float           # BPM ước tính


@dataclass
class ChordSegment:
    start: float
    end: float
    label: str                        # ký hiệu Harte
    confidence: float | None = None   # khoảng [0, 1]; None nếu model không cung cấp


@dataclass
class ChordModelOutput:
    """Đầu ra của một model nhận diện hợp âm (chords/*.py)."""

    model_name: str
    segments: list[ChordSegment]
    # Xác suất theo từng khung. Không bắt buộc, nhưng cần có để kết hợp model và tính độ tin cậy.
    frame_times: np.ndarray | None = None   # (T,) thời điểm giữa mỗi khung
    frame_probs: np.ndarray | None = None   # (T, V), mỗi hàng cộng lại bằng 1
    vocabulary: list[str] | None = None     # V nhãn Harte, ứng với các cột của frame_probs
    # Thông tin phụ, ví dụ xác suất nốt bass của Chord-CNN-LSTM: {"bass_probs": (T, 13)}
    extras: dict[str, np.ndarray] = field(default_factory=dict)


def segments_to_lab(segments: list[ChordSegment], path: Path) -> None:
    """Ghi ra file .lab (bắt đầu, kết thúc, nhãn), định dạng mir_eval đọc được."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in segments:
            f.write(f"{s.start:.6f}\t{s.end:.6f}\t{s.label}\n")


def lab_to_segments(path: Path) -> list[ChordSegment]:
    out: list[ChordSegment] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                out.append(ChordSegment(float(parts[0]), float(parts[1]), parts[2]))
    return out
