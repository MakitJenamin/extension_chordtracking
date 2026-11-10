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


@dataclass
class KeyEstimate:
    """Giọng của bài (analysis.py)."""

    tonic: str               # "C", "F#", "Bb"...
    mode: str                # "major" hoặc "minor"
    confidence: float | None = None


@dataclass
class KeyRegion:
    """Một vùng giọng của bài (analysis.estimate_key_regions). Các vùng phủ liên tục [0, duration]."""

    start: float
    end: float
    key: KeyEstimate


@dataclass
class Section:
    """Một đoạn của bài, kèm vòng hợp âm (analysis.py)."""

    start: float
    end: float
    label: str                    # "intro", "verse", "chorus", "bridge", "outro"... hoặc "A", "B" nếu không đặt tên được
    progression: list[str]        # vòng hợp âm lặp lại của đoạn, nhãn Harte, ví dụ ["C:maj", "G:maj", "A:min", "F:maj"]
    roman: list[str]              # cùng vòng đó ở dạng bậc La Mã, ví dụ ["I", "V", "vi", "IV"]


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


# ---------------------------------------------------------------------------
# Giai điệu (melody). Cao độ dùng số MIDI (60 = C4). Bên trong luôn giữ quãng tám đầy đủ.


@dataclass
class MelodyNote:
    start: float
    end: float
    pitch: int                        # số MIDI
    confidence: float | None = None   # khoảng [0, 1]; None nếu model không cung cấp
    source: str | None = None         # nhạc cụ hoặc "voice" nếu model có nhãn nhạc cụ, ngược lại None


@dataclass
class PitchTrack:
    """Đường cao độ theo khung (đầu ra của model dò f0 như RMVPE)."""

    times: np.ndarray                     # (T,) thời điểm mỗi khung
    f0_hz: np.ndarray                     # (T,) tần số cơ bản, 0 ở khung không có giai điệu
    confidence: np.ndarray | None = None  # (T,) khoảng [0, 1]


@dataclass
class MelodyModelOutput:
    """Đầu ra thô của một model melody hoặc chép nốt (chordtrack/melody/*.py).

    Model chép nốt nhiều nhạc cụ (YourMT3+, MuScriptor, Basic Pitch) điền `notes` với mọi nốt nghe được,
    kể cả đa âm; việc chọn dòng giai điệu làm ở bước sau. Model dò f0 (RMVPE) điền `pitch_track`.
    """

    model_name: str
    notes: list[MelodyNote] = field(default_factory=list)
    pitch_track: PitchTrack | None = None
    extras: dict[str, np.ndarray] = field(default_factory=dict)
