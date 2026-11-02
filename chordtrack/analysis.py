"""Bước 5 và 9: nhận diện giọng, bậc La Mã, chia đoạn và tìm vòng hợp âm.

BẢN KHUNG: hiện trả về rỗng. Agent phụ trách analysis sẽ thay phần thân các hàm.
Chữ ký hàm là hợp đồng với pipeline.py, không được đổi.
"""
from __future__ import annotations

from chordtrack.types import BeatGrid, ChordSegment, KeyEstimate, Section


def estimate_key(segments: list[ChordSegment]) -> KeyEstimate | None:
    return None


def to_roman(label: str, key: KeyEstimate) -> str:
    """Đổi một nhãn Harte sang bậc La Mã theo giọng, ví dụ ("A:min", C major) -> "vi"."""
    return "?"


def analyze_structure(
    segments: list[ChordSegment],
    beats: BeatGrid | None,
    duration: float,
    key: KeyEstimate | None,
) -> list[Section]:
    return []
