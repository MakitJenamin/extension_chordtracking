"""Bước 6 và 7: kết hợp nhiều model hợp âm và hậu xử lý thành một chuỗi hợp âm cuối cùng.

BẢN KHUNG: hiện chỉ trả lại kết quả của ChordMini và lấp các khoảng trống. Agent phụ trách
fusion sẽ thay phần thân hàm `fuse`. Chữ ký hàm là hợp đồng với pipeline.py, không được đổi.
"""
from __future__ import annotations

from pathlib import Path

from chordtrack.types import BeatGrid, ChordModelOutput, ChordSegment


def fuse(
    models: dict[str, ChordModelOutput],
    beats: BeatGrid | None,
    bass_audio: Path | None,
    duration: float,
) -> list[ChordSegment]:
    """Trả về chuỗi hợp âm phủ liên tục từ 0 tới `duration`, mỗi đoạn có confidence.

    models: các khóa đã biết là "chordmini" (BTC-PL) và "cnnlstm"; có thể thiếu khóa.
    beats: lưới phách để bỏ phiếu theo phách; None nếu không có.
    bass_audio: file WAV chỉ có bass (Stems.bass); None nếu không có.
    """
    base = models["chordmini"].segments if "chordmini" in models else next(iter(models.values())).segments
    return fill_gaps(base, duration)


def fill_gaps(segments: list[ChordSegment], duration: float) -> list[ChordSegment]:
    """Lấp khoảng trống: kéo dài đoạn trước tới đầu đoạn sau, thêm "N" ở đầu và cuối nếu thiếu."""
    out: list[ChordSegment] = []
    t = 0.0
    for s in sorted(segments, key=lambda x: x.start):
        if s.end <= t:
            continue
        start = max(s.start, t)
        if start > t:
            if out:
                out[-1] = ChordSegment(out[-1].start, start, out[-1].label, out[-1].confidence)
            else:
                out.append(ChordSegment(0.0, start, "N", None))
        out.append(ChordSegment(start, s.end, s.label, s.confidence))
        t = s.end
    if t < duration:
        out.append(ChordSegment(t, duration, "N", None))
    return out
