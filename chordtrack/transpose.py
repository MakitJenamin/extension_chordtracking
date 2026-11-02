"""Dịch giọng hợp âm, và chế độ "phím trắng" cho piano.

Người chơi piano hay dịch mọi bài về phím trắng: giọng trưởng chơi như C trưởng, giọng thứ chơi
như A thứ, rồi dùng nút transpose của đàn điện để đàn phát ra đúng tông. Gặp chỗ đổi giọng (ví dụ
điệp khúc cuối nâng +2) thì đổi giá trị transpose giữa bài.

Quy ước: **transpose = +k** nghĩa là bấm phím C thì đàn phát ra nốt C+k (k nửa cung).
Hợp âm hiển thị để bấm = hợp âm gốc dịch đi **-k**.
"""
from __future__ import annotations

import re

from chordtrack.symbols import FLAT_NAMES, harte_to_symbol, pitch_class
from chordtrack.types import KeyEstimate

WHITE_MAJOR = KeyEstimate("C", "major")
WHITE_MINOR = KeyEstimate("A", "minor")

_ROOT_RE = re.compile(r"^([A-G][b#]*)(.*)$")


def wrap_semitones(k: int) -> int:
    """Đưa số nửa cung về khoảng [-6, +5] (ví dụ 7 -> -5, 6 -> -6)."""
    return (int(k) + 6) % 12 - 6


def white_key_offset(key: KeyEstimate) -> int:
    """Số nửa cung đặt cho nút transpose của đàn để chơi giọng `key` trên phím trắng.

    Giọng trưởng quy về C trưởng, giọng thứ quy về A thứ. Giá trị nằm trong [-6, +5], chọn trị
    tuyệt đối nhỏ nhất; trường hợp cách đúng 6 nửa cung (F#/Gb trưởng, D#/Eb thứ) thì quy ước
    lấy **-6** (đàn dịch xuống), không dùng +6.

    Ví dụ: D trưởng -> +2, Eb trưởng -> +3, F# trưởng -> -6, C# thứ -> +4, F thứ -> -4.
    """
    target = 0 if key.mode == "major" else 9
    return wrap_semitones(pitch_class(key.tonic) - target)


def white_key_reference(key: KeyEstimate) -> KeyEstimate:
    """Giọng phím trắng tương ứng: C trưởng hoặc A thứ."""
    return WHITE_MINOR if key.mode == "minor" else WHITE_MAJOR


def transpose_label(harte: str, semitones: int) -> str:
    """Dịch nhãn Harte đi `semitones` nửa cung (dương là lên).

    Giữ nguyên loại hợp âm, phần mở rộng trong ngoặc và bậc bass (bass ghi theo bậc tính từ nốt
    gốc nên tự đúng): ("D:min7/b7", -2) -> "C:min7/b7", ("E:maj/3", 1) -> "F:maj/3".
    Nốt gốc mới viết theo bảng 12 nốt dùng dấu giáng (Db, Eb, Ab, Bb), riêng F# giữ dấu thăng.
    Như vậy hợp âm thuộc giọng C trưởng/A thứ luôn ra nốt trắng (C, Dm, Em, F, G, Am, Bdim).
    Nhãn "C" viết tắt (không có dấu ":") giữ dạng viết tắt. "N", "X" và nhãn không đọc được giữ nguyên.
    """
    label = harte.strip()
    if label in ("N", "X", ""):
        return label
    m = _ROOT_RE.match(label)
    if m is None:
        return label
    root, rest = m.groups()
    try:
        pc = pitch_class(root)
    except KeyError:
        return label
    return FLAT_NAMES[(pc + int(semitones)) % 12] + rest


def white_key_label(harte: str, key: KeyEstimate) -> str:
    """Nhãn Harte để bấm trên phím trắng khi chơi giọng `key` (dịch đi -white_key_offset)."""
    return transpose_label(harte, -white_key_offset(key))


def white_key_symbol(harte: str, key: KeyEstimate) -> str:
    """Ký hiệu hợp âm để bấm trên phím trắng, ví dụ ("E:maj", D trưởng) -> "D"."""
    return harte_to_symbol(white_key_label(harte, key), white_key_reference(key))
