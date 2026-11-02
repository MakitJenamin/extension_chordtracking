"""Đổi nhãn Harte ("A:min7", "C:maj/5") sang ký hiệu cho người chơi đàn ("Am7", "C/G").

Tên nốt gốc viết lại theo bảng 12 nốt: mặc định dùng dấu giáng (Bb, Eb, Ab, Db), riêng F#
giữ dấu thăng. Nếu biết giọng thì viết theo giọng: giọng thăng (G, D, A, E... trưởng) dùng
dấu thăng, giọng giáng (F, Bb, Eb... trưởng) dùng dấu giáng. Nốt bass viết theo chữ cái
của quãng tính từ nốt gốc, nên A/C# đúng chứ không thành A/Db.
"""
from __future__ import annotations

import re

from chordtrack.types import KeyEstimate

NO_CHORD = "N.C."
UNKNOWN_CHORD = "?"

LETTERS = "CDEFGAB"
LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ALL_FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# Nốt chủ của giọng trưởng tương đương (giọng thứ thì cộng 3 nửa cung).
_SHARP_MAJOR_TONICS = {7, 2, 9, 4, 11, 6, 1}   # G D A E B F# C#

# Loại hợp âm Harte (shorthand) -> hậu tố ký hiệu.
QUALITY_SUFFIX = {
    "maj": "",
    "min": "m",
    "aug": "aug",
    "dim": "dim",
    "sus2": "sus2",
    "sus4": "sus4",
    "5": "5",
    "1": "",
    "7": "7",
    "maj7": "maj7",
    "min7": "m7",
    "minmaj7": "m(maj7)",
    "dim7": "dim7",
    "hdim7": "m7b5",
    "maj6": "6",
    "min6": "m6",
    "9": "9",
    "maj9": "maj9",
    "min9": "m9",
    "11": "11",
    "min11": "m11",
    "13": "13",
    "maj13": "maj13",
    "min13": "m13",
}

# Tổ hợp loại + nốt thêm hay gặp, viết theo cách quen dùng.
SPECIAL = {
    ("sus4", ("b7",)): "7sus4",
    ("sus4", ("9", "b7")): "9sus4",
    ("sus4", ("13", "9", "b7")): "13sus4",
    ("sus2", ("b7",)): "7sus2",
    ("aug", ("b7",)): "aug7",
    ("maj", ("9",)): "add9",
    ("maj", ("2",)): "add2",
    ("maj", ("4",)): "add4",
    ("maj", ("11",)): "add11",
    ("min", ("9",)): "m(add9)",
    ("min", ("2",)): "m(add2)",
    ("min", ("4",)): "m(add4)",
    ("min", ("11",)): "m(add11)",
    ("maj6", ("9",)): "6/9",
    ("min7", ("b5",)): "m7b5",
}

_DEGREE_SEMITONES = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11}
_LABEL_RE = re.compile(r"^([A-G])([b#]*):?([^/(]*)(?:\(([^)]*)\))?(?:/([b#]*\d+))?$")


def pitch_class(note: str) -> int:
    """"C" -> 0, "Bb" -> 10, "F#" -> 6, "Cb" -> 11."""
    pc = LETTER_PC[note[0]]
    for acc in note[1:]:
        pc += 1 if acc == "#" else -1
    return pc % 12


def _sharp_key(key: KeyEstimate | None) -> bool | None:
    """True: giọng dùng dấu thăng, False: dấu giáng, None: không rõ (C trưởng, A thứ, không có giọng)."""
    if key is None:
        return None
    try:
        tonic = pitch_class(key.tonic)
    except (KeyError, IndexError):
        return None
    major_tonic = (tonic + 3) % 12 if key.mode == "minor" else tonic
    if major_tonic == 0:
        return None
    if major_tonic in (6, 1):  # F#/Gb, C#/Db: theo cách viết tên nốt chủ
        return "#" in key.tonic
    if major_tonic in _SHARP_MAJOR_TONICS:
        return True
    return False


def note_name(pc: int, key: KeyEstimate | None = None) -> str:
    """Tên nốt cho lớp cao độ `pc` (C=0), viết theo giọng nếu có."""
    sharp = _sharp_key(key)
    if sharp is True:
        return SHARP_NAMES[pc % 12]
    if sharp is False:
        return ALL_FLAT_NAMES[pc % 12]
    return FLAT_NAMES[pc % 12]


def _degree_semitones(degree: str) -> tuple[int, int]:
    """"b7" -> (7, 10), "#4" -> (4, 6), "9" -> (9, 14). Trả về (bậc, số nửa cung)."""
    acc = 0
    i = 0
    while i < len(degree) and degree[i] in "b#":
        acc += 1 if degree[i] == "#" else -1
        i += 1
    num = int(degree[i:])
    if num < 1:
        raise ValueError(degree)
    octave, base = divmod(num - 1, 7)
    return num, 12 * octave + _DEGREE_SEMITONES[base + 1] + acc


def bass_note(root: str, degree: str) -> str:
    """Nốt bass của quãng `degree` tính từ `root`, viết theo chữ cái của quãng.

    ("C", "5") -> "G", ("A", "3") -> "C#", ("Eb", "b7") -> "Db". Nếu cần dấu kép thì viết
    theo bảng 12 nốt.
    """
    num, semis = _degree_semitones(degree)
    pc = (pitch_class(root) + semis) % 12
    letter = LETTERS[(LETTERS.index(root[0]) + num - 1) % 7]
    diff = (pc - LETTER_PC[letter]) % 12
    if diff == 0:
        return letter
    if diff == 1:
        return letter + "#"
    if diff == 11:
        return letter + "b"
    return FLAT_NAMES[pc]


def _suffix(quality: str, extensions: tuple[str, ...]) -> str:
    # Bỏ các nốt bị lược ("*3") khỏi phần hiển thị; giữ thứ tự ổn định để tra bảng.
    adds = tuple(sorted((e for e in extensions if e and not e.startswith("*")), reverse=True))
    if not quality:
        # Chỉ có danh sách nốt, ví dụ "C:(1,5)" -> power chord, "C:(1)" -> nốt đơn.
        if set(adds) <= {"1", "5"}:
            return "5" if "5" in adds else ""
        quality = "maj"
        adds = tuple(e for e in adds if e not in ("1", "3", "5"))
    if "*3" in extensions and quality in ("maj", "min") and not adds:
        return "5"  # bỏ nốt 3: power chord
    if (quality, adds) in SPECIAL:
        return SPECIAL[(quality, adds)]
    base = QUALITY_SUFFIX.get(quality, quality)
    if not adds:
        return base
    return f"{base}({','.join(adds)})"


def harte_to_symbol(label: str, key: KeyEstimate | None = None) -> str:
    """Nhãn Harte -> ký hiệu hợp âm. Nhãn không đọc được thì trả lại nguyên văn."""
    label = label.strip()
    if label in ("N", ""):
        return NO_CHORD
    if label == "X":
        return UNKNOWN_CHORD
    m = _LABEL_RE.match(label)
    if m is None:
        return label
    letter, acc, quality, ext, bass = m.groups()
    try:
        root_pc = pitch_class(letter + acc)
    except KeyError:
        return label
    root = note_name(root_pc, key)
    if not quality and ext is None:
        quality = "maj"  # "C" không có dấu ":" là C trưởng
    extensions = tuple(e.strip() for e in ext.split(",")) if ext else ()
    symbol = root + _suffix(quality, extensions)
    if bass and bass != "1":
        try:
            symbol += "/" + bass_note(root, bass)
        except (ValueError, KeyError):
            return label
    return symbol
