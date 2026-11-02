import pytest

from chordtrack.symbols import harte_to_symbol
from chordtrack.transpose import (
    transpose_label,
    white_key_label,
    white_key_offset,
    white_key_symbol,
    wrap_semitones,
)
from chordtrack.types import KeyEstimate

# Đủ 24 giọng (kể cả cách viết khác của cùng cao độ): (nốt chủ, mode, transpose mong đợi)
MAJOR = [("C", 0), ("Db", 1), ("C#", 1), ("D", 2), ("Eb", 3), ("D#", 3), ("E", 4), ("F", 5),
         ("F#", -6), ("Gb", -6), ("G", -5), ("Ab", -4), ("G#", -4), ("A", -3), ("Bb", -2), ("A#", -2),
         ("B", -1), ("Cb", -1)]
MINOR = [("A", 0), ("Bb", 1), ("A#", 1), ("B", 2), ("C", 3), ("C#", 4), ("Db", 4), ("D", 5),
         ("D#", -6), ("Eb", -6), ("E", -5), ("F", -4), ("F#", -3), ("Gb", -3), ("G", -2), ("G#", -1),
         ("Ab", -1)]


@pytest.mark.parametrize("tonic, expected", MAJOR)
def test_white_key_offset_major(tonic, expected):
    assert white_key_offset(KeyEstimate(tonic, "major")) == expected


@pytest.mark.parametrize("tonic, expected", MINOR)
def test_white_key_offset_minor(tonic, expected):
    assert white_key_offset(KeyEstimate(tonic, "minor")) == expected


def test_white_key_offset_covers_all_24_keys_in_range():
    seen = set()
    for pc, name in enumerate(["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]):
        for mode, target in (("major", 0), ("minor", 9)):
            k = white_key_offset(KeyEstimate(name, mode))
            assert -6 <= k <= 5
            assert (target + k) % 12 == pc        # bấm nốt chủ phím trắng, đàn phát ra đúng nốt chủ
            seen.add((pc, mode))
    assert len(seen) == 24


def test_examples_from_spec():
    assert white_key_offset(KeyEstimate("D", "major")) == 2
    assert white_key_offset(KeyEstimate("Eb", "major")) == 3
    assert white_key_offset(KeyEstimate("F#", "major")) == -6
    assert white_key_offset(KeyEstimate("C#", "minor")) == 4
    assert white_key_offset(KeyEstimate("F", "minor")) == -4


def test_wrap_semitones():
    assert [wrap_semitones(k) for k in (0, 5, 6, 7, 11, 12, -1, -6, -7)] == [0, 5, -6, -5, -1, 0, -1, -6, 5]


@pytest.mark.parametrize("label, k, expected", [
    ("C:maj", 2, "D:maj"),
    ("D:min7/b7", -2, "C:min7/b7"),      # hợp âm 7 đảo: bass theo bậc nên giữ nguyên
    ("E:maj/3", 1, "F:maj/3"),
    ("G:7/3", -7, "C:7/3"),
    ("A:sus4", 3, "C:sus4"),
    ("F#:dim", 5, "B:dim"),
    ("B:hdim7", 1, "C:hdim7"),
    ("Bb:maj7", 2, "C:maj7"),
    ("C#:min(9)", -1, "C:min(9)"),
    ("C", 5, "F"),                       # nhãn viết tắt giữ dạng viết tắt
    ("C/5", 7, "G/5"),
    ("C:maj", 1, "Db:maj"),
    ("C:maj", 6, "F#:maj"),
    ("A:min", 12, "A:min"),
    ("N", 3, "N"),
    ("X", 3, "X"),
])
def test_transpose_label(label, k, expected):
    assert transpose_label(label, k) == expected


def test_transpose_roundtrip_all_offsets():
    for k in range(-12, 13):
        for label in ("C:maj", "Eb:min7/b3", "F#:sus2", "Ab:dim7", "B:7/b7"):
            back = transpose_label(transpose_label(label, k), -k)
            assert harte_to_symbol(back) == harte_to_symbol(label)


def test_white_key_symbols_major():
    d = KeyEstimate("D", "major")
    labels = ["D:maj", "E:min", "F#:min", "G:maj", "A:7", "B:min", "C#:dim", "D:maj/3", "A:sus4", "B:min7/b7"]
    assert [white_key_symbol(lab, d) for lab in labels] == ["C", "Dm", "Em", "F", "G7", "Am", "Bdim", "C/E",
                                                             "Gsus4", "Am7/G"]


def test_white_key_symbols_minor():
    fsm = KeyEstimate("F#", "minor")   # transpose -3: F#m -> Am
    assert white_key_offset(fsm) == -3
    labels = ["F#:min", "D:maj7", "B:min7", "A:maj/5", "E:7", "C#:maj", "G#:hdim7"]
    assert [white_key_symbol(lab, fsm) for lab in labels] == ["Am", "Fmaj7", "Dm7", "C/G", "G7", "E", "Bm7b5"]


def test_white_key_label_is_inverse_of_transpose():
    eb = KeyEstimate("Eb", "major")
    assert white_key_label("Ab:maj/3", eb) == "F:maj/3"
    assert transpose_label(white_key_label("Ab:maj/3", eb), white_key_offset(eb)) == "Ab:maj/3"
