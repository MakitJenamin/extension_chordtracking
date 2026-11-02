import numpy as np
import pytest

from chordtrack.analysis import (
    analyze_structure,
    estimate_key,
    find_progression,
    key_to_mir_eval,
    normalize_label,
    simplify_label,
    to_roman,
)
from chordtrack.types import BeatGrid, ChordSegment, KeyEstimate

C_MAJOR = KeyEstimate("C", "major", 1.0)
A_MINOR = KeyEstimate("A", "minor", 1.0)


def _song(labels: list[str], dur: float = 2.0, repeat: int = 1) -> list[ChordSegment]:
    out, t = [], 0.0
    for _ in range(repeat):
        for lab in labels:
            out.append(ChordSegment(t, t + dur, lab))
            t += dur
    return out


# ---------------------------------------------------------------- to_roman

@pytest.mark.parametrize("label, key, expected", [
    ("A:min", C_MAJOR, "vi"),
    ("G:7", C_MAJOR, "V7"),
    ("F:maj7", C_MAJOR, "IVmaj7"),
    ("C:maj/5", C_MAJOR, "I/5"),
    ("C:maj/3", C_MAJOR, "I/3"),
    ("Bb:maj", C_MAJOR, "bVII"),
    ("B:dim", C_MAJOR, "vii°"),
    ("E:maj", A_MINOR, "V"),
    ("N", C_MAJOR, "N.C."),
    ("C", C_MAJOR, "I"),                 # nhãn tắt của Harte: C = C:maj
    ("D:min7", C_MAJOR, "ii7"),
    ("B:hdim7", C_MAJOR, "viiø7"),
    ("B:dim7", C_MAJOR, "vii°7"),
    ("Ab:maj", C_MAJOR, "bVI"),
    ("Eb:maj", C_MAJOR, "bIII"),
    ("Db:maj", C_MAJOR, "bII"),
    ("G:sus4", C_MAJOR, "Vsus4"),
    ("C:aug", C_MAJOR, "I+"),
    ("A:min/b3", C_MAJOR, "vi/b3"),
    ("G:7/b7", C_MAJOR, "V7/b7"),
    ("C:maj", A_MINOR, "III"),
    ("G:maj", A_MINOR, "VII"),
    ("D:min", A_MINOR, "iv"),
    ("A:min", A_MINOR, "i"),
    ("G#:dim", A_MINOR, "#vii°"),
    ("F:maj", A_MINOR, "VI"),
    ("Bb:maj", A_MINOR, "bII"),
    ("C:(1,3,5)", C_MAJOR, "I"),          # chỉ có ngoặc, không có shorthand
    ("D:min(9)", C_MAJOR, "ii"),          # phần mở rộng trong ngoặc bị bỏ qua
    ("X", C_MAJOR, "?"),
])
def test_to_roman(label, key, expected):
    assert to_roman(label, key) == expected


def test_to_roman_enharmonic_tonic():
    assert to_roman("F:maj", KeyEstimate("Bb", "major")) == "V"
    assert to_roman("F:maj", KeyEstimate("A#", "major")) == "V"
    assert to_roman("C#:min", KeyEstimate("E", "major")) == "vi"


# ---------------------------------------------------------------- nhãn

def test_simplify_and_normalize():
    assert simplify_label("A:min/b7") == "A:min"
    assert simplify_label("F:maj7") == "F:maj"
    assert simplify_label("C") == "C:maj"
    assert simplify_label("G:7") == "G:maj"
    assert simplify_label("B:hdim7") == "B:dim"
    assert simplify_label("N") == "N"
    assert normalize_label("C") == "C:maj"
    assert normalize_label("C/5") == "C:maj/5"
    assert normalize_label("A:min7") == "A:min7"
    assert normalize_label("N") == "N"


# ---------------------------------------------------------------- estimate_key

def test_key_c_major_pop_loop():
    k = estimate_key(_song(["C:maj", "G:maj", "A:min", "F:maj"], repeat=8) + [ChordSegment(64, 68, "C:maj")])
    assert (k.tonic, k.mode) == ("C", "major")
    assert 0.0 <= k.confidence <= 1.0


def test_key_relative_minor():
    # cùng bộ nốt với C major nhưng xoay quanh A minor, có E trưởng (V của A minor)
    segs = _song(["A:min", "D:min", "E:7", "A:min", "F:maj", "G:maj", "E:maj", "A:min"], repeat=6)
    k = estimate_key(segs)
    assert (k.tonic, k.mode) == ("A", "minor")


def test_key_flat_spelling():
    k = estimate_key(_song(["Bb:maj", "Eb:maj", "F:7", "Bb:maj", "G:min", "C:min", "F:maj", "Bb:maj"], repeat=4))
    assert (k.tonic, k.mode) == ("Bb", "major")
    assert key_to_mir_eval(k) == "Bb major"
    k = estimate_key(_song(["F#:min", "B:min", "C#:7", "F#:min"], repeat=6))
    assert (k.tonic, k.mode) == ("F#", "minor")


def test_key_g_major_with_d7_cadences():
    k = estimate_key(_song(["G:maj", "C:maj", "D:7", "G:maj", "E:min", "A:min", "D:7", "G:maj"], repeat=4))
    assert (k.tonic, k.mode) == ("G", "major")


def test_key_empty():
    assert estimate_key([]) is None
    assert estimate_key([ChordSegment(0, 10, "N")]) is None
    assert key_to_mir_eval(None) == "X"


# ---------------------------------------------------------------- find_progression

def test_find_progression_basic_loop():
    seq = ["C:maj", "G:maj", "A:min", "F:maj"] * 4
    assert find_progression(seq) == ["C:maj", "G:maj", "A:min", "F:maj"]


def test_find_progression_drops_repeats_and_n():
    seq = ["N", "C:maj", "C:maj", "G:maj", "N", "A:min", "F:maj", "F:maj"] * 3
    assert find_progression(seq) == ["C:maj", "G:maj", "A:min", "F:maj"]


def test_find_progression_multiple_sequences():
    line = ["C:maj", "G:maj", "A:min", "F:maj", "C:maj", "G:maj", "F:maj", "C:maj"]
    assert find_progression([line, line, line]) == line


def test_find_progression_no_repeat():
    seq = ["C:maj", "D:min", "E:min", "F:maj", "G:maj"]
    assert find_progression(seq) == seq
    assert find_progression([]) == []


# ---------------------------------------------------------------- analyze_structure

def _check_continuous(sections, duration):
    assert sections, "phải có ít nhất một đoạn"
    assert sections[0].start == 0.0
    assert sections[-1].end == pytest.approx(duration)
    for a, b in zip(sections, sections[1:]):
        assert a.end == b.start
    for s in sections:
        assert s.end > s.start


def _verse_chorus_song():
    verse = ["C:maj", "G:maj", "A:min", "F:maj"] * 2
    chorus = ["F:maj", "G:maj", "E:min", "A:min", "D:min", "G:maj", "C:maj", "C:maj"]
    intro = ["C:maj", "C:maj", "F:maj", "F:maj"]
    labels = intro + (verse * 2 + chorus * 2) * 3 + ["C:maj"]
    return _song(labels, dur=2.0)


def test_structure_continuous_and_labels():
    segs = _verse_chorus_song()
    duration = segs[-1].end + 1.5
    key = estimate_key(segs)
    assert (key.tonic, key.mode) == ("C", "major")
    sections = analyze_structure(segs, None, duration, key)
    _check_continuous(sections, duration)
    assert 3 <= len(sections) <= 20
    progs = {tuple(s.roman) for s in sections}
    assert ("I", "V", "vi", "IV") in progs
    for s in sections:
        assert len(s.roman) == len(s.progression)


def test_structure_with_beat_grid():
    segs = _verse_chorus_song()
    duration = segs[-1].end
    beats = np.arange(0.3, duration, 0.5)
    grid = BeatGrid(beats=beats, downbeats=beats[::4], tempo=120.0)
    sections = analyze_structure(segs, grid, duration, None)
    _check_continuous(sections, duration)
    assert all(s.roman == [] for s in sections)
    assert any(s.progression == ["C:maj", "G:maj", "A:min", "F:maj"] for s in sections)


def test_structure_edge_cases():
    _check_continuous(analyze_structure([], None, 30.0, None), 30.0)
    short = [ChordSegment(0, 2, "C:maj"), ChordSegment(2, 4, "G:maj")]
    _check_continuous(analyze_structure(short, None, 4.0, C_MAJOR), 4.0)
    # nhãn lệch khỏi [0, duration] và có khoảng trống
    gappy = [ChordSegment(1.0, 3.0, "C:maj"), ChordSegment(5.0, 9.0, "G:maj"), ChordSegment(9.0, 40.0, "F:maj")]
    _check_continuous(analyze_structure(gappy, None, 35.0, C_MAJOR), 35.0)
    assert analyze_structure(short, None, 0.0, None) == []
