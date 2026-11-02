import pytest

from chordtrack.paths import CHORDMINI_DIR, CNNLSTM_DIR
from chordtrack.symbols import bass_note, harte_to_symbol, note_name
from chordtrack.types import KeyEstimate


@pytest.mark.parametrize("label, symbol", [
    ("A:min7", "Am7"),
    ("C:maj/5", "C/G"),
    ("C:maj/3", "C/E"),
    ("G:7", "G7"),
    ("F:maj7", "Fmaj7"),
    ("D:sus4", "Dsus4"),
    ("B:dim", "Bdim"),
    ("B:hdim7", "Bm7b5"),
    ("N", "N.C."),
    ("X", "?"),
    ("C:maj", "C"),
    ("C", "C"),
    ("A:min", "Am"),
    ("C:aug", "Caug"),
    ("C:sus2", "Csus2"),
    ("C:dim7", "Cdim7"),
    ("C:maj6", "C6"),
    ("A:min6", "Am6"),
    ("C:minmaj7", "Cm(maj7)"),
    ("C:9", "C9"),
    ("C:maj9", "Cmaj9"),
    ("D:min9", "Dm9"),
    ("C:11", "C11"),
    ("C:13", "C13"),
    ("G:sus4(b7)", "G7sus4"),
    ("C:maj(9)", "Cadd9"),
    ("C:7(#9)", "C7(#9)"),
    ("C:(1,5)", "C5"),
    ("C:(1,b3,5)", "Cm"),
    ("C:(1,b3,b5)", "Cdim"),
    ("C:(1,3,5,b7)", "C7"),
    ("C:sus4(b7,9)", "C9sus4"),
    ("C:sus4(9,b7)", "C9sus4"),
    ("C:sus4(b7,9,13)", "C13sus4"),
    ("C:maj6(9)", "C6/9"),
    ("C:maj(*3)", "C5"),
    # bass là quãng tính từ nốt gốc
    ("F:maj/2", "F/G"),
    ("C:maj/b7", "C/Bb"),
    ("C:min/b3", "Cm/Eb"),
    ("C:min/5", "Cm/G"),
    ("C:min/b7", "Cm/Bb"),
    ("C:min/2", "Cm/D"),
    ("A:maj/3", "A/C#"),
    ("E:maj/3", "E/G#"),
    ("Eb:7/b7", "Eb7/Db"),
    ("G:maj/1", "G"),
    # nốt gốc viết bằng dấu giáng, trừ F#
    ("A#:min", "Bbm"),
    ("C#:maj", "Db"),
    ("D#:7", "Eb7"),
    ("G#:min7", "Abm7"),
    ("F#:min7/b7", "F#m7/E"),
    ("Gb:maj", "F#"),
    ("Db:maj/5", "Db/Ab"),
])
def test_harte_to_symbol(label, symbol):
    assert harte_to_symbol(label) == symbol


def test_spelling_follows_key():
    e_major = KeyEstimate("E", "major")
    assert harte_to_symbol("G#:min", e_major) == "G#m"
    assert harte_to_symbol("C#:min7", e_major) == "C#m7"
    f_major = KeyEstimate("F", "major")
    assert harte_to_symbol("A#:maj", f_major) == "Bb"
    gb_major = KeyEstimate("Gb", "major")
    assert harte_to_symbol("F#:maj", gb_major) == "Gb"
    # C trưởng / A thứ: bảng mặc định
    assert harte_to_symbol("A#:maj", KeyEstimate("A", "minor")) == "Bb"
    assert note_name(6, KeyEstimate("C", "major")) == "F#"
    # bậc mượn trong giọng thăng vẫn viết giáng
    g_major = KeyEstimate("G", "major")
    assert harte_to_symbol("A#:maj", g_major) == "Bb"
    assert harte_to_symbol("D#:maj", g_major) == "Eb"
    assert harte_to_symbol("A#:maj/3", g_major) == "Bb/D"
    assert harte_to_symbol("F#:dim", g_major) == "F#dim"
    assert harte_to_symbol("C#:min7", KeyEstimate("C#", "minor")) == "C#m7"
    assert note_name(1, KeyEstimate("D#", "minor")) == "C#"


def test_bass_note_letter_spelling():
    assert bass_note("C", "5") == "G"
    assert bass_note("A", "3") == "C#"
    assert bass_note("Bb", "3") == "D"
    assert bass_note("F#", "3") == "A#"
    assert bass_note("Ab", "b3") == "Cb"
    assert bass_note("C", "9") == "D"


def test_unparseable_label_returned_as_is():
    assert harte_to_symbol("garbage") == "garbage"


def _template_labels(name: str) -> list[str]:
    path = CNNLSTM_DIR / "data" / f"{name}_chord_list.txt"
    if not path.exists():
        pytest.skip(f"thiếu {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


@pytest.mark.parametrize("name", ["submission", "ismir2017", "extended"])
def test_cnnlstm_vocabularies_all_mapped(name):
    for label in _template_labels(name):
        sym = harte_to_symbol(label)
        assert ":" not in sym and "(" not in sym.replace("m(maj7)", ""), (label, sym)


def test_cnnlstm_submission_exact():
    expected = {
        "C:min/b7": "Cm/Bb", "C:min/2": "Cm/D", "C:maj/b7": "C/Bb", "C:maj/2": "C/D",
        "C:sus4(b7)": "C7sus4", "C:sus2": "Csus2", "C:sus4": "Csus4", "C:13": "C13", "C:11": "C11",
        "C:min9": "Cm9", "C:9": "C9", "C:maj9": "Cmaj9", "C:dim7": "Cdim7", "C:hdim7": "Cm7b5",
        "C:min7": "Cm7", "C:7": "C7", "C:maj7": "Cmaj7", "C:min/5": "Cm/G", "C:min/b3": "Cm/Eb",
        "C:maj/5": "C/G", "C:maj/3": "C/E", "C:dim": "Cdim", "C:aug": "Caug", "C:min": "Cm",
        "C:maj": "C", "N": "N.C.",
    }
    labels = _template_labels("submission")
    assert set(labels) == set(expected)
    for label in labels:
        assert harte_to_symbol(label) == expected[label]


def test_cnnlstm_full_vocabulary_never_crashes():
    for label in _template_labels("full"):
        assert harte_to_symbol(label)


def test_chordmini_vocabulary_all_mapped():
    if not CHORDMINI_DIR.exists():
        pytest.skip("thiếu ChordMini")
    from chordtrack.chords.chordmini import vocabulary

    vocab = vocabulary()
    symbols = [harte_to_symbol(label) for label in vocab]
    for label, sym in zip(vocab, symbols):
        assert ":" not in sym and "(" not in sym.replace("m(maj7)", ""), (label, sym)
    # 12 nốt x 14 loại + N + X, không có hai nhãn trùng ký hiệu
    assert len(set(symbols)) == len(vocab)
