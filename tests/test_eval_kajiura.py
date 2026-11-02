"""Test bộ đọc bản hợp âm ChordWiki (ChordPro -> Harte) và hàm chấm điểm theo chuỗi của eval/kajiura.py."""
from __future__ import annotations

import numpy as np
import pytest

from chordtrack.types import ChordSegment
from eval.kajiura import (
    bigram_f1,
    chord_sequence,
    dtw_align,
    lcs_length,
    parse_chord,
    parse_chordpro,
    reduce_label,
    score_sequence,
    segments_in_window,
    split_tokens,
    transpose,
    transpose_sequence,
)


@pytest.mark.parametrize("symbol, harte", [
    ("C", "C:maj"),
    ("Am", "A:min"),
    ("Am7", "A:min7"),
    ("G7", "G:7"),
    ("CM7", "C:maj7"),
    ("Cmaj7", "C:maj7"),
    ("C△7", "C:maj7"),
    ("DbM", "Db:maj"),
    ("F#m7-5", "F#:hdim7"),
    ("Bm7b5", "B:hdim7"),
    ("Cm7(b5)", "C:hdim7"),
    ("Cdim", "C:dim"),
    ("Cdim7", "C:dim7"),
    ("Caug", "C:aug"),
    ("C+5", "C:aug"),
    ("Csus4", "C:sus4"),
    ("Csus2", "C:sus2"),
    ("C7sus4", "C:sus4(b7)"),
    ("A7(sus4)", "A:sus4(b7)"),
    ("Cadd9", "C:maj(9)"),
    ("C(9)", "C:maj(9)"),
    ("Cm(9)", "C:min(9)"),
    ("Cmadd9", "C:min(9)"),
    ("C6", "C:maj6"),
    ("Cm6", "C:min6"),
    ("C69", "C:maj6(9)"),
    ("C6/9", "C:maj6(9)"),
    ("C9", "C:9"),
    ("C7(9)", "C:9"),
    ("CM7(9)", "C:maj9"),
    ("Eb7(b9)", "Eb:7(b9)"),
    ("B7#9", "B:7(#9)"),
    ("C7-5", "C:7(*5,b5)"),
    ("G5", "G:5"),
    ("CmM7", "C:minmaj7"),
    # Hợp âm đảo: /X và onX (kiểu Nhật)
    ("C/E", "C:maj/3"),
    ("ConE", "C:maj/3"),
    ("Am7onG", "A:min7/b7"),
    ("Am/C", "A:min/b3"),
    ("G#/B#", "G#:maj/3"),
    ("C/C", "C:maj"),
    # Ký tự toàn chiều rộng, ngoặc, N.C.
    ("Ｃｍ７", "C:min7"),
    ("(Bm)", "B:min"),
    ("N.C.", "N"),
    ("N.C", "N"),
    ("NC", "N"),
])
def test_parse_chord(symbol, harte):
    assert parse_chord(symbol) == harte


def test_parsed_labels_are_valid_for_mir_eval():
    import mir_eval

    for s in ("CM7(9)", "C7-5", "A7(sus4)", "Bb5(add9)", "D(11)", "Gaug(b9)", "Dsus4(9)", "Am7onG"):
        mir_eval.chord.encode(parse_chord(s))


def test_split_tokens_drops_bars_and_annotations():
    assert split_tokens("| Am | F G |") == ["Am", "F", "G"]
    assert split_tokens("( E /F# ) C#m7") == ["E", "/F#", "C#m7"]
    assert split_tokens("C7 (Dr.fill) (break) N.C. (↓) ～ >") == ["C7", "N.C."]
    assert split_tokens("C-G") == ["C", "G"]
    assert split_tokens("Cm7-5") == ["Cm7-5"]


def test_parse_chordpro():
    text = "\n".join([
        "{yt:abcdefghijk}",
        "{key:Am}",
        "{c:Intro}",
        "[Am G6 F G]",
        "[C /B Am] x2",
        "{key:F#m}",
        "[N.C.][F#m ConE]",
        "[Dr.fill]",
    ])
    sh = parse_chordpro(text, "demo")
    assert sh.meta["yt"] == "abcdefghijk"
    assert sh.keys == [(0, "Am"), (10, "F#m")]
    assert sh.sections == [(0, "Intro")]
    assert sh.chords == ["A:min", "G:maj6", "F:maj", "G:maj",
                         "C:maj", "C:maj/7", "A:min", "C:maj", "C:maj/7", "A:min",
                         "N", "F#:min", "C:maj/3"]
    assert sh.parse_rate == 1.0


def test_parse_chordpro_counts_unparsed():
    sh = parse_chordpro("[C Cm7(q) G]")
    assert sh.chords == ["C:maj", "G:maj"]
    assert sh.unparsed["Cm7(q)"] == 1
    assert sh.parse_rate == pytest.approx(2 / 3)


def test_reduce_label():
    assert reduce_label("C:maj7/3", "majmin") == "C:maj"
    assert reduce_label("A:hdim7", "majmin") == "A:min"
    assert reduce_label("G:sus4(b7)", "majmin") == "G:sus"
    assert reduce_label("C:maj7", "sevenths") == "C:maj7"
    assert reduce_label("G:9", "sevenths") == "G:7"
    assert reduce_label("B:hdim7", "sevenths") == "B:hdim7"
    assert reduce_label("B:dim7", "sevenths") == "B:dim7"
    assert reduce_label("C:sus4(7)", "sevenths") == "C:susmaj7"
    assert reduce_label("C:sus4(b7)", "sevenths") == "C:sus7"
    assert reduce_label("C:maj(9)", "sevenths") == "C:maj"
    assert reduce_label("C#:min7", "sevenths") == "Db:min7"
    assert reduce_label("N", "majmin") is None
    assert reduce_label("X", "majmin") is None


def test_chord_sequence_collapses_repeats_and_drops_n():
    labels = ["C:maj", "C:maj7", "N", "C:maj", "A:min", "A:min7", "F:maj"]
    assert chord_sequence(labels, "majmin") == ["C:maj", "A:min", "F:maj"]
    assert chord_sequence(labels, "sevenths") == ["C:maj", "C:maj7", "C:maj", "A:min", "A:min7", "F:maj"]


def test_transpose():
    assert transpose("C:maj/3", 2) == "D:maj/3"
    assert transpose("B:min7", 1) == "C:min7"
    assert transpose("N", 5) == "N"
    assert transpose("F#:hdim7", 12) == "F#:hdim7"


def test_lcs_length():
    assert lcs_length(list("ABCBDAB"), list("BDCABA")) == 4
    assert lcs_length([], ["A"]) == 0
    assert lcs_length(["A"] * 100, ["A"] * 70) == 70
    rng = np.random.default_rng(0)
    for _ in range(20):    # so với quy hoạch động thường
        a = list(rng.choice(list("ABCD"), size=rng.integers(1, 30)))
        b = list(rng.choice(list("ABCD"), size=rng.integers(1, 30)))
        dp = np.zeros((len(a) + 1, len(b) + 1), dtype=int)
        for i in range(len(a)):
            for j in range(len(b)):
                dp[i + 1, j + 1] = dp[i, j] + 1 if a[i] == b[j] else max(dp[i, j + 1], dp[i + 1, j])
        assert lcs_length(a, b) == dp[-1, -1]


def test_score_identical_sequences_is_one():
    seq = ["C:maj", "G:maj", "A:min", "F:maj"] * 4
    res = score_sequence(seq, seq)
    assert res["f1"] == 1.0 and res["precision"] == 1.0 and res["recall"] == 1.0
    assert res["bigram_f1"] == 1.0
    assert res["shift"] == 0


def test_score_transposed_matches_only_when_shift_allowed():
    ref = ["C:maj", "G:maj", "A:min", "F:maj", "D:min", "E:maj"]
    est = transpose_sequence(ref, 3)        # đáp án ghi theo capo 3
    assert score_sequence(ref, est)["f1"] < 0.5
    res = score_sequence(ref, est, shifts=range(12))
    assert res["f1"] == 1.0 and res["shift"] == 3


def test_score_partial_match():
    ref = ["C:maj", "G:maj", "A:min", "F:maj"]
    est = ["C:maj", "E:min", "G:maj", "A:min", "F:maj"]   # thêm một hợp âm
    res = score_sequence(ref, est)
    assert res["recall"] == 1.0
    assert res["precision"] == pytest.approx(4 / 5)
    assert bigram_f1(ref, est) == pytest.approx(2 * 2 / (3 + 4))


def test_segments_in_window():
    segs = [ChordSegment(0, 2, "C:maj"), ChordSegment(2, 4, "G:maj"), ChordSegment(4, 6, "A:min")]
    assert segments_in_window(segs, 2.5, 4.5) == ["G:maj", "A:min"]


def test_dtw_align_recovers_order():
    # 3 hợp âm, mỗi hợp âm khớp tốt với một vùng; đầu và cuối là vùng "ngoài" (chi phí cao với mọi hợp âm).
    T = 40
    cost = np.full((3, T), 0.9)
    cost[0, 5:15] = 0.1
    cost[1, 15:25] = 0.1
    cost[2, 25:35] = 0.1
    total, path = dtw_align(cost, outside=0.5)
    assert list(path[:5]) == [0] * 5
    assert set(path[6:14]) == {1} and set(path[16:24]) == {2} and set(path[26:34]) == {3}
    assert list(path[36:]) == [4] * 4
    assert np.all(np.diff(path) >= 0)
