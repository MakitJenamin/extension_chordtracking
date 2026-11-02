import csv

import mir_eval
import numpy as np

from chordtrack.types import ChordSegment
from eval import datasets as D


def test_song_name():
    assert D.song_name("11_-_Abbey_Road/02_-_Something.lab") == "11_02_Something"
    assert D.song_name("10CD1_-_The_Beatles/CD1_-_11_-_Black_Bird.lab") == "10CD1_11_Black_Bird"
    assert D.song_name("03_-_A_Hard_Day's_Night/07_-_Can't_Buy_Me_Love.lab") == "03_07_Can_t_Buy_Me_Love"


def test_core_title():
    assert D._core_title("The Beatles - Help!") == "Help!"
    assert D._core_title("Let It Be (Remastered 2009)") == "Let It Be"
    assert D._core_title("The Beatles - The Beatles - Hello, Goodbye (Remastered 2015)") == "Hello, Goodbye"
    assert D._core_title("The Beatles - Ticket To Ride - 2009 Remaster [Stereo]") == "Ticket To Ride"
    # "Help!" không được khớp nhầm với "With A Little Help From My Friends"
    assert D._norm("Help!") != D._norm(D._core_title("With A Little Help From My Friends (Remastered 2009)"))


def test_mapping_csv_is_valid():
    rows = D.read_mapping()
    assert len(rows) >= 20
    names = [r["name"] for r in rows]
    assert len(set(names)) == len(names)
    assert len({r["video_id"] for r in rows}) == len(rows)
    for r in rows:
        assert r["name"] == D.song_name(r["lab_file"])
        assert len(r["video_id"]) == 11
    with open(D.MAPPING_CSV, encoding="utf-8", newline="") as f:
        assert tuple(csv.DictReader(f).fieldnames) == D.CSV_FIELDS


def test_shift_segments_positive_offset_pads_with_n():
    segs = [ChordSegment(0.0, 1.0, "C:maj"), ChordSegment(1.0, 2.0, "G:maj")]
    out = D.shift_segments(segs, 0.5, audio_dur=2.2)
    assert [(s.start, s.end, s.label) for s in out] == [(0.0, 0.5, "N"), (0.5, 1.5, "C:maj"), (1.5, 2.2, "G:maj")]


def test_shift_segments_negative_offset_clips():
    segs = [ChordSegment(0.0, 0.3, "N"), ChordSegment(0.3, 2.0, "C:maj")]
    out = D.shift_segments(segs, -0.5, audio_dur=10.0)
    assert [(s.start, s.end, s.label) for s in out] == [(0.0, 1.5, "C:maj")]


def test_label_chroma_and_best_offset_recovers_shift():
    hop = D.ALIGN_HOP / D.ALIGN_SR
    rng = np.random.default_rng(0)
    roots = rng.integers(0, 12, size=40)
    qualities = rng.choice(["maj", "min", "7"], size=40)
    segs, t = [ChordSegment(0.0, 1.0, "N")], 1.0
    for r, q in zip(roots, qualities):
        d = float(rng.uniform(1.0, 3.0))
        segs.append(ChordSegment(t, t + d, f"{['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][r]}:{q}"))
        t += d
    n = int(t / hop) + 200
    lch, mask = D.label_chroma(segs, n, hop)
    assert not mask[: int(1.0 / hop) - 1].any()  # đoạn N đầu không có hợp âm
    _root, bitmap, _bass = mir_eval.chord.encode(segs[1].label)
    np.testing.assert_array_equal(lch[int(1.5 / hop)] > 0, np.roll(bitmap, roots[0]) > 0)

    # "Audio" giả: chroma mẫu trễ 0.4 giây, cộng nhiễu.
    lag_true = int(round(0.4 / hop))
    ach = np.roll(lch, lag_true, axis=0) + 0.3 * rng.random(lch.shape)
    ach /= np.linalg.norm(ach, axis=1, keepdims=True)
    lag, score, score0 = D.best_offset(ach, lch, mask, max_lag=int(2.0 / hop))
    assert lag == lag_true
    assert score > score0


def test_check_alignment_ignores_fade_out_length_but_rejects_drift():
    max_lag = 87
    # đuôi fade-out dài ngắn khác nhau (lệch 8 s) nhưng đầu và cuối bài cùng độ lệch: đạt
    assert D.check_alignment(-7.9, 10, max_lag, 0.62, 0.21, 0.21) == ""
    # độ lệch cuối bài khác đầu bài quá MAX_DRIFT: loại
    assert "cuối bài" in D.check_alignment(2.9, -4, max_lag, 0.65, -0.12, 1.58)
    assert "chạm biên" in D.check_alignment(0.0, max_lag, max_lag, 0.9, 0.0, 0.0)
    assert "khớp kém" in D.check_alignment(0.0, 0, max_lag, 0.3, 0.0, 0.0)
    assert "bản khác" in D.check_alignment(D.MAX_DURATION_DIFF + 1, 0, max_lag, 0.9, 0.0, 0.0)


def test_window_mask_keeps_only_window():
    mask = np.ones(100, dtype=bool)
    mask[10] = False
    w = D.window_mask(mask, 0.5, 2.0, 0.1)
    assert w.sum() == 14 and not w[10] and w[5] and not w[20]


def test_shift_segments_trims_to_common_end():
    segs = [ChordSegment(0.0, 5.0, "C:maj"), ChordSegment(5.0, 10.0, "G:maj")]
    out = D.shift_segments(segs, 0.0, 7.0)
    assert out[-1].end == 7.0 and len(out) == 2
