import numpy as np
import pytest

from chordtrack.beats import beat_synchronous_segments, estimate_tempo
from chordtrack.types import BeatGrid


def _grid(beats):
    b = np.asarray(beats, dtype=float)
    return BeatGrid(beats=b, downbeats=b[::4], tempo=estimate_tempo(b))


def _check_cover(segs, duration):
    assert segs[0][0] == 0.0
    assert segs[-1][1] == duration
    for (a, b), (c, _) in zip(segs[:-1], segs[1:]):
        assert b == c
    assert all(b > a for a, b in segs)


def test_segments_basic():
    segs = beat_synchronous_segments(_grid([0.5, 1.0, 1.5]), 2.0)
    assert segs == [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]
    _check_cover(segs, 2.0)


def test_segments_beat_at_zero_and_end():
    segs = beat_synchronous_segments(_grid([0.0, 1.0, 2.0]), 2.0)
    assert segs == [(0.0, 1.0), (1.0, 2.0)]


def test_segments_drop_out_of_range_and_duplicates():
    segs = beat_synchronous_segments(_grid([-0.2, 0.5, 0.5, 1.0, 3.0]), 2.0)
    assert segs == [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0)]


def test_segments_no_beats():
    assert beat_synchronous_segments(_grid([]), 3.0) == [(0.0, 3.0)]
    assert beat_synchronous_segments(_grid([1.0]), 0.0) == []


def test_tempo_from_median():
    beats = np.arange(0, 10, 60 / 72)
    beats[5] += 0.1  # một phách lệch không làm sai trung vị
    assert estimate_tempo(beats) == pytest.approx(72, abs=0.5)
    assert estimate_tempo(np.array([1.0])) == 0.0


@pytest.mark.gpu
def test_track_beats_sample_30s():
    from chordtrack.beats import track_beats
    from chordtrack.paths import SAMPLE_AUDIO

    if not SAMPLE_AUDIO.exists():
        pytest.skip("không có bài thử")
    grid = track_beats(SAMPLE_AUDIO, max_duration=30.0)
    assert np.all(np.diff(grid.beats) > 0)
    assert np.all(grid.beats < 30.0)
    assert np.isin(grid.downbeats, grid.beats).all()
    assert 60 < grid.tempo < 85  # Let It Be khoảng 72-73 BPM
    assert len(grid.beats) >= 25
    ratio = len(grid.downbeats) / len(grid.beats)
    assert 0.18 < ratio < 0.32
    _check_cover(beat_synchronous_segments(grid, 30.0), 30.0)
