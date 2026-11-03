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


# ---------------------------------------------------------------- làm sạch phách, ô nhịp

from chordtrack import beats as B  # noqa: E402


def test_clean_beats_drops_extra_and_recenters_misplaced():
    beats = np.arange(0, 12, 0.6)
    extra = np.sort(np.append(beats, 3.3))                 # phách thừa giữa 3.0 và 3.6
    np.testing.assert_allclose(B.clean_beats(extra), beats)
    moved = beats.copy()
    moved[10] = 6.3                                         # phách 6.0 bị đặt lệch sang 6.3 (sát 6.6)
    moved = np.delete(moved, 11)                            # ... và phách 6.6 bị mất
    out = B.clean_beats(np.sort(np.append(moved, 6.45)))   # còn chèn thêm một phách sát bên
    assert np.all(np.diff(out) > 0.4)
    np.testing.assert_allclose(B.clean_beats(beats), beats)


def _synthetic(meter, n_bars, ibi=0.5, phase_shift_at=None):
    beats = np.arange(n_bars * meter) * ibi + 1.0
    pos = np.arange(len(beats)) % meter
    if phase_shift_at is not None:
        pos[phase_shift_at:] = (np.arange(len(beats) - phase_shift_at) + 0) % meter
    return beats, pos


def test_refine_downbeats_detects_three_four_from_activations():
    beats, pos = _synthetic(3, 40)
    probs = np.where(pos == 0, 0.8, 0.1)
    probs[[7, 20]] = 0.9                       # vài phách lẻ beat_this tưởng là phách mạnh
    downs, meter = B.refine_downbeats(beats, probs)
    assert meter == 3
    np.testing.assert_allclose(downs, beats[pos == 0])


def test_refine_downbeats_uses_chord_changes_when_activations_are_ambiguous():
    beats, pos = _synthetic(4, 30)
    probs = np.full(len(beats), 0.3)
    changes = beats[pos == 0][::2] + 0.05     # đổi hợp âm mỗi 2 ô, ở phách 1, lệch chút
    downs, meter = B.refine_downbeats(beats, probs, changes)
    assert meter == 4
    assert set(np.round(beats[pos == 0][::2], 3)) <= set(np.round(downs, 3))
    assert np.all(np.isin(np.round(downs, 3), np.round(beats[pos == 0], 3)))


def test_refine_downbeats_recovers_phase_shift():
    beats, pos = _synthetic(4, 40, phase_shift_at=82)   # ô thứ 21 chỉ có 2 phách, rồi về lại 4/4
    probs = np.where(pos == 0, 0.9, 0.05)
    downs, meter = B.refine_downbeats(beats, probs)
    assert meter == 4
    np.testing.assert_allclose(downs, beats[pos == 0])


def test_refine_downbeats_short_input():
    downs, meter = B.refine_downbeats(np.array([1.0]), None)
    assert meter == 4 and list(downs) == [1.0]


def test_beat_downbeat_probs_takes_local_max():
    frames = np.zeros(200)
    frames[51] = 0.9
    np.testing.assert_allclose(B.beat_downbeat_probs(np.array([1.0, 2.0]), frames), [0.9, 0.0])


def test_postprocess_cleans_and_refines():
    beats, pos = _synthetic(3, 30, ibi=0.6)
    frames = np.zeros(int(beats[-1] * B.FPS) + 10)
    frames[np.round(beats[pos == 0] * B.FPS).astype(int)] = 0.95
    raw = np.sort(np.append(beats, beats[10] + 0.3))
    tr = B.BeatTracking(BeatGrid(beats=raw, downbeats=raw[::4], tempo=100.0), frames)
    from chordtrack.types import ChordSegment
    segs = [ChordSegment(0, beats[3], "C:maj"), ChordSegment(beats[3], beats[9], "G:maj"),
            ChordSegment(beats[9], 60, "N")]
    assert B.chord_change_times(segs) == [beats[3]]
    grid, meter = B.postprocess(tr, segs)
    assert meter == 3
    np.testing.assert_allclose(grid.beats, beats)
    np.testing.assert_allclose(grid.downbeats, beats[pos == 0])
