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


def test_fix_double_time_run_and_extra_beats():
    from chordtrack.beats import fix_double_time

    g = 0.9
    normal = np.arange(0, 20 * g, g)                                # 20 phách đều
    double = normal[-1] + g / 2 * np.arange(1, 17)                  # 16 phách ở tốc độ gấp đôi
    tail = double[-1] + g * np.arange(1, 11)
    beats = np.concatenate([normal, double, tail])
    out = fix_double_time(beats)
    assert len(out) == 20 + 8 + 10
    assert np.allclose(np.diff(out), g)
    # một phách thừa lẻ tẻ (hai khoảng g/2 liền nhau): bỏ phách giữa
    extra = np.sort(np.append(normal, normal[7] + g / 2))
    assert np.allclose(fix_double_time(extra), normal)
    # không có gì sai thì giữ nguyên
    assert np.allclose(fix_double_time(normal), normal)


def test_lock_meter_keeps_4_4_through_two_beat_accents():
    from chordtrack.beats import lock_meter

    # 4/4 rõ trong 8 ô nhịp, rồi 8 ô nhịp model nhấn cứ 2 phách một lần (đổi hợp âm mỗi 2 phách)
    act = np.full(64, 0.05)
    act[0:32:4] = 0.9
    act[32::2] = 0.6
    down = lock_meter(act, 4, switch_penalty=6.0)
    assert np.array_equal(down, np.arange(0, 64, 4))


def test_lock_meter_follows_strong_evidence_of_real_bar_change():
    from chordtrack.beats import lock_meter

    # một ô 2/4 thật (bằng chứng rất mạnh) rồi 4/4 lệch pha: chấp nhận đổi
    act = np.full(42, 0.01)
    act[[0, 4, 8, 12, 16]] = 0.99
    act[[18, 22, 26, 30, 34, 38]] = 0.99
    down = lock_meter(act, 4, switch_penalty=6.0)
    assert list(down) == [0, 4, 8, 12, 16, 18, 22, 26, 30, 34, 38]


def test_meter_from_downbeats():
    from chordtrack.beats import meter_from_downbeats

    beats = np.arange(30, dtype=float)
    assert meter_from_downbeats(beats, beats[::3]) == 3
    assert meter_from_downbeats(beats, beats[:1]) == 4


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
