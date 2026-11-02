"""Nhận giọng theo vùng (đổi giọng, nâng tông) và liên kết đoạn với vùng giọng."""
import numpy as np
import pytest

from chordtrack import analysis as A
from chordtrack.transpose import transpose_label
from chordtrack.types import BeatGrid, ChordSegment, KeyEstimate, KeyRegion, Section

BAR = 2.0   # 120 BPM, 4/4: một hợp âm mỗi ô


def _song(blocks: list[tuple[list[str], int]]) -> tuple[list[ChordSegment], BeatGrid, float]:
    segs, t = [], 0.0
    for labels, repeat in blocks:
        for lab in labels * repeat:
            segs.append(ChordSegment(t, t + BAR, lab))
            t += BAR
    beats = np.arange(0.0, t, BAR / 4)
    return segs, BeatGrid(beats=beats, downbeats=beats[::4], tempo=120.0), t


def _names(regions):
    return [(r.key.tonic, r.key.mode) for r in regions]


POP = ["C:maj", "A:min", "F:maj", "G:maj"]          # I–vi–IV–V
POP2 = ["C:maj", "G:maj", "A:min", "F:maj"]         # I–V–vi–IV
MINOR = ["A:min", "F:maj", "C:maj", "G:maj"]        # i–VI–III–VII


@pytest.mark.parametrize("prog, n_before, n_after", [
    (POP, 4, 4), (POP, 4, 3), (POP, 8, 4), (POP2, 4, 4), (POP2, 6, 3), (MINOR, 4, 4), (MINOR, 8, 4),
])
def test_key_change_up_two_semitones(prog, n_before, n_after):
    up = [transpose_label(lab, 2) for lab in prog]
    segs, grid, dur = _song([(prog, n_before), (up, n_after)])
    regions = A.estimate_key_regions(segs, grid, dur)
    tonic0 = prog[0].split(":")[0] if prog is not MINOR else "A"
    mode = "minor" if prog is MINOR else "major"
    assert len(regions) == 2
    assert _names(regions) == [(tonic0, mode), (transpose_label(tonic0, 2), mode)]
    t_mod = n_before * len(prog) * BAR
    assert abs(regions[1].start - t_mod) < BAR          # lệch dưới 1 ô nhịp
    assert regions[0].start == 0.0 and regions[-1].end == pytest.approx(dur)
    assert regions[0].end == regions[1].start


def test_key_change_without_beats_uses_fallback_grid():
    up = [transpose_label(lab, 2) for lab in POP]
    segs, _, dur = _song([(POP, 4), (up, 4)])
    regions = A.estimate_key_regions(segs, None, dur)
    assert _names(regions) == [("C", "major"), ("D", "major")]
    assert abs(regions[1].start - 32.0) < BAR


def test_single_key_song_is_one_region_equal_to_estimate_key():
    # kiểu "Let It Be": I–V–vi–IV, có I/5 và bVII ở điệp khúc
    verse = ["C:maj", "G:maj", "A:min", "F:maj", "C:maj", "G:maj", "F:maj", "C:maj"]
    chorus = ["A:min", "C:maj/5", "F:maj", "C:maj", "C:maj", "G:maj", "F:maj", "C:maj"]
    bridge = ["F:maj", "C:maj", "Bb:maj", "F:maj", "C:maj"]
    segs, grid, dur = _song([(verse, 2), (chorus, 2), (verse, 2), (bridge, 1), (chorus, 2), (verse, 1)])
    regions = A.estimate_key_regions(segs, grid, dur)
    assert len(regions) == 1
    assert regions[0].key == A.estimate_key(segs)
    assert _names(regions) == [("C", "major")]
    assert (regions[0].start, regions[0].end) == (0.0, pytest.approx(dur))


def test_short_detour_is_not_a_region():
    # 2 ô hợp âm lạ giữa bài (ngắn hơn MIN_REGION_BARS) không tạo vùng giọng mới
    segs, grid, dur = _song([(POP, 4), (["Eb:maj", "Bb:maj"], 1), (POP, 4)])
    regions = A.estimate_key_regions(segs, grid, dur)
    assert _names(regions) == [("C", "major")]


def test_regions_empty_and_no_chords():
    assert A.estimate_key_regions([], None, 10.0) == []
    assert A.estimate_key_regions([ChordSegment(0, 10, "N")], None, 10.0) == []
    assert A.estimate_key_regions([ChordSegment(0, 10, "C:maj")], None, 0.0) == []


def test_regions_cover_duration_and_boundary_snap():
    up = [transpose_label(lab, 2) for lab in POP]
    segs, grid, dur = _song([(POP, 4), (up, 4)])
    # ranh giới đoạn ở 32 s được ưu tiên
    regions = A.estimate_key_regions(segs, grid, dur, boundaries=[16.0, 32.0, 48.0])
    assert regions[1].start == pytest.approx(32.0)
    for a, b in zip(regions, regions[1:]):
        assert a.end == b.start


def test_main_region_key_and_region_index():
    regs = [KeyRegion(0, 100, KeyEstimate("C", "major")), KeyRegion(100, 130, KeyEstimate("D", "major")),
            KeyRegion(130, 150, KeyEstimate("C", "major"))]
    assert A.main_region_key(regs).tonic == "C"
    assert A.main_region_key([]) is None
    assert A.region_index(regs, 10, 20) == 0
    assert A.region_index(regs, 95, 125) == 1      # giao nhiều nhất
    assert A.region_index(regs, 140, 140) == 2     # đoạn rỗng: vùng chứa điểm đầu
    assert A.region_index(regs, 200, 200) == 2
    assert A.region_index([], 0, 1) is None


def test_relate_sections_finds_transposed_chorus():
    regs = [KeyRegion(0, 32, KeyEstimate("C", "major")), KeyRegion(32, 48, KeyEstimate("D", "major"))]
    secs = [
        Section(0, 16, "verse", ["A:min", "F:maj", "G:maj", "C:maj"], []),
        Section(16, 32, "chorus", ["C:maj", "G:maj", "A:min", "F:maj"], []),
        Section(32, 48, "B", ["D:maj", "A:maj", "B:min", "G:maj"], []),
    ]
    info = A.relate_sections_to_keys(secs, regs)
    assert [i["key_index"] for i in info] == [0, 0, 1]
    assert info[2]["roman"] == ["I", "V", "vi", "IV"]
    assert info[2]["transposed_from"] == 1 and info[2]["semitones"] == 2
    assert info[0]["transposed_from"] is None and info[1]["transposed_from"] is None


def test_relate_sections_rotation_and_no_false_match():
    regs = [KeyRegion(0, 20, KeyEstimate("A", "minor")), KeyRegion(20, 40, KeyEstimate("B", "minor")),
            KeyRegion(40, 60, KeyEstimate("C", "major"))]
    secs = [
        Section(0, 20, "A", ["A:min", "F:maj", "C:maj", "G:maj"], []),
        Section(20, 40, "B", ["D:maj", "A:maj", "B:min", "G:maj"], []),    # III–VII–i–VI: xoay vòng, B thứ
        Section(40, 60, "C", ["A:min", "F:maj", "C:maj", "G:maj"], []),    # vùng C trưởng: khác mode, không tính
    ]
    info = A.relate_sections_to_keys(secs, regs)
    assert info[1]["transposed_from"] == 0 and info[1]["semitones"] == 2
    assert info[2]["transposed_from"] is None


def test_relate_sections_half_step_up_read_as_relative_minor():
    # "And I Love Her": verse ở E trưởng, đoạn solo nâng nửa cung nhưng vùng bị đọc là D thứ (tương đối của F)
    regs = [KeyRegion(0, 30, KeyEstimate("E", "major")), KeyRegion(30, 60, KeyEstimate("D", "minor"))]
    secs = [Section(0, 30, "verse", ["F#:min", "C#:min", "A:maj", "B:maj"], []),
            Section(30, 60, "C", ["G:min", "D:min", "Bb:maj", "C:7"], [])]
    info = A.relate_sections_to_keys(secs, regs)
    assert info[1]["transposed_from"] == 0 and info[1]["semitones"] == 1
    assert info[1]["roman"] == ["iv", "i", "VI", "VII7"]


def _key(tonic, mode="major"):
    return KeyEstimate(tonic=tonic, mode=mode, confidence=1.0)


def test_merge_weak_regions_short_or_one_chord():
    # Kiểu "Nơi này có anh": intro chỉ một hợp âm Ab (9,6 s) trước cả bài C trưởng -> một vùng C
    segs, _, dur = _song([(["Ab:maj"], 4), (POP, 6)])
    regions = [KeyRegion(0.0, 8.0, _key("Ab")), KeyRegion(8.0, dur, _key("C"))]
    out = A._merge_weak_regions(segs, regions)
    assert [(r.start, r.end, r.key.tonic) for r in out] == [(0.0, dur, "C")]


def test_merge_weak_regions_keeps_real_modulation_and_joins_same_key():
    up = [transpose_label(lab, 2) for lab in POP]
    segs, _, dur = _song([(POP, 3), (["Eb:maj", "Bb:maj"], 1), (POP, 1), (up, 3)])
    # vùng Eb ngắn (4 s) kẹp giữa hai vùng C: gộp vào C, rồi nối hai vùng C liền nhau; vùng D (24 s) giữ nguyên
    regions = [KeyRegion(0.0, 24.0, _key("C")), KeyRegion(24.0, 28.0, _key("Eb")),
               KeyRegion(28.0, 36.0 + 0.0, _key("C")), KeyRegion(36.0, dur, _key("D"))]
    out = A._merge_weak_regions(segs, regions)
    assert [(r.start, r.key.tonic) for r in out] == [(0.0, "C"), (36.0, "D")]
    assert out[-1].end == dur


def test_merge_weak_regions_single_region_untouched():
    segs, _, dur = _song([(["C:maj"], 2)])
    regions = [KeyRegion(0.0, dur, _key("C"))]
    assert A._merge_weak_regions(segs, regions) == regions
