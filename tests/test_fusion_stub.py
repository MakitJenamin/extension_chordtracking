from chordtrack.fusion import fill_gaps
from chordtrack.types import ChordSegment


def test_fill_gaps_covers_whole_duration():
    segs = [ChordSegment(0.5, 1.0, "C:maj"), ChordSegment(1.2, 2.0, "G:maj")]
    out = fill_gaps(segs, 3.0)
    assert out[0].start == 0.0 and out[-1].end == 3.0
    for a, b in zip(out, out[1:]):
        assert a.end == b.start
    assert [s.label for s in out] == ["N", "C:maj", "G:maj", "N"]
