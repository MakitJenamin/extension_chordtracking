from chordtrack.types import ChordSegment, lab_to_segments, segments_to_lab


def test_lab_roundtrip(tmp_path):
    segs = [ChordSegment(0.0, 1.5, "C:maj"), ChordSegment(1.5, 3.0, "G:maj/3")]
    p = tmp_path / "x.lab"
    segments_to_lab(segs, p)
    back = lab_to_segments(p)
    assert [(s.start, s.end, s.label) for s in back] == [(0.0, 1.5, "C:maj"), (1.5, 3.0, "G:maj/3")]
