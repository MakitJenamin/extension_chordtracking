import mir_eval
import numpy as np
import pytest

from chordtrack.chords import cnnlstm
from chordtrack.paths import CNNLSTM_DIR, MAIN_ROOT, SAMPLE_AUDIO
from chordtrack.types import lab_to_segments

REF_LAB = MAIN_ROOT / "output" / "cnnlstm.lab"


def test_root_probs():
    rng = np.random.default_rng(0)
    triad = rng.random((5, 73))
    triad /= triad.sum(axis=1, keepdims=True)
    roots = cnnlstm._root_probs(triad)
    assert roots.shape == (5, 13)
    np.testing.assert_allclose(roots.sum(axis=1), 1.0)
    # nốt gốc D (pc 2) = D:maj (3) + D:min (15) + ... + D:aug (63)
    np.testing.assert_allclose(roots[:, 3], triad[:, [3, 15, 27, 39, 51, 63]].sum(axis=1))


def test_segments_merge_and_confidence():
    vocab = ["N", "C:maj", "G:maj"]
    tags = ["N", "C:maj", "C:maj", "G:maj"]
    probs = np.array([[0.5, 0.5, 0], [0, 1, 0], [0, 0.5, 0.5], [0, 0, 1]])
    segs = cnnlstm._segments(tags, 0.1, probs, vocab)
    assert [(s.label, s.start, s.end) for s in segs] == [
        ("N", 0.0, 0.1), ("C:maj", 0.1, pytest.approx(0.3)), ("G:maj", pytest.approx(0.3), pytest.approx(0.4))
    ]
    assert segs[1].confidence == pytest.approx(0.75)


def test_invalid_chord_dict():
    with pytest.raises(ValueError):
        cnnlstm.run_cnnlstm(SAMPLE_AUDIO, chord_dict="xyz")


@pytest.fixture(scope="module")
def output():
    if not SAMPLE_AUDIO.exists() or not CNNLSTM_DIR.exists():
        pytest.skip("thiếu audio thử hoặc chord-cnn-lstm-model")
    return cnnlstm.run_cnnlstm(SAMPLE_AUDIO)


@pytest.mark.gpu
def test_output_shapes(output):
    T = len(output.frame_times)
    V = len(output.vocabulary)
    assert output.frame_probs.shape == (T, V)
    assert len(set(output.vocabulary)) == V and output.vocabulary[0] == "N"
    for label in output.vocabulary:
        mir_eval.chord.encode(label)
    np.testing.assert_allclose(output.frame_probs.sum(axis=1), 1.0, atol=1e-6)
    for key, width in [("bass_probs", 13), ("root_probs", 13), ("triad_probs", 73)]:
        arr = output.extras[key]
        assert arr.shape == (T, width)
        np.testing.assert_allclose(arr.sum(axis=1), 1.0, atol=1e-4)
    for s in output.segments:
        assert 0.0 <= s.confidence <= 1.0


@pytest.mark.gpu
def test_matches_reference_lab(output):
    if not REF_LAB.exists():
        pytest.skip(f"không có {REF_LAB}")
    ref = lab_to_segments(REF_LAB)
    got = output.segments
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        assert g.label == r.label
        assert g.start == pytest.approx(r.start, abs=1e-3)
        assert g.end == pytest.approx(r.end, abs=1e-3)


@pytest.mark.gpu
def test_c_over_g_in_chorus(output):
    # Điệp khúc Am – C/G – F – C: khoảng giây 40–42 là C/G
    seg = next(s for s in output.segments if s.start <= 41.0 < s.end)
    assert seg.label == "C:maj/5"
    # nốt bass G (pc 7, cột 8) trội trong đoạn này
    i0, i1 = np.searchsorted(output.frame_times, [seg.start, seg.end])
    bass = output.extras["bass_probs"][i0:i1].mean(axis=0)
    assert int(np.argmax(bass)) == 8
