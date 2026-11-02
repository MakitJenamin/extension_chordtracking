import mir_eval
import numpy as np
import pytest

from chordtrack.chords import chordmini
from chordtrack.paths import CHORDMINI_DIR, MAIN_ROOT, SAMPLE_AUDIO
from chordtrack.types import lab_to_segments

REF_LAB = MAIN_ROOT / "output" / "chordmini_btc" / "QDYfEBY9NM4.lab"

needs_repo = pytest.mark.skipif(not CHORDMINI_DIR.exists(), reason="chưa có third_party_src/ChordMini")


def test_to_harte():
    assert chordmini.to_harte("C") == "C:maj"
    assert chordmini.to_harte("A:min") == "A:min"
    assert chordmini.to_harte("F#:maj7") == "F#:maj7"
    assert chordmini.to_harte("N") == "N"
    assert chordmini.to_harte("X") == "X"


@needs_repo
def test_vocabulary_is_valid_harte():
    vocab = chordmini.vocabulary()
    assert len(vocab) == 170 and len(set(vocab)) == 170
    assert vocab[1] == "C:maj" and vocab[0] == "C:min" and vocab[-1] == "N"
    for label in vocab:
        mir_eval.chord.encode(label)  # ném lỗi nếu không phải Harte hợp lệ


def test_segments_drop_short_and_confidence():
    vocab = ["A", "B", "C"]
    preds = np.array([0] * 10 + [1] * 2 + [2] * 10)
    probs = np.full((22, 3), 1 / 3)
    probs[:10, 0] = 0.8
    segs = chordmini._segments_from_predictions(preds, 0.1, probs, vocab)
    # đoạn "B" dài 0.2 s < 0.5 s bị bỏ, để lại khoảng trống như script gốc
    assert [s.label for s in segs] == ["A", "C"]
    assert segs[0].confidence == pytest.approx(0.8)
    assert segs[1].start == pytest.approx(1.2)


def test_invalid_variant():
    with pytest.raises(ValueError):
        chordmini.run_chordmini(SAMPLE_AUDIO, variant="xyz")


@pytest.fixture(scope="module")
def btc_output():
    if not SAMPLE_AUDIO.exists() or not CHORDMINI_DIR.exists():
        pytest.skip("thiếu audio thử hoặc ChordMini")
    return chordmini.run_chordmini(SAMPLE_AUDIO, variant="btc")


@pytest.mark.gpu
def test_btc_output_shapes(btc_output):
    out = btc_output
    T = len(out.frame_times)
    assert out.frame_probs.shape == (T, 170)
    assert len(out.vocabulary) == 170
    np.testing.assert_allclose(out.frame_probs.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(np.diff(out.frame_times) > 0)
    assert out.frame_times[-1] < 244.0
    for s in out.segments:
        assert 0.0 <= s.confidence <= 1.0
        assert s.label in out.vocabulary


@pytest.mark.gpu
def test_btc_matches_reference_lab(btc_output):
    if not REF_LAB.exists():
        pytest.skip(f"không có {REF_LAB}")
    ref = lab_to_segments(REF_LAB)
    got = btc_output.segments
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        assert g.label == chordmini.to_harte(r.label)
        assert g.start == pytest.approx(r.start, abs=1e-3)
        assert g.end == pytest.approx(r.end, abs=1e-3)


@pytest.mark.gpu
def test_btc_finds_let_it_be_progression(btc_output):
    labels = [s.label for s in btc_output.segments[:4]]
    assert labels == ["C:maj", "G:maj", "A:min", "F:maj7"]


@pytest.mark.gpu
def test_2e1d_runs():
    if not SAMPLE_AUDIO.exists() or not CHORDMINI_DIR.exists():
        pytest.skip("thiếu audio thử hoặc ChordMini")
    out = chordmini.run_chordmini(SAMPLE_AUDIO, variant="2e1d")
    np.testing.assert_allclose(out.frame_probs.sum(axis=1), 1.0, atol=1e-6)
    labels = {s.label for s in out.segments}
    assert {"C:maj", "G:maj", "A:min"} <= labels
