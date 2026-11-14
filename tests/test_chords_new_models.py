"""Test cho wrapper ChordFormer (phần CPU; phần GPU đánh dấu `gpu`)."""
import numpy as np
import pytest

from chordtrack.chords import chordformer
from chordtrack.paths import CHORDFORMER_DIR, SAMPLE_AUDIO


def test_chordformer_invalid_chord_dict():
    with pytest.raises(ValueError):
        chordformer.run_chordformer(SAMPLE_AUDIO, chord_dict="xyz")


@pytest.mark.gpu
@pytest.mark.skipif(not (CHORDFORMER_DIR / "cache_data").exists() or not SAMPLE_AUDIO.exists(),
                    reason="thiếu checkpoint ChordFormer hoặc audio mẫu")
def test_chordformer_let_it_be():
    out = chordformer.run_chordformer(SAMPLE_AUDIO)
    assert out.frame_probs.shape[1] == len(out.vocabulary)
    np.testing.assert_allclose(out.frame_probs.sum(axis=1), 1.0, atol=1e-6)
    assert out.extras["bass_probs"].shape == (len(out.frame_times), 13)
    labels = {s.label for s in out.segments if 30 < s.start < 60}
    assert {"C:maj", "G:maj", "A:min", "F:maj"} <= labels
