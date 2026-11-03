"""Test cho wrapper ChordFormer và consonance-ACE (phần CPU; phần GPU đánh dấu `gpu`)."""
import numpy as np
import pytest

from chordtrack.chords import chordformer, consonance
from chordtrack.paths import CHORDFORMER_DIR, CONSONANCE_DIR, SAMPLE_AUDIO


def test_consonance_label_logprobs_prefers_matching_notes():
    vocab = ["N", "C:maj", "C:min", "A:min", "X"]
    root = np.zeros((2, 13))
    root[0, 0] = 1.0          # gốc C
    root[1, 12] = 1.0         # N
    notes = np.full((2, 12), 0.01)
    notes[0, [0, 4, 7]] = 0.99   # C E G
    lp = consonance.label_logprobs(root, notes, vocab)
    assert lp.shape == (2, 5)
    assert np.argmax(lp[0]) == 1          # C:maj
    assert lp[0, 1] > lp[0, 2]            # C:maj hơn C:min (E chứ không phải Eb)
    assert np.argmax(lp[1]) == 0          # N
    assert np.isneginf(lp[:, 4]).all()    # X bị loại


def test_consonance_bass_layout():
    p = np.zeros((1, 13))
    p[0, 4] = 1.0     # E (C..B = 0..11)
    p2 = np.zeros((1, 13))
    p2[0, 12] = 1.0   # N
    assert consonance._to_ours(p)[0, 1 + 4] == 1.0
    assert consonance._to_ours(p2)[0, 0] == 1.0


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


@pytest.mark.gpu
@pytest.mark.skipif(not (CONSONANCE_DIR / "ACE" / "checkpoints").exists() or not SAMPLE_AUDIO.exists(),
                    reason="thiếu checkpoint consonance-ACE hoặc audio mẫu")
def test_consonance_let_it_be():
    out = consonance.run_consonance(SAMPLE_AUDIO)
    assert out.frame_probs.shape == (len(out.frame_times), len(out.vocabulary))
    np.testing.assert_allclose(out.frame_probs.sum(axis=1), 1.0, atol=1e-6)
    assert out.frame_times[-1] < 244
    labels = {s.label for s in out.segments if 30 < s.start < 60 and s.end - s.start > 0.5}
    assert {"C:maj", "G:maj", "A:min"} <= labels
