import numpy as np

from chordtrack.melody.io import load_output, save_output
from chordtrack.types import MelodyModelOutput, MelodyNote, PitchTrack


def test_roundtrip(tmp_path):
    out = MelodyModelOutput(
        "m",
        [MelodyNote(0.0, 1.0, 60, 0.5, "voice"), MelodyNote(1.0, 2.0, 62)],
        PitchTrack(np.arange(3.0), np.array([0.0, 220.0, 440.0]), np.array([0.1, 0.9, 0.8])),
        {"a": np.ones(2)},
    )
    p = tmp_path / "x.npz"
    save_output(out, p)
    back = load_output(p)
    assert back.model_name == "m"
    assert back.notes == out.notes
    assert np.allclose(back.pitch_track.f0_hz, out.pitch_track.f0_hz)
    assert np.allclose(back.extras["a"], 1)


def test_empty(tmp_path):
    p = tmp_path / "e.npz"
    save_output(MelodyModelOutput("e"), p)
    back = load_output(p)
    assert back.notes == [] and back.pitch_track is None
