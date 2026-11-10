import numpy as np
import pytest

from chordtrack.melody import yourmt3


def test_source_name():
    names = {0: "Acoustic Piano", 48: "String Ensemble", 64: "Soprano/Alto Sax", 52: "Choir and Voice"}
    assert yourmt3.source_name(100, False, names) == "voice"
    assert yourmt3.source_name(101, False, names) == "voice_chorus"
    assert yourmt3.source_name(36, True, names) == "drums"
    assert yourmt3.source_name(0, False, names) == "acoustic_piano"
    assert yourmt3.source_name(48, False, names) == "string_ensemble"
    assert yourmt3.source_name(64, False, names) == "soprano_alto_sax"
    assert yourmt3.source_name(52, False, names) == "choir_and_voice"
    assert yourmt3.source_name(7, False, names) == "program_7"


@pytest.mark.gpu
def test_transcribe_synthetic(tmp_path):
    """Âm sắc gần piano: C4 – E4 – G4, mỗi nốt 1 s. Phải nhận ra ít nhất hai trong ba cao độ."""
    sf = pytest.importorskip("soundfile")
    if not yourmt3.CHECKPOINT.exists():
        pytest.skip("thiếu checkpoint YourMT3")
    from chordtrack import gpu

    sr = 16000
    t = np.arange(sr) / sr
    env = np.exp(-3 * t)
    parts = []
    for midi in (60, 64, 67):
        f = 440 * 2 ** ((midi - 69) / 12)
        tone = sum(np.sin(2 * np.pi * f * k * t) / k for k in range(1, 6)) * env
        parts.append(tone)
    y = np.concatenate(parts + [np.zeros(sr)]) * 0.3
    p = tmp_path / "c_e_g.wav"
    sf.write(p, y.astype(np.float32), sr)
    with gpu.gpu_lock():
        try:
            out = yourmt3.transcribe(p)
        finally:
            yourmt3.offload()
    assert out.model_name == "yourmt3"
    pitched = {n.pitch for n in out.notes if n.source != "drums"}
    assert len(pitched & {60, 64, 67}) >= 2, out.notes
    assert all(0 <= n.start < n.end <= 4.0 + 1e-6 for n in out.notes)
    assert len(out.extras["note_program"]) == len(out.notes)
