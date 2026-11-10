import json

import numpy as np
import pytest
import soundfile as sf

from chordtrack.melody import rmvpe
from chordtrack.types import MelodyModelOutput, MelodyNote


def _sine(path, hz=440.0, sec=2.0, sr=22050):
    t = np.arange(int(sec * sr)) / sr
    sf.write(path, 0.3 * np.sin(2 * np.pi * hz * t), sr)
    return path


def test_rmvpe_decode_peak_and_threshold():
    sal = np.zeros((3, 360), dtype=np.float32)
    sal[0, 100] = 0.9          # bin 100: 1997.38 + 2000 cent so với 10 Hz
    sal[1, 100] = 0.02         # dưới ngưỡng 0.03
    f0, f0_raw, peak = rmvpe.decode(sal)
    expect = 10 * 2 ** ((1997.3794084376191 + 2000) / 1200)
    assert f0[0] == pytest.approx(expect, rel=1e-4)
    assert f0[1] == 0 and f0_raw[1] == pytest.approx(expect, rel=1e-4)
    assert peak[1] == pytest.approx(0.02) and f0[2] == 0


def test_rmvpe_model_matches_weights():
    torch = pytest.importorskip("torch")
    if not rmvpe.WEIGHTS.exists():
        pytest.skip("chưa tải rmvpe.pt")
    keys = set(rmvpe._build_model().state_dict())
    assert keys == set(torch.load(rmvpe.WEIGHTS, map_location="cpu", weights_only=True))


@pytest.mark.gpu
def test_rmvpe_sine_440(tmp_path):
    if not rmvpe.WEIGHTS.exists():
        pytest.skip("chưa tải rmvpe.pt")
    out = rmvpe.transcribe(_sine(tmp_path / "a.wav"))
    pt = out.pitch_track
    assert np.allclose(np.diff(pt.times), 0.01)
    mid = pt.f0_hz[20:-20]
    assert (mid > 0).mean() > 0.95
    assert np.median(mid) == pytest.approx(440, rel=0.01)
    assert {"salience_max", "f0_raw_hz"} <= set(out.extras)


def test_rmvpe_chunking_matches_full(monkeypatch):
    """Chạy theo đoạn cho kết quả gần như chạy cả bài (CPU, model nhỏ ngẫu nhiên không cần weights)."""
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    model = rmvpe._build_model().eval()
    monkeypatch.setattr(rmvpe, "load_model", lambda: model)
    y = np.random.default_rng(0).standard_normal(16000 * 4).astype(np.float32) * 0.1
    full = rmvpe._salience(y, "cpu")
    monkeypatch.setattr(rmvpe, "CHUNK_FRAMES", 96)
    monkeypatch.setattr(rmvpe, "CONTEXT_FRAMES", 64)
    chunked = rmvpe._salience(y, "cpu")
    assert full.shape == chunked.shape == (len(y) // 160 + 1, 360)
    assert np.abs(full - chunked).max() < 0.05


def test_basic_pitch_sine_is_a4(tmp_path):
    pytest.importorskip("basic_pitch")
    from chordtrack.melody import basic_pitch

    out = basic_pitch.transcribe(_sine(tmp_path / "a.wav"))
    assert out.model_name == "basic_pitch"
    longest = max(out.notes, key=lambda n: n.end - n.start)
    assert longest.pitch == 69
    assert all(0 <= n.confidence <= 1 for n in out.notes)


def test_cache_runner_skips_existing(tmp_path, monkeypatch, capsys):
    from chordtrack.melody import io
    from eval.melody import cache_runner

    ds = tmp_path / "eval" / "toy"
    ds.mkdir(parents=True)
    rows = [{"track_id": "t1", "audio": str(tmp_path / "t1.wav")}, {"track_id": "t2", "audio": str(tmp_path / "t2.wav")}]
    (ds / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.setattr(cache_runner, "MELODY_EVAL_DIR", tmp_path / "eval")
    monkeypatch.setattr(io.paths, "CACHE_DIR", tmp_path / "cache")
    calls = []

    def fake(audio):
        calls.append(audio.name)
        return MelodyModelOutput("fake", [MelodyNote(0, 1, 60, 0.5)])

    assert cache_runner.main("fake", fake, []) == 0
    assert cache_runner.main("fake", fake, ["--dataset", "toy"]) == 0
    assert calls == ["t1.wav", "t2.wav"]
    back = io.load_output(io.cache_path("fake", "toy", "t1"))
    assert back.notes[0].pitch == 60 and "runtime_sec" in back.extras
    with pytest.raises(SystemExit):
        cache_runner.main("fake", fake, ["--dataset", "missing"])
