import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from chordtrack import paths, separation
from chordtrack.separation import SAMPLE_RATE, normalize_audio, separate

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="cần ffmpeg trong PATH")


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


@needs_ffmpeg
def test_normalize_audio_to_44k_stereo(tmp_path):
    sr = 22050
    t = np.arange(sr) / sr
    src = tmp_path / "in.wav"
    sf.write(src, 0.3 * np.sin(2 * np.pi * 440 * t), sr)  # mono 22.05 kHz
    dst = tmp_path / "mix.wav"
    normalize_audio(src, dst)
    info = sf.info(dst)
    assert info.samplerate == SAMPLE_RATE and info.channels == 2
    assert abs(info.duration - 1.0) < 0.01
    assert not list(tmp_path.glob("*.part.wav"))


def test_cache_skips_existing_outputs(tmp_path, monkeypatch):
    for name in ["mix", "vocals", "instrumental", "bass", "no_bass"]:
        sf.write(tmp_path / f"{name}.wav", np.zeros((10, 2)), SAMPLE_RATE)

    def boom(*a, **k):
        raise AssertionError("không được chạy lại bước đã có cache")

    monkeypatch.setattr(separation, "normalize_audio", boom)
    monkeypatch.setattr(separation, "_run_roformer", boom)
    monkeypatch.setattr(separation, "_run_demucs", boom)
    stems = separate(tmp_path / "khong_ton_tai.wav", tmp_path)
    assert stems.bass == tmp_path / "bass.wav"
    assert stems.no_bass == tmp_path / "no_bass.wav"


def test_cache_reruns_only_missing_step(tmp_path, monkeypatch):
    for name in ["mix", "vocals", "instrumental"]:
        sf.write(tmp_path / f"{name}.wav", np.zeros((10, 2)), SAMPLE_RATE)
    calls = []

    def fake_demucs(instrumental, bass, no_bass):
        calls.append("demucs")
        for p in (bass, no_bass):
            sf.write(p, np.zeros((10, 2)), SAMPLE_RATE)

    monkeypatch.setattr(separation, "_run_roformer", lambda *a: calls.append("roformer"))
    monkeypatch.setattr(separation, "_run_demucs", fake_demucs)
    separate(tmp_path / "x.wav", tmp_path)
    assert calls == ["demucs"]


def test_rerun_vocal_step_invalidates_bass_step(tmp_path, monkeypatch):
    # instrumental.wav bị mất: Roformer chạy lại, nên bass/no_bass cũ phải được tạo lại theo
    for name in ["mix", "vocals", "bass", "no_bass"]:
        sf.write(tmp_path / f"{name}.wav", np.zeros((10, 2)), SAMPLE_RATE)
    calls = []

    def fake(step):
        def run(_, a, b):
            calls.append(step)
            for p in (a, b):
                sf.write(p, np.zeros((10, 2)), SAMPLE_RATE)
        return run

    monkeypatch.setattr(separation, "_run_roformer", fake("roformer"))
    monkeypatch.setattr(separation, "_run_demucs", fake("demucs"))
    separate(tmp_path / "x.wav", tmp_path)
    assert calls == ["roformer", "demucs"]


@needs_ffmpeg
def test_different_source_redoes_everything(tmp_path, monkeypatch):
    out = tmp_path / "out"
    calls = []

    def fake(step):
        def run(_, a, b):
            calls.append(step)
            for p in (a, b):
                sf.write(p, np.zeros((10, 2)), SAMPLE_RATE)
        return run

    monkeypatch.setattr(separation, "_run_roformer", fake("roformer"))
    monkeypatch.setattr(separation, "_run_demucs", fake("demucs"))
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    sf.write(a, np.zeros((22050, 2)), 22050)
    sf.write(b, np.zeros((44100, 2)), 22050)
    separate(a, out)
    separate(a, out)  # cùng nguồn: dùng cache
    assert calls == ["roformer", "demucs"]
    separate(b, out)  # nguồn khác: tách lại từ đầu
    assert calls == ["roformer", "demucs"] * 2
    assert abs(sf.info(out / "mix.wav").duration - 2.0) < 0.01


@pytest.mark.gpu
@needs_ffmpeg
def test_separate_first_20s_of_sample(tmp_path):
    if not paths.SAMPLE_AUDIO.exists():
        pytest.skip(f"thiếu {paths.SAMPLE_AUDIO}")
    clip = tmp_path / "clip.wav"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(paths.SAMPLE_AUDIO),
                    "-t", "20", str(clip)], check=True)

    # separate() tự lấy gpu_lock() cho từng model
    stems = separate(clip, tmp_path / "out")

    audio = {}
    for name in ["mix", "vocals", "instrumental", "bass", "no_bass"]:
        p = getattr(stems, name)
        assert p.exists(), name
        data, sr = sf.read(p, dtype="float32", always_2d=True)
        assert sr == SAMPLE_RATE, name
        assert data.shape[1] == 2, name
        assert abs(data.shape[0] / sr - 20.0) < 0.1, name
        assert np.isfinite(data).all(), name
        audio[name] = data

    for name in ["mix", "vocals", "instrumental", "no_bass"]:
        assert _rms(audio[name]) > 1e-2, f"{name} gần như im lặng"
    # 20 giây đầu "Let It Be" chưa có bass (bass vào ở khoảng giây 60), nên chỉ đòi không phải
    # toàn số 0. Việc bass có tín hiệu thật đã kiểm tra trên cả bài, xem mô tả PR.
    assert np.abs(audio["bass"]).max() > 0

    # Intro 0–8 s chỉ có piano; giọng hát bắt đầu khoảng giây 12.
    sr = SAMPLE_RATE
    v = audio["vocals"]
    intro, sung = _rms(v[: 8 * sr]), _rms(v[12 * sr: 20 * sr])
    assert sung > 10 * intro, (intro, sung)
