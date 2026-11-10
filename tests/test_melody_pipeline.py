"""Test dò giai điệu cho server (chordtrack.melody.pipeline): model giả, chỉ CPU; test GPU thật đánh dấu `gpu`."""
from __future__ import annotations

import random

import numpy as np
import pytest
import soundfile as sf

from chordtrack import paths
from chordtrack.melody import line as L
from chordtrack.melody import pipeline as mp
from chordtrack.types import MelodyModelOutput, MelodyNote as N

VID = "abcdefghijk"


def _random_outputs(seed: int) -> tuple[MelodyModelOutput, MelodyModelOutput]:
    rng = random.Random(seed)
    voice, fill = [], []
    t = 0.0
    while t < 30:
        d = rng.uniform(0.1, 0.8)
        src = rng.choice(["voice", "voice", "acoustic_piano", "electric_bass", "violin", "voice_chorus"])
        voice.append(N(t, t + d, rng.randint(45, 80), None, src))
        if rng.random() < 0.5:
            voice.append(N(t, t + d * 1.5, rng.randint(40, 70), None, "acoustic_piano"))
        fill.append(N(t + 0.02, t + d, rng.randint(50, 85), rng.uniform(0.2, 1.0),
                      rng.choice(["voice", "violin", "acoustic_piano", "flutes"])))
        t += d * rng.uniform(0.6, 1.4)
    return MelodyModelOutput("yourmt3", voice), MelodyModelOutput("muscriptor_small", fill)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_build_line_matches_eval_choice(seed):
    """Hằng số trong pipeline phải cho đúng dòng mà eval chấm với cấu hình CV đã chọn
    (`a=source_viterbi b=salience_nodur gate=a pad=0.6`)."""
    from eval.melody import score as S

    voice, fill = _random_outputs(seed)
    cfg = next(c for c in S.NN_GRID if c == dict(a="source_viterbi", b="salience_nodur", gate="a", pad=0.6))
    fam_a, ci_a = S.NN_KEYS[cfg["a"]]
    fam_b, ci_b = S.NN_KEYS[cfg["b"]]
    line_a = L.notes_to_line(voice.notes, S.params_of(S.note_families()[fam_a][ci_a]))
    line_b = L.notes_to_line(fill.notes, S.params_of(S.note_families()[fam_b][ci_b]))
    expect = L.combine_gated(line_a, line_b, L.voice_intervals(voice.notes, pad=cfg["pad"]))
    if mp.OCTAVE_FIX:
        expect = L.fix_octave_jumps(expect, **mp.OCTAVE_FIX)
    got = mp.build_line(voice, fill)
    assert got == expect and len(got) > 10
    assert all(a.end <= b.start + 1e-9 for a, b in zip(got, got[1:]))


def test_fix_octave_jumps():
    def line(pitches, dur=0.2, gap=0.0):
        out, t = [], 0.0
        for p in pitches:
            out.append(N(t, t + dur, p, 0.9, "voice"))
            t += dur + gap
        return out

    fix = lambda notes, **kw: [n.pitch for n in L.fix_octave_jumps(notes, **kw)]  # noqa: E731
    assert fix(line([60, 62, 76, 65, 64])) == [60, 62, 64, 65, 64]             # một nốt vọt lên quãng tám
    assert fix(line([67, 66, 54, 53, 64, 65])) == [67, 66, 66, 65, 64, 65]      # hai nốt tụt xuống
    assert fix(line([60, 62, 87, 65])) == [60, 62, 63, 65]                      # nhảy hai quãng tám
    assert fix(line([60, 62, 76, 77, 78, 65], dur=0.3), max_run=0.6) == [60, 62, 76, 77, 78, 65]   # đoạn dài: giữ
    assert fix(line([60, 62, 74, 72])) == [60, 62, 74, 72]                      # không quay lại: giữ
    assert fix(line([60, 62, 76, 65], gap=1.5)) == [60, 62, 76, 65]             # khác câu: giữ
    assert fix(line([60, 69, 60])) == [60, 69, 60]                              # quãng 6: không phải nhảy quãng tám
    notes = line([60, 76, 62])
    out = L.fix_octave_jumps(notes)
    assert out[1].start == notes[1].start and out[1].confidence == notes[1].confidence


@pytest.fixture
def fake_models(tmp_path, monkeypatch):
    """CACHE_DIR tạm có sẵn mix.wav của VID; hai model trả kết quả giả và đếm số lần chạy."""
    monkeypatch.setattr(paths, "CACHE_DIR", tmp_path / "cache")
    d = paths.CACHE_DIR / VID
    d.mkdir(parents=True)
    sf.write(d / "mix.wav", np.zeros(44100 * 3, dtype=np.float32), 44100)
    voice, fill = _random_outputs(0)
    calls = {"yourmt3": 0, "muscriptor": 0}

    def fake_yourmt3(audio, out_path):
        calls["yourmt3"] += 1
        assert audio == d / "mix.wav"
        return MelodyModelOutput("yourmt3", list(voice.notes), None, {"peak_vram_mib": np.array(2600.0)})

    def fake_muscriptor(audio):
        calls["muscriptor"] += 1
        return MelodyModelOutput("muscriptor_small", list(fill.notes), None, {"peak_vram_mb": np.array(650.0)})

    monkeypatch.setattr(mp, "run_yourmt3", fake_yourmt3)
    monkeypatch.setattr(mp, "run_muscriptor", fake_muscriptor)
    monkeypatch.setattr(mp.audio_mod, "fetch", lambda s: pytest.fail("không được tải lại audio đã có"))
    return calls


def test_analyze_melody_contract_and_cache(fake_models, monkeypatch):
    events = []
    res = mp.analyze_melody(f"https://youtu.be/{VID}", progress=lambda s, f: events.append((s, f)))
    assert events == [(s, f) for s in mp.STEPS for f in (0.0, 1.0)]
    assert res["song_id"] == VID and res["model"] == "yourmt3>muscriptor_small"
    assert res["code_revision"] == mp.code_revision() and res["duration"] == pytest.approx(3.0)
    notes = res["notes"]
    assert notes and set(notes[0]) == {"start", "end", "pitch", "confidence", "uncertain", "source"}
    assert all(a["end"] <= b["start"] for a, b in zip(notes, notes[1:]))
    for n in notes:
        assert isinstance(n["pitch"], int) and 0.0 <= n["confidence"] <= 1.0
        assert n["uncertain"] == (n["confidence"] < mp.UNCERTAIN_THRESHOLD)
    assert res["peak_vram_mb"] == {"yourmt3": 2600.0, "muscriptor_small": 650.0}
    assert mp.result_path(VID).exists()
    assert fake_models == {"yourmt3": 1, "muscriptor": 1}

    # Có cache: trả ngay, không báo bước nào.
    events.clear()
    assert mp.analyze_melody(VID, progress=lambda s, f: events.append((s, f)))["notes"] == notes
    assert events == [] and mp.cached_melody(VID)["notes"] == notes

    # Mã ghép đổi (code_revision khác): chạy lại nhưng dùng lại đầu ra thô của hai model.
    monkeypatch.setattr(mp, "code_revision", lambda: "khac")
    assert mp.cached_melody(VID) is None and mp.stale_melody(VID) is not None
    assert mp.analyze_melody(VID)["code_revision"] == "khac"
    assert fake_models == {"yourmt3": 1, "muscriptor": 1}

    # force: chạy lại cả hai model.
    mp.analyze_melody(VID, force=True)
    assert fake_models == {"yourmt3": 2, "muscriptor": 2}


def test_raw_cache_invalidated_when_wrapper_changes(fake_models, monkeypatch):
    mp.analyze_melody(VID)
    mp.result_path(VID).unlink()
    mp._wrapper_revision.cache_clear()
    real = mp._wrapper_revision.__wrapped__
    monkeypatch.setattr(mp, "_wrapper_revision", lambda m: "moi" if m == "yourmt3" else real(m))
    mp.analyze_melody(VID)
    assert fake_models == {"yourmt3": 2, "muscriptor": 1}


def test_run_yourmt3_reports_child_error(tmp_path, monkeypatch):
    import subprocess

    class Done:
        returncode = 1
        stdout = ""
        stderr = "Traceback...\nFileNotFoundError: Thiếu checkpoint YourMT3: x.ckpt"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Done())
    with pytest.raises(RuntimeError, match="Thiếu checkpoint YourMT3"):
        mp.run_yourmt3(tmp_path / "a.wav", tmp_path / "raw" / "yourmt3.npz")


def test_melody_ready_keys():
    r = mp.melody_ready()
    assert set(r) == {"yourmt3_code", "yourmt3_pydeps", "yourmt3_checkpoint", "muscriptor_package",
                      "muscriptor_weights"}
    assert all(isinstance(v, bool) for v in r.values())


@pytest.mark.gpu
def test_let_it_be_real(tmp_path, monkeypatch):
    """Chạy thật hai model trên "Let It Be". Không bọc `gpu_lock()` ở đây: tiến trình con YourMT3+ và
    `muscriptor.transcribe` tự lấy khóa; giữ khóa ở tiến trình cha thì tiến trình con chờ mãi."""
    if not paths.SAMPLE_AUDIO.exists() or not all(mp.melody_ready().values()):
        pytest.skip("thiếu audio mẫu hoặc model dò giai điệu")
    monkeypatch.setattr(paths, "CACHE_DIR", tmp_path / "cache")
    res = mp.analyze_melody(str(paths.SAMPLE_AUDIO))
    notes = res["notes"]
    assert 200 < len(notes) < 2000 and res["duration"] == pytest.approx(243, abs=2)
    # "When I find myself in times of trouble" bắt đầu khoảng 13 s, giọng ở quãng G3–E4.
    first = [n for n in notes if 12.5 < n["start"] < 20 and n["source"] == "voice"]
    assert len(first) >= 6 and all(52 <= n["pitch"] <= 67 for n in first)
    assert any(n["uncertain"] for n in notes) and not all(n["uncertain"] for n in notes)
