"""Test dò giai điệu cho server (chordtrack.melody.pipeline): model giả, chỉ CPU; test GPU thật đánh dấu `gpu`."""
from __future__ import annotations

import random
from pathlib import Path

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
        if audio == d / "vocals.wav":     # YourMT3+ trên stem giọng (biến thể tách giọng): nốt giọng cao hơn 1 cung
            calls["yourmt3_vocals"] = calls.get("yourmt3_vocals", 0) + 1
            notes = [N(n.start, n.end, n.pitch + 2, None, n.source) for n in voice.notes if n.source == "voice"]
            return MelodyModelOutput("yourmt3", notes, None, {"peak_vram_mib": np.array(1500.0)})
        calls["yourmt3"] += 1
        assert audio == d / "mix.wav"
        return MelodyModelOutput("yourmt3", list(voice.notes), None, {"peak_vram_mib": np.array(2600.0)})

    def fake_separation(mix, cache_dir):
        calls["separate"] = calls.get("separate", 0) + 1
        assert mix == d / "mix.wav" and cache_dir == d
        sf.write(d / "vocals.wav", np.zeros(44100 * 3, dtype=np.float32), 44100)
        return {"seconds": 1.0, "peak_vram_mb": 1600.0}

    def fake_muscriptor(audio):
        calls["muscriptor"] += 1
        return MelodyModelOutput("muscriptor_small", list(fill.notes), None, {"peak_vram_mb": np.array(650.0)})

    monkeypatch.setattr(mp, "run_yourmt3", fake_yourmt3)
    monkeypatch.setattr(mp, "run_muscriptor", fake_muscriptor)
    monkeypatch.setattr(mp, "run_separation", fake_separation)
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
    monkeypatch.setattr(mp, "code_revision", lambda variant="combo": "khac")
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

    seen = {}

    class FakePopen:
        def __init__(self, cmd, *, stdin, stdout, stderr, **kw):
            seen["cmd"] = cmd
            stderr.write("Traceback...\nFileNotFoundError: Thiếu checkpoint YourMT3: x.ckpt".encode("utf-8"))

        def wait(self, timeout=None):
            return 1

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    with pytest.raises(RuntimeError, match="Thiếu checkpoint YourMT3"):
        mp.run_yourmt3(tmp_path / "a.wav", tmp_path / "raw" / "yourmt3.npz")
    assert seen["cmd"][1:3] == ["-m", "chordtrack.melody.yourmt3"] and "--parent-pid" in seen["cmd"]


def test_child_exits_with_parent_and_imports_work(tmp_path):
    """Tiến trình con vẫn import được trong lúc theo dõi tiến trình cha, và thoát khi cha chết."""
    import subprocess
    import sys

    child = ("import sys,time; from chordtrack.melody.yourmt3 import exit_with_parent; "
             "exit_with_parent(int(sys.argv[1])); import numpy; print('ok', flush=True); time.sleep(60)")
    parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        p = subprocess.Popen([sys.executable, "-c", child, str(parent.pid)], stdout=subprocess.PIPE, text=True)
        assert p.stdout.readline().strip() == "ok"
        parent.kill()
        parent.wait()
        assert p.wait(timeout=20) == 3
    finally:
        parent.kill()


def test_cached_raw_steps_only_report_done(fake_models, monkeypatch):
    """Bước dùng lại đầu ra thô chỉ báo xong, để thời gian ~0 không lọt vào thống kê ETA."""
    mp.analyze_melody(VID)
    monkeypatch.setattr(mp, "code_revision", lambda variant="combo": "khac")
    events = []
    mp.analyze_melody(VID, progress=lambda s, f: events.append((s, f)))
    assert ("melody_voice", 0.0) not in events and ("melody_fill", 0.0) not in events
    assert ("melody_voice", 1.0) in events and ("melody_line", 0.0) in events


def test_stale_melody_checks_version(fake_models):
    mp.analyze_melody(VID)
    from chordtrack.audio import read_json, write_json

    data = read_json(mp.result_path(VID))
    write_json(mp.result_path(VID), {**data, "version": 99})
    assert mp.stale_melody(VID) is None and mp.cached_melody(VID) is None


def test_melody_ready_keys():
    r = mp.melody_ready()
    assert set(r) == {"yourmt3_code", "yourmt3_pydeps", "yourmt3_checkpoint", "muscriptor_package",
                      "muscriptor_weights", "separator_package"}
    assert all(isinstance(v, bool) for v in r.values())
    ok = {k: True for k in r}
    v = mp.variant_ready({**ok, "muscriptor_weights": False})
    assert v == {"combo": False, "yourmt3": True, "combo_vocals": False, "yourmt3_vocals": True}
    v = mp.variant_ready({**ok, "separator_package": False})
    assert v == {"combo": True, "yourmt3": True, "combo_vocals": False, "yourmt3_vocals": False}
    assert not any(mp.variant_ready({**ok, "yourmt3_checkpoint": False}).values())


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_build_line_yourmt3_matches_eval_choice(seed):
    """Biến thể "yourmt3" = cấu hình CV của YourMT3+ một mình (`yourmt3|0`: source_viterbi, chỉ số 1 của lưới)."""
    from eval.melody import score as S

    voice, _ = _random_outputs(seed)
    cfg = S.note_families()["source_viterbi"][1]
    assert S.cfg_text(cfg) == "jump_penalty=0.5 cut_penalty=0.15 chorus_as_voice=False"
    expect = L.notes_to_line(voice.notes, S.params_of(cfg))
    if mp.OCTAVE_FIX:
        expect = L.fix_octave_jumps(expect, **mp.OCTAVE_FIX)
    got = mp.build_line_yourmt3(voice)
    assert got == expect and len(got) > 10


def test_yourmt3_variant_reuses_raw_and_has_own_cache(fake_models):
    """Biến thể "yourmt3": bước riêng, file riêng, dùng lại đầu ra thô YourMT3+ của lần chạy ghép."""
    with pytest.raises(ValueError):
        mp.analyze_melody(VID, variant="khac")
    events = []
    res = mp.analyze_melody(VID, variant="yourmt3", progress=lambda s, f: events.append((s, f)))
    assert events == [(s, f) for s in ("fetch", "melody_voice", "melody_line") for f in (0.0, 1.0)]
    assert res["model"] == "yourmt3" and res["peak_vram_mb"] == {"yourmt3": 2600.0}
    assert "melody_fill" not in res["timing"]
    assert fake_models == {"yourmt3": 1, "muscriptor": 0}
    assert mp.result_path(VID, "yourmt3").name == "melody_yourmt3.json" and mp.result_path(VID, "yourmt3").exists()
    assert mp.cached_melody(VID) is None and mp.stale_melody(VID) is None      # bản ghép chưa có
    assert mp.cached_melody(VID, "yourmt3")["notes"] == res["notes"]

    # Ghép sau đó: dùng lại đầu ra thô YourMT3+, chỉ chạy MuScriptor; hai file không đè nhau.
    combo = mp.analyze_melody(VID)
    assert fake_models == {"yourmt3": 1, "muscriptor": 1}
    assert combo["model"] == "yourmt3>muscriptor_small" and mp.cached_melody(VID, "yourmt3")["model"] == "yourmt3"

    # Có đầu ra thô (vd từ lần chạy ghép) thì bản YourMT3+ không chạy lại model.
    mp.result_path(VID, "yourmt3").unlink()
    events.clear()
    again = mp.analyze_melody(VID, variant="yourmt3", progress=lambda s, f: events.append((s, f)))
    assert again["notes"] == res["notes"] and fake_models == {"yourmt3": 1, "muscriptor": 1}
    assert ("melody_voice", 0.0) not in events and ("melody_voice", 1.0) in events

    # force: chỉ chạy lại YourMT3+; bản ghép dựng từ đầu ra thô cũ thành "cũ" (vẫn đọc được qua stale).
    mp.analyze_melody(VID, variant="yourmt3", force=True)
    assert fake_models == {"yourmt3": 2, "muscriptor": 1}
    assert mp.cached_melody(VID) is None and mp.stale_melody(VID)["code_revision"] == mp.RAW_CHANGED
    assert mp.cached_melody(VID, "yourmt3") is not None
    combo2 = mp.analyze_melody(VID)                          # dò lại bản ghép: dùng đầu ra thô mới, không chạy model
    assert fake_models == {"yourmt3": 2, "muscriptor": 1} and mp.cached_melody(VID) == combo2
    assert mp.cached_melody(VID, "yourmt3") is not None


def test_code_revision_per_variant(monkeypatch):
    """Bản YourMT3+ không phụ thuộc mã MuScriptor; mọi biến thể có loại job, Literal của server khớp VARIANTS."""
    import typing

    from chordtrack import jobs, server
    from chordtrack.melody import muscriptor

    mp.code_revision.cache_clear()
    try:
        y3, combo = mp.code_revision("yourmt3"), mp.code_revision("combo")
        assert y3 != combo
        real = mp._file_hash
        monkeypatch.setattr(mp, "_file_hash", lambda *f: real(*f) + ("x" if Path(muscriptor.__file__) in f else ""))
        mp.code_revision.cache_clear()
        assert mp.code_revision("yourmt3") == y3 and mp.code_revision("combo") != combo
    finally:
        mp.code_revision.cache_clear()
    assert set(typing.get_args(server.MelodyModel)) == set(mp.BASE_MODELS)
    assert {mp.variant_key(m, v) for m in mp.BASE_MODELS for v in (False, True)} == set(mp.VARIANTS)
    assert all(mp.VARIANTS[mp.variant_key(s.base, s.vocals)] is s for s in mp.VARIANTS.values())
    assert set(jobs.MELODY_KIND_OF) == set(mp.VARIANTS) and all(k in jobs.KINDS for k in jobs.MELODY_KIND_OF.values())
    assert all(jobs.KINDS[jobs.MELODY_KIND_OF[v]].steps == s.steps for v, s in mp.VARIANTS.items())


def test_build_line_yourmt3_vocals():
    """Bản YourMT3+ tách giọng: dòng từ stem giọng trong cổng giọng của stem (nới GATE_PAD), dòng mix ngoài cổng."""
    stem = MelodyModelOutput("yourmt3", [N(10.0, 10.5, 64, None, "voice"), N(10.5, 11.0, 65, None, "voice"),
                                         N(10.0, 11.0, 40, None, "acoustic_piano")])
    mix = MelodyModelOutput("yourmt3", [N(t, t + 0.5, 72, None, "violin") for t in np.arange(8.0, 13.0, 0.5)])
    got = mp.build_line_yourmt3_vocals(stem, mix)
    assert [n.pitch for n in got if 10.0 <= n.start < 11.0] == [64, 65]        # giọng từ stem, bỏ bè đệm lọt vào
    assert all(n.pitch == 72 for n in got if n.end <= 10.0 - mp.GATE_PAD or n.start >= 11.0 + mp.GATE_PAD)
    assert not any(n.pitch == 72 and 9.5 < n.start < 11.5 for n in got)          # trong cổng không có dòng mix
    assert all(a.end <= b.start + 1e-9 for a, b in zip(got, got[1:]))
    # Stem không có nốt voice (bài không lời): cả dòng lấy từ mix.
    empty = MelodyModelOutput("yourmt3", [N(10.0, 11.0, 40, None, "acoustic_piano")])
    assert mp.build_line_yourmt3_vocals(empty, mix) == mp.build_line_yourmt3(mix)


def test_vocals_variants_reuse_raw_and_stem(fake_models):
    """Bản tách giọng: tách stem một lần (dùng lại vocals.wav), YourMT3+ trên stem dùng chung giữa hai bản, dùng lại
    YourMT3+ mix / MuScriptor đã có; file và model riêng; force tách lại cả stem."""
    mp.analyze_melody(VID, variant="yourmt3")                         # đã có YourMT3+ trên mix
    assert fake_models == {"yourmt3": 1, "muscriptor": 0}
    events = []
    res = mp.analyze_melody(VID, variant="yourmt3_vocals", progress=lambda s, f: events.append((s, f)))
    assert res["model"] == "yourmt3[vocals]"
    assert mp.result_path(VID, "yourmt3_vocals").name == "melody_yourmt3_vocals.json"
    assert fake_models == {"yourmt3": 1, "muscriptor": 0, "separate": 1, "yourmt3_vocals": 1}
    assert ("melody_separate", 0.0) in events and ("melody_voice_mix", 0.0) not in events
    assert [s for s, f in events if f == 1.0] == list(mp.VARIANTS["yourmt3_vocals"].steps)
    assert res["peak_vram_mb"] == {"roformer": 1600.0, "yourmt3_vocals": 1500.0, "yourmt3": 2600.0}
    assert res["notes"] and mp.cached_melody(VID, "yourmt3")["model"] == "yourmt3"

    combo = mp.analyze_melody(VID, variant="combo_vocals")           # dùng lại stem và YourMT3+ trên stem
    assert combo["model"] == "yourmt3[vocals]>muscriptor_small"
    assert mp.result_path(VID, "combo_vocals").name == "melody_vocals.json"
    assert fake_models == {"yourmt3": 1, "muscriptor": 1, "separate": 1, "yourmt3_vocals": 1}
    assert set(combo["timing"]) >= {"melody_separate", "melody_voice", "melody_fill"}
    assert "melody_voice_mix" not in combo["timing"]

    # force bản YourMT3+ tách giọng: tách lại, chạy lại YourMT3+ trên stem và trên mix; các bản dùng chung đầu ra thô
    # bị chạy lại thành cũ.
    mp.analyze_melody(VID, variant="yourmt3_vocals", force=True)
    assert fake_models == {"yourmt3": 2, "muscriptor": 1, "separate": 2, "yourmt3_vocals": 2}
    assert mp.stale_melody(VID, "combo_vocals")["code_revision"] == mp.RAW_CHANGED
    assert mp.stale_melody(VID, "yourmt3")["code_revision"] == mp.RAW_CHANGED
    assert mp.cached_melody(VID, "yourmt3_vocals") is not None


def test_vocals_raw_tied_to_stem_file(fake_models):
    """Đầu ra thô YourMT3+ trên stem gắn với đúng vocals.wav: stem bị tách lại (vd pipeline hợp âm) thì chạy lại."""
    mp.analyze_melody(VID, variant="yourmt3_vocals")
    assert fake_models["yourmt3_vocals"] == 1
    mp.result_path(VID, "yourmt3_vocals").unlink()
    mp.analyze_melody(VID, variant="yourmt3_vocals")              # stem không đổi: dùng lại
    assert fake_models["yourmt3_vocals"] == 1 and fake_models["separate"] == 1
    mp.result_path(VID, "yourmt3_vocals").unlink()
    sf.write(paths.CACHE_DIR / VID / "vocals.wav", np.zeros(44100 * 2, dtype=np.float32), 44100)
    mp.analyze_melody(VID, variant="yourmt3_vocals")              # stem khác: chạy lại YourMT3+ trên stem
    assert fake_models["yourmt3_vocals"] == 2 and fake_models["separate"] == 1 and fake_models["yourmt3"] == 1


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
