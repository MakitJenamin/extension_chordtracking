import json

import numpy as np
import pytest
import soundfile as sf

from eval.melody import datasets as ds


def _wav(path, sr=22050, seconds=1.0, channels=1):
    y = np.zeros((int(sr * seconds), channels), dtype=np.float32)
    sf.write(str(path), y, sr)
    return path


def test_manifest_roundtrip(tmp_path):
    wav = _wav(tmp_path / "a.wav", channels=2)
    ref = tmp_path / "a.csv"
    ds.write_notes_csv(np.array([[0.5, 1.0, 60], [1.2, 1.5, 62.0]]), ref)
    e = ds.make_entry("a", "toy", wav, "vocal", ref_notes=ref, offset=0.25)
    assert e["sr"] == 22050 and e["channels"] == 2 and e["duration"] == pytest.approx(1.0)
    m = tmp_path / "toy" / "manifest.jsonl"
    ds.write_manifest([e], m)
    back = ds.read_manifest(m)
    assert back == [json.loads(json.dumps(e))]
    ref_data = ds.load_reference(back[0])
    assert ref_data["f0"] is None
    assert np.allclose(ref_data["notes"], [[0.75, 1.25, 60], [1.45, 1.75, 62]])


def test_manifest_validation(tmp_path):
    with pytest.raises(ValueError):
        ds.write_manifest([{"track_id": "x"}], tmp_path / "m.jsonl")
    with pytest.raises(ValueError):
        ds.make_entry("a", "toy", _wav(tmp_path / "a.wav"), "piano")


def test_f0_csv_and_mel(tmp_path):
    mel = tmp_path / "x.mel"
    mel.write_text("0.00\t0\n0.01\t440.5\n0.02\t-3\n", encoding="utf-8")
    t, hz = ds.parse_mel(mel)
    assert np.allclose(t, [0, 0.01, 0.02]) and np.allclose(hz, [0, 440.5, 0])
    out = tmp_path / "x.csv"
    ds.write_f0_csv(t, hz, out)
    t2, hz2 = ds.read_f0_csv(out, offset=1.0)
    assert np.allclose(t2, t + 1.0) and np.allclose(hz2, hz)


def test_notes_csv_skips_comments(tmp_path):
    p = tmp_path / "n.csv"
    p.write_text("# onset,offset,midi\n1,2,60\n\n", encoding="utf-8")
    assert ds.read_notes_csv(p).tolist() == [[1.0, 2.0, 60.0]]
    p.write_text("", encoding="utf-8")
    assert ds.read_notes_csv(p).shape == (0, 3)


def test_mirst_decide():
    assert ds.mirst_decide({"lag": 0.3, "score": 1.2, "margin": 0.5}) == (True, 0.3)
    assert ds.mirst_decide({"lag": 0.012, "score": 1.2, "margin": 0.5}) == (True, 0.0)
    assert ds.mirst_decide({"lag": 0.3, "score": 0.1, "margin": 0.5})[0] is False
    assert ds.mirst_decide({"lag": 0.3, "score": 1.2, "margin": 0.02})[0] is False


def test_score_offsets_no_notes_in_range():
    S = np.zeros((61, 500))
    assert ds.mirst_decide(ds.score_offsets(S, np.zeros((0, 3))))[0] is False
    assert ds.mirst_decide(ds.score_offsets(S, np.array([[0.1, 0.5, 20.0]])))[0] is False


def test_safe_extract_rejects_sibling_prefix(tmp_path):
    import zipfile

    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../extracted.tmp_evil/x.txt", "x")
    with pytest.raises(RuntimeError):
        ds._safe_extract(z, tmp_path / "extracted")
    assert not (tmp_path / "extracted").exists()


def test_redecide_keeps_missing_and_unestimated(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "MELODY_DIR", tmp_path)
    d = tmp_path / "mir_st500"
    for tid in ("401", "402"):
        _wav((d / "audio").mkdir(parents=True, exist_ok=True) or d / "audio" / f"{tid}.wav")
        ds.write_notes_csv(np.array([[0.1, 0.5, 60]]), d / "ref" / f"{tid}.csv")
    good = {"lag": 0.3, "score": 1.5, "margin": 0.5}
    report = {"401": {"url": "u1", "status": "ok", "offset": 0.0},                     # --skip-offset, không có est
              "402": {"url": "u2", "status": "mismatch", "offset": 0.0, "est": good},
              "403": {"url": "u3", "status": "missing", "est": good}}                 # mất audio
    (d / "offsets.json").write_text(json.dumps(report), encoding="utf-8")
    entries = ds.redecide_mir_st500(0.8, 0.15)
    assert [(e["track_id"], e["offset"]) for e in entries] == [("401", 0.0), ("402", 0.3)]
    back = json.loads((d / "offsets.json").read_text(encoding="utf-8"))
    assert back["403"]["status"] == "missing"


def test_estimate_offset_synthetic(tmp_path):
    # Giai điệu tổng hợp lệch 0.4 s so với nhãn: phải tìm lại được độ lệch.
    sr = 22050
    rng = np.random.default_rng(0)
    notes, t = [], 1.0
    while t < 20:
        d = rng.uniform(0.2, 0.6)
        notes.append([t, t + d, int(rng.integers(55, 80))])
        t += d + rng.uniform(0.0, 0.2)
    notes = np.array(notes, dtype=float)
    y = np.zeros(int(sr * 23))
    for on, off, midi in notes:
        a, b = int((on + 0.4) * sr), int((off + 0.4) * sr)
        tt = np.arange(b - a) / sr
        f = 440 * 2 ** ((midi - 69) / 12)
        y[a:b] += sum(np.sin(2 * np.pi * f * k * tt) / k for k in (1, 2, 3))
    y += 0.05 * rng.standard_normal(len(y))
    wav = tmp_path / "s.wav"
    sf.write(str(wav), (0.2 * y).astype(np.float32), sr)
    est = ds.estimate_offset(wav, notes)
    assert est["lag"] == pytest.approx(0.4, abs=0.03)
    assert ds.mirst_decide(est) == (True, est["lag"])
    assert est["score"] > est["score0"] and abs(est["null"]) < 0.3
