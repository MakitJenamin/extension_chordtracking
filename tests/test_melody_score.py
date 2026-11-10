"""Test phần chấm điểm giai điệu (eval.melody.score) bằng dữ liệu tổng hợp, chỉ CPU."""
from __future__ import annotations

import json

import numpy as np
import pytest

from chordtrack import paths
from chordtrack.melody import line as L
from chordtrack.melody.io import cache_path, save_output
from chordtrack.types import ChordSegment, MelodyModelOutput, MelodyNote as N, PitchTrack
from eval.melody import score as S


def test_f0_metrics_perfect_and_octave():
    t = np.arange(0, 1, 0.01)
    ref = np.where(t < 0.5, 440.0, 0.0)
    v, w = S.f0_metrics(t, ref, t, ref)
    assert v["oa"] == pytest.approx(1.0) and v["rpa"] == pytest.approx(1.0) and v["vfa"] == pytest.approx(0.0)
    assert w["oa"] == 100 and w["rpa"] == 50 and w["vfa"] == 50
    v, _ = S.f0_metrics(t, ref, t, ref * 2)           # sai quãng tám: RCA đúng, RPA sai
    assert v["rpa"] == pytest.approx(0.0) and v["rca"] == pytest.approx(1.0)


def test_note_metrics_onset_tolerance():
    ref = np.array([[0.0, 0.5, 60], [0.5, 1.0, 62]])
    v, w = S.note_metrics(ref, [N(0.03, 0.5, 60), N(0.6, 1.0, 62)])
    assert v["note_p"] == pytest.approx(0.5) and v["note_r"] == pytest.approx(0.5) and w["note_f1"] == 2
    v, _ = S.note_metrics(ref, [])
    assert v["note_f1"] == 0.0


def test_free_metrics_in_chord_and_jumps():
    chords = [ChordSegment(0.0, 2.0, "C:maj"), ChordSegment(2.0, 4.0, "N")]
    est = [N(0.0, 0.5, 60), N(0.5, 1.0, 62), N(1.0, 1.5, 79), N(2.0, 3.0, 64)]
    v, w = S.free_metrics(est, 6.0, chords)
    assert v["in_chord"] == pytest.approx(2 / 3) and w["in_chord"] == 3      # bỏ nốt ở đoạn N
    assert v["coverage"] == pytest.approx(2.5 / 6)
    assert v["octave_jumps_pm"] == pytest.approx(2 / 0.1)                    # 62 -> 79 -> 64 trong 6 giây
    assert v["median_dur"] == pytest.approx(0.5)
    assert S.chord_pcs("A:min7") == {9, 0, 4, 7}
    assert S.chord_pcs("C:maj/5") == {0, 4, 7}


@pytest.fixture
def fake_data(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "EVAL_DIR", tmp_path / "eval")
    monkeypatch.setattr(paths, "CACHE_DIR", tmp_path / "cache")
    rng = np.random.default_rng(0)
    # bộ có ref_f0: 4 bài, giai điệu bậc thang, model f0 chép gần đúng; model nốt có cả bè đệm thấp
    d = paths.EVAL_DIR / "melody" / "fake_f0"
    d.mkdir(parents=True)
    lines = []
    for i in range(4):
        tid = f"t{i}"
        t = np.arange(0, 4, 0.01)
        midi = 60 + 2 * (np.floor(t * 2) % 5) + i
        hz = L.midi_to_hz(midi)
        hz[(t % 0.5) > 0.45] = 0.0
        np.savetxt(d / f"{tid}.csv", np.c_[t, hz], delimiter=",", fmt="%.4f")
        lines.append({"track_id": tid, "dataset": "fake_f0", "audio": "x.wav", "ref_notes": None,
                      "ref_f0": str(d / f"{tid}.csv"), "melody_type": "vocal", "duration": 4.0})
        noisy = np.where(hz > 0, hz * 2 ** (rng.normal(0, 0.1, len(t)) / 12), 0.0)
        save_output(MelodyModelOutput("f0m", pitch_track=PitchTrack(t, noisy, np.ones_like(t))),
                    cache_path("f0m", "fake_f0", tid))
        notes = [N(k * 0.5, k * 0.5 + 0.45, int(60 + 2 * (k % 5) + i), 0.8, "voice") for k in range(8)]
        notes += [N(0.0, 4.0, 48, 0.9, "piano")]
        save_output(MelodyModelOutput("notem", notes=notes), cache_path("notem", "fake_f0", tid))
    (d / "manifest.jsonl").write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
    # bộ không đáp án, có chord_ref
    d2 = paths.EVAL_DIR / "melody" / "fake_free"
    d2.mkdir(parents=True)
    (d2 / "a.lab").write_text("0.0\t4.0\tC:maj\n", encoding="utf-8")
    (d2 / "manifest.jsonl").write_text(json.dumps(
        {"track_id": "a", "dataset": "fake_free", "audio": "x.wav", "ref_notes": None, "ref_f0": None,
         "melody_type": "mixed", "duration": 4.0, "chord_ref": "a.lab"}), encoding="utf-8")
    save_output(MelodyModelOutput("notem", notes=[N(0, 1, 60, 0.9, None), N(1, 2, 61, 0.9, None)]),
                cache_path("notem", "fake_free", "a"))
    # lưới phách cho một bài
    bd = paths.CACHE_DIR / "melody" / "_beats" / "fake_f0"
    bd.mkdir(parents=True)
    np.savez(bd / "t0.npz", beats=np.arange(0, 4, 0.5), downbeats=np.arange(0, 4, 2.0))
    return tmp_path


def test_score_end_to_end(fake_data, capsys):
    out_dir = fake_data / "results"
    assert S.main(["--jobs", "1", "--folds", "2", "--boot", "200", "--out-dir", str(out_dir)]) == 0
    printed = capsys.readouterr().out
    assert "fake_f0" in printed and "fake_free" in printed
    summary = list(__import__("csv").DictReader(open(out_dir / "summary.csv", encoding="utf-8")))
    cv = {(r["dataset"], r["model"], r["family"], r["quant"]): r for r in summary if r["selection"] == "cv"}
    # mọi họ đều có mặt, kể cả combo và bản căn phách
    fams = {(m, f) for (_, m, f, _) in cv}
    assert {("notem", "skyline"), ("notem", "viterbi"), ("notem", "source"), ("notem", "salience"),
            ("f0m", "f0raw"), ("f0m", "f0notes"), ("f0m+notem", "combo")} <= fams
    assert ("fake_f0", "notem", "viterbi", "1") in cv
    # model nốt chép đúng giai điệu (bè piano thấp bị bỏ nhờ ưu tiên giọng/cao độ)
    assert float(cv[("fake_f0", "notem", "viterbi", "0")]["oa"]) > 0.9
    assert float(cv[("fake_f0", "f0m", "f0notes", "0")]["rpa"]) > 0.9
    assert int(cv[("fake_f0", "notem", "viterbi", "0")]["n_tracks"]) == 4
    free = cv[("fake_free", "notem", "viterbi", "0")]
    assert float(free["in_chord"]) == pytest.approx(0.5)
    assert float(free["coverage"]) == pytest.approx(0.5)
    choices = json.loads((out_dir / "cv_choices.json").read_text(encoding="utf-8"))
    assert len(choices["choices"]["notem|viterbi|0"]["folds"]) == 2
    assert (out_dir / "per_track.csv").stat().st_size > 0


def test_offset_shifts_reference(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("onset,offset,midi\n1.0,2.0,60\n", encoding="utf-8")
    ref = S.load_refs({"_dir": str(tmp_path), "ref_notes": str(p), "offset": 0.5})
    assert ref["notes"][0].tolist() == [1.5, 2.5, 60]
    assert ref["f0"][1][np.searchsorted(ref["f0"][0], 2.0)] == pytest.approx(L.midi_to_hz(60))
    v, _ = S.note_metrics(ref["notes"], [N(1.5, 2.5, 60)])
    assert v["note_f1"] == pytest.approx(1.0)


def test_dataset_primary_uses_note_f1_for_note_datasets():
    entries = {"st": [{"ref_notes": "a.csv", "ref_f0": None}], "f0": [{"ref_f0": "b.csv"}], "free": [{}]}
    assert S.dataset_primary(entries) == {"st": "note_f1", "f0": "oa", "free": None}
