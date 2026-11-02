import json
import math

import mir_eval
import pytest

from chordtrack.types import ChordSegment, segments_to_lab
from eval.evaluate import METRICS, aggregate, evaluate_pair, main

REF = [
    ChordSegment(0.0, 1.0, "N"),
    ChordSegment(1.0, 3.0, "C:maj"),
    ChordSegment(3.0, 5.0, "G:maj/3"),
    ChordSegment(5.0, 7.0, "A:min7"),
    ChordSegment(7.0, 9.0, "F:maj"),
]


def test_identical_labels_score_one():
    r = evaluate_pair(REF, REF)
    for m in METRICS:
        assert r[m] == pytest.approx(1.0), m
    assert r["seg"] == pytest.approx(1.0)
    assert r["duration"] == pytest.approx(9.0)


def test_wrong_root_drops_by_its_duration():
    # Đoán sai nốt gốc của đoạn C (2 giây) thành D. Đoạn N (1 giây) vẫn được tính ở mọi mức.
    est = [ChordSegment(s.start, s.end, "D:maj" if s.label == "C:maj" else s.label) for s in REF]
    r = evaluate_pair(REF, est)
    for m in METRICS:
        assert r[m] == pytest.approx(7.0 / 9.0), m
    assert r["seg"] == pytest.approx(1.0)


def test_inversion_only_counts_at_inv_levels():
    # Đoán G thay vì G/B: đúng ở majmin, sai ở majmin_inv.
    est = [ChordSegment(s.start, s.end, "G:maj" if s.label == "G:maj/3" else s.label) for s in REF]
    r = evaluate_pair(REF, est)
    assert r["majmin"] == pytest.approx(1.0)
    assert r["majmin_inv"] == pytest.approx(7.0 / 9.0)
    assert r["sevenths_inv"] == pytest.approx(7.0 / 9.0)


def test_seventh_quality():
    # A:min thay vì A:min7: đúng ở majmin/root, sai ở sevenths.
    est = [ChordSegment(s.start, s.end, "A:min" if s.label == "A:min7" else s.label) for s in REF]
    r = evaluate_pair(REF, est)
    assert r["root"] == pytest.approx(1.0)
    assert r["majmin"] == pytest.approx(1.0)
    assert r["sevenths"] == pytest.approx(7.0 / 9.0)


def test_matches_mir_eval_evaluate():
    est = [
        ChordSegment(0.3, 2.5, "C:maj7"),
        ChordSegment(2.5, 4.0, "E:min"),
        ChordSegment(4.0, 6.2, "A:min"),
        ChordSegment(6.2, 10.0, "F:maj/5"),
    ]
    r = evaluate_pair(REF, est)
    ref_i = [[s.start, s.end] for s in REF]
    est_i = [[s.start, s.end] for s in est]
    import numpy as np

    expected = mir_eval.chord.evaluate(np.array(ref_i), [s.label for s in REF], np.array(est_i), [s.label for s in est])
    for m in METRICS:
        assert r[m] == pytest.approx(expected[m]), m
    for m in ("underseg", "overseg", "seg"):
        assert r[m] == pytest.approx(expected[m]), m


def test_unknown_ref_excluded_from_denominator():
    ref = REF + [ChordSegment(9.0, 10.0, "X")]
    r = evaluate_pair(ref, REF)
    assert r["majmin"] == pytest.approx(1.0)
    assert r["weights"]["majmin"] == pytest.approx(9.0)


def test_empty_estimate_and_lab_files(tmp_path):
    segments_to_lab(REF, tmp_path / "ref.lab")
    (tmp_path / "est.lab").write_text("", encoding="utf-8")
    r = evaluate_pair(tmp_path / "ref.lab", tmp_path / "est.lab")
    assert r["root"] == pytest.approx(1.0 / 9.0)  # chỉ đoạn N là đúng


def test_aggregate_is_duration_weighted():
    short = [ChordSegment(0.0, 1.0, "C:maj")]
    long = [ChordSegment(0.0, 3.0, "C:maj")]
    res = {
        "a": evaluate_pair(short, [ChordSegment(0.0, 1.0, "D:maj")]),  # 0 / 1 giây
        "b": evaluate_pair(long, long),  # 3 / 3 giây
    }
    agg = aggregate(res)
    assert agg["root"] == pytest.approx(0.75)
    assert agg["n_files"] == 2


def test_cli(tmp_path, capsys):
    ref_dir, est_dir = tmp_path / "ref", tmp_path / "est"
    segments_to_lab(REF, ref_dir / "song1.lab")
    segments_to_lab(REF, est_dir / "song1.lab")
    segments_to_lab(REF, ref_dir / "song2.lab")  # thiếu nhãn đoán
    out = tmp_path / "res.json"
    assert main(["--ref-dir", str(ref_dir), "--est-dir", str(est_dir), "--out", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["mean"]["n_files"] == 1
    assert data["missing"] == ["song2"]
    assert math.isclose(data["files"]["song1"]["mirex"], 1.0)
    assert "TRUNG BÌNH" in capsys.readouterr().out
