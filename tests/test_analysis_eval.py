import numpy as np

from eval import eval_analysis as E


def test_read_keylab_and_main_key(tmp_path):
    p = tmp_path / "k.lab"
    p.write_text("0.000\t10.0\tKey\tC\n10.0\t100.0\tKey\tA:minor\n100.0\t120.0\tKey\tG:mixolydian\n"
                 "120.0\t121.0\tSilence\n", encoding="utf-8")
    rows = E.read_keylab(p)
    assert [k for _, _, k in rows] == ["C major", "A minor", "G major"]
    assert E.main_key(rows) == "A minor"


def test_main_key_modal_is_skipped(tmp_path):
    p = tmp_path / "k.lab"
    p.write_text("0.0\t50.0\tKey\tC#:modal\n", encoding="utf-8")
    assert E.main_key(E.read_keylab(p)) is None
    assert E.main_key([]) is None


def test_read_seglab(tmp_path):
    p = tmp_path / "s.lab"
    p.write_text("0.000\t13.392\t\tintro\n13.392\t38.821\t\tverse\n38.821 52.048 refrain\n", encoding="utf-8")
    iv, labels = E.read_seglab(p)
    assert labels == ["intro", "verse", "refrain"]
    np.testing.assert_allclose(iv[-1], [38.821, 52.048])


def test_score_segments_perfect_and_uniform():
    ref_iv = np.array([[0.0, 10.0], [10.0, 30.0], [30.0, 50.0], [50.0, 60.0]])
    ref_lab = ["intro", "verse", "verse", "outro"]
    r = E.score_segments(ref_iv, ref_lab, ref_iv, ["a", "b", "b", "c"])
    assert r["F@3"] == 1.0 and r["pair_F"] == 1.0
    iv, lab = E.uniform_baseline(60.0, 15.0)
    assert iv[0, 0] == 0.0 and iv[-1, 1] == 60.0 and len(lab) == 4
    assert E.score_key("C major", "C major") == {"weighted": 1.0, "correct": 1.0}
    assert E.score_key("C major", "G major")["weighted"] == 0.5
