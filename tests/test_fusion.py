"""Test logic fusion, không cần GPU hay weights: dữ liệu giả lập."""
import numpy as np
import pytest

from chordtrack import fusion as F
from chordtrack.types import BeatGrid, ChordModelOutput, ChordSegment

CM_VOCAB = ["N", "C:maj", "G:maj", "A:min", "F:maj", "C:7"]


def _covers(segs, duration):
    assert segs[0].start == 0.0
    assert segs[-1].end == pytest.approx(duration)
    for a, b in zip(segs, segs[1:]):
        assert a.end == pytest.approx(b.start)
        assert b.end > b.start


def _model(name, times, probs, vocab, extras=None):
    return ChordModelOutput(name, [], np.asarray(times), np.asarray(probs, dtype=float), vocab, extras or {})


def _one_hot_track(plan, hop, vocab, conf=0.9):
    """plan: [(nhãn, giây)] -> (times, probs) với nhãn đúng có xác suất conf."""
    labels = []
    for lab, dur in plan:
        labels += [vocab.index(lab)] * int(round(dur / hop))
    n = len(labels)
    probs = np.full((n, len(vocab)), (1 - conf) / (len(vocab) - 1))
    probs[np.arange(n), labels] = conf
    return (np.arange(n) + 0.5) * hop, probs


# ---------------------------------------------------------------- gán bass

def test_with_bass_adds_degree_for_chord_tones():
    assert F.with_bass("C:maj", 7) == "C:maj/5"       # C/G
    assert F.with_bass("C:maj", 4) == "C:maj/3"       # C/E
    assert F.with_bass("A:min", 0) == "A:min/b3"      # Am/C
    assert F.with_bass("G:7", 5) == "G:7/b7"          # G7/F
    assert F.with_bass("C:aug", 8) == "C:aug/#5"


def test_with_bass_keeps_label_for_root_non_chord_tone_or_no_chord():
    assert F.with_bass("C:maj", 0) == "C:maj"         # bass là nốt gốc
    assert F.with_bass("C:maj", 2) == "C:maj"         # D không thuộc C
    assert F.with_bass("C:maj", 11) == "C:maj"        # B không thuộc C (không phải maj7)
    assert F.with_bass("N", 4) == "N"
    assert F.with_bass("C:maj/3", 7) == "C:maj/3"     # đã có bass thì giữ nguyên


def test_harte_bass_labels_are_valid_for_mir_eval():
    import mir_eval

    for lab, pc in [("C:maj", 7), ("A:min", 0), ("G:7", 5), ("C:dim7", 9), ("C:aug", 8), ("F:maj7", 4)]:
        out = F.with_bass(lab, pc)
        root, _bitmap, bass = mir_eval.chord.encode(out)
        assert (root + bass) % 12 == pc, out


def test_pick_bass_threshold():
    dist = np.zeros(12)
    dist[7], dist[0] = 0.45, 0.35
    assert F.pick_bass(dist, "C:maj", 0.4) == "C:maj/5"
    assert F.pick_bass(dist, "C:maj", 0.5) == "C:maj"


def test_assign_bass_per_segment_and_per_beat():
    hop = 0.1
    times = (np.arange(40) + 0.5) * hop
    bass = np.full((40, 12), 0.01)
    bass[:24, 0] = 0.9    # 0–2.4 s bass C
    bass[24:, 4] = 0.9    # 2.4–4 s bass E
    bass = bass / bass.sum(axis=1, keepdims=True)
    segs = [(0.0, 4.0, 1)]  # C:maj cả 4 giây
    vocab = CM_VOCAB
    # Cả đoạn: C nhiều hơn E, argmax là C (nốt gốc) -> không đảo
    assert [s[3] for s in F.assign_bass(segs, vocab, bass, times, 0.3)] == ["C:maj"]
    # Theo phách: tách thành C rồi C/E
    beats = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]
    out = F.assign_bass(segs, vocab, bass, times, 0.3, beats)
    assert [(s[0], s[1], s[3]) for s in out] == [(0.0, 2.0, "C:maj"), (2.0, 4.0, "C:maj/3")]


def test_bass_distribution_sources():
    cnn = np.tile(np.eye(12)[0], (5, 1))
    chroma = np.tile(np.eye(12)[7] * 10, (5, 1))
    p = F.FusionParams(bass_source="both", stem_weight=0.5)
    d = F.bass_distribution(cnn, chroma, p)
    assert d[0, 0] == pytest.approx(0.5) and d[0, 7] == pytest.approx(0.5)
    assert F.bass_distribution(cnn, chroma, F.FusionParams(bass_source="none")) is None
    assert np.argmax(F.bass_distribution(cnn, chroma, F.FusionParams(bass_source="stem"))[0]) == 7
    # thiếu một nguồn thì dùng nguồn còn lại
    assert np.argmax(F.bass_distribution(None, chroma, p)[0]) == 7


def test_stem_bass_probs_silence_is_uniform():
    chroma = np.zeros((4, 12))
    chroma[:3, 9] = 1.0
    p = F.stem_bass_probs(chroma)
    assert p[0, 9] == pytest.approx(1.0)
    np.testing.assert_allclose(p[3], np.full(12, 1 / 12))


# ---------------------------------------------------------------- ánh xạ và trộn

def test_cnnlstm_mapping_drops_inversions_and_reduces_quality():
    src = ["N", "C:maj", "C:maj/5", "Eb:min9", "G:9", "D:sus4(b7)"]
    dst = ["N", "C:maj", "D#:min7", "G:7", "D:sus4", "C:min"]
    probs = np.array([[0.1, 0.2, 0.3, 0.1, 0.2, 0.1]])
    out = F.cnnlstm_to_chordmini(probs, src, dst)
    np.testing.assert_allclose(out, [[0.1, 0.5, 0.1, 0.2, 0.1, 0.0]])


def test_resample_probs_averages_and_fills():
    src_t = np.array([0.05, 0.15, 0.25, 0.35])
    src = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
    out = F.resample_probs(src_t, src, np.array([0.1, 0.3, 0.5, 0.7]))
    np.testing.assert_allclose(out[0], [0.5, 0.5])
    np.testing.assert_allclose(out[1], [1.0, 0.0])
    np.testing.assert_allclose(out[3], [1.0, 0.0])  # ngoài phạm vi: khung gần nhất


def test_mix_log_weights():
    a = np.array([[0.9, 0.1]])
    b = np.array([[0.1, 0.9]])
    np.testing.assert_allclose(F.mix_log(a, b, 0.0), a, atol=1e-5)
    np.testing.assert_allclose(F.mix_log(a, b, 0.5), [[0.5, 0.5]], atol=1e-5)


# ---------------------------------------------------------------- làm mượt

def test_viterbi_removes_flicker():
    obs = np.full((20, 2), np.log(0.4))
    obs[:, 0] = np.log(0.6)
    obs[10] = [np.log(0.3), np.log(0.7)]   # một khung chớp nháy
    path = F.viterbi_uniform(obs, 0.9)
    assert (path == 0).all()
    assert (F.viterbi_uniform(obs, 0.5)[10] == 1)


def test_beat_voting_snaps_changes_to_beats():
    hop = 0.1
    times = (np.arange(40) + 0.5) * hop
    probs = np.full((40, 3), 0.05)
    probs[:, 0] = 0.9
    probs[23:, :] = 0.05
    probs[23:, 1] = 0.9  # model đổi hợp âm ở 2.3 s, phách ở 2.0 s
    beats = BeatGrid(np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0]), 60.0)
    p = F.FusionParams(smoothing="beat")
    labels = F.decode(probs, times, beats, 4.0, p)
    segs = F.frames_to_segments(labels, times, 4.0)
    # phách 2–3 s: 3 khung C, 7 khung G -> G; đổi hợp âm đúng ở phách 2.0 s
    assert [(round(a, 6), round(b, 6), i) for a, b, i in segs] == [(0.0, 2.0, 0), (2.0, 4.0, 1)]
    labels = F.decode(probs, times, beats, 4.0, F.FusionParams(smoothing="beat_hmm", self_prob=0.9))
    assert [i for *_, i in F.frames_to_segments(labels, times, 4.0)] == [0, 1]


def test_beat_smoothing_without_beats_falls_back():
    probs = np.array([[0.9, 0.1]] * 5)
    times = (np.arange(5) + 0.5) * 0.1
    assert (F.decode(probs, times, None, 0.5, F.FusionParams(smoothing="beat")) == 0).all()


def test_merge_short_absorbs_into_better_neighbour():
    times = (np.arange(30) + 0.5) * 0.1
    probs = np.full((30, 3), 0.1)
    probs[:, 0] = 0.8
    probs[10:12, 2] = 0.85   # đoạn 1.0–1.2 s nhãn 2, ngắn
    segs = [(0.0, 1.0, 0), (1.0, 1.2, 2), (1.2, 3.0, 1)]
    out = F.merge_short(segs, probs, times, 0.5)
    assert out == [(0.0, 1.2, 0), (1.2, 3.0, 1)]


# ---------------------------------------------------------------- fill_gaps và fuse

def test_fill_gaps_covers_whole_duration():
    segs = [ChordSegment(0.5, 1.0, "C:maj"), ChordSegment(1.2, 2.0, "G:maj")]
    out = F.fill_gaps(segs, 3.0)
    _covers(out, 3.0)
    assert [s.label for s in out] == ["N", "C:maj", "G:maj", "N"]


def test_fill_gaps_clips_past_duration():
    out = F.fill_gaps([ChordSegment(0.0, 5.0, "C:maj")], 3.0)
    assert [(s.start, s.end) for s in out] == [(0.0, 3.0)]


def _synthetic_models(duration=16.0):
    hop_cm = F.FRAME_HOP
    plan = [("C:maj", 4.0), ("G:maj", 4.0), ("A:min", 4.0), ("F:maj", 4.0)]
    t_cm, p_cm = _one_hot_track(plan, hop_cm, CM_VOCAB)
    cn_vocab = ["N", "C:maj", "C:maj/5", "G:maj", "A:min", "F:maj"]
    cn_plan = [("C:maj/5", 4.0), ("G:maj", 4.0), ("A:min", 4.0), ("F:maj", 4.0)]
    t_cn, p_cn = _one_hot_track(cn_plan, 0.025, cn_vocab, conf=0.8)
    bass = np.full((len(t_cn), 13), 0.01)
    bass_pc = {0: 7, 1: 7, 2: 9, 3: 5}  # C/G, G, Am, F
    for i, t in enumerate(t_cn):
        bass[i, 1 + bass_pc[min(3, int(t // 4))]] = 0.9
    bass /= bass.sum(axis=1, keepdims=True)
    return {
        "chordmini": _model("chordmini_btc", t_cm, p_cm, CM_VOCAB),
        "cnnlstm": _model("cnnlstm", t_cn, p_cn, cn_vocab, {"bass_probs": bass}),
    }


@pytest.mark.parametrize("smoothing", ["frame", "beat", "hmm", "beat_hmm"])
def test_fuse_synthetic_progression_with_inversion(smoothing):
    duration = 16.5
    models = _synthetic_models()
    beats = BeatGrid(np.arange(0.0, 16.0, 1.0), np.arange(0.0, 16.0, 4.0), 60.0)
    p = F.FusionParams(cnn_weight=0.3, smoothing=smoothing, bass_source="cnn", bass_threshold=0.5)
    out = F.fuse(models, beats, None, duration, params=p)
    _covers(out, duration)
    labels = [s.label for s in out]
    assert labels[:4] == ["C:maj/5", "G:maj", "A:min", "F:maj"]
    for s in out:
        assert s.confidence is None or 0.0 <= s.confidence <= 1.0
    assert out[0].confidence > 0.5


def test_fuse_default_params_and_missing_inputs():
    models = _synthetic_models()
    out = F.fuse(models, None, None, 16.0)
    _covers(out, 16.0)
    # chỉ có CNN-LSTM: dùng từ vựng của nó (đã có hợp âm đảo)
    out = F.fuse({"cnnlstm": models["cnnlstm"]}, None, None, 16.0)
    _covers(out, 16.0)
    assert out[0].label == "C:maj/5"
    # không có xác suất theo khung: lấp khoảng trống cho đoạn sẵn có
    seg_only = ChordModelOutput("x", [ChordSegment(1.0, 2.0, "C:maj", 0.7)])
    out = F.fuse({"chordmini": seg_only}, None, None, 3.0)
    _covers(out, 3.0)
    assert [s.label for s in out] == ["N", "C:maj", "N"]


def test_fuse_with_bass_stem_file(tmp_path):
    import soundfile as sf

    sr = 11025
    t = np.arange(int(sr * 8.0)) / sr
    # 0–4 s bass G2 (98 Hz) dưới hợp âm C -> C/G; 4–8 s bass G2 dưới G -> G (nốt gốc)
    y = 0.5 * np.sin(2 * np.pi * 98.0 * t)
    sf.write(tmp_path / "bass.wav", y, sr)
    plan = [("C:maj", 4.0), ("G:maj", 4.0)]
    t_cm, p_cm = _one_hot_track(plan, F.FRAME_HOP, CM_VOCAB)
    models = {"chordmini": _model("chordmini_btc", t_cm, p_cm, CM_VOCAB)}
    p = F.FusionParams(smoothing="hmm", bass_source="stem", bass_threshold=0.3)
    out = F.fuse(models, None, tmp_path / "bass.wav", 8.0, params=p)
    _covers(out, 8.0)
    assert [s.label for s in out] == ["C:maj/5", "G:maj"]


# ---------------------------------------------------------------- các lỗi đã sửa sau review

def test_cnnlstm_abstains_on_labels_it_cannot_produce():
    cm_vocab = ["N", "C:maj", "C:maj6", "X"]
    cm = _model("chordmini_btc", [0.05], [[0.0, 0.2, 0.7, 0.1]], cm_vocab)
    cn = _model("cnnlstm", [0.05], [[0.0, 1.0]], ["N", "C:maj"])
    p_cn = F.cnnlstm_on_chordmini(cn, cm, np.array([0.05]), cm.frame_probs)
    np.testing.assert_allclose(p_cn, [[0.0, 0.2, 0.7, 0.1]])
    mixed = F.mix_models(cm.frame_probs, p_cn, cm_vocab, 0.35, "full")
    assert np.argmax(mixed[0]) == 2   # C:maj6 không bị loại vì CNN-LSTM không có nhãn này


def test_cnnlstm_bass_keeps_no_bass_mass():
    bass = np.zeros((1, 13))
    bass[0, 0], bass[0, 1 + 7] = 0.9, 0.1
    cn = _model("cnnlstm", [0.05], [[1.0]], ["N"], {"bass_probs": bass})
    b = F.cnnlstm_bass(cn, np.array([0.05]))
    assert b[0, 7] == pytest.approx(0.1)
    assert F.pick_bass(b[0], "C:maj", 0.4) == "C:maj"   # model nói không có bass -> không thêm /5


def test_mapping_unknown_minor_quality_stays_minor():
    out = F.cnnlstm_to_chordmini(np.array([[1.0]]), ["A:min11"], ["A:maj", "A:min", "A:min7"])
    np.testing.assert_allclose(out, [[0.0, 0.0, 1.0]])
    out = F.cnnlstm_to_chordmini(np.array([[1.0]]), ["A:min(9)"], ["A:maj", "A:min"])
    np.testing.assert_allclose(out, [[0.0, 1.0]])


def test_beats_outside_song_fall_back_to_frames():
    probs = np.full((40, 2), 0.1)
    probs[:20, 0] = probs[20:, 1] = 0.9
    times = (np.arange(40) + 0.5) * 0.1
    beats = BeatGrid(np.array([0.0]), np.array([0.0]), 60.0)
    labels = F.decode(probs, times, beats, 4.0, F.FusionParams(smoothing="beat"))
    assert labels[0] == 0 and labels[-1] == 1


def test_invalid_bass_source_raises():
    with pytest.raises(ValueError):
        F.bass_distribution(None, None, F.FusionParams(bass_source="stems"))


def test_family_mix_quality_weight():
    """quality_weight: 0 giữ loại hợp âm (C so với C7) của ChordMini, 1 lấy của CNN-LSTM; phân bố họ
    (gốc + maj/min) vẫn trộn theo cnn_weight và không đổi khi quality_weight đổi."""
    vocab = ["N", "C:maj", "C:7", "A:min"]
    p_cm = np.array([[0.05, 0.6, 0.15, 0.2]])
    p_cn = np.array([[0.05, 0.1, 0.6, 0.25]])
    fam_idx = [[0], [1, 2], [3]]
    q0 = F.mix_models(p_cm, p_cn, vocab, 0.5, "family", 0.0)
    q1 = F.mix_models(p_cm, p_cn, vocab, 0.5, "family", 1.0)
    qh = F.mix_models(p_cm, p_cn, vocab, 0.5, "family", 0.5)
    for m in (q0, q1, qh):
        np.testing.assert_allclose(m.sum(axis=1), 1.0)
    for f in fam_idx:   # phân bố họ như nhau
        np.testing.assert_allclose(q0[0, f].sum(), q1[0, f].sum(), atol=1e-9)
        np.testing.assert_allclose(q0[0, f].sum(), qh[0, f].sum(), atol=1e-9)
    assert q0[0, 1] / q0[0, 2] == pytest.approx(0.6 / 0.15, rel=1e-4)    # tỉ lệ C : C7 của ChordMini
    assert q1[0, 1] / q1[0, 2] == pytest.approx(0.1 / 0.6, rel=1e-4)     # tỉ lệ C : C7 của CNN-LSTM
    assert q0[0, 1] / q0[0, 2] > qh[0, 1] / qh[0, 2] > q1[0, 1] / q1[0, 2]
    np.testing.assert_allclose(q0, F.mix_models(p_cm, p_cn, vocab, 0.5, "family"))   # mặc định như cũ


def test_quality_weight_alone_still_mixes():
    """cnn_weight=0 nhưng quality_weight>0: vẫn dùng CNN-LSTM cho loại hợp âm."""
    vocab = ["N", "C:maj", "C:7"]
    cm = _model("chordmini_btc", [0.05], [[0.0, 0.8, 0.2]], vocab)
    cn = _model("cnnlstm", [0.05], [[0.0, 0.1, 0.9]], vocab)
    times = np.array([0.05])
    p, _ = F.chord_probs({"chordmini": cm, "cnnlstm": cn}, times, 0.0, "family", 1.0)
    assert np.argmax(p[0]) == 2
    p, _ = F.chord_probs({"chordmini": cm, "cnnlstm": cn}, times, 0.0, "family", 0.0)
    assert np.argmax(p[0]) == 1


def test_quality_weight_keeps_chordmini_share_for_abstained_labels():
    """CNN-LSTM không có maj6: trong họ C, maj6 giữ tỉ lệ của ChordMini, không bị đẩy lên khi CNN-LSTM
    gần như không đặt khối nào vào họ C."""
    vocab = ["N", "C:maj", "C:maj6", "G:maj"]
    p_cm = np.array([[0.0, 0.9, 0.05, 0.05]])
    p_cn = np.array([[0.0, 0.001, 0.05, 0.949]])     # cột maj6 đã chép từ ChordMini (cnnlstm_on_chordmini)
    abstain = np.array([False, False, True, False])
    m = F.mix_models(p_cm, p_cn, vocab, 0.0, "family", 1.0, abstain)
    assert m[0, 1] / m[0, 2] == pytest.approx(0.9 / 0.05, rel=1e-3)
    bad = F.mix_models(p_cm, p_cn, vocab, 0.0, "family", 1.0)   # không có mặt nạ: maj6 thắng sai
    assert bad[0, 2] > bad[0, 1]


# ---------------------------------------------------------------- trộn ba model (ChordFormer)

def _three_models(n=40):
    times = (np.arange(n) + 0.5) * F.FRAME_HOP
    cm_vocab = ["N", "C:maj", "C:maj7", "A:min"]
    cn_vocab = ["N", "C:maj", "C:maj7", "A:min"]
    p_cm = np.tile([0.05, 0.25, 0.05, 0.65], (n, 1))     # ChordMini nghiêng về Am
    p_cn = np.tile([0.05, 0.10, 0.80, 0.05], (n, 1))     # CNN-LSTM: Cmaj7
    p_cf = np.tile([0.05, 0.70, 0.20, 0.05], (n, 1))     # ChordFormer: C
    bass_cn = np.zeros((n, 13)); bass_cn[:, 1 + 0] = 1.0  # C
    bass_cf = np.zeros((n, 13)); bass_cf[:, 1 + 4] = 1.0  # E
    return times, {
        "chordmini": _model("chordmini", times, p_cm, cm_vocab),
        "cnnlstm": _model("cnnlstm", times, p_cn, cn_vocab, {"bass_probs": bass_cn}),
        "chordformer": _model("chordformer", times, p_cf, cn_vocab, {"bass_probs": bass_cf}),
    }


def test_chord_probs_multi_weights_and_fallback():
    times, models = _three_models()
    p = F.FusionParams(cnn_weight=0.3, cf_weight=0.6, quality_weight=0.0)
    probs, vocab = F.chord_probs_multi(models, times, p)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    fam_c = probs[0, vocab.index("C:maj")] + probs[0, vocab.index("C:maj7")]
    assert fam_c > probs[0, vocab.index("A:min")]           # 0.9 trọng số cho họ C thắng ChordMini
    # cf_weight = 0 và quality_weight = 0: chỉ còn ChordMini + CNN-LSTM, giống mix_models
    p2 = F.FusionParams(cnn_weight=0.6, cf_weight=0.0, quality_source="chordformer", quality_weight=0.0)
    a, _ = F.chord_probs_multi(models, times, p2)
    b, _ = F.chord_probs(models, times, 0.6, "family", 0.0)
    np.testing.assert_allclose(a, b, atol=1e-9)
    # Thiếu ChordFormer: trọng số chia lại, không lỗi
    no_cf = {k: v for k, v in models.items() if k != "chordformer"}
    c, _ = F.chord_probs_multi(no_cf, times, p)
    np.testing.assert_allclose(c.sum(axis=1), 1.0)


def test_model_bass_sources():
    times, models = _three_models()
    cn = F.model_bass(models, times, "cnnlstm")
    cf = F.model_bass(models, times, "chordformer")
    both = F.model_bass(models, times, "both")
    assert cn[0].argmax() == 0 and cf[0].argmax() == 4
    np.testing.assert_allclose(both, (cn + cf) / 2)
    only_cn = {k: v for k, v in models.items() if k != "chordformer"}
    np.testing.assert_allclose(F.model_bass(only_cn, times, "chordformer"), cn)   # thiếu thì dùng model còn lại
    with pytest.raises(ValueError):
        F.model_bass(models, times, "xyz")


def test_default_uses_chordformer_and_fuse_runs():
    assert F.uses_chordformer(F.DEFAULT_PARAMS)
    assert not F.uses_chordformer(F.FusionParams(cf_weight=0.0, bass_model="cnnlstm"))
    times, models = _three_models()
    segs = F.fuse(models, None, None, float(times[-1] + F.FRAME_HOP / 2))
    assert segs and segs[0].label.startswith("C:")
