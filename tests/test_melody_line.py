"""Test chọn dòng giai điệu (chordtrack.melody.line) bằng dữ liệu tổng hợp, chỉ CPU."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from chordtrack.melody import line as L
from chordtrack.types import BeatGrid, MelodyModelOutput, MelodyNote as N, PitchTrack


def _no_overlap(notes):
    return all(a.end <= b.start + 1e-9 for a, b in zip(notes, notes[1:])) and all(n.end > n.start for n in notes)


def test_source_class():
    assert L.source_class("Voice") == L.VOICE
    assert L.source_class("choir") == L.VOICE
    assert L.source_class("violin") == L.MELODIC
    assert L.source_class("bassoon") == L.MELODIC
    assert L.source_class("electric bass") == L.EXCLUDED
    assert L.source_class("drums") == L.EXCLUDED
    assert L.source_class("synth pad") == L.EXCLUDED
    assert L.source_class("piano") == L.ACCOMP
    assert L.source_class(None) == L.UNKNOWN
    assert L.source_class("33") == L.EXCLUDED     # GM: electric bass
    assert L.source_class("40") == L.MELODIC      # GM: violin
    assert L.source_class("program:53") == L.VOICE


@pytest.mark.parametrize("strategy", ["skyline", "salience", "viterbi"])
def test_two_overlapping_notes(strategy):
    notes = [N(0.0, 1.0, 60, 0.9, "piano"), N(0.5, 1.5, 72, 0.8, "voice"), N(0.0, 2.0, 36, 0.9, "bass")]
    out = L.notes_to_line(notes, L.LineParams(strategy=strategy))
    assert _no_overlap(out)
    assert [n.pitch for n in out] == [60, 72]           # bass bị bỏ, nốt trên lấy phần chồng nhau
    assert out[0].end == pytest.approx(0.5) and out[1].start == pytest.approx(0.5)
    assert all(0 <= n.confidence <= 1 for n in out)
    assert out[0].confidence < 0.9                       # chỉ giữ một nửa nốt: kém chắc hơn


def test_skyline_keeps_bass_when_not_excluded():
    notes = [N(0.0, 1.0, 36, 0.9, "bass")]
    assert L.notes_to_line(notes, L.LineParams(strategy="skyline", exclude_accomp=False, pitch_min=30))[0].pitch == 36
    assert L.notes_to_line(notes, L.LineParams(strategy="skyline")) == []


def test_source_strict_prefers_voice_over_higher_instrument():
    notes = [N(0.0, 1.0, 64, 0.5, "voice"), N(0.0, 1.0, 76, 0.9, "piano")]
    out = L.notes_to_line(notes, L.LineParams(strategy="salience", source_mode="strict"))
    assert [n.pitch for n in out] == [64]
    out = L.notes_to_line(notes, L.LineParams(strategy="skyline"))
    assert [n.pitch for n in out] == [76]


def test_viterbi_avoids_octave_flicker():
    # Giai điệu 67 đều; một nốt quãng tám trên ngắn chen vào giữa. Viterbi không nhảy lên, salience thì có.
    notes = [N(0.0, 2.0, 67, 0.8, None), N(0.9, 1.1, 79, 0.8, None)]
    p = L.LineParams(strategy="viterbi", jump_penalty=1.0, cut_penalty=0.3)
    assert [n.pitch for n in L.notes_to_line(notes, p)] == [67]
    assert [n.pitch for n in L.notes_to_line(notes, L.LineParams(strategy="salience", w_dur=0.0))] == [67, 79, 67]


def test_min_dur_drops_short_output_notes():
    notes = [N(0.0, 0.02, 60, 0.9, None), N(0.1, 0.5, 62, 0.9, None)]
    out = L.notes_to_line(notes, L.LineParams(strategy="skyline", min_input_dur=0.0, min_dur=0.05))
    assert [n.pitch for n in out] == [62]


def test_empty_inputs():
    assert L.notes_to_line([]) == []
    assert L.f0_to_line(PitchTrack(np.array([]), np.array([]))) == []
    assert L.extract_line(MelodyModelOutput("x")) == []


def _vibrato_track(tuning: float = 0.0):
    t = np.arange(0, 2, 0.01)
    m = np.where(t < 1, 60.0, 64.0) + 0.4 * np.sin(2 * np.pi * 5.5 * t) + tuning
    f = L.midi_to_hz(m)
    f[(t > 1.5) & (t < 1.6)] = 0.0
    return PitchTrack(t, f, np.ones_like(t))


def test_f0_vibrato_is_one_note():
    out = L.f0_to_line(_vibrato_track())
    assert [n.pitch for n in out] == [60, 64, 64]        # khoảng lặng 0.1 s tách nốt 64 làm hai
    assert out[0].start == pytest.approx(0.0, abs=0.01)
    assert out[0].end == pytest.approx(1.0, abs=0.03)
    assert _no_overlap(out)


def test_f0_tuning_offset_compensated():
    out = L.f0_to_line(_vibrato_track(tuning=0.45))     # lệch gần nửa cung: không bù sẽ thành 60/61 lẫn lộn
    assert [n.pitch for n in out][:2] == [60, 64]
    assert L.estimate_tuning(np.array([60.3, 62.3, 65.3])) == pytest.approx(0.3)


def test_f0_voicing_threshold_and_gap_fill():
    t = np.arange(0, 1, 0.01)
    f = np.full(len(t), L.midi_to_hz(69))
    conf = np.ones(len(t))
    conf[50:52] = 0.1                                    # 20 ms không chắc: lấp lại
    out = L.f0_to_line(PitchTrack(t, f, conf), L.LineParams(voicing_threshold=0.5))
    assert len(out) == 1 and out[0].pitch == 69
    conf[30:60] = 0.1
    out = L.f0_to_line(PitchTrack(t, f, conf), L.LineParams(voicing_threshold=0.5))
    assert len(out) == 2


def test_f0_step_splits_and_glitch_does_not():
    t = np.arange(0, 1, 0.01)
    m = np.full(len(t), 60.0)
    m[50:] = 62.0
    m[20:23] = 67.0                                      # nhiễu 30 ms: không thành nốt riêng
    out = L.f0_to_line(PitchTrack(t, L.midi_to_hz(m)), L.LineParams(median_frames=1))
    assert [n.pitch for n in out] == [60, 62]
    assert out[1].start == pytest.approx(0.495, abs=0.011)


def test_notes_to_f0_and_monophonic():
    notes = [N(0.0, 1.0, 69), N(0.5, 1.5, 81)]
    mono = L.make_monophonic(notes)
    assert _no_overlap(mono) and mono[0].end == 0.5
    f0 = L.notes_to_f0(notes, np.array([0.25, 0.75, 1.25, 2.0]))
    assert f0 == pytest.approx([440.0, 880.0, 880.0, 0.0])


def test_combine_lines_fills_gaps_only():
    primary = [N(0.0, 1.0, 60), N(3.0, 4.0, 62)]
    secondary = [N(0.5, 2.0, 70), N(2.0, 5.0, 72)]
    out = L.combine_lines(primary, secondary, min_gap=0.3)
    assert [(n.start, n.end, n.pitch) for n in out] == [(0.0, 1.0, 60), (1.0, 2.0, 70), (2.0, 3.0, 72),
                                                        (3.0, 4.0, 62), (4.0, 5.0, 72)]


def test_quantize_to_beats():
    grid = BeatGrid(np.arange(0, 5, 0.5), np.arange(0, 5, 2.0), 120.0)
    notes = [N(0.13, 0.49, 60), N(0.52, 0.54, 62), N(1.30, 1.70, 64)]
    out = L.quantize_to_beats(notes, grid, subdivisions=4, max_shift=0.05)
    assert [(round(n.start, 3), round(n.end, 3)) for n in out] == [(0.125, 0.5), (0.5, 0.625), (1.3, 1.7)]
    assert _no_overlap(out)


def test_review_regressions():
    assert L.source_class("Tenor Sax") == L.MELODIC
    assert L.source_class("47") == L.EXCLUDED                  # GM timpani
    # hai nốt nhanh cùng căn về một điểm: không được mất nốt
    grid = BeatGrid(np.arange(0, 4, 0.5), np.arange(0, 4, 2.0), 120.0)
    out = L.quantize_to_beats([N(0.0, 0.06, 60), N(0.06, 0.2, 62)], grid, subdivisions=4, max_shift=0.08)
    assert [n.pitch for n in out] == [60, 62] and _no_overlap(out)
    # f0_to_line không sửa confidence của người gọi
    t = np.arange(0, 1, 0.01)
    conf = np.ones(len(t))
    conf[50:52] = 0.1
    before = conf.copy()
    L.f0_to_line(PitchTrack(t, np.full(len(t), 440.0), conf), L.LineParams(voicing_threshold=0.5))
    assert np.array_equal(conf, before)


def test_elementary_matches_bruteforce():
    rng = np.random.default_rng(1)
    notes = [N(float(s), float(s + d), int(p)) for s, d, p in
             zip(rng.uniform(0, 10, 60), rng.uniform(0.05, 2, 60), rng.integers(50, 80, 60))]
    bounds, active = L._elementary(notes)
    for k, a in enumerate(active):
        m = (bounds[k] + bounds[k + 1]) / 2
        assert sorted(a.tolist()) == [i for i, n in enumerate(notes) if n.start <= m < n.end]


def test_chorus_as_voice_and_gated_combine():
    assert L.source_class("voice_chorus") == L.VOICE
    assert L.source_class("voice_chorus", chorus_as_voice=False) == L.MELODIC
    assert L.source_class("voice", chorus_as_voice=False) == L.VOICE
    notes = [N(0.0, 1.0, 64, None, "voice_chorus"), N(0.0, 1.0, 72, None, "violin")]
    p = L.LineParams(strategy="salience", source_mode="strict")
    assert [n.pitch for n in L.notes_to_line(notes, p)] == [64]
    assert [n.pitch for n in L.notes_to_line(notes, replace(p, chorus_as_voice=False))] == [72]
    # cổng giọng: dòng f0 trong khoảng có giọng (nới 0.1 s), dòng model nốt ngoài khoảng đó
    gate = L.voice_intervals([N(1.0, 2.0, 60, None, "voice"), N(1.95, 3.0, 62, None, "voice"),
                              N(0.0, 5.0, 40, None, "piano")], pad=0.1)
    assert gate == [(0.9, 3.1)]
    primary = [N(0.5, 3.5, 67)]
    secondary = [N(0.0, 1.0, 70), N(3.0, 5.0, 72)]
    out = L.combine_gated(primary, secondary, gate)
    assert [(round(n.start, 2), round(n.end, 2), n.pitch) for n in out] == [(0.0, 0.9, 70), (0.9, 3.1, 67),
                                                                            (3.1, 5.0, 72)]
