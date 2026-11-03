"""Test pipeline không cần GPU: thay các bước model bằng hàm giả."""
import json

import numpy as np
import pytest
import soundfile as sf

from chordtrack import cli, paths, pipeline
from chordtrack.audio import AudioSource
from chordtrack.types import BeatGrid, ChordModelOutput, ChordSegment, KeyEstimate, Section, Stems

DURATION = 8.0
SR = 8000
ALL_STEPS = ["fetch", "normalize", "beats", "chordmini", "cnnlstm", "chordformer"]        # mặc định: không tách nguồn
RERUN_STEPS = ["fetch", "beats", "chordmini", "cnnlstm", "chordformer"]                     # mix.wav đã có trong cache
SEPARATION_STEPS = ["fetch", "separate", "beats", "chordmini", "cnnlstm", "chordformer"]


def check_schema(result: dict) -> None:
    """Dạng JSON mà extension sẽ đọc."""
    assert set(result) == {"version", "song_id", "source", "duration", "tempo", "meter", "key", "keys", "modulations",
                           "beats", "downbeats", "chords", "sections", "timing", "pipeline"}
    assert result["version"] == 1
    assert isinstance(result["song_id"], str)
    assert {"type", "title", "url"} <= set(result["source"])
    assert isinstance(result["duration"], float)
    assert isinstance(result["meter"], int) and result["meter"] >= 1
    assert result["key"] is None or set(result["key"]) == {"tonic", "mode", "confidence"}
    keys = result["keys"]
    for k in keys:
        assert set(k) == {"start", "end", "tonic", "mode", "confidence", "white_key_transpose"}
        assert -6 <= k["white_key_transpose"] <= 5
    if keys:
        assert keys[0]["start"] == 0.0 and keys[-1]["end"] == pytest.approx(result["duration"], abs=1e-3)
        for a, b in zip(keys, keys[1:]):
            assert a["end"] == pytest.approx(b["start"])
    assert len(result["modulations"]) == max(0, len(keys) - 1)
    for m, b in zip(result["modulations"], keys[1:]):
        assert set(m) == {"time", "from", "to", "semitones", "transpose_from", "transpose_to"}
        assert m["time"] == b["start"] and m["transpose_to"] == b["white_key_transpose"]
    assert all(isinstance(t, float) for t in result["beats"] + result["downbeats"])
    assert set(result["downbeats"]) <= set(result["beats"])
    chords = result["chords"]
    assert chords and chords[0]["start"] == 0.0 and chords[-1]["end"] == pytest.approx(result["duration"], abs=1e-3)
    for a, b in zip(chords, chords[1:]):
        assert a["end"] == pytest.approx(b["start"])
    for c in chords:
        assert set(c) == {"start", "end", "label", "symbol", "roman", "white_key_symbol", "key_index",
                          "confidence", "uncertain"}
        assert isinstance(c["uncertain"], bool)
        assert c["key_index"] is None or 0 <= c["key_index"] < len(keys)
        if c["key_index"] is None:
            assert c["roman"] is None and c["white_key_symbol"] is None
    base = {"start", "end", "label", "progression", "roman", "key_index"}
    for s in result["sections"]:
        assert base <= set(s) and set(s) - base <= {"transposed_from", "semitones"}
    assert set(result["timing"]) == set(pipeline.TIMING_KEYS)
    assert {"separation_model", "bass_model", "chord_models"} <= set(result["pipeline"])
    json.dumps(result)  # phải ghi ra JSON được


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    """Thay mọi bước nặng bằng hàm giả, ghi lại thứ tự và đối số."""
    monkeypatch.setattr(paths, "CACHE_DIR", tmp_path / "cache")
    wav = tmp_path / "song.wav"
    sf.write(wav, np.zeros((int(DURATION * SR), 2)), SR)
    calls = []

    def write(p):
        sf.write(p, np.zeros((int(DURATION * SR), 2)), SR)

    def fake_fetch(source):
        calls.append(("fetch", source))
        return AudioSource(song_id="abcdefghijk", path=wav, type="youtube", video_id="abcdefghijk",
                           url="https://www.youtube.com/watch?v=abcdefghijk", title="Bài thử")

    def fake_separate(audio_path, out_dir):
        calls.append(("separate", out_dir.name))
        out_dir.mkdir(parents=True, exist_ok=True)
        stems = Stems(*(out_dir / f"{n}.wav" for n in pipeline.STEM_NAMES))
        for n in pipeline.STEM_NAMES:
            write(getattr(stems, n))
        return stems

    def fake_normalize(src, dst):
        calls.append(("normalize", dst.name))
        write(dst)

    def fake_beats(path):
        calls.append(("beats", path.name))
        beats = np.arange(0.5, DURATION, 0.5)
        return BeatGrid(beats=beats, downbeats=beats[::4], tempo=120.0)

    def fake_chordmini(path):
        calls.append(("chordmini", path.name))
        segs = [ChordSegment(0.5, 2.0, "C:maj", 0.9), ChordSegment(2.0, 4.0, "G:maj", 0.4),
                ChordSegment(4.0, 6.0, "A:min7", 0.8), ChordSegment(6.0, 7.5, "F:maj/3", 0.7)]
        return ChordModelOutput("chordmini_btc", segs)

    def fake_cnnlstm(path):
        calls.append(("cnnlstm", path.name))
        return ChordModelOutput("cnnlstm_submission", [ChordSegment(0.0, DURATION, "C:maj", 0.5)])

    def fake_chordformer(path):
        calls.append(("chordformer", path.name))
        return ChordModelOutput("chordformer_submission", [ChordSegment(0.0, DURATION, "C:maj", 0.5)])

    monkeypatch.setattr(pipeline.audio_mod, "fetch", fake_fetch)
    monkeypatch.setattr(pipeline.separation, "separate", fake_separate)
    monkeypatch.setattr(pipeline.separation, "normalize_audio", fake_normalize)
    monkeypatch.setattr(pipeline, "track_beats", fake_beats)
    monkeypatch.setattr(pipeline, "run_chordmini", fake_chordmini)
    monkeypatch.setattr(pipeline, "run_cnnlstm", fake_cnnlstm)
    monkeypatch.setattr(pipeline, "run_chordformer", fake_chordformer)
    return calls


def test_analyze_order_stems_and_schema(fake_env):
    result = pipeline.analyze("abcdefghijk")
    assert [c[0] for c in fake_env] == ALL_STEPS   # mặc định không gọi separate
    assert ("beats", "mix.wav") in fake_env
    assert ("chordmini", "mix.wav") in fake_env and ("cnnlstm", "mix.wav") in fake_env
    check_schema(result)
    assert result["song_id"] == "abcdefghijk"
    assert result["source"] == {"type": "youtube", "video_id": "abcdefghijk", "title": "Bài thử",
                                "url": "https://www.youtube.com/watch?v=abcdefghijk"}
    assert result["duration"] == pytest.approx(DURATION)
    assert result["tempo"] == 120.0
    assert result["meter"] == 4
    assert result["key"]["tonic"] == "C" and result["key"]["mode"] == "major"
    assert [c["symbol"] for c in result["chords"]] == ["N.C.", "C", "G", "Am7", "F/A", "N.C."]
    by_symbol = {c["symbol"]: c for c in result["chords"]}
    assert by_symbol["G"]["uncertain"] is True and by_symbol["C"]["uncertain"] is False
    assert result["pipeline"]["separation_model"] is None
    assert result["pipeline"]["bass_model"] is None
    assert result["pipeline"]["chord_models"] == ["chordmini-btc", "cnnlstm", "chordformer"]
    assert result["timing"]["total"] >= result["timing"]["beats"]
    assert (paths.CACHE_DIR / "abcdefghijk" / "result.json").exists()


def test_default_does_not_separate_and_fuses_without_stem(fake_env, monkeypatch):
    seen = {}
    real_fuse = pipeline.fusion.fuse

    def spy_fuse(models, beats, bass_audio, duration, params=None):
        seen["bass_audio"], seen["params"] = bass_audio, params
        return real_fuse(models, beats, bass_audio, duration, params)

    monkeypatch.setattr(pipeline.fusion, "fuse", spy_fuse)
    assert pipeline.PipelineConfig().use_separation is False
    pipeline.analyze("abcdefghijk")
    assert "separate" not in [c[0] for c in fake_env]
    assert seen["bass_audio"] is None and seen["params"].bass_source == "cnn"
    pipeline.analyze("abcdefghijk", use_separation=True)
    assert "separate" in [c[0] for c in fake_env]
    assert seen["bass_audio"].name == "bass.wav" and seen["params"].bass_source == "both"


def test_separation_on_runs_separate_and_reports_models(fake_env):
    result = pipeline.analyze("abcdefghijk", use_separation=True)
    assert [c[0] for c in fake_env] == SEPARATION_STEPS
    assert ("beats", "mix.wav") in fake_env and ("chordmini", "mix.wav") in fake_env
    check_schema(result)
    assert result["pipeline"]["separation_model"] == "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    assert result["pipeline"]["bass_model"] == "htdemucs_ft"
    # cache riêng, không đè kết quả mặc định
    assert not (paths.CACHE_DIR / "abcdefghijk" / "result.json").exists()
    assert list((paths.CACHE_DIR / "abcdefghijk").glob("result-*.json"))


def test_cache_not_mixed_between_separation_modes(fake_env):
    plain = pipeline.analyze("abcdefghijk")
    sep = pipeline.analyze("abcdefghijk", use_separation=True)
    fake_env.clear()
    assert pipeline.analyze("abcdefghijk") == plain
    assert pipeline.analyze("abcdefghijk", use_separation=True) == sep
    assert fake_env == []
    assert plain["pipeline"]["separation_model"] is None and sep["pipeline"]["separation_model"] is not None
    # result.json do bản cũ ghi khi mặc định còn tách nguồn: không được dùng cho cấu hình không tách
    (paths.CACHE_DIR / "abcdefghijk" / "result.json").write_text(json.dumps(sep), encoding="utf-8")
    fake_env.clear()
    again = pipeline.analyze("abcdefghijk")
    assert [c[0] for c in fake_env] == RERUN_STEPS
    assert again["pipeline"]["separation_model"] is None


def test_result_cache_and_force(fake_env):
    first = pipeline.analyze("abcdefghijk")
    fake_env.clear()
    again = pipeline.analyze("abcdefghijk")
    assert fake_env == []  # có cache: không chạy bước nào, kể cả fetch
    assert again == first
    fake_env.clear()
    pipeline.analyze("abcdefghijk", force=True)
    assert [c[0] for c in fake_env] == RERUN_STEPS


def test_youtube_cache_checked_before_fetch(fake_env):
    pipeline.analyze("abcdefghijk")
    fake_env.clear()
    pipeline.analyze("https://youtu.be/abcdefghijk")
    assert fake_env == []  # không gọi fetch (không cần mạng)


def test_cache_invalidated_when_code_changes(fake_env, monkeypatch):
    pipeline.analyze("abcdefghijk")
    monkeypatch.setattr(pipeline, "code_revision", lambda: "khac")
    fake_env.clear()
    result = pipeline.analyze("abcdefghijk")
    assert [c[0] for c in fake_env] == RERUN_STEPS
    assert result["pipeline"]["code_revision"] == "khac"


def test_separation_vram_only_when_step_ran(fake_env, monkeypatch):
    stats = {"roformer": {"peak_vram_mb": 1600.0, "seconds": 50.0}}
    fake = pipeline.separation.separate

    def separate_writing_stats(audio_path, out_dir):
        stems = fake(audio_path, out_dir)
        (out_dir / pipeline.separation.STATS_FILE).write_text(json.dumps(stats), encoding="utf-8")
        return stems

    monkeypatch.setattr(pipeline.separation, "separate", separate_writing_stats)
    first = pipeline.analyze("abcdefghijk", use_separation=True)
    assert first["pipeline"]["peak_vram_mb"]["roformer"] == 1600.0
    again = pipeline.analyze("abcdefghijk", force=True, use_separation=True)  # stats không đổi = stem lấy từ cache
    assert again["pipeline"]["peak_vram_mb"]["roformer"] is None


def test_no_separation_runs_everything_on_mix(fake_env):
    # chord_stem khác mix bị bỏ qua khi không tách nguồn
    result = pipeline.analyze("abcdefghijk", use_separation=False, chord_stem="no_bass")
    assert [c[0] for c in fake_env] == ["fetch", "normalize", "beats", "chordmini", "cnnlstm", "chordformer"]
    assert all(c[1] == "mix.wav" for c in fake_env[1:])
    check_schema(result)
    assert result["pipeline"]["separation_model"] is None
    assert result["pipeline"]["chord_stem"] == "mix"
    assert (paths.CACHE_DIR / "abcdefghijk" / "result.json").exists()   # cùng cấu hình mặc định


def test_chord_stem_parameter(fake_env):
    pipeline.analyze("abcdefghijk", use_separation=True, chord_stem="no_bass", beat_stem="instrumental")
    assert ("chordmini", "no_bass.wav") in fake_env and ("beats", "instrumental.wav") in fake_env
    with pytest.raises(ValueError):
        pipeline.analyze("abcdefghijk", chord_stem="drums")


def test_key_roman_and_sections(fake_env, monkeypatch):
    key = KeyEstimate("C", "major", 0.9)
    romans = {"C:maj": "I", "G:maj": "V", "A:min7": "vi7", "F:maj/3": "IV6", "G:maj/3": "V6", "A:min": "vi",
              "F:maj": "IV"}
    monkeypatch.setattr(pipeline.analysis, "estimate_key", lambda segs: key)
    monkeypatch.setattr(pipeline.analysis, "to_roman", lambda label, k: romans[label])
    monkeypatch.setattr(pipeline.analysis, "analyze_structure", lambda segs, beats, dur, k: [
        Section(0.0, DURATION, "verse", ["C:maj", "G:maj/3", "A:min", "F:maj"], ["I", "V6", "vi", "IV"])])
    result = pipeline.analyze("abcdefghijk")
    check_schema(result)
    assert result["key"] == {"tonic": "C", "mode": "major", "confidence": 0.9}
    assert [c["roman"] for c in result["chords"]] == [None, "I", "V", "vi7", "IV6", None]
    assert [c["white_key_symbol"] for c in result["chords"]] == ["N.C.", "C", "G", "Am7", "F/A", "N.C."]
    assert {c["key_index"] for c in result["chords"]} == {0}
    assert result["keys"] == [{"start": 0.0, "end": DURATION, "tonic": "C", "mode": "major", "confidence": 0.9,
                               "white_key_transpose": 0}]
    assert result["modulations"] == []
    assert result["sections"] == [{"start": 0.0, "end": DURATION, "label": "verse", "key_index": 0,
                                   "progression": ["C", "G/B", "Am", "F"], "roman": ["I", "V6", "vi", "IV"]}]


def test_sections_roman_null_without_key(fake_env, monkeypatch):
    monkeypatch.setattr(pipeline.analysis, "estimate_key", lambda segs: None)
    monkeypatch.setattr(pipeline.analysis, "estimate_key_regions", lambda *a, **k: [])
    monkeypatch.setattr(pipeline.analysis, "analyze_structure", lambda segs, beats, dur, k: [
        Section(0.0, DURATION, "A", ["C:maj"], [])])
    result = pipeline.analyze("abcdefghijk")
    check_schema(result)
    assert result["sections"][0]["roman"] is None and result["sections"][0]["progression"] == ["C"]
    assert result["keys"] == [] and result["modulations"] == []
    assert all(c["key_index"] is None and c["white_key_symbol"] is None for c in result["chords"])


def _modulating_song():
    """16 ô C trưởng (I–V–vi–IV) rồi 16 ô D trưởng, mỗi hợp âm một ô 2 s."""
    segs, t = [], 0.0
    for lab in ["C:maj", "G:maj", "A:min", "F:maj"] * 4 + ["D:maj", "A:maj/3", "B:min", "G:maj"] * 4:
        segs.append(ChordSegment(t, t + 2.0, lab, 0.9))
        t += 2.0
    beats = np.arange(0.0, t, 0.5)
    return segs, BeatGrid(beats=beats, downbeats=beats[::4], tempo=120.0), t


def _modulating_result(with_boundaries: bool = True) -> dict:
    segs, grid, dur = _modulating_song()
    key = pipeline.analysis.estimate_key(segs)
    sections = pipeline.analysis.analyze_structure(segs, grid, dur, key)
    bounds = [s.start for s in sections[1:]] if with_boundaries else None
    regions = pipeline.analysis.estimate_key_regions(segs, grid, dur, bounds)
    src = AudioSource(song_id="x", path=None, type="file", title="nâng tông")
    return pipeline.build_result(src, duration=dur, grid=grid, segments=segs, key=key, sections=sections,
                                 timing={}, config=pipeline.PipelineConfig(), regions=regions)


def test_build_result_with_key_change():
    result = _modulating_result()
    check_schema(result)
    assert [(k["tonic"], k["mode"], k["white_key_transpose"]) for k in result["keys"]] == [
        ("C", "major", 0), ("D", "major", 2)]
    (mod,) = result["modulations"]
    assert mod["from"] == "C major" and mod["to"] == "D major" and mod["semitones"] == 2
    assert (mod["transpose_from"], mod["transpose_to"]) == (0, 2)
    assert abs(mod["time"] - 32.0) < 2.0
    late = [c for c in result["chords"] if c["start"] >= 34.0]
    assert {c["key_index"] for c in late} == {1}
    # bậc La Mã theo giọng của vùng, ký hiệu phím trắng là nốt trắng
    assert [c["roman"] for c in late[:4]] == ["V/3", "vi", "IV", "I"]
    assert [c["white_key_symbol"] for c in late[:4]] == ["G/B", "Am", "F", "C"]
    assert [c["symbol"] for c in late[:4]] == ["A/C#", "Bm", "G", "D"]
    early = [c for c in result["chords"] if c["end"] <= 30.0]
    assert {c["key_index"] for c in early} == {0}
    assert [c["white_key_symbol"] for c in early[:4]] == ["C", "G", "Am", "F"]
    # đoạn ở vùng D có cùng vòng bậc La Mã với một đoạn ở vùng C: nhận ra là nâng tông
    secs = result["sections"]
    moved = [s for s in secs if s.get("transposed_from") is not None]
    assert moved, secs
    for s in moved:
        src_sec = secs[s["transposed_from"]]
        assert s["semitones"] == 2 and s["label"] == src_sec["label"]
        assert s["key_index"] == 1 and src_sec["key_index"] == 0


def test_cli_prints_modulation_and_white_keys():
    result = _modulating_result(with_boundaries=False)
    text = cli.summarize(result)
    assert "C major -> D major" in text and "đặt transpose đàn: +2" in text
    lines = cli.white_key_lines(result)
    assert "[transpose 0]" in lines[0] and "[transpose +2]" in lines[2]
    assert lines[1].split()[:4] == ["C", "G", "Am", "F"] and lines[3].split()[:4] == ["C", "G/B", "Am", "F"]


@pytest.mark.parametrize("t, s", [(0, "0:00.00"), (59.999, "1:00.00"), (61.5, "1:01.50"), (243.03, "4:03.03")])
def test_cli_fmt_time(t, s):
    assert cli._fmt_time(t) == s


def test_cli_writes_json_and_summary(fake_env, tmp_path, capsys):
    out = tmp_path / "out" / "r.json"
    assert cli.main(["abcdefghijk", "-o", str(out)]) == 0
    check_schema(json.loads(out.read_text(encoding="utf-8")))
    text = capsys.readouterr().out
    assert "Bài thử" in text and "Am7" in text and "Thời gian" in text
    assert "separate" not in [c[0] for c in fake_env]


@pytest.mark.parametrize("flags, separate", [([], False), (["--no-separation"], False), (["--separation"], True),
                                             (["--separation", "--chord-stem", "no_bass"], True)])
def test_cli_separation_flags(fake_env, flags, separate):
    assert cli.main(["abcdefghijk", *flags]) == 0
    assert ("separate" in [c[0] for c in fake_env]) is separate


def test_cli_rejects_stem_without_separation(fake_env):
    with pytest.raises(SystemExit):
        cli.main(["abcdefghijk", "--chord-stem", "instrumental"])
    assert fake_env == []


def test_main_key_replaces_compromise_key():
    from chordtrack.types import KeyRegion

    c, d, g = KeyEstimate("C", "major"), KeyEstimate("D", "major"), KeyEstimate("G", "major")
    regions = [KeyRegion(0, 100, c), KeyRegion(100, 140, d)]
    assert pipeline.main_key(g, regions) is c                      # G không thuộc vùng nào
    assert pipeline.main_key(d, regions) is d                      # giọng cả bài có trong một vùng: giữ
    assert pipeline.main_key(g, [KeyRegion(0, 100, c)]) is g       # một vùng: giữ estimate_key
    assert pipeline.main_key(None, regions) is None


def test_relabel_transposed_renames_whole_cluster():
    secs = [{"label": "verse", "key_index": 0}, {"label": "chorus", "key_index": 0},
            {"label": "B", "key_index": 1, "transposed_from": 1, "semitones": 2},
            {"label": "B", "key_index": 1}]
    pipeline._relabel_transposed(secs)
    assert [s["label"] for s in secs] == ["verse", "chorus", "chorus", "chorus"]
    # nhãn cũ còn được dùng ở vùng của đoạn gốc: chỉ đổi đoạn nâng tông
    secs = [{"label": "A", "key_index": 0}, {"label": "B", "key_index": 0},
            {"label": "A", "key_index": 1, "transposed_from": 1, "semitones": 2}]
    pipeline._relabel_transposed(secs)
    assert [s["label"] for s in secs] == ["A", "B", "B"]


def _grid(beats, downbeats):
    return BeatGrid(beats=np.asarray(beats, dtype=float), downbeats=np.asarray(downbeats, dtype=float), tempo=0.0)


def test_estimate_meter_most_common_bar_length():
    beats = np.arange(0.0, 30.0, 0.5)
    assert pipeline.estimate_meter(_grid(beats, beats[::4])) == 4
    assert pipeline.estimate_meter(_grid(beats, beats[::3])) == 3
    # nhịp 3/4 có vài ô 2 và 4 phách (bắt nhịp sai chỗ): vẫn là 3
    downs = [beats[i] for i in (0, 3, 6, 9, 11, 14, 17, 21, 24, 27)]
    assert pipeline.estimate_meter(_grid(beats, downs)) == 3
    # phách mạnh lệch phách vì làm tròn vẫn đếm đúng
    assert pipeline.estimate_meter(_grid(beats, beats[::4] + 0.0004)) == 4
    # hòa: lấy số nhỏ hơn
    assert pipeline.estimate_meter(_grid(beats, [beats[i] for i in (0, 3, 6, 10, 14)])) == 3


def test_estimate_meter_defaults_to_4():
    beats = np.arange(0.0, 10.0, 0.5)
    assert pipeline.estimate_meter(None) == 4
    assert pipeline.estimate_meter(_grid(beats, [])) == 4
    assert pipeline.estimate_meter(_grid(beats, beats[:1])) == 4
    assert pipeline.estimate_meter(_grid([], [])) == 4
