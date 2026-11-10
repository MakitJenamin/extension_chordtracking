"""Test wrapper MuScriptor bằng model giả (không cần weights); test GPU thật đánh dấu `gpu`."""
import json

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("muscriptor")
torch = pytest.importorskip("torch")

from muscriptor.events import NoteEndEvent, NoteStartEvent, ProgressEvent  # noqa: E402
from muscriptor.tokenizer.mt3 import MT3_FULL_PLUS_GROUP_NAMES  # noqa: E402

from chordtrack.melody import muscriptor as M  # noqa: E402
from chordtrack.melody.io import cache_path, load_output  # noqa: E402


def _start(i, pitch, t, inst):
    return NoteStartEvent(pitch=pitch, start_time=t, index=i, instrument=inst)


def test_group_names_match_muscriptor():
    assert M.INSTRUMENT_GROUPS == tuple(MT3_FULL_PLUS_GROUP_NAMES)
    assert MT3_FULL_PLUS_GROUP_NAMES[M.VOICE_SOURCE] == 17


def test_events_to_notes_pairs_and_drops_drums():
    a = _start(0, 60, 1.0, "voice")
    d = _start(1, 36, 1.2, "drums")
    b = _start(2, 64, 0.5, "acoustic_piano")
    events = [ProgressEvent(0, 2), a, d, NoteEndEvent(1.21, d), b,
              NoteEndEvent(2.0, a), NoteEndEvent(0.5, b)]   # b dài 0: kéo thành tối thiểu
    confs = iter([0.9, 0.4])
    notes, dt, dp = M.events_to_notes(events, lambda: next(confs))
    assert [(n.pitch, n.source, n.confidence) for n in notes] == [(64, "acoustic_piano", 0.4), (60, "voice", 0.9)]
    assert notes[0].end == pytest.approx(0.5 + M.MIN_NOTE_DURATION)
    assert notes[1].start == 1.0 and notes[1].end == 2.0
    assert dt.tolist() == [1.2] and dp.tolist() == [36]


def test_shift_notes_clamps_at_zero():
    notes, _, _ = M.events_to_notes([_start(0, 60, 0.01, "voice"), NoteEndEvent(1.0, _start(0, 60, 0.01, "voice"))])
    out = M.shift_notes(notes, 0.02)
    assert out[0].start == 0.0 and out[0].end == pytest.approx(0.98)
    assert M.shift_notes(notes, 0.0) is notes


def test_prob_recorder_takes_max_softmax():
    class LM:
        def _compute_logits(self, *a, **k):
            return torch.tensor([[0.0, np.log(3.0)]])

    lm = LM()
    rec = M._ProbRecorder(lm)
    assert rec() is None
    lm._compute_logits()
    assert rec() == pytest.approx(0.75)


def test_ensure_weights_auth_error(monkeypatch, tmp_path):
    import huggingface_hub
    from huggingface_hub.errors import GatedRepoError

    monkeypatch.setattr(M, "weights_dir", lambda size: tmp_path / size)

    class Resp:
        status_code = 401
        headers: dict = {}
        request = None

    def deny(*a, **k):
        raise GatedRepoError("401 gated", response=Resp())

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", deny)
    with pytest.raises(M.MuscriptorAuthError, match="hf auth login"):
        M.ensure_weights("small")
    with pytest.raises(ValueError):
        M.ensure_weights("huge")


class _FakeLM:
    """Mỗi bước sinh gọi _compute_logits một lần; xác suất token được chọn lần lượt 0.5, 0.8, ..."""

    def __init__(self):
        self.step = 0

    def _compute_logits(self, *a, **k):
        p = [0.5, 0.8, 0.6][self.step % 3]
        self.step += 1
        return torch.log(torch.tensor([[p, 1 - p]]))


class _FakeTM:
    def __init__(self):
        self._model = _FakeLM()
        self.kwargs = None

    def transcribe(self, audio, **kwargs):
        self.kwargs = kwargs
        a = _start(0, 67, 0.2, "voice")
        b = _start(1, 48, 0.3, "acoustic_bass")
        self._model._compute_logits()      # token cao độ của a
        yield a
        self._model._compute_logits()      # token cao độ của b
        yield b
        self._model._compute_logits()      # token kết thúc, không phải nốt mới
        yield NoteEndEvent(1.0, a)
        yield NoteEndEvent(1.5, b)


def _wav(path, seconds=1.0):
    sf.write(path, np.zeros(int(16000 * seconds), dtype=np.float32), 16000)
    return path


def test_transcribe_with_fake_model(monkeypatch, tmp_path):
    from muscriptor.transcription_model import TranscriptionModel

    fake = _FakeTM()
    monkeypatch.setattr(M, "ensure_weights", lambda size: tmp_path / "model.safetensors")
    monkeypatch.setattr(TranscriptionModel, "load_model", classmethod(lambda cls, *a, **k: fake))
    monkeypatch.setattr(M, "_detect_onset_delay", lambda path, onsets: 0.02)
    out = M.transcribe(_wav(tmp_path / "a.wav"), "medium", device="cpu")
    assert out.model_name == "muscriptor_medium"
    assert [(n.pitch, n.source) for n in out.notes] == [(67, "voice"), (48, "acoustic_bass")]
    assert [n.confidence for n in out.notes] == pytest.approx([0.5, 0.8])
    assert out.notes[0].start == pytest.approx(0.18)
    assert float(out.extras["cfg_coef"]) == 1.0 and str(out.extras["dtype"]) == "float32"
    assert float(out.extras["onset_delay"]) == 0.02
    assert float(out.extras["duration_s"]) == pytest.approx(1.0)
    assert fake.kwargs["cfg_coef"] == 1.0 and fake.kwargs["prelude_forcing"] is True


def test_cache_script_skips_existing(monkeypatch, tmp_path):
    from chordtrack import paths
    from chordtrack.types import MelodyModelOutput, MelodyNote
    from eval.melody import cache_muscriptor as C

    monkeypatch.setattr(paths, "CACHE_DIR", tmp_path / "cache")
    root = tmp_path / "melody"
    monkeypatch.setattr(C, "MANIFEST_ROOT", root)
    (root / "ds").mkdir(parents=True)
    rows = [{"track_id": t, "dataset": "ds", "audio": str(tmp_path / f"{t}.wav"), "ref_notes": None,
             "ref_f0": None, "melody_type": "vocal"} for t in ("t1", "t2")]
    (root / "ds" / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    calls = []

    def fake_transcribe(audio, size, dtype=None):
        calls.append(audio)
        x = {k: np.array(1.0) for k in ("duration_s", "load_s", "infer_s", "peak_vram_mb")}
        return MelodyModelOutput(M.model_name(size), [MelodyNote(0, 1, 60, 0.5, "voice")], None, x)

    monkeypatch.setattr(M, "transcribe", fake_transcribe)
    assert C.main(["--size", "small"]) == 0
    assert len(calls) == 2
    out = load_output(cache_path("muscriptor_small", "ds", "t1"))
    assert out.notes[0].source == "voice"
    assert C.main(["--size", "small", "--dataset", "ds"]) == 0
    assert len(calls) == 2                       # đã có cache, không chạy lại
    with pytest.raises(FileNotFoundError):
        C.read_manifests(["khong_co"])


@pytest.mark.gpu
def test_real_small_on_sample(tmp_path):
    from chordtrack import paths

    if not paths.SAMPLE_AUDIO.exists():
        pytest.skip("không có audio mẫu")
    try:
        M.ensure_weights("small")
    except M.MuscriptorAuthError as e:
        pytest.skip(str(e))
    clip = tmp_path / "clip.wav"
    y, sr = sf.read(str(paths.SAMPLE_AUDIO), frames=None)
    sf.write(clip, y[: 15 * sr], sr)                # 15 s đầu Let It Be: piano
    out = M.transcribe(clip, "small", correct_onset_delay=False)
    assert len(out.notes) > 10
    assert all(0 <= n.pitch < 128 and 0 < n.confidence <= 1 for n in out.notes)
    assert "acoustic_piano" in {n.source for n in out.notes}
