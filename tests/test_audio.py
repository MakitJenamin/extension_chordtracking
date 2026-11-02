import json

import numpy as np
import pytest
import soundfile as sf

from chordtrack import audio, paths


@pytest.mark.parametrize("source, vid", [
    ("QDYfEBY9NM4", "QDYfEBY9NM4"),
    ("https://www.youtube.com/watch?v=QDYfEBY9NM4", "QDYfEBY9NM4"),
    ("https://www.youtube.com/watch?list=PL1&v=QDYfEBY9NM4&t=30s", "QDYfEBY9NM4"),
    ("youtube.com/watch?v=QDYfEBY9NM4", "QDYfEBY9NM4"),
    ("https://youtu.be/QDYfEBY9NM4?si=abc", "QDYfEBY9NM4"),
    ("https://m.youtube.com/watch?v=QDYfEBY9NM4", "QDYfEBY9NM4"),
    ("https://music.youtube.com/watch?v=QDYfEBY9NM4", "QDYfEBY9NM4"),
    ("https://www.youtube.com/shorts/QDYfEBY9NM4", "QDYfEBY9NM4"),
    ("https://www.youtube.com/embed/QDYfEBY9NM4", "QDYfEBY9NM4"),
    ("https://example.com/watch?v=QDYfEBY9NM4", None),
    ("https://notyoutube.com/watch?v=QDYfEBY9NM4", None),
    ("https://www.youtube.com/watch?v=short", None),
    ("not a video", None),
])
def test_parse_video_id(source, vid):
    assert audio.parse_video_id(source) == vid


def test_local_file_song_id(tmp_path):
    p = tmp_path / "Bài hát (live).wav"
    sf.write(p, np.zeros((22050, 1)), 22050)
    song_id, path = audio.fetch_audio(str(p))
    assert path == p.resolve()
    stem, digest = song_id.rsplit("-", 1)
    assert stem == "B_i_h_t_live" and len(digest) == 8
    src = audio.fetch(str(p))
    assert src.type == "file" and src.duration == pytest.approx(1.0)
    assert src.to_json()["type"] == "file"
    # cùng nội dung -> cùng hash; khác nội dung -> khác hash
    q = tmp_path / "copy.wav"
    q.write_bytes(p.read_bytes())
    assert audio.fetch_audio(str(q))[0].rsplit("-", 1)[1] == digest
    sf.write(q, np.ones((22050, 1)) * 0.1, 22050)
    assert audio.fetch_audio(str(q))[0].rsplit("-", 1)[1] != digest


def test_invalid_source():
    with pytest.raises(ValueError):
        audio.fetch("khong-phai-video")


def test_youtube_cached_does_not_download(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "AUDIO_DIR", tmp_path)
    sf.write(tmp_path / "abcdefghijk.wav", np.zeros((44100, 2)), 44100)
    (tmp_path / "abcdefghijk.info.json").write_text(json.dumps({"title": "Bài thử", "duration": 1}), encoding="utf-8")

    def boom(*a, **k):
        raise AssertionError("không được gọi mạng khi đã có cache")

    monkeypatch.setattr(audio, "download_youtube", boom)
    monkeypatch.setattr(audio, "_extract_info", boom)
    src = audio.fetch("https://youtu.be/abcdefghijk")
    assert src.song_id == "abcdefghijk" and src.path == tmp_path / "abcdefghijk.wav"
    assert src.title == "Bài thử" and src.duration == 1.0
    assert src.to_json() == {"type": "youtube", "video_id": "abcdefghijk", "title": "Bài thử",
                             "url": "https://www.youtube.com/watch?v=abcdefghijk"}


def test_youtube_cached_audio_without_metadata_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "AUDIO_DIR", tmp_path)
    sf.write(tmp_path / "abcdefghijk.wav", np.zeros((44100, 2)), 44100)

    def offline(*a, **k):
        raise OSError("mất mạng")

    monkeypatch.setattr(audio, "_extract_info", offline)
    src = audio.fetch("abcdefghijk")
    assert src.title is None and src.duration == pytest.approx(1.0)
    # lần lỗi đã được ghi lại: lần sau không gọi mạng nữa
    monkeypatch.setattr(audio, "_extract_info", lambda *a: pytest.fail("gọi mạng lại"))
    assert audio.fetch("abcdefghijk").title is None


def test_youtube_downloads_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "AUDIO_DIR", tmp_path)
    calls = []

    def fake_download(video_id, audio_dir):
        calls.append(video_id)
        sf.write(audio_dir / f"{video_id}.wav", np.zeros((44100, 2)), 44100)
        return {"title": "Mới", "duration": 1, "webpage_url": f"https://www.youtube.com/watch?v={video_id}"}

    monkeypatch.setattr(audio, "download_youtube", fake_download)
    src = audio.fetch("abcdefghijk")
    assert calls == ["abcdefghijk"] and src.title == "Mới"
