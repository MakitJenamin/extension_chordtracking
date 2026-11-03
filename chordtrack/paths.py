"""Đường dẫn tới tài nguyên dùng chung, không nằm trong git.

Mặc định là thư mục gốc của repo (nơi clone về). Trong git worktree (không có `third_party_src`) thì trỏ vào
checkout chính, vì weights và dữ liệu chỉ nằm ở đó. Đổi được bằng biến môi trường `CHORDTRACK_ROOT`,
`CHORDTRACK_THIRD_PARTY`, `CHORDTRACK_DATA`.
"""
from __future__ import annotations

import os
from pathlib import Path


def _default_root() -> Path:
    here = Path(__file__).resolve().parents[1]
    if (here / "third_party_src").exists():
        return here
    git = here / ".git"
    if git.is_file():  # worktree: ".git" là file "gitdir: <checkout chính>/.git/worktrees/<tên>"
        try:
            gitdir = Path(git.read_text(encoding="utf-8").split(":", 1)[1].strip())
            main = gitdir.parents[2]
            if (main / "third_party_src").exists():
                return main
        except (OSError, IndexError):
            pass
    return here


MAIN_ROOT = Path(os.environ.get("CHORDTRACK_ROOT") or _default_root())

THIRD_PARTY = Path(os.environ.get("CHORDTRACK_THIRD_PARTY", MAIN_ROOT / "third_party_src"))
CHORDMINI_DIR = THIRD_PARTY / "ChordMini"
CNNLSTM_DIR = THIRD_PARTY / "chord-cnn-lstm-model"
CHORDFORMER_DIR = THIRD_PARTY / "ChordFormer"        # github.com/mwaseemrandhawa/ChordFormer, checkpoint lấy từ commit 0d1071a
CONSONANCE_DIR = THIRD_PARTY / "consonance-ACE"     # github.com/andreamust/consonance-ACE (MIT)

DATA_DIR = Path(os.environ.get("CHORDTRACK_DATA", MAIN_ROOT / "data"))
AUDIO_DIR = DATA_DIR / "audio"          # audio tải bằng yt-dlp, đặt tên theo <videoId>.wav
MODELS_DIR = DATA_DIR / "models"        # weights tải thêm (BS-Roformer, HTDemucs, beat_this...)
CACHE_DIR = DATA_DIR / "cache"          # stem và kết quả trung gian, theo <videoId>/
EVAL_DIR = DATA_DIR / "eval"            # dữ liệu đánh giá: nhãn chuẩn kèm audio

# Bài thử đã tải sẵn: The Beatles - Let It Be (Remastered 2009), 243 giây
SAMPLE_AUDIO = AUDIO_DIR / "QDYfEBY9NM4.wav"
