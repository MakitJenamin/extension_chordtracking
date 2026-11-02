"""Đường dẫn tới tài nguyên dùng chung, không nằm trong git.

Các thư mục này chỉ có ở checkout chính, không có trong worktree, nên mặc định trỏ vào
checkout chính. Đổi được bằng biến môi trường.
"""
from __future__ import annotations

import os
from pathlib import Path

MAIN_ROOT = Path(os.environ.get("CHORDTRACK_ROOT", r"C:\Users\MakitJenamin\Documents\extension_chordtracking"))

THIRD_PARTY = Path(os.environ.get("CHORDTRACK_THIRD_PARTY", MAIN_ROOT / "third_party_src"))
CHORDMINI_DIR = THIRD_PARTY / "ChordMini"
CNNLSTM_DIR = THIRD_PARTY / "chord-cnn-lstm-model"

DATA_DIR = Path(os.environ.get("CHORDTRACK_DATA", MAIN_ROOT / "data"))
AUDIO_DIR = DATA_DIR / "audio"          # audio tải bằng yt-dlp, đặt tên theo <videoId>.wav
MODELS_DIR = DATA_DIR / "models"        # weights tải thêm (BS-Roformer, HTDemucs, beat_this...)
CACHE_DIR = DATA_DIR / "cache"          # stem và kết quả trung gian, theo <videoId>/
EVAL_DIR = DATA_DIR / "eval"            # dữ liệu đánh giá: nhãn chuẩn kèm audio

# Bài thử đã tải sẵn: The Beatles - Let It Be (Remastered 2009), 243 giây
SAMPLE_AUDIO = AUDIO_DIR / "QDYfEBY9NM4.wav"
