"""Cache kết quả basic_pitch trên mọi bộ đánh giá giai điệu.

    python -m eval.melody.cache_basic_pitch [--dataset BỘ ...] [--audio FILE] [--force]
"""
from __future__ import annotations

import sys

from chordtrack.melody import basic_pitch as model
from eval.melody.cache_runner import main

if __name__ == "__main__":
    sys.exit(main(model.MODEL_NAME, model.transcribe))
