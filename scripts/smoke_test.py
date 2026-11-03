"""Chạy thử sau khi cài: tạo audio hợp âm C rồi G (mỗi hợp âm 4 giây), chạy đủ các model và trộn kết quả.

    python scripts/smoke_test.py

Không cần mạng hay YouTube. Thành công thì in thiết bị và các hợp âm nhận được, mã thoát 0.
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SR = 22050
CHORDS = [("C:maj", (48, 60, 64, 67)), ("G:maj", (43, 59, 62, 67))]   # MIDI: bass + bộ ba


def synth(path: Path, seconds: float = 4.0) -> None:
    t = np.arange(int(seconds * SR)) / SR
    env = np.minimum(1.0, t / 0.02) * np.exp(-t / 3.0)
    parts = []
    for _label, notes in CHORDS:
        x = sum(np.sin(2 * np.pi * 440 * 2 ** ((m - 69) / 12) * t * k) / k
                for m in notes for k in (1, 2, 3))
        parts.append(x * env)
    y = np.concatenate(parts * 2)
    sf.write(str(path), (0.2 * y / np.abs(y).max()).astype(np.float32), SR)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    import torch

    from chordtrack import fusion
    from chordtrack.beats import track_beats
    from chordtrack.chords.chordformer import run_chordformer
    from chordtrack.chords.chordmini import run_chordmini
    from chordtrack.chords.cnnlstm import run_cnnlstm

    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Thiết bị: {dev}")
    with tempfile.TemporaryDirectory() as d:
        wav = Path(d) / "smoke.wav"
        synth(wav)
        duration = sf.info(str(wav)).duration
        t0 = time.perf_counter()
        beats = track_beats(wav)
        models = {"chordmini": run_chordmini(wav), "cnnlstm": run_cnnlstm(wav)}
        if fusion.uses_chordformer(fusion.DEFAULT_PARAMS):
            models["chordformer"] = run_chordformer(wav)
        segs = fusion.fuse(models, beats, None, duration)
        print(f"Chạy {len(models)} model trong {time.perf_counter() - t0:.1f} s (gồm nạp model)")
    labels = [s.label.split("/")[0] for s in segs if s.end - s.start >= 1.0]
    print("Hợp âm nhận được:", " ".join(labels))
    roots = {lab.split(":")[0] for lab in labels}
    if not {"C", "G"} <= roots:
        print("⚠️ Không nhận ra đủ C và G: model có thể chưa cài đúng.", file=sys.stderr)
        return 1
    print("OK: cài đặt chạy được.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
