"""Chạy thử sau khi cài: tạo audio hợp âm C rồi G (mỗi hợp âm 4 giây), chạy đủ các model và trộn kết quả.

    python scripts/smoke_test.py
    python scripts/smoke_test.py --melody   # thêm: chạy YourMT3+ (dò giai điệu) trên một chuỗi nốt tổng hợp

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


MELODY = (60, 62, 64, 65, 67, 65, 64, 62)    # C D E F G F E D, mỗi nốt 0,5 giây


def synth_melody(path: Path, note_s: float = 0.5) -> None:
    t = np.arange(int(note_s * SR)) / SR
    env = np.minimum(1.0, t / 0.01) * np.minimum(1.0, (note_s - t) / 0.05)
    parts = [sum(np.sin(2 * np.pi * 440 * 2 ** ((m - 69) / 12) * t * k) / k for k in (1, 2, 3, 4)) * env
             for m in MELODY * 2]
    y = np.concatenate([np.zeros(SR // 2), *parts, np.zeros(SR // 2)])
    sf.write(str(path), (0.3 * y / np.abs(y).max()).astype(np.float32), SR)


def melody_check() -> int:
    """YourMT3+ (tiến trình con, tự lấy gpu_lock) phải ra được phần lớn các nốt của chuỗi tổng hợp."""
    from chordtrack.melody import pipeline as melody_pipeline

    ready = melody_pipeline.variant_ready(melody_pipeline.melody_ready())
    if not ready.get("yourmt3"):
        print("⚠️ Chưa cài đủ YourMT3+ (dò giai điệu): chạy lại scripts/fetch_models.py --melody", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as d:
        wav = Path(d) / "melody.wav"
        synth_melody(wav)
        t0 = time.perf_counter()
        out = melody_pipeline.run_yourmt3(wav, Path(d) / "yourmt3.npz")
        print(f"YourMT3+ chạy trong {time.perf_counter() - t0:.1f} s (gồm nạp model), {len(out.notes)} nốt")
    got = {n.pitch % 12 for n in out.notes}
    want = {m % 12 for m in MELODY}
    if len(got & want) < len(want) - 1:
        print(f"⚠️ YourMT3+ không nhận ra chuỗi nốt (được {sorted(got)}, cần {sorted(want)})", file=sys.stderr)
        return 1
    print("OK: dò giai điệu (YourMT3+) chạy được. Cách dò Kỹ (MuScriptor): "
          + ("có" if ready.get("combo") else "chưa bật, xem README mục Dò giai điệu"))
    return 0


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
    if "--melody" in sys.argv[1:]:
        return melody_check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
