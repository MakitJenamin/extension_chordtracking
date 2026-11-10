"""Tạo <Melody> giả (đúng dạng docs/melody_api.md) cho harness từ cache model có sẵn, khi backend chưa có /melody.

    <venv python> extension/dev/make_dev_melody.py

Ghi ra `data/cache/dev_melody/<song_id>.json` (không commit). serve.py trả các file này cho `/melody`.
- devmagia_01 (Magia): YourMT3+ trong khoảng có giọng hát, MuScriptor small ở phần còn lại (giống cách đã chốt).
- QDYfEBY9NM4 (Let It Be): chỉ có cache Basic Pitch, nên lấy dòng nốt của Basic Pitch (chỉ để thử giao diện).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from chordtrack import paths  # noqa: E402
from chordtrack.melody import io, line  # noqa: E402

UNCERTAIN = 0.5
OUT = paths.CACHE_DIR / "dev_melody"


def to_json(song_id: str, model: str, notes, duration: float) -> dict:
    out = []
    for n in sorted(notes, key=lambda x: x.start):
        c = 0.5 if n.confidence is None else max(0.0, min(1.0, float(n.confidence)))
        out.append({"start": round(n.start, 3), "end": round(n.end, 3), "pitch": int(n.pitch),
                    "confidence": round(c, 3), "uncertain": c < UNCERTAIN, "source": n.source or "voice"})
    return {"song_id": song_id, "model": model, "code_revision": "dev", "duration": duration, "notes": out}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    y = io.cache_path("yourmt3", "kajiura", "magia")
    m = io.cache_path("muscriptor_small", "kajiura", "magia")
    if y.exists() and m.exists():
        yo, mo = io.load_output(y), io.load_output(m)
        gate = line.voice_intervals(yo.notes)
        voice = line.extract_line(yo)
        fill = line.extract_line(mo)
        notes = line.combine_gated(voice, fill, gate)
        jobs.append(("devmagia_01", "yourmt3>muscriptor_small", notes, float(mo.extras.get("duration_s", 0)) or 260.0))
    b = io.cache_path("basic_pitch", "adhoc", "QDYfEBY9NM4")
    if b.exists():
        bo = io.load_output(b)
        jobs.append(("QDYfEBY9NM4", "basic_pitch(dev)", line.extract_line(bo), 243.1))
    for sid, model, notes, dur in jobs:
        p = OUT / f"{sid}.json"
        p.write_text(json.dumps(to_json(sid, model, notes, dur), ensure_ascii=False), encoding="utf-8")
        print(f"{p}: {len(notes)} notes")


if __name__ == "__main__":
    main()
