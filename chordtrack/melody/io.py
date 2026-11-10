"""Lưu và đọc MelodyModelOutput ra một file .npz (dùng cho cache kết quả thô của từng model).

Đường dẫn cache chuẩn: `paths.CACHE_DIR / "melody" / <model_name> / <dataset> / <track_id>.npz`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from chordtrack import paths
from chordtrack.types import MelodyModelOutput, MelodyNote, PitchTrack


def cache_path(model_name: str, dataset: str, track_id: str) -> Path:
    return paths.CACHE_DIR / "melody" / model_name / dataset / f"{track_id}.npz"


def save_output(out: MelodyModelOutput, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = out.notes
    arrays: dict[str, np.ndarray] = {
        "model_name": np.array(out.model_name),
        "note_start": np.array([x.start for x in n], dtype=np.float64),
        "note_end": np.array([x.end for x in n], dtype=np.float64),
        "note_pitch": np.array([x.pitch for x in n], dtype=np.int16),
        "note_conf": np.array([np.nan if x.confidence is None else x.confidence for x in n], dtype=np.float32),
        "note_source": np.array(json.dumps([x.source for x in n])),
    }
    if out.pitch_track is not None:
        pt = out.pitch_track
        arrays["pt_times"] = np.asarray(pt.times, dtype=np.float64)
        arrays["pt_f0"] = np.asarray(pt.f0_hz, dtype=np.float32)
        if pt.confidence is not None:
            arrays["pt_conf"] = np.asarray(pt.confidence, dtype=np.float32)
    for k, v in out.extras.items():
        arrays[f"x_{k}"] = np.asarray(v)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def load_output(path: Path) -> MelodyModelOutput:
    with np.load(path, allow_pickle=False) as z:
        sources = json.loads(str(z["note_source"]))
        notes = [
            MelodyNote(float(s), float(e), int(p), None if np.isnan(c) else float(c), src)
            for s, e, p, c, src in zip(z["note_start"], z["note_end"], z["note_pitch"], z["note_conf"], sources)
        ]
        pt = None
        if "pt_times" in z:
            pt = PitchTrack(z["pt_times"], z["pt_f0"], z["pt_conf"] if "pt_conf" in z else None)
        extras = {k[2:]: z[k] for k in z.files if k.startswith("x_")}
        return MelodyModelOutput(str(z["model_name"]), notes, pt, extras)
