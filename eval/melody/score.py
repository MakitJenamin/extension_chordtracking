"""Chấm dòng giai điệu của mọi (model × chiến lược × bộ dữ liệu) từ cache, chỉ dùng CPU.

    python -m eval.melody.score [--datasets a,b] [--models m1,m2] [--exclude muscriptor_medium] [--folds 3]
                                [--jobs 6] [--boot 2000] [--no-beats] [--out-dir EVAL_DIR/results/melody]

Thêm model mới (vd muscriptor_*) chỉ cần có cache rồi chạy lại lệnh trên: model tự được tìm trong cache.

Đầu vào:
- Manifest `EVAL_DIR/melody/<dataset>/manifest.jsonl` (xem `eval/melody/README.md`). Trường dùng thêm nếu có:
  `offset` (giây, cộng vào mọi thời điểm của đáp án để khớp audio), `duration`, `chord_ref` (.lab nhãn Harte).
- Cache `chordtrack.melody.io.cache_path(model, dataset, track_id)`. Model là mọi thư mục trong
  `CACHE_DIR/melody/` không bắt đầu bằng `_`. Model có `notes` chấm các họ chiến lược nốt đa âm, model có
  `pitch_track` chấm các họ f0; nếu model lưu f0 thô (`extras` `f0_raw_hz` + `salience_max`, như RMVPE) thì ngưỡng
  voicing được dò lại trên bản thô (`RAW_VOICING`). Mỗi cặp (model f0, model nốt) chấm thêm họ `combo`
  (`combine_lines`: lấp khoảng trống của dòng f0) và `combo_gated` (`combine_gated`: dòng f0 ở chỗ model nốt có
  nhãn giọng hát, dòng model nốt ở phần còn lại). Các cặp model nốt A:B trong `--nn-pairs` (mặc định YourMT3+ và
  MuScriptor small, cả hai chiều) chấm họ `nn_gated`: dòng của A ở đoạn có giọng (cổng theo nốt `voice` của A hoặc
  của B), dòng của B ở phần còn lại.
- Lưới phách (tùy chọn) `CACHE_DIR/melody/_beats/<dataset>/<track_id>.npz` với mảng `beats`, `downbeats`.
  Có thì chấm thêm bản căn phách của mọi cấu hình (cột `quant` = 1).

Chỉ số:
- Bộ có `ref_f0`: `mir_eval.melody` OA, RPA, RCA, VR, VFA trên lưới thời gian của đáp án (dòng nốt đổi ra f0).
- Bộ có `ref_notes`: `mir_eval.transcription` P/R/F1 onset-only (±50 ms) và onset+offset, cộng các chỉ số mức
  khung như trên (đáp án và dự đoán cùng đổi ra f0 lưới 10 ms).
- Mọi bộ (kể cả không đáp án): % thời gian có nốt, % nốt thuộc hợp âm đang phát (nếu có `chord_ref`), số bước
  nhảy > 1 quãng tám mỗi phút, số nốt mỗi phút, độ dài nốt trung vị.
Mỗi chỉ số có trọng số theo bài (số khung, số nốt hoặc thời lượng) để tính cả trung bình theo bài lẫn theo
thời lượng.

**Chọn tham số bằng cross-validation theo bài:** chỉ số mục tiêu mỗi bài là OA (bộ f0) hoặc F1 onset (bộ nốt).
Bài của mọi bộ có đáp án gộp lại, chia `--folds` phần (bài thứ i của mỗi bộ, theo `track_id`, vào phần
i mod k). Mỗi phần: với mỗi (model, họ, căn phách) chọn cấu hình tốt nhất trên các phần còn lại, chấm phần giữ
lại. Dòng `cv` là điểm các phần giữ lại ghép lại. Dòng `full` chọn trên cả bộ (lạc quan, và là cấu hình dùng cho
bộ không đáp án). Dòng `default` là cấu hình đầu tiên của lưới (= `LineParams` mặc định của họ đó; riêng model
có f0 thô thì là ngưỡng 0.03 trên bản thô, tương đương ngưỡng model tự cắt).
Mức model (`cv_model`, `full_model`): cùng cách trên nhưng mỗi (model, căn phách) chọn trong mọi cấu hình của mọi
họ (trừ `f0raw`), nên cả họ chiến lược lẫn tham số đều được chọn bằng CV. Đây là bảng chính. Biến thể mục tiêu
`(RPA + VR) / 2` (`cv_model_rpa_vr`) không phạt VFA: ADC2004 / MIREX05 / MIR-ST500 chỉ gán nhãn giọng hát, nên nốt
nhạc cụ ở đoạn không hát bị tính là báo nhầm dù đúng mục tiêu sản phẩm.

Ra `out-dir`: `per_track.csv` (mọi bài, mọi cấu hình), `summary.csv`, `cv_choices.json`, `report.md` (bảng theo
model và theo từng họ), và in bảng theo model kèm so cặp hai model tốt nhất mỗi bộ (bootstrap 95% chênh lệch
trung bình theo bài, p Wilcoxon). Bộ nào có model chỉ cache một phần bài thì in thêm bảng phụ trên
các bài mọi model đều có (`cv_model_common`).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from chordtrack import paths
from chordtrack.melody import line as L
from chordtrack.melody.io import cache_path, load_output
from chordtrack.types import BeatGrid, MelodyNote, PitchTrack, lab_to_segments

FRAME_HOP = 0.01
ONSET_TOL = 0.05
F0_METRICS = ["oa", "rpa", "rca", "vr", "vfa"]
NOTE_METRICS = ["note_p", "note_r", "note_f1", "note_off_f1"]
FREE_METRICS = ["coverage", "in_chord", "octave_jumps_pm", "notes_pm", "median_dur"]
ALL_METRICS = F0_METRICS + NOTE_METRICS + FREE_METRICS

# --------------------------------------------------------------------------- lưới cấu hình


def _grid(base: L.LineParams, **axes) -> list[dict]:
    """Tích Đề-các các trục; giá trị đầu mỗi trục là mặc định, nên cấu hình đầu là mặc định của họ."""
    out = [{}]
    for k, vals in axes.items():
        out = [dict(c, **{k: v}) for c in out for v in vals]
    return [{"_base": base, **c} for c in out]


def note_families() -> dict[str, list[dict]]:
    P = L.LineParams
    return {
        "skyline": _grid(P(strategy="skyline", source_mode="ignore"),
                         min_dur=[0.06, 0.03, 0.1], exclude_accomp=[True, False]),
        "salience": _grid(P(strategy="salience", source_mode="bonus"),
                          w_conf=[1.0, 0.0], w_dur=[0.3, 0.0], w_pitch=[0.5, 0.0, 1.0]),
        # ưu tiên nguồn: giọng hát > nhạc cụ giai điệu > không rõ > đệm, trong từng khoảng
        "source": _grid(P(strategy="salience", source_mode="strict"),
                        w_pitch=[0.5, 1.0], min_dur=[0.06, 0.1], chorus_as_voice=[True, False]),
        "source_viterbi": _grid(P(strategy="viterbi", source_mode="strict"),
                                jump_penalty=[0.5, 1.0], cut_penalty=[0.15, 0.05], chorus_as_voice=[True, False]),
        "viterbi": _grid(P(strategy="viterbi"),
                         jump_penalty=[0.5, 0.25, 1.0], cut_penalty=[0.15, 0.05, 0.4], w_pitch=[0.5, 1.0]),
    }


# Ngưỡng voicing thử cho model có f0 thô (RMVPE: ngưỡng trên salience_max; model tự cắt ở 0.03).
RAW_VOICING = [0.03, 0.01, 0.05, 0.1, 0.2, 0.3]


def f0_families(has_raw: bool, has_conf: bool) -> dict[str, list[dict]]:
    P = L.LineParams
    vt = RAW_VOICING if has_raw else ([0.0, 0.3, 0.5] if has_conf else [0.0])
    return {
        "f0raw": _grid(P(), voicing_threshold=vt),
        "f0notes": _grid(P(), voicing_threshold=vt, median_frames=[5, 1, 9], split_semitones=[0.6, 0.8],
                         split_min_dur=[0.08, 0.12]),
    }


# Ghép model f0 (đoạn có giọng) + model nốt (đoạn còn lại). `vt`: ngưỡng voicing của dòng f0 (tham số tách nốt
# để mặc định), `sec`: họ chiến lược của model nốt (cấu hình mặc định của họ đó).
COMBO_VT = [0.03, 0.1, 0.2]
COMBO_SEC = ["viterbi", "source"]
COMBO_PAD = [0.1, 0.3]


def combo_families() -> dict[str, list[dict]]:
    return {
        # lấp mọi khoảng trống >= min_gap giây của dòng f0 bằng dòng của model nốt
        "combo": [dict(vt=v, min_gap=g, sec=c) for v in COMBO_VT for g in (0.3, 1.0) for c in COMBO_SEC],
        # dòng f0 ở chỗ model nốt thấy giọng hát (nới pad giây), dòng của model nốt ở phần còn lại
        "combo_gated": [dict(vt=v, pad=d, sec=c) for v in COMBO_VT for d in COMBO_PAD for c in COMBO_SEC],
    }


# Ghép hai model nốt (A ở đoạn có giọng, B ở phần còn lại). Cặp có thứ tự; đổi bằng --nn-pairs.
NN_PAIRS: list[tuple[str, str]] = [("yourmt3", "muscriptor_small"), ("muscriptor_small", "yourmt3")]
NN_PAD = [0.1, 0.3, 0.6]


def _cfg_index(family: str, **kv) -> tuple[str, int]:
    """(họ, chỉ số) của cấu hình trong lưới `note_families()` có các tham số `kv`."""
    for ci, cfg in enumerate(note_families()[family]):
        if all(cfg.get(k) == v for k, v in kv.items()):
            return family, ci
    raise KeyError(f"{family} {kv}")


# Dòng của A và B lấy từ các cấu hình này của lưới nốt đa âm (đã tính sẵn khi chấm từng họ).
NN_KEYS = {
    "source_viterbi": _cfg_index("source_viterbi"),                  # ưu tiên giọng + Viterbi, mặc định
    "viterbi": _cfg_index("viterbi"),
    "salience_nodur": _cfg_index("salience", w_conf=1.0, w_dur=0.0, w_pitch=0.5),   # CV chọn cho MuScriptor
}


def nn_families() -> dict[str, list[dict]]:
    return {"nn_gated": [dict(a=a, b=b, gate=g, pad=d) for a in ("source_viterbi", "viterbi")
                         for b in ("salience_nodur", "source_viterbi", "viterbi")
                         for g in ("a", "b") for d in NN_PAD]}


NN_GRID = nn_families()["nn_gated"]
NN_SEP = ">"      # tên model ghép hai model nốt: "A>B" (khác "f0+notes" của combo)


def nn_name(a: str, b: str) -> str:
    return f"{a}{NN_SEP}{b}"


def model_parts(name: str) -> list[str]:
    """Các model thành phần của một tên model (đơn, combo "f0+notes" hoặc ghép "A>B")."""
    return re.split(r"[+>]", name)


def parse_nn_pairs(text: str) -> list[tuple[str, str]]:
    pairs = []
    for item in text.split(","):
        if not item.strip():
            continue
        parts = [p.strip() for p in item.split(":")]
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"--nn-pairs: '{item}' phải có dạng A:B")
        pairs.append((parts[0], parts[1]))
    return pairs


def params_of(cfg: dict) -> L.LineParams:
    return replace(cfg["_base"], **{k: v for k, v in cfg.items() if k != "_base"})


def cfg_text(cfg: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in cfg.items() if k != "_base") or "mặc định"


# --------------------------------------------------------------------------- dữ liệu


def read_manifests(names: list[str] | None) -> dict[str, list[dict]]:
    root = paths.EVAL_DIR / "melody"
    out: dict[str, list[dict]] = {}
    for m in sorted(root.glob("*/manifest.jsonl")):
        ds = m.parent.name
        if names and ds not in names:
            continue
        rows = []
        for ln in m.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                e = json.loads(ln)
                e.setdefault("dataset", ds)
                e["_dir"] = str(m.parent)
                rows.append(e)
        out[ds] = sorted(rows, key=lambda e: str(e["track_id"]))
    return out


def _resolve(p: str | None, base: str) -> Path | None:
    if not p:
        return None
    q = Path(p)
    return q if q.is_absolute() else Path(base) / q


def discover_models(datasets: list[str], only: list[str] | None, exclude: list[str] | None = None) -> list[str]:
    """Các model có ít nhất một file cache ở một trong các bộ."""
    root = paths.CACHE_DIR / "melody"
    out: list[str] = []
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith("_") or (only and d.name not in only) or d.name in (exclude or []):
            continue
        if any(next((d / ds).glob("*.npz"), None) for ds in datasets):
            out.append(d.name)
    return out


def load_beats(dataset: str, track_id: str, beats_dir: Path | None) -> BeatGrid | None:
    if beats_dir is None:
        return None
    p = beats_dir / dataset / f"{track_id}.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        b = np.asarray(z["beats"], dtype=np.float64)
        db = np.asarray(z["downbeats"], dtype=np.float64) if "downbeats" in z else b[:0]
    return BeatGrid(b, db, 0.0)


# --------------------------------------------------------------------------- chỉ số


def f0_metrics(ref_t: np.ndarray, ref_hz: np.ndarray, est_t: np.ndarray, est_hz: np.ndarray) -> tuple[dict, dict]:
    import mir_eval.melody as mm

    ref_v, ref_c, est_v, est_c = mm.to_cent_voicing(ref_t, ref_hz, est_t, est_hz)
    rv = np.asarray(ref_v, dtype=np.float64)
    n_v, n_u = float(rv.sum()), float((1 - rv).sum())
    vr, vfa = mm.voicing_measures(ref_v, est_v)
    vals = {"oa": mm.overall_accuracy(ref_v, ref_c, est_v, est_c),
            "rpa": mm.raw_pitch_accuracy(ref_v, ref_c, est_v, est_c) if n_v else math.nan,
            "rca": mm.raw_chroma_accuracy(ref_v, ref_c, est_v, est_c) if n_v else math.nan,
            "vr": vr if n_v else math.nan, "vfa": vfa if n_u else math.nan}
    wts = {"oa": float(len(rv)), "rpa": n_v, "rca": n_v, "vr": n_v, "vfa": n_u}
    return {k: float(v) for k, v in vals.items()}, wts


def note_metrics(ref: np.ndarray, est: list[MelodyNote]) -> tuple[dict, dict]:
    """`ref`: (N, 3) onset, offset, midi."""
    import mir_eval.transcription as mt

    w = float(len(ref))
    if len(ref) == 0:
        return {k: math.nan for k in NOTE_METRICS}, {k: 0.0 for k in NOTE_METRICS}
    if not est:
        return {k: 0.0 for k in NOTE_METRICS}, {k: w for k in NOTE_METRICS}
    ri, rp = ref[:, :2], L.midi_to_hz(ref[:, 2])
    ei = np.array([[n.start, n.end] for n in est])
    ep = L.midi_to_hz([n.pitch for n in est])
    p, r, f, _ = mt.precision_recall_f1_overlap(ri, rp, ei, ep, onset_tolerance=ONSET_TOL, offset_ratio=None)
    _, _, f_off, _ = mt.precision_recall_f1_overlap(ri, rp, ei, ep, onset_tolerance=ONSET_TOL)
    return ({"note_p": float(p), "note_r": float(r), "note_f1": float(f), "note_off_f1": float(f_off)},
            {k: w for k in NOTE_METRICS})


def chord_pcs(label: str) -> set[int] | None:
    import mir_eval.chord as mc

    try:
        root, bitmap, bass = mc.encode(label)
    except Exception:
        return None
    if root < 0:
        return None
    pcs = {(root + i) % 12 for i in np.nonzero(bitmap)[0]}
    if bass >= 0:
        pcs.add((root + bass) % 12)
    return pcs


def free_metrics(est: list[MelodyNote], duration: float, chords: list | None) -> tuple[dict, dict]:
    dur = max(duration, 1e-6)
    lens = np.array([n.end - n.start for n in est])
    vals = {"coverage": float(lens.sum() / dur) if len(est) else 0.0,
            "notes_pm": len(est) / (dur / 60.0),
            "median_dur": float(np.median(lens)) if len(est) else math.nan}
    wts = {"coverage": dur, "notes_pm": dur, "median_dur": dur if len(est) else 0.0}
    jumps = sum(1 for a, b in zip(est[:-1], est[1:]) if abs(b.pitch - a.pitch) > 12 and b.start - a.end < 1.0)
    vals["octave_jumps_pm"], wts["octave_jumps_pm"] = jumps / (dur / 60.0), dur
    vals["in_chord"], wts["in_chord"] = math.nan, 0.0
    if chords:
        starts = np.array([c.start for c in chords])
        ends = np.array([c.end for c in chords])
        pcs = [chord_pcs(c.label) for c in chords]
        hit = tot = 0
        for n in est:
            ov = np.minimum(ends, n.end) - np.maximum(starts, n.start)
            k = int(np.argmax(ov))
            if ov[k] <= 0 or pcs[k] is None:
                continue
            tot += 1
            hit += (n.pitch % 12) in pcs[k]
        if tot:
            vals["in_chord"], wts["in_chord"] = hit / tot, float(tot)
    return vals, wts


def score_line(entry: dict, ref: dict, est: list[MelodyNote], raw_f0: tuple | None = None) -> tuple[dict, dict]:
    """Mọi chỉ số của một dòng giai điệu trên một bài. `raw_f0` = (times, hz): chấm f0 thô thay cho dòng nốt
    ở các chỉ số mức khung."""
    vals: dict = {}
    wts: dict = {}
    frame_ref = ref.get("f0")
    if frame_ref is not None:
        et, eh = raw_f0 if raw_f0 is not None else (frame_ref[0], L.notes_to_f0(est, frame_ref[0]))
        v, w = f0_metrics(frame_ref[0], frame_ref[1], et, eh)
        vals.update(v), wts.update(w)
    if ref.get("notes") is not None and raw_f0 is None:
        v, w = note_metrics(ref["notes"], est)
        vals.update(v), wts.update(w)
    v, w = free_metrics(est, ref["duration"], ref.get("chords"))
    vals.update(v), wts.update(w)
    return vals, wts


def load_refs(entry: dict) -> dict:
    from eval.melody.datasets import load_reference

    base, off = entry["_dir"], float(entry.get("offset") or 0.0)
    loaded = load_reference({**entry, "ref_f0": _resolve(entry.get("ref_f0"), base),
                             "ref_notes": _resolve(entry.get("ref_notes"), base)})   # đã cộng offset
    ref: dict = {}
    end = 0.0
    if loaded["f0"] is not None:
        ref["f0"] = loaded["f0"]
        end = max(end, float(ref["f0"][0][-1]) if len(ref["f0"][0]) else 0.0)
    if loaded["notes"] is not None:
        a = loaded["notes"]
        a = a[a[:, 1] > a[:, 0]]
        ref["notes"] = a
        end = max(end, float(a[:, 1].max()) if len(a) else 0.0)
        if "f0" not in ref:      # chỉ số mức khung: đổi đáp án nốt ra f0 lưới 10 ms
            t = np.arange(0.0, end + FRAME_HOP, FRAME_HOP)
            mono = [MelodyNote(float(s), float(e), int(round(m))) for s, e, m in a]
            ref["f0"] = (t, L.notes_to_f0(mono, t))
    p = _resolve(entry.get("chord_ref"), base)
    if p is not None and p.exists() and entry.get("chord_ref_ok", True) is not False:
        segs = lab_to_segments(p)
        for s in segs:
            s.start += off
            s.end += off
        ref["chords"] = segs
    ref["duration"] = float(entry.get("duration") or 0.0) or end
    return ref


# --------------------------------------------------------------------------- chấm một bài


def _primary(entry: dict) -> str | None:
    if entry.get("ref_f0"):
        return "oa"
    if entry.get("ref_notes"):
        return "note_f1"
    return None


def score_track(task: tuple) -> list[dict]:
    entry, models, beats_dir = task[:3]
    nn_pairs = task[3] if len(task) > 3 else NN_PAIRS
    ds, tid = entry["dataset"], str(entry["track_id"])
    try:
        ref = load_refs(entry)
    except Exception as e:  # pragma: no cover - dữ liệu hỏng: báo và bỏ bài
        print(f"[{ds}/{tid}] lỗi đọc đáp án: {e}", file=sys.stderr, flush=True)
        return []
    grid = load_beats(ds, tid, beats_dir)
    outs = {}
    for m in models:
        p = cache_path(m, ds, tid)
        if p.exists():
            outs[m] = load_output(p)
    if not ref["duration"]:
        ends = [n.end for o in outs.values() for n in o.notes]
        ends += [float(o.pitch_track.times[-1]) for o in outs.values() if o.pitch_track is not None
                 and len(o.pitch_track.times)]
        ref["duration"] = max(ends, default=0.0)

    rows: list[dict] = []

    def emit(model: str, family: str, ci: int, cfg: dict | None, est: list[MelodyNote], raw=None):
        variants = [(0, est)]
        if grid is not None and raw is None:
            variants.append((1, L.quantize_to_beats(est, grid)))
        for q, e in variants:
            vals, wts = score_line(entry, ref, e, raw)
            rows.append({"dataset": ds, "track_id": tid, "model": model, "family": family, "cfg": ci, "quant": q,
                         "params": "" if cfg is None else cfg_text(cfg), "duration": ref["duration"],
                         **{k: vals.get(k, math.nan) for k in ALL_METRICS},
                         **{f"w_{k}": wts.get(k, 0.0) for k in ALL_METRICS}})

    note_lines: dict[str, dict[str, list[MelodyNote]]] = {}   # model -> họ -> dòng của cấu hình mặc định
    # model -> (họ, cấu hình) -> dòng, chỉ giữ các cấu hình mà nn_gated dùng
    nn_lines: dict[str, dict[tuple[str, int], list[MelodyNote]]] = {}
    nn_needed = set(NN_KEYS.values())
    f0_tracks: dict[str, tuple[PitchTrack, bool]] = {}
    for m, o in outs.items():
        if o.notes:
            note_lines[m], nn_lines[m] = {}, {}
            for fam, cfgs in note_families().items():
                for ci, cfg in enumerate(cfgs):
                    est = L.notes_to_line(o.notes, params_of(cfg))
                    emit(m, fam, ci, cfg, est)
                    if (fam, ci) in nn_needed:
                        nn_lines[m][(fam, ci)] = est
                    if ci == 0:
                        note_lines[m][fam] = est
        if o.pitch_track is not None:
            pt = L.raw_pitch_track(o)          # f0 chưa cắt ngưỡng nếu model lưu (RMVPE)
            has_raw = pt is not o.pitch_track
            f0_tracks[m] = (pt, has_raw)
            fams = f0_families(has_raw, pt.confidence is not None)
            default_by_vt: dict[float, list[MelodyNote]] = {}   # dòng của tham số tách nốt mặc định, theo ngưỡng
            d0 = fams["f0notes"][0]
            for ci, cfg in enumerate(fams["f0notes"]):
                est = L.f0_to_line(pt, params_of(cfg))
                emit(m, "f0notes", ci, cfg, est)
                if all(cfg[k] == d0[k] for k in cfg if k not in ("_base", "voicing_threshold")):
                    default_by_vt[cfg["voicing_threshold"]] = est
            # f0 thô chỉ khác dòng nốt ở chỉ số mức khung: bài không đáp án thì bỏ.
            for ci, cfg in enumerate(fams["f0raw"] if ref.get("f0") is not None else []):
                vt = cfg.get("voicing_threshold", 0.0)
                hz = np.asarray(pt.f0_hz, dtype=np.float64).copy()
                if pt.confidence is not None and vt > 0:
                    hz[np.asarray(pt.confidence) < vt] = 0.0
                # các chỉ số không cần đáp án lấy từ dòng nốt mặc định cùng ngưỡng
                emit(m, "f0raw", ci, cfg, default_by_vt[vt], raw=(np.asarray(pt.times, dtype=np.float64), hz))
    base_f0 = params_of(f0_families(True, True)["f0notes"][0])
    cfams = combo_families()
    # Cổng giọng theo model nốt. Model không gắn nhãn nguồn nào (Basic Pitch) thì bỏ họ combo_gated. Model có
    # nhãn nhưng bài này không có giọng thì cổng rỗng (= dòng của model nốt), để mọi cấu hình chấm cùng một bộ bài.
    gates = {mn: {d: L.voice_intervals(outs[mn].notes, pad=d) for d in COMBO_PAD} for mn in note_lines
             if any(n.source is not None for n in outs[mn].notes)}
    for mf, (pt, has_raw) in f0_tracks.items():
        if has_raw:
            prim = {v: L.f0_to_line(pt, replace(base_f0, voicing_threshold=v)) for v in COMBO_VT}
        else:                     # không có f0 thô: không dò ngưỡng, chỉ một giá trị `vt`
            prim = {COMBO_VT[0]: L.f0_to_line(pt, replace(base_f0, voicing_threshold=0.0))}
        for mn, lines in note_lines.items():
            if mf == mn:
                continue
            name = f"{mf}+{mn}"
            for ci, c in enumerate(cfams["combo"]):
                if c["vt"] in prim:
                    emit(name, "combo", ci, c, L.combine_lines(prim[c["vt"]], lines[c["sec"]], min_gap=c["min_gap"]))
            for ci, c in enumerate(cfams["combo_gated"] if mn in gates else []):
                if c["vt"] in prim:
                    emit(name, "combo_gated", ci, c,
                         L.combine_gated(prim[c["vt"]], lines[c["sec"]], gates[mn][c["pad"]]))
    # Ghép hai model nốt: dòng của A ở đoạn có giọng (cổng theo nốt `voice` của A hoặc của B), dòng của B ở phần
    # còn lại. Mọi cấu hình đều chấm trên mọi bài có cache của cả hai model (model không có nốt nào ở bài này thì
    # dòng rỗng, bài không có giọng thì cổng rỗng = dòng của B), để CV so trên cùng một bộ bài.
    voice_gates: dict[tuple[str, float], list] = {}
    for a, b in nn_pairs:
        if a not in outs or b not in outs:
            continue
        for m in (a, b):
            for d in NN_PAD:
                if (m, d) not in voice_gates:
                    voice_gates[(m, d)] = L.voice_intervals(outs[m].notes, pad=d)
        for ci, c in enumerate(NN_GRID):
            line_a = nn_lines.get(a, {}).get(NN_KEYS[c["a"]], [])
            line_b = nn_lines.get(b, {}).get(NN_KEYS[c["b"]], [])
            gate = voice_gates[(a if c["gate"] == "a" else b, c["pad"])]
            emit(nn_name(a, b), "nn_gated", ci, c, L.combine_gated(line_a, line_b, gate))
    return rows


# --------------------------------------------------------------------------- tổng hợp


def _agg(rows: list[dict]) -> dict:
    out = {"n_tracks": len(rows), "minutes": sum(r["duration"] for r in rows) / 60.0}
    for k in ALL_METRICS:
        v = np.array([r[k] for r in rows], dtype=np.float64)
        w = np.array([r[f"w_{k}"] for r in rows], dtype=np.float64)
        ok = np.isfinite(v) & (w > 0)
        out[k] = float(v[ok].mean()) if ok.any() else math.nan
        out[f"{k}_tw"] = float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else math.nan
    return out


def track_objective(r: dict, primary: str, objective: str) -> float:
    """Điểm mục tiêu của một dòng (một bài, một cấu hình).

    `objective="primary"`: OA (bộ f0) hoặc F1 onset (bộ nốt).
    `objective="rpa_vr"`: (RPA + VR) / 2 ở mức khung, cho mọi bộ có đáp án (bộ nốt dùng f0 đổi từ nốt). Không phạt
    VFA: các bộ ADC2004 / MIREX05 / MIR-ST500 chỉ gán nhãn giọng hát, nên nốt nhạc cụ ở đoạn không hát bị tính là
    báo nhầm dù đúng mục tiêu sản phẩm (giai điệu nổi nhất bất kể nguồn).
    """
    if objective == "rpa_vr":
        a, b = r["rpa"], r["vr"]
        return 0.5 * (a + b) if np.isfinite(a) and np.isfinite(b) else math.nan
    return r[primary]


def select_cv(rows: list[dict], entries: dict[str, list[dict]], folds: int,
              level: str = "family", objective: str = "primary") -> tuple[list[dict], dict]:
    """Chọn cấu hình bằng CV theo bài. Trả về (các dòng đã chọn, kèm cột `selection`, và mô tả lựa chọn).

    `level="family"`: mỗi (model, họ, căn phách) chọn trong lưới của họ đó; nhãn `default`, `full`, `cv`.
    `level="model"`: mỗi (model, căn phách) chọn trong mọi cấu hình của mọi họ (trừ `f0raw`, không phải dòng
    nốt), nên họ chiến lược cũng được chọn bằng CV; nhãn `full_model`, `cv_model`, cột `family` = `*`.
    `level="model_ds"`: như `model` nhưng chọn riêng trong từng bộ có đáp án (trần của model theo loại nhạc,
    không phải một cấu hình chung); nhãn `full_model_ds`, `cv_model_ds`. Bộ không đáp án bị bỏ.

    Hàm mục tiêu cân bằng theo bộ: trung bình của (trung bình chỉ số mục tiêu theo bài trong từng bộ), để bộ
    nhiều bài (MIR-ST500) không lấn các bộ khác. `objective` xem `track_objective`; khác "primary" thì nhãn
    `selection` có thêm hậu tố `_<objective>`.
    """
    prim = {(ds, str(e["track_id"])): _primary(e) for ds, es in entries.items() for e in es}
    fold_of = {}
    for ds, es in entries.items():
        labelled = [e for e in es if _primary(e)]
        for i, e in enumerate(labelled):
            fold_of[(ds, str(e["track_id"]))] = i % folds
    groups: dict[tuple, dict[tuple, list[dict]]] = {}
    for r in rows:
        if level == "family":
            key, ck = (r["model"], r["family"], r["quant"]), (r["cfg"],)
        else:
            if r["family"] == "f0raw":
                continue
            key, ck = (r["model"], r["quant"]), (r["family"], r["cfg"])
            if level == "model_ds":
                if not prim.get((r["dataset"], r["track_id"])):
                    continue
                key = key + (r["dataset"],)
        groups.setdefault(key, {}).setdefault(ck, []).append(r)

    def score(cfg_rows: list[dict], keep) -> float:
        by_ds: dict[str, list[float]] = {}
        for r in cfg_rows:
            t = (r["dataset"], r["track_id"])
            m = prim.get(t)
            if m and keep(t):
                v = track_objective(r, m, objective)
                if np.isfinite(v):
                    by_ds.setdefault(r["dataset"], []).append(v)
        return float(np.mean([np.mean(v) for v in by_ds.values()])) if by_ds else -math.inf

    sfx = {"family": "", "model": "_model", "model_ds": "_model_ds"}[level]
    if objective != "primary":
        sfx += f"_{objective}"
    chosen: list[dict] = []
    choices: dict = {}

    def tag(r: dict, sel: str) -> dict:
        out = {**r, "selection": sel + sfx}
        if level != "family":
            out["family"] = "*"     # họ và cấu hình được chọn ghi trong `choices`
        return out

    for key, by_cfg in groups.items():
        # Thứ tự xuất hiện trong lưới (họ theo thứ tự khai báo, cấu hình mặc định đứng đầu mỗi họ):
        # hòa điểm thì lấy cấu hình đứng trước.
        cfgs = sorted(by_cfg, key=lambda c: c[-1]) if level == "family" else list(by_cfg)
        order = {c: i for i, c in enumerate(cfgs)}
        pick = lambda keep: max(cfgs, key=lambda c: (score(by_cfg[c], keep), -order[c]))  # noqa: E731
        full = pick(lambda t: t in fold_of)
        desc = {"full": list(full), "full_params": by_cfg[full][0]["params"], "folds": []}
        if level == "family":
            chosen.extend(tag(r, "default") for r in by_cfg[cfgs[0]])
        chosen.extend(tag(r, "full") for r in by_cfg[full])
        for f in range(folds):
            best = pick(lambda t, f=f: t in fold_of and fold_of[t] != f)
            desc["folds"].append(list(best))
            chosen.extend(tag(r, "cv") for r in by_cfg[best] if fold_of.get((r["dataset"], r["track_id"])) == f)
        # bộ không đáp án: dùng cấu hình chọn trên cả bộ có đáp án
        chosen.extend(tag(r, "cv") for r in by_cfg[full] if (r["dataset"], r["track_id"]) not in fold_of)
        choices["|".join(map(str, key))] = desc
    return chosen, choices


def summarize(chosen: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in chosen:
        groups.setdefault((r["selection"], r["dataset"], r["model"], r["family"], r["quant"]), []).append(r)
    return [{"selection": s, "dataset": d, "model": m, "family": f, "quant": q, **_agg(rs)}
            for (s, d, m, f, q), rs in sorted(groups.items())]


def dataset_primary(entries: dict[str, list[dict]]) -> dict[str, str | None]:
    """Chỉ số mục tiêu của mỗi bộ (giống hàm mục tiêu của CV): OA nếu có ref_f0, F1 onset nếu chỉ có ref_notes."""
    return {ds: next((m for m in map(_primary, es) if m), None) for ds, es in entries.items()}


def compare_top2(chosen: list[dict], summary: list[dict], primary: dict[str, str | None], n_boot: int,
                 seed: int = 0, selection: str = "cv") -> list[dict]:
    """So cặp hai cấu hình (model, họ, căn phách) tốt nhất theo chỉ số mục tiêu (điểm CV) trên mỗi bộ."""
    from eval.ablation_kajiura import bootstrap_diff

    rng = np.random.default_rng(seed)
    out = []
    for ds in sorted({s["dataset"] for s in summary}):
        rows = [s for s in summary if s["selection"] == selection and s["dataset"] == ds]
        metric = primary.get(ds)
        if metric is None or len(rows) < 2:
            continue
        top = sorted(rows, key=lambda s: -s[metric] if np.isfinite(s[metric]) else math.inf)[:2]
        per = []
        for s in top:
            per.append({r["track_id"]: r[metric] for r in chosen if r["selection"] == selection and r["dataset"] == ds
                        and (r["model"], r["family"], r["quant"]) == (s["model"], s["family"], s["quant"])})
        common = sorted(set(per[0]) & set(per[1]))
        if len(common) < 2:
            continue
        d = bootstrap_diff(np.array([per[0][t] for t in common]), np.array([per[1][t] for t in common]), n_boot, rng)
        out.append({"dataset": ds, "metric": metric, "a": _name(top[0]), "b": _name(top[1]), "n": len(common), **d})
    return out


def compare_named(chosen: list[dict], selection: str, pairs: list[tuple[str, str]], metrics: list[str],
                  n_boot: int, seed: int = 0) -> list[dict]:
    """So cặp (model a, model b) theo tên trên mọi bộ có đáp án, với từng chỉ số có giá trị ở bộ đó
    (bootstrap 95% chênh lệch trung bình theo bài, p Wilcoxon), trên các bài cả hai đều có."""
    from eval.ablation_kajiura import bootstrap_diff

    rng = np.random.default_rng(seed)
    per: dict[tuple[str, str], dict[str, dict]] = {}
    for r in chosen:
        if r["selection"] == selection and r["quant"] == 0:
            per.setdefault((r["dataset"], r["model"]), {})[r["track_id"]] = r
    out = []
    for ds in sorted({d for d, _ in per}):
        for a, b in pairs:
            ra, rb = per.get((ds, a)), per.get((ds, b))
            if not ra or not rb:
                continue
            common = sorted(set(ra) & set(rb))
            for m in metrics:
                xa = np.array([ra[t][m] for t in common], dtype=np.float64)
                xb = np.array([rb[t][m] for t in common], dtype=np.float64)
                ok = np.isfinite(xa) & np.isfinite(xb)
                if ok.sum() < 2:
                    continue
                d = bootstrap_diff(xa[ok], xb[ok], n_boot, rng)
                out.append({"dataset": ds, "metric": m, "a": a, "b": b, "n": int(ok.sum()), **d})
    return out


def _name(s: dict) -> str:
    base = s["model"] if s["family"] == "*" else f"{s['model']}/{s['family']}"
    return base + ("+beats" if s["quant"] else "")


# --------------------------------------------------------------------------- in và ghi


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _fmt(v, pct: bool = True) -> str:
    if v is None or not np.isfinite(v):
        return "–"
    return f"{100 * v:.1f}" if pct else f"{v:.2f}"


COL_NAMES = {"oa": "OA", "rpa": "RPA", "rca": "RCA", "vr": "VR", "vfa": "VFA", "note_f1": "F1 onset",
             "note_off_f1": "F1 on+off", "note_p": "P onset", "note_r": "R onset", "coverage": "% có nốt",
             "in_chord": "% trong hợp âm", "octave_jumps_pm": "nhảy >8ve/phút", "notes_pm": "nốt/phút",
             "median_dur": "độ dài trung vị (s)"}


def tables_md(summary: list[dict], primary: dict[str, str | None], selection: str, title: str,
              compact: bool = False, top: int = 0, objective: str = "primary") -> list[str]:
    """Bảng markdown mỗi bộ. `compact`: chỉ các cột chính (OA/RPA/RCA/VR/VFA, F1, số không cần đáp án).
    Dòng sắp theo mục tiêu `objective` (xem `track_objective`; bộ không đáp án sắp theo % có nốt);
    `top` > 0: chỉ `top` dòng đầu mỗi bộ."""
    out: list[str] = []
    for ds in sorted({s["dataset"] for s in summary}):
        rows = [s for s in summary if s["dataset"] == ds and s["selection"] == selection]
        if not rows:
            continue
        p = primary.get(ds)
        n = max(s["n_tracks"] for s in rows)
        mins = max(s["minutes"] for s in rows)
        if p == "oa":
            cols = ["oa", "rpa", "rca", "vr", "vfa"]
        elif p == "note_f1":
            cols = ["note_f1", "note_off_f1"] + ([] if compact else ["note_p", "note_r"]) + ["oa", "rpa", "vr", "vfa"]
        else:
            cols = []
        if p is None or not compact:
            cols += ["coverage", "in_chord", "octave_jumps_pm", "notes_pm", "median_dur"]
        if p is None and not any(np.isfinite(s["in_chord"]) for s in rows):
            cols.remove("in_chord")
        out.append(f"\n### {ds} ({n} bài, {mins:.1f} phút) — {title}\n")
        out.append("| model | bài | " + " | ".join(COL_NAMES[c] for c in cols) + " |")
        out.append("|---" * (len(cols) + 2) + "|")
        def rank(s: dict) -> float:
            v = track_objective(s, p, objective) if p else s["coverage"]
            return -(v if np.isfinite(v) else -1)

        ranked = sorted(rows, key=rank)
        for s in ranked[:top] if top > 0 else ranked:
            cells = []
            for c in cols:
                pct = c not in ("octave_jumps_pm", "median_dur", "notes_pm")
                v, tw = _fmt(s[c], pct), _fmt(s[c + "_tw"], pct)
                cells.append(v if compact or v == tw else f"{v} ({tw})")
            out.append(f"| {_name(s)} | {s['n_tracks']} | " + " | ".join(cells) + " |")
    return out


def tests_md(tests: list[dict], title: str) -> list[str]:
    if not tests:
        return []
    out = [f"\n### {title} (a − b, điểm %, CI bootstrap 95%, thắng/thua, p Wilcoxon)\n"]
    for t in tests:
        out.append(f"- {t['dataset']} {COL_NAMES.get(t['metric'], t['metric'])}: {t['a']} − {t['b']} = "
                   f"{100 * t['mean']:+.2f} [{100 * t['lo']:+.2f}, {100 * t['hi']:+.2f}], {t['wins']}/{t['losses']}, "
                   f"p={t['p_wilcoxon']:.3f} (n={t['n']})")
    return out


def balanced_rank_md(summary: list[dict], primary: dict[str, str | None], selection: str, title: str) -> list[str]:
    """Xếp hạng model theo trung bình cân bằng các bộ có đáp án của chỉ số mục tiêu từng bộ (OA hoặc F1 onset);
    chỉ các model có mặt ở mọi bộ có đáp án."""
    labelled = sorted(ds for ds, p in primary.items() if p)
    tab: dict[str, dict[str, dict]] = {}
    for s in summary:
        if s["selection"] == selection and s["quant"] == 0 and s["dataset"] in labelled:
            tab.setdefault(s["model"], {})[s["dataset"]] = s
    ranked = sorted(((float(np.mean([d[ds][primary[ds]] for ds in labelled])), m, d) for m, d in tab.items()
                     if len(d) == len(labelled)), key=lambda x: -x[0])
    if not ranked:
        return []
    out = [f"\n### {title}\n", "| model | trung bình | " + " | ".join(
        f"{ds} {COL_NAMES[primary[ds]]}" for ds in labelled) + " |", "|---" * (len(labelled) + 2) + "|"]
    for v, m, d in ranked:
        out.append(f"| {m} | {100 * v:.1f} | " + " | ".join(
            f"{_fmt(d[ds][primary[ds]])} ({d[ds]['n_tracks']})" for ds in labelled) + " |")
    return out


def named_tests_md(tests: list[dict], title: str) -> list[str]:
    if not tests:
        return []
    out = [f"\n### {title} (a − b, điểm %, CI bootstrap 95%, thắng/thua, p Wilcoxon)\n",
           "| bộ | a − b | chỉ số | chênh | CI 95% | thắng/thua | p | n |", "|---|---|---|---|---|---|---|---|"]
    for t in tests:
        out.append(f"| {t['dataset']} | {t['a']} − {t['b']} | {COL_NAMES.get(t['metric'], t['metric'])} | "
                   f"{100 * t['mean']:+.2f} | [{100 * t['lo']:+.2f}, {100 * t['hi']:+.2f}] | {t['wins']}/{t['losses']} | "
                   f"{t['p_wilcoxon']:.3f} | {t['n']} |")
    return out


def choices_md(choices: dict, title: str = "Cấu hình chọn trên cả các bộ có đáp án (dùng cho bộ không đáp án)"
               ) -> list[str]:
    out = [f"\n### {title}\n"]
    for key, d in sorted(choices.items()):
        model, quant, *ds = key.split("|")
        fam, ci = d["full"]
        folds = ", ".join(f"{f}#{c}" for f, c in d["folds"])
        where = f" [{ds[0]}]" if ds else ""
        out.append(f"- {model}{'+beats' if quant == '1' else ''}{where}: {fam}#{ci} `{d['full_params']}` "
                   f"(các phần CV: {folds})")
    return out


def main(argv: list[str] | None = None) -> int:
    from eval.evaluate import _utf8_console

    _utf8_console()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", help="danh sách bộ, phân cách bằng dấu phẩy (mặc định: mọi manifest)")
    ap.add_argument("--models", help="danh sách model trong cache (mặc định: tất cả)")
    ap.add_argument("--exclude", default="muscriptor_medium",
                    help="model bỏ qua, phân cách bằng dấu phẩy (mặc định: muscriptor_medium, đã quyết định không dùng)")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--no-beats", action="store_true", help="không chấm bản căn phách")
    ap.add_argument("--out-dir", type=Path, default=paths.EVAL_DIR / "results" / "melody")
    ap.add_argument("--reuse", action="store_true",
                    help="không chấm lại: đọc out-dir/per_track.csv, chỉ chọn cấu hình và in bảng")
    ap.add_argument("--nn-pairs", default=",".join(f"{a}:{b}" for a, b in NN_PAIRS),
                    help="cặp ghép hai model nốt A:B (A ở đoạn có giọng, B ở phần còn lại), phân cách bằng dấu phẩy")
    args = ap.parse_args(argv)
    nn_pairs = parse_nn_pairs(args.nn_pairs)

    entries = read_manifests(args.datasets.split(",") if args.datasets else None)
    if not entries:
        print("Không tìm thấy manifest nào trong", paths.EVAL_DIR / "melody", file=sys.stderr)
        return 1
    if args.reuse:
        rows = [r for r in read_per_track(args.out_dir / "per_track.csv") if r["dataset"] in entries]
        if args.models:
            keep = set(args.models.split(","))
            rows = [r for r in rows if set(model_parts(r["model"])) <= keep]
        singles = sorted({r["model"] for r in rows if len(model_parts(r["model"])) == 1})
        return report_from_rows(rows, entries, singles, args)
    only = args.models.split(",") if args.models else None
    # model ghi rõ trong --models thì không bị --exclude bỏ
    exclude = [x for x in args.exclude.split(",") if x and not (only and x in only)]
    models = discover_models(list(entries), only, exclude)
    if not models:
        print("Không có cache model nào trong", paths.CACHE_DIR / "melody", file=sys.stderr)
        return 1
    print(f"Bộ: {', '.join(f'{d} ({len(e)} bài)' for d, e in entries.items())}; model: {', '.join(models)}",
          flush=True)
    beats_dir = None if args.no_beats else paths.CACHE_DIR / "melody" / "_beats"
    tasks = [(e, models, beats_dir, nn_pairs) for es in entries.values() for e in es]
    rows: list[dict] = []
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            for i, rs in enumerate(pool.imap_unordered(score_track, tasks), 1):
                rows.extend(rs)
                if i % 20 == 0 or i == len(tasks):
                    print(f"  {i}/{len(tasks)} bài", flush=True)
    else:
        for e in tasks:
            rows.extend(score_track(e))
    if not rows:
        print("Không có bài nào có cache.", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "per_track.csv", rows)
    return report_from_rows(rows, entries, models, args)


def read_per_track(path: Path) -> list[dict]:
    """Đọc lại `per_track.csv` (để chọn cấu hình và in bảng lại mà không chấm lại)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k, v in r.items():
                if k in ("cfg", "quant"):
                    r[k] = int(v)
                elif k not in ("dataset", "track_id", "model", "family", "params"):
                    r[k] = float(v) if v not in ("", "nan") else math.nan
            rows.append(r)
    return rows


def common_subset(chosen: list[dict], selection: str) -> list[dict]:
    """Ở bộ nào có model không phủ đủ bài (vd MuScriptor large chỉ cache một phần), lấy các bài mọi model đều có
    và gắn nhãn `<selection>_common` để so công bằng trên cùng tập bài. Bộ mà mọi model phủ như nhau thì bỏ."""
    tracks: dict[str, dict[str, set]] = {}
    for r in chosen:
        if r["selection"] == selection:
            tracks.setdefault(r["dataset"], {}).setdefault(r["model"], set()).add(r["track_id"])
    keep: dict[str, set] = {}
    for ds, by_model in tracks.items():
        common = set.intersection(*by_model.values())
        if common and len(common) < max(len(v) for v in by_model.values()):
            keep[ds] = common
    return [{**r, "selection": selection + "_common"} for r in chosen
            if r["selection"] == selection and r["track_id"] in keep.get(r["dataset"], ())]


def report_from_rows(rows: list[dict], entries: dict[str, list[dict]], models: list[str], args) -> int:
    chosen_f, choices_f = select_cv(rows, entries, args.folds, level="family")
    chosen_m, choices_m = select_cv(rows, entries, args.folds, level="model")
    chosen_d, choices_d = select_cv(rows, entries, args.folds, level="model_ds")
    chosen_c = common_subset(chosen_m, "cv_model")
    summary = summarize(chosen_f + chosen_m + chosen_d + chosen_c)
    primary = dataset_primary(entries)
    tests_m = compare_top2(chosen_m, summary, primary, args.boot, selection="cv_model")
    tests_c = compare_top2(chosen_c, summary, primary, args.boot, selection="cv_model_common")
    tests_d = compare_top2(chosen_d, summary, primary, args.boot, selection="cv_model_ds")
    tests_f = compare_top2(chosen_f, summary, primary, args.boot, selection="cv")
    # Biến thể mục tiêu không phạt VFA, và so cặp các bản ghép hai model nốt với từng thành phần của nó.
    chosen_r, choices_r = select_cv(rows, entries, args.folds, level="model", objective="rpa_vr")
    summary += summarize(chosen_r)
    nn_models = sorted({r["model"] for r in rows if r["family"] == "nn_gated"})
    pairs = [(nm, part) for nm in nn_models for part in model_parts(nm)]
    metrics = ["oa", "rpa", "vr", "note_f1"]
    tests_nn = compare_named(chosen_m, "cv_model", pairs, metrics, args.boot)
    tests_nn_r = compare_named(chosen_r, "cv_model_rpa_vr", pairs, metrics, args.boot)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "summary.csv", summary)
    (args.out_dir / "cv_choices.json").write_text(json.dumps(
        {"model": choices_m, "model_rpa_vr": choices_r, "model_per_dataset": choices_d, "family": choices_f,
         "tests_model": tests_m, "tests_model_common": tests_c, "tests_model_per_dataset": tests_d,
         "tests_family": tests_f, "tests_nn": tests_nn, "tests_nn_rpa_vr": tests_nn_r},
        indent=1, ensure_ascii=False), encoding="utf-8")
    brief = (["\n## Tóm tắt: top cấu hình mỗi bộ"]
             + balanced_rank_md(summary, primary, "cv_model", "Xếp hạng chung, mục tiêu OA / F1 onset (trung bình "
                                                              "cân bằng các bộ có đáp án; trong ngoặc: số bài)")
             + balanced_rank_md(summary, primary, "cv_model_rpa_vr", "Xếp hạng chung, cấu hình chọn theo (RPA+VR)/2 "
                                                                     "(điểm vẫn là OA / F1 onset)")
             + tables_md(summary, primary, "cv_model", "top 5, cấu hình chung, mục tiêu OA / F1 onset",
                         compact=True, top=5)
             + tables_md(summary, primary, "cv_model_rpa_vr", "top 5, cấu hình chung, mục tiêu (RPA+VR)/2 "
                                                             "(không phạt VFA; ⚠️ thưởng cả việc bật nốt mọi lúc, "
                                                             "xem cột VFA)", compact=True, top=5, objective="rpa_vr")
             + named_tests_md(tests_nn, "Ghép hai model nốt so với từng thành phần (mục tiêu OA / F1 onset)")
             + named_tests_md(tests_nn_r, "Ghép hai model nốt so với từng thành phần (mục tiêu (RPA+VR)/2)")
             + choices_md({k: v for k, v in choices_m.items() if k.split("|")[0] in nn_models},
                          "Cấu hình ghép chọn được (mục tiêu OA / F1 onset)")
             + choices_md({k: v for k, v in choices_r.items() if k.split("|")[0] in nn_models},
                          "Cấu hình ghép chọn được (mục tiêu (RPA+VR)/2)"))
    head = (tables_md(summary, primary, "cv_model", "mỗi model ở MỘT cấu hình chung cho mọi bộ (họ + tham số chọn "
                                                    "bằng CV theo bài, mục tiêu cân bằng theo bộ)", compact=True)
            + tests_md(tests_m, "So cặp hai model tốt nhất (cấu hình chung)") + choices_md(choices_m))
    if chosen_c:       # bảng phụ: chỉ các bài mọi model đều có cache
        head += (tables_md(summary, primary, "cv_model_common", "BẢNG PHỤ: chỉ các bài mọi model đều có cache, "
                                                                "cấu hình chung như trên", compact=True)
                 + tests_md(tests_c, "So cặp hai model tốt nhất trên tập con chung"))
    per_ds = (tables_md(summary, primary, "cv_model_ds", "mỗi model, cấu hình chọn riêng trong bộ này bằng CV theo "
                                                         "bài (trần theo loại nhạc)", compact=True)
              + tests_md(tests_d, "So cặp hai model tốt nhất (cấu hình riêng từng bộ)")
              + choices_md(choices_d, "Cấu hình chọn trên cả từng bộ"))
    detail = (tables_md(summary, primary, "cv", "từng họ chiến lược, điểm CV, trung bình theo bài "
                                                "(theo thời lượng trong ngoặc)")
              + tests_md(tests_f, "So cặp hai (model, họ) tốt nhất"))
    report = ["# Kết quả dò giai điệu", f"\nModel: {', '.join(models)}. CV {args.folds} phần theo bài; mục tiêu "
              "OA (bộ f0) hoặc F1 onset (MIR-ST500), cân bằng theo bộ; biến thể (RPA+VR)/2. Điểm %. Lưu ý: "
              "ADC2004, MIREX05, MIR-ST500 chỉ gán nhãn giọng hát, nên nốt nhạc cụ ở đoạn không hát bị tính VFA "
              "và làm giảm OA dù đúng mục tiêu sản phẩm (giai điệu nổi nhất bất kể nguồn)."] + brief + \
             ["\n## Theo model, cấu hình chung"] + head + \
             ["\n## Theo model, cấu hình riêng từng bộ"] + per_ds + ["\n## Theo họ chiến lược"] + detail
    (args.out_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(brief + ["\n## Theo model, cấu hình chung"] + head))
    print(f"\nĐã ghi {args.out_dir} (report.md có thêm bảng theo từng họ chiến lược)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
