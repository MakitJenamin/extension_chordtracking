"""Chấm dòng giai điệu của mọi (model × chiến lược × bộ dữ liệu) từ cache, chỉ dùng CPU.

    python -m eval.melody.score [--datasets a,b] [--models m1,m2] [--folds 3] [--jobs 6] [--boot 2000]
                                [--no-beats] [--out-dir EVAL_DIR/results/melody]

Đầu vào:
- Manifest `EVAL_DIR/melody/<dataset>/manifest.jsonl` (xem `eval/melody/README.md`). Trường dùng thêm nếu có:
  `offset` (giây, cộng vào mọi thời điểm của đáp án để khớp audio), `duration`, `chord_ref` (.lab nhãn Harte).
- Cache `chordtrack.melody.io.cache_path(model, dataset, track_id)`. Model là mọi thư mục trong
  `CACHE_DIR/melody/` không bắt đầu bằng `_`. Model có `notes` chấm các họ chiến lược nốt đa âm, model có
  `pitch_track` chấm các họ f0. Mỗi cặp (model f0, model nốt) chấm thêm họ `combo` (`combine_lines`).
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
bộ không đáp án). Dòng `default` là cấu hình đầu tiên của lưới (= `LineParams` mặc định của họ đó).

Ra `out-dir`: `per_track.csv` (mọi bài, mọi cấu hình), `summary.csv`, `cv_choices.json`, và in bảng markdown kèm
so cặp hai cấu hình tốt nhất mỗi bộ (bootstrap 95% chênh lệch trung bình theo bài, p Wilcoxon).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from chordtrack import paths
from chordtrack.melody import line as L
from chordtrack.melody.io import cache_path, load_output
from chordtrack.types import BeatGrid, MelodyNote, lab_to_segments

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
        "source": _grid(P(strategy="salience", source_mode="strict"),
                        w_pitch=[0.5, 1.0], min_dur=[0.06, 0.1]),
        "viterbi": _grid(P(strategy="viterbi"),
                         jump_penalty=[0.5, 0.25, 1.0], cut_penalty=[0.15, 0.05, 0.4], w_pitch=[0.5, 1.0]),
    }


def f0_families(has_conf: bool) -> dict[str, list[dict]]:
    P = L.LineParams
    vt = [0.0, 0.3, 0.5] if has_conf else [0.0]
    return {
        "f0raw": _grid(P(), voicing_threshold=vt),
        "f0notes": _grid(P(), median_frames=[5, 1, 9], split_semitones=[0.6, 0.8],
                         split_min_dur=[0.08, 0.12], voicing_threshold=vt[:1] + vt[2:]),
    }


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


def read_numeric_csv(path: Path, ncols: int) -> np.ndarray:
    """Đọc csv số (bỏ dòng tiêu đề hay dòng hỏng), phân cách bằng dấu phẩy, tab hoặc khoảng trắng."""
    rows = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        parts = ln.replace(",", " ").replace(";", " ").split()
        try:
            rows.append([float(x) for x in parts[:ncols]])
        except ValueError:
            continue
    rows = [r for r in rows if len(r) == ncols]
    return np.array(rows, dtype=np.float64).reshape(-1, ncols)


def discover_models(datasets: list[str], only: list[str] | None) -> list[str]:
    """Các model có ít nhất một file cache ở một trong các bộ."""
    root = paths.CACHE_DIR / "melody"
    out: list[str] = []
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith("_") or (only and d.name not in only):
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
    base, off = entry["_dir"], float(entry.get("offset") or 0.0)
    ref: dict = {}
    end = 0.0
    p = _resolve(entry.get("ref_f0"), base)
    if p is not None:
        a = read_numeric_csv(p, 2)
        ref["f0"] = (a[:, 0] + off, np.maximum(a[:, 1], 0.0))
        end = max(end, float(a[-1, 0] + off) if len(a) else 0.0)
    p = _resolve(entry.get("ref_notes"), base)
    if p is not None:
        a = read_numeric_csv(p, 3)
        a[:, :2] += off
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
    entry, models, beats_dir = task
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

    defaults_f0, defaults_notes = {}, {}
    for m, o in outs.items():
        if o.notes:
            for fam, cfgs in note_families().items():
                for ci, cfg in enumerate(cfgs):
                    est = L.notes_to_line(o.notes, params_of(cfg))
                    emit(m, fam, ci, cfg, est)
                    if fam == "viterbi" and ci == 0:
                        defaults_notes[m] = est
        if o.pitch_track is not None:
            pt = o.pitch_track
            fams = f0_families(pt.confidence is not None)
            # f0 thô chỉ khác dòng nốt ở chỉ số mức khung: bài không đáp án thì bỏ (trùng f0notes mặc định).
            for ci, cfg in enumerate(fams["f0raw"] if ref.get("f0") is not None else []):
                hz = np.asarray(pt.f0_hz, dtype=np.float64).copy()
                vt = cfg.get("voicing_threshold", 0.0)
                if pt.confidence is not None and vt > 0:
                    hz[np.asarray(pt.confidence) < vt] = 0.0
                est = L.f0_to_line(pt, params_of(cfg))      # cho các chỉ số không cần đáp án
                emit(m, "f0raw", ci, cfg, est, raw=(np.asarray(pt.times, dtype=np.float64), hz))
            for ci, cfg in enumerate(fams["f0notes"]):
                est = L.f0_to_line(pt, params_of(cfg))
                emit(m, "f0notes", ci, cfg, est)
                if ci == 0:
                    defaults_f0[m] = est
    for mf, lf in defaults_f0.items():
        for mn, ln in defaults_notes.items():
            if mf != mn:
                emit(f"{mf}+{mn}", "combo", 0, None, L.combine_lines(lf, ln))
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


def select_cv(rows: list[dict], entries: dict[str, list[dict]], folds: int) -> tuple[list[dict], dict]:
    """Trả về (các dòng đã chọn, kèm cột `selection`, và mô tả lựa chọn)."""
    prim = {(ds, str(e["track_id"])): _primary(e) for ds, es in entries.items() for e in es}
    fold_of = {}
    for ds, es in entries.items():
        labelled = [e for e in es if _primary(e)]
        for i, e in enumerate(labelled):
            fold_of[(ds, str(e["track_id"]))] = i % folds
    groups: dict[tuple, dict[int, list[dict]]] = {}
    for r in rows:
        groups.setdefault((r["model"], r["family"], r["quant"]), {}).setdefault(r["cfg"], []).append(r)

    def objective(cfg_rows: list[dict], keep) -> float:
        v = [r[prim[(r["dataset"], r["track_id"])]] for r in cfg_rows
             if keep((r["dataset"], r["track_id"])) and prim[(r["dataset"], r["track_id"])]]
        v = [x for x in v if np.isfinite(x)]
        return float(np.mean(v)) if v else -math.inf

    chosen: list[dict] = []
    choices: dict = {}
    for key, by_cfg in groups.items():
        cfgs = sorted(by_cfg)
        full = max(cfgs, key=lambda c: (objective(by_cfg[c], lambda t: t in fold_of), -c))
        desc = {"full": full, "folds": []}
        for r in by_cfg[cfgs[0]]:
            chosen.append({**r, "selection": "default"})
        for r in by_cfg[full]:
            chosen.append({**r, "selection": "full"})
        for f in range(folds):
            train = lambda t, f=f: t in fold_of and fold_of[t] != f   # noqa: E731
            best = max(cfgs, key=lambda c: (objective(by_cfg[c], train), -c))
            desc["folds"].append(best)
            for r in by_cfg[best]:
                t = (r["dataset"], r["track_id"])
                if fold_of.get(t) == f:
                    chosen.append({**r, "selection": "cv"})
        # bộ không đáp án: dùng cấu hình chọn trên cả bộ có đáp án
        for r in by_cfg[full]:
            if (r["dataset"], r["track_id"]) not in fold_of:
                chosen.append({**r, "selection": "cv"})
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
                 seed: int = 0) -> list[dict]:
    """So cặp hai cấu hình (model, họ, căn phách) tốt nhất theo chỉ số mục tiêu (điểm CV) trên mỗi bộ."""
    from eval.ablation_kajiura import bootstrap_diff

    rng = np.random.default_rng(seed)
    out = []
    for ds in sorted({s["dataset"] for s in summary}):
        rows = [s for s in summary if s["selection"] == "cv" and s["dataset"] == ds]
        metric = primary.get(ds)
        if metric is None or len(rows) < 2:
            continue
        top = sorted(rows, key=lambda s: -s[metric] if np.isfinite(s[metric]) else math.inf)[:2]
        per = []
        for s in top:
            per.append({r["track_id"]: r[metric] for r in chosen if r["selection"] == "cv" and r["dataset"] == ds
                        and (r["model"], r["family"], r["quant"]) == (s["model"], s["family"], s["quant"])})
        common = sorted(set(per[0]) & set(per[1]))
        if len(common) < 2:
            continue
        d = bootstrap_diff(np.array([per[0][t] for t in common]), np.array([per[1][t] for t in common]), n_boot, rng)
        out.append({"dataset": ds, "metric": metric, "a": _name(top[0]), "b": _name(top[1]), "n": len(common), **d})
    return out


def _name(s: dict) -> str:
    return f"{s['model']}/{s['family']}" + ("+beats" if s["quant"] else "")


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


def print_tables(summary: list[dict], tests: list[dict], primary: dict[str, str | None]) -> None:
    for ds in sorted({s["dataset"] for s in summary}):
        rows = [s for s in summary if s["dataset"] == ds and s["selection"] == "cv"]
        has_f0 = any(np.isfinite(s["oa"]) for s in rows)
        has_notes = any(np.isfinite(s["note_f1"]) for s in rows)
        print(f"\n### {ds} ({rows[0]['n_tracks'] if rows else 0} bài, {rows[0]['minutes'] if rows else 0:.1f} phút) "
              f"— điểm CV, trung bình theo bài (theo thời lượng trong ngoặc)\n")
        cols = (["oa", "rpa", "rca", "vr", "vfa"] if has_f0 else []) + (["note_f1", "note_off_f1"] if has_notes else [])
        cols += ["coverage", "in_chord", "octave_jumps_pm", "median_dur"]
        print("| model / chiến lược | " + " | ".join(cols) + " |")
        print("|---" * (len(cols) + 1) + "|")
        key = primary.get(ds) or "coverage"
        for s in sorted(rows, key=lambda s: -(s[key] if np.isfinite(s[key]) else -1)):
            cells = []
            for c in cols:
                pct = c not in ("octave_jumps_pm", "median_dur")
                cells.append(f"{_fmt(s[c], pct)} ({_fmt(s[c + '_tw'], pct)})")
            print(f"| {_name(s)} | " + " | ".join(cells) + " |")
    if tests:
        print("\n### So cặp hai cấu hình tốt nhất (a − b, điểm %, CI bootstrap 95%, thắng/thua, p Wilcoxon)\n")
        for t in tests:
            print(f"- {t['dataset']} {t['metric']}: {t['a']} − {t['b']} = {100 * t['mean']:+.2f} "
                  f"[{100 * t['lo']:+.2f}, {100 * t['hi']:+.2f}], {t['wins']}/{t['losses']}, "
                  f"p={t['p_wilcoxon']:.3f} (n={t['n']})")


def main(argv: list[str] | None = None) -> int:
    from eval.evaluate import _utf8_console

    _utf8_console()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", help="danh sách bộ, phân cách bằng dấu phẩy (mặc định: mọi manifest)")
    ap.add_argument("--models", help="danh sách model trong cache (mặc định: tất cả)")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--no-beats", action="store_true", help="không chấm bản căn phách")
    ap.add_argument("--out-dir", type=Path, default=paths.EVAL_DIR / "results" / "melody")
    args = ap.parse_args(argv)

    entries = read_manifests(args.datasets.split(",") if args.datasets else None)
    if not entries:
        print("Không tìm thấy manifest nào trong", paths.EVAL_DIR / "melody", file=sys.stderr)
        return 1
    models = discover_models(list(entries), args.models.split(",") if args.models else None)
    if not models:
        print("Không có cache model nào trong", paths.CACHE_DIR / "melody", file=sys.stderr)
        return 1
    print(f"Bộ: {', '.join(f'{d} ({len(e)} bài)' for d, e in entries.items())}; model: {', '.join(models)}",
          flush=True)
    beats_dir = None if args.no_beats else paths.CACHE_DIR / "melody" / "_beats"
    tasks = [(e, models, beats_dir) for es in entries.values() for e in es]
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

    chosen, choices = select_cv(rows, entries, args.folds)
    summary = summarize(chosen)
    primary = dataset_primary(entries)
    tests = compare_top2(chosen, summary, primary, args.boot)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "per_track.csv", rows)
    write_csv(args.out_dir / "summary.csv", summary)
    (args.out_dir / "cv_choices.json").write_text(json.dumps({"choices": choices, "tests": tests}, indent=1,
                                                             ensure_ascii=False), encoding="utf-8")
    print_tables(summary, tests, primary)
    print(f"\nĐã ghi {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
