"""Tải và chuẩn bị các bộ đánh giá dò giai điệu thành manifest (định dạng ở `eval/melody/README.md`).

    python -m eval.melody.datasets kajiura      # audio Kajiura có sẵn, không có đáp án giai điệu
    python -m eval.melody.datasets sanity       # "Let It Be", không có đáp án
    python -m eval.melody.datasets mir_st500    # tập test 100 bài (401–500), tải YouTube, kiểm tra lệch thời gian
    python -m eval.melody.datasets orchset      # Zenodo 1289786, 64 đoạn giao hưởng, đáp án f0
    python -m eval.melody.datasets adc2004      # LabROSA, 20 đoạn, đáp án f0
    python -m eval.melody.datasets mirex05      # LabROSA, 13 đoạn công khai, đáp án f0
    python -m eval.melody.datasets summary      # số bài, tổng thời lượng mỗi bộ

Dữ liệu (không commit) ở `EVAL_DIR/melody/<dataset>/`: `manifest.jsonl`, `audio/<track_id>.wav`,
`ref/<track_id>.csv`, `raw/` (file tải về gốc). Ngoài các trường bắt buộc, mỗi dòng manifest có thêm
`sr`, `channels`, `duration` (đọc từ file WAV) và có thể có `offset` (giây, cộng vào thời gian đáp án).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

from chordtrack.paths import EVAL_DIR, SAMPLE_AUDIO

MELODY_DIR = EVAL_DIR / "melody"
MELODY_TYPES = ("vocal", "instrumental", "mixed")
REQUIRED_FIELDS = ("track_id", "dataset", "audio", "ref_notes", "ref_f0", "melody_type")

# Audio tải từ YouTube được chuyển về WAV PCM 16-bit, 44.1 kHz, stereo.
YT_SR = 44100
YT_CHANNELS = 2


# ---------------------------------------------------------------- manifest

def dataset_dir(dataset: str) -> Path:
    return MELODY_DIR / dataset


def manifest_path(dataset: str) -> Path:
    return dataset_dir(dataset) / "manifest.jsonl"


def audio_info(path: Path) -> dict:
    import soundfile as sf

    info = sf.info(str(path))
    return {"sr": int(info.samplerate), "channels": int(info.channels), "duration": round(float(info.duration), 3)}


def make_entry(track_id: str, dataset: str, audio: Path, melody_type: str, ref_notes: Path | None = None,
               ref_f0: Path | None = None, **extra) -> dict:
    if melody_type not in MELODY_TYPES:
        raise ValueError(f"melody_type không hợp lệ: {melody_type}")
    audio = Path(audio).resolve()
    entry = {"track_id": str(track_id), "dataset": dataset, "audio": str(audio),
             "ref_notes": None if ref_notes is None else str(Path(ref_notes).resolve()),
             "ref_f0": None if ref_f0 is None else str(Path(ref_f0).resolve()),
             "melody_type": melody_type}
    entry.update(audio_info(audio))
    entry.update(extra)
    return entry


def write_manifest(entries: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for e in entries:
            missing = [k for k in REQUIRED_FIELDS if k not in e]
            if missing:
                raise ValueError(f"{e.get('track_id')}: thiếu trường {missing}")
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_manifest(dataset_or_path: str | Path) -> list[dict]:
    """Đọc manifest theo tên bộ (`"orchset"`) hoặc đường dẫn file."""
    p = Path(dataset_or_path)
    if not p.suffix:
        p = manifest_path(str(dataset_or_path))
    entries = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                missing = [k for k in REQUIRED_FIELDS if k not in e]
                if missing:
                    raise ValueError(f"{p}: {e.get('track_id')} thiếu trường {missing}")
                entries.append(e)
    return entries


# ---------------------------------------------------------------- đáp án dạng csv

def write_notes_csv(notes: np.ndarray, path: Path) -> None:
    """notes: (N, 3) onset, offset (giây), midi."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for on, off, midi in np.asarray(notes, dtype=float).reshape(-1, 3):
            w.writerow([f"{on:.4f}", f"{off:.4f}", f"{midi:g}"])


def read_notes_csv(path: Path | str, offset: float = 0.0) -> np.ndarray:
    """Đọc csv onset,offset,midi -> mảng (N, 3). `offset` (giây) cộng vào onset/offset."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if row and not row[0].lstrip().startswith("#"):
                rows.append([float(row[0]), float(row[1]), float(row[2])])
    out = np.array(rows, dtype=float).reshape(-1, 3)
    out[:, :2] += offset
    return out


def write_f0_csv(times: np.ndarray, hz: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for t, h in zip(np.asarray(times, float), np.asarray(hz, float)):
            w.writerow([f"{t:.4f}", f"{max(h, 0.0):.3f}"])


def read_f0_csv(path: Path | str, offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Đọc csv time,hz (0 = không có giai điệu). Chấp nhận dấu phẩy, tab hoặc khoảng trắng."""
    t, h = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.replace(",", " ").split()
            if len(parts) >= 2 and not parts[0].startswith("#"):
                t.append(float(parts[0]))
                h.append(float(parts[1]))
    return np.array(t) + offset, np.maximum(np.array(h), 0.0)


def load_reference(entry: dict) -> dict:
    """Đọc đáp án của một dòng manifest, đã cộng `offset`. Trả về {"notes": (N,3)|None, "f0": (t, hz)|None}."""
    off = float(entry.get("offset") or 0.0)
    return {"notes": read_notes_csv(entry["ref_notes"], off) if entry.get("ref_notes") else None,
            "f0": read_f0_csv(entry["ref_f0"], off) if entry.get("ref_f0") else None}


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, md5: str | None = None) -> Path:
    import requests

    if dest.exists() and (md5 is None or _md5(dest) == md5):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    if md5 and _md5(tmp) != md5:
        raise RuntimeError(f"Sai MD5 khi tải {url}")
    tmp.replace(dest)
    return dest


# ---------------------------------------------------------------- Kajiura và sanity (không có đáp án)

def prepare_kajiura() -> list[dict]:
    from eval import kajiura

    report = json.loads(kajiura.REPORT_PATH.read_text(encoding="utf-8")) if kajiura.REPORT_PATH.exists() else {}
    entries = []
    for wav in sorted(kajiura.AUDIO_DIR.glob("*.wav")):
        name = wav.stem
        lab = kajiura.REF_DIR / f"{name}.lab"
        entries.append(make_entry(name, "kajiura", wav, "mixed",
                                  chord_ref=str(lab.resolve()) if lab.exists() else None,
                                  chord_ref_ok=bool(report.get(name, {}).get("ok", False))))
    write_manifest(entries, manifest_path("kajiura"))
    return entries


def prepare_sanity() -> list[dict]:
    entries = [make_entry("let_it_be", "sanity", SAMPLE_AUDIO, "vocal", video_id=SAMPLE_AUDIO.stem)]
    write_manifest(entries, manifest_path("sanity"))
    return entries


# ---------------------------------------------------------------- Orchset

def _safe_extract(zpath: Path, dest: Path) -> None:
    """Giải nén vào thư mục tạm rồi đổi tên, để lần giải nén bị ngắt giữa chừng không để lại `dest` thiếu file."""
    if dest.exists():
        return
    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    with zipfile.ZipFile(zpath) as zf:
        for m in zf.namelist():
            if not (tmp / m).resolve().is_relative_to(tmp.resolve()):
                raise RuntimeError(f"Đường dẫn lạ trong zip: {m}")
        zf.extractall(tmp)
    tmp.rename(dest)


ORCHSET_URL = "https://zenodo.org/api/records/1289786/files/Orchset_dataset_0.zip/content"
ORCHSET_MD5 = "cf6fe52d64624f61ee116c752fb318ca"


def parse_mel(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """File .mel của Orchset: mỗi dòng `time<TAB>hz` (bước 10 ms, 0 = không có giai điệu)."""
    return read_f0_csv(path)


def prepare_orchset() -> list[dict]:
    d = dataset_dir("orchset")
    z = _download(ORCHSET_URL, d / "raw" / "Orchset_dataset_0.zip", ORCHSET_MD5)
    extract = d / "raw" / "extracted"
    _safe_extract(z, extract)
    mels = sorted(extract.rglob("*.mel"))
    if not mels:
        raise RuntimeError(f"Không thấy file .mel trong {extract}")
    wavs = {p.stem: p for p in extract.rglob("*.wav") if "mono" in [x.lower() for x in p.parts]}
    entries = []
    for mel in mels:
        wav = wavs.get(mel.stem)
        if wav is None:
            print(f"[bỏ] {mel.stem}: không có audio mono", file=sys.stderr)
            continue
        t, hz = parse_mel(mel)
        ref = d / "ref" / f"{mel.stem}.csv"
        write_f0_csv(t, hz, ref)
        entries.append(make_entry(mel.stem, "orchset", wav, "instrumental", ref_f0=ref))
    write_manifest(entries, manifest_path("orchset"))
    return entries


# ---------------------------------------------------------------- ADC2004, MIREX05 (bản LabROSA công khai)

LABROSA_URL = "http://labrosa.ee.columbia.edu/projects/melody/"
# Bản MIREX05 công khai là 13 bài "training set" LabROSA (bộ test 25 bài của MIREX không công bố); các bài báo
# melody extraction (MSNet, FTANet, TONet...) chấm trên chính 13 bài này, phần giọng hát là train01–train09.
LABROSA_SETS = {
    "adc2004": ("adc2004_full_set.zip", "d707611df52c53b1762e7ebf093003f6",
                lambda name: "instrumental" if name.startswith(("jazz", "midi")) else "vocal"),
    "mirex05": ("mirex05TrainFiles.zip", "0a25871c801b91bd775dff9dc5b8f60c",
                lambda name: "instrumental" if name.startswith(("train10", "train11", "train12", "train13")) else "vocal"),
}


def prepare_labrosa(dataset: str) -> list[dict]:
    """ADC2004 (20 đoạn) hoặc MIREX05 (13 đoạn): WAV mono 44.1 kHz + `<tên>REF.txt` (time hz)."""
    fname, md5, kind = LABROSA_SETS[dataset]
    d = dataset_dir(dataset)
    z = _download(LABROSA_URL + fname, d / "raw" / fname, md5)
    extract = d / "raw" / "extracted"
    _safe_extract(z, extract)
    entries = []
    for refp in sorted(extract.rglob("*REF.txt")):
        name = refp.name[:-len("REF.txt")]
        wavs = sorted(refp.parent.glob(f"{name}*.wav"))   # train13REF.txt đi với train13MIDI.wav
        if not wavs:
            print(f"[bỏ] {name}: không có audio", file=sys.stderr)
            continue
        t, hz = read_f0_csv(refp)
        ref = d / "ref" / f"{wavs[0].stem}.csv"
        write_f0_csv(t, hz, ref)
        entries.append(make_entry(wavs[0].stem, dataset, wavs[0], kind(name), ref_f0=ref))
    write_manifest(entries, manifest_path(dataset))
    return entries


# ---------------------------------------------------------------- MIR-ST500

MIRST_RAW_URL = "https://raw.githubusercontent.com/york135/singing_transcription_ICASSP2021/master/MIR-ST500_20210206/"
MIRST_TEST_IDS = [str(i) for i in range(401, 501)]   # split chính thức: 1–400 train, 401–500 test
# Thứ tự thử: web_embedded tải được mà không cần cookie (máy này bị YouTube đòi xác minh "not a bot").
YT_CLIENTS = [("youtube:player_client=web_embedded", "bestaudio"), (None, "bestaudio"),
              ("youtube:player_client=mweb", "bestaudio/best")]


def _yt_download(url: str, out_stem: Path, cookies: str | None = None) -> tuple[Path | None, str]:
    """Tải audio gốc (không chuyển định dạng) về `<out_stem>.<ext>`. Trả về (đường dẫn, lỗi cuối)."""
    cookies = cookies or os.environ.get("CHORDTRACK_YT_COOKIES") or None
    err = ""
    for client, fmt in YT_CLIENTS:
        cmd = [sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "--no-playlist", "-f", fmt,
               "-o", f"{out_stem}.%(ext)s", url]
        if client:
            cmd[3:3] = ["--extractor-args", client]
        if cookies:
            cmd[3:3] = ["--cookies", cookies]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        got = [p for p in out_stem.parent.glob(out_stem.name + ".*") if not p.name.endswith((".part", ".ytdl"))]
        if res.returncode == 0 and got:
            return got[0], ""
        err = (res.stderr.strip().splitlines() or ["?"])[-1]
        if cookies:
            err = err.replace(cookies, "<cookies>")
        if any(k in err for k in ("Video unavailable", "video is unavailable", "Private video", "removed")):
            break
    return None, err


def _to_wav(src: Path, dst: Path) -> str:
    """Chuyển sang WAV 44.1 kHz stereo 16-bit. Trả về "" nếu thành công, ngược lại là thông báo lỗi của ffmpeg."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.wav")
    res = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-ar", str(YT_SR), "-ac", str(YT_CHANNELS),
                          "-c:a", "pcm_s16le", str(tmp)], capture_output=True, text=True, encoding="utf-8",
                         errors="replace")
    if res.returncode != 0:
        tmp.unlink(missing_ok=True)
        return "ffmpeg: " + ((res.stderr.strip().splitlines() or ["?"])[-1])
    tmp.replace(dst)
    return ""


_SAL_SR, _SAL_HOP, _SAL_LO, _SAL_BINS = 22050, 256, 36, 61   # salience: MIDI 36–96, khung 11.6 ms


def pitch_salience(audio: Path) -> np.ndarray:
    """Salience cao độ (_SAL_BINS × khung): CQT 1 bin/nửa cung từ MIDI 36, log, trừ trung vị theo thời gian của
    từng bin (bỏ nốt kéo dài của nhạc đệm), cộng điều hòa (bậc 1–4), chuẩn hóa z theo khung."""
    import librosa

    y, _ = librosa.load(str(audio), sr=_SAL_SR, mono=True)
    C = np.abs(librosa.cqt(y, sr=_SAL_SR, hop_length=_SAL_HOP, fmin=librosa.midi_to_hz(_SAL_LO),
                           n_bins=_SAL_BINS + 24, bins_per_octave=12))
    L = np.log1p(100 * C / (C.max() + 1e-9))
    L = L - np.median(L, axis=1, keepdims=True)
    S = np.zeros((_SAL_BINS, L.shape[1]))
    for h, w in ((0, 1.0), (12, 0.5), (19, 0.33), (24, 0.25)):
        S += w * L[h:h + _SAL_BINS]
    return (S - S.mean(axis=0)) / (S.std(axis=0) + 1e-9)


_NO_MATCH = {"lag": 0.0, "score": 0.0, "score0": 0.0, "margin": 0.0, "null": 0.0, "z": 0.0}


def score_offsets(S: np.ndarray, notes: np.ndarray, max_lag: float = 2.0) -> dict:
    """Điểm của độ lệch `lag` = trung bình salience tại cao độ nhãn ở khung `t + lag`, trên mọi khung có nốt.

    Trả về lag tốt nhất (giây, cộng vào thời gian nhãn), điểm tại lag đó (`score`), điểm tại 0 (`score0`),
    `margin` = đỉnh trừ điểm cao nhất ở các lag cách đỉnh > 0.25 s (độ nhọn của đỉnh), `null` = điểm khi xáo
    trộn cao độ (mức ngẫu nhiên, khoảng 0) và `z` = (đỉnh − trung vị) / độ lệch chuẩn của đường điểm.
    Không có nốt nào trong dải MIDI 36–96 thì trả về điểm 0 (bài bị loại).
    """
    n_bins, n_frames = S.shape
    fps = _SAL_SR / _SAL_HOP
    frames, pitches = [], []
    for on, off, midi in notes:
        p = int(round(midi)) - _SAL_LO
        if not 0 <= p < n_bins:
            continue
        a, b = int(round(on * fps)), int(round(off * fps))
        frames.append(np.arange(a, max(b, a + 1)))
        pitches.append(np.full(max(b, a + 1) - a, p))
    if not frames:
        return dict(_NO_MATCH)
    frames = np.concatenate(frames)
    pitches = np.concatenate(pitches)
    lags = np.arange(-int(max_lag * fps), int(max_lag * fps) + 1)
    scores = np.full(len(lags), np.nan)
    for i, lag in enumerate(lags):
        f = frames + lag
        ok = (f >= 0) & (f < n_frames)
        if ok.sum() > 0.5 * len(f):
            scores[i] = S[pitches[ok], f[ok]].mean()
    if np.isnan(scores).all():
        return dict(_NO_MATCH)
    best = int(np.nanargmax(scores))
    valid = scores[~np.isnan(scores)]
    z = (scores[best] - np.median(valid)) / (valid.std() + 1e-9)
    far = (np.abs(lags - lags[best]) > 0.25 * fps) & ~np.isnan(scores)
    margin = scores[best] - scores[far].max() if far.any() else 0.0
    rng = np.random.default_rng(0)
    f = frames + lags[best]
    ok = (f >= 0) & (f < n_frames)
    null = float(np.mean([S[rng.permutation(n_bins)[pitches[ok]], f[ok]].mean() for _ in range(5)]))
    s0 = scores[lags == 0][0]
    return {"lag": round(float(lags[best] / fps), 3), "score": round(float(scores[best]), 3),
            "score0": round(float(0.0 if np.isnan(s0) else s0), 3), "margin": round(float(margin), 3),
            "null": round(null, 3), "z": round(float(z), 2)}


def estimate_offset(audio: Path, notes: np.ndarray, max_lag: float = 2.0) -> dict:
    """Ước lượng độ lệch giữa nhãn nốt và audio bằng pitch salience (CPU, không model)."""
    return score_offsets(pitch_salience(audio), notes, max_lag)


MIRST_MIN_SCORE = 0.8     # bài khớp: 1.4–1.9; nhãn của bài khác: khoảng 0
MIRST_MIN_MARGIN = 0.15   # bài khớp: ≥ 0.45; nhãn của bài khác: < 0.05
MIRST_MIN_SHIFT = 0.05    # lệch nhỏ hơn mức này (vài khung 11.6 ms) coi là 0


def mirst_decide(est: dict, min_score: float = MIRST_MIN_SCORE, min_margin: float = MIRST_MIN_MARGIN) -> tuple[bool, float]:
    """Từ kết quả `estimate_offset`: (giữ bài?, offset dùng). Ngưỡng chọn theo phân bố thật (xem README)."""
    if est["score"] < min_score or est["margin"] < min_margin:
        return False, 0.0
    return True, est["lag"] if abs(est["lag"]) >= MIRST_MIN_SHIFT else 0.0


def _mirst_estimate(wav: Path, notes: np.ndarray) -> dict:
    """Tìm trong ±2 s; nếu không khớp thì tìm rộng ±10 s (bản YouTube có thể thêm đoạn mở đầu)."""
    S = pitch_salience(wav)
    est = score_offsets(S, notes, 2.0)
    if not mirst_decide(est)[0]:
        wide = score_offsets(S, notes, 10.0)
        if mirst_decide(wide)[0]:
            wide["wide"] = True
            return wide
    return est


def prepare_mir_st500(ids: list[str] | None = None, cookies: str | None = None, skip_offset: bool = False) -> list[dict]:
    d = dataset_dir("mir_st500")
    raw = d / "raw"
    labels_p = _download(MIRST_RAW_URL + "MIR-ST500_corrected.json", raw / "MIR-ST500_corrected.json")
    links_p = _download(MIRST_RAW_URL + "MIR-ST500_link.json", raw / "MIR-ST500_link.json")
    labels = json.loads(labels_p.read_text(encoding="utf-8"))
    links = json.loads(links_p.read_text(encoding="utf-8"))
    report_p = d / "offsets.json"
    report = json.loads(report_p.read_text(encoding="utf-8")) if report_p.exists() else {}

    bad = sorted(set(ids or []) - set(MIRST_TEST_IDS))
    if bad:
        raise ValueError(f"Chỉ dùng tập test 401–500, id không hợp lệ: {bad}")
    for tid in ids or MIRST_TEST_IDS:
        r = report.setdefault(tid, {"url": links[tid]})
        wav = d / "audio" / f"{tid}.wav"
        ref = d / "ref" / f"{tid}.csv"
        notes = np.array(labels[tid], dtype=float).reshape(-1, 3)
        write_notes_csv(notes, ref)
        if not wav.exists():
            src, err = _yt_download(links[tid], raw / "yt" / tid, cookies)
            if src is None:
                r.update(status="missing", error=err)
                print(f"[mất] {tid}: {err}", flush=True)
                report_p.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
                continue
            err = _to_wav(src, wav)
            src.unlink(missing_ok=True)
            if err:
                r.update(status="missing", error=err)
                print(f"[lỗi] {tid}: {err}", flush=True)
                report_p.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
                continue
        r.pop("error", None)
        if not skip_offset and "est" not in r:
            r["est"] = _mirst_estimate(wav, notes)
        if "est" in r:
            keep, off = mirst_decide(r["est"])
            r.update(status="ok" if keep else "mismatch", offset=off)
        else:
            r.update(status="ok", offset=0.0)
        print(f"{tid}: {r['status']} {r.get('est')}", flush=True)
        report_p.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")

    return _mirst_write_manifest(report)


def _mirst_write_manifest(report: dict) -> list[dict]:
    d = dataset_dir("mir_st500")
    entries = []
    for tid in MIRST_TEST_IDS:
        r = report.get(tid, {})
        wav = d / "audio" / f"{tid}.wav"
        if r.get("status") == "ok" and wav.exists():
            entries.append(make_entry(tid, "mir_st500", wav, "vocal", ref_notes=d / "ref" / f"{tid}.csv",
                                      offset=r["offset"], url=r["url"]))
    write_manifest(entries, manifest_path("mir_st500"))
    return entries


def redecide_mir_st500(min_score: float, min_margin: float) -> list[dict]:
    """Chọn lại bài giữ/loại theo ngưỡng mới mà không tải hay tính lại salience.

    Chỉ đổi bài đã có `est` và vẫn còn audio; bài mất hoặc chưa ước lượng giữ nguyên trạng thái.
    """
    d = dataset_dir("mir_st500")
    report_p = d / "offsets.json"
    report = json.loads(report_p.read_text(encoding="utf-8"))
    for tid, r in report.items():
        if "est" in r and r.get("status") != "missing" and (d / "audio" / f"{tid}.wav").exists():
            keep, off = mirst_decide(r["est"], min_score, min_margin)
            r.update(status="ok" if keep else "mismatch", offset=off)
    report_p.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return _mirst_write_manifest(report)


# ---------------------------------------------------------------- tổng hợp

def summary() -> list[tuple[str, int, float]]:
    rows = []
    for m in sorted(MELODY_DIR.glob("*/manifest.jsonl")):
        es = read_manifest(m)
        rows.append((m.parent.name, len(es), sum(e.get("duration", 0.0) for e in es)))
    for name, n, dur in rows:
        print(f"{name:12s} {n:4d} bài  {dur / 60:7.1f} phút")
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.melody.datasets")
    ap.add_argument("dataset", choices=["kajiura", "sanity", "mir_st500", "orchset", "adc2004", "mirex05", "summary"])
    ap.add_argument("--ids", nargs="*", help="mir_st500: chỉ xử lý các id này")
    ap.add_argument("--cookies", default=None, help="file cookie Netscape (hoặc biến CHORDTRACK_YT_COOKIES)")
    ap.add_argument("--skip-offset", action="store_true", help="mir_st500: không ước lượng lệch thời gian")
    ap.add_argument("--redecide", nargs=2, type=float, metavar=("MIN_SCORE", "MIN_MARGIN"),
                    help="mir_st500: chọn lại bài theo ngưỡng mới, không tải/tính lại")
    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):   # console Windows mặc định cp1252, không in được tiếng Việt
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    if a.dataset != "mir_st500" and (a.ids or a.redecide or a.skip_offset or a.cookies):
        ap.error("--ids, --cookies, --skip-offset, --redecide chỉ dùng cho mir_st500")
    if a.dataset == "summary":
        summary()
        return 0
    if a.dataset == "mir_st500" and a.redecide:
        redecide_mir_st500(*a.redecide)
    else:
        fn = {"kajiura": prepare_kajiura, "sanity": prepare_sanity, "orchset": prepare_orchset,
              "adc2004": lambda: prepare_labrosa("adc2004"), "mirex05": lambda: prepare_labrosa("mirex05")}.get(a.dataset)
        if fn:
            fn()
        else:
            prepare_mir_st500(a.ids, a.cookies, a.skip_offset)
    es = read_manifest(a.dataset)
    print(f"{a.dataset}: {len(es)} bài, {sum(e['duration'] for e in es) / 60:.1f} phút -> {manifest_path(a.dataset)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
