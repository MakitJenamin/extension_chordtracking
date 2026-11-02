"""Dựng bộ đánh giá Isophonics The Beatles trong `chordtrack.paths.EVAL_DIR`.

Cấu trúc thư mục tạo ra (không commit, `data/` đã gitignore):

    EVAL_DIR/isophonics/
        chordlab/<Album>/<Track>.lab   nhãn gốc Isophonics (khớp bản CD gốc)
        audio/<name>.wav               audio tải từ YouTube bằng yt-dlp
        ref/<name>.lab                 nhãn chuẩn đã căn theo audio, chỉ gồm bài đạt yêu cầu
        alignment.json                 báo cáo căn thời gian: độ lệch, bài bị loại và lý do

Bảng ánh xạ bài sang videoId nằm ở `eval/isophonics_youtube.csv` (có commit).

Các lệnh:
    python -m eval.datasets labels   # tải và giải nén nhãn
    python -m eval.datasets search   # tìm videoId cho CANDIDATES, ghi EVAL_DIR/isophonics/youtube_candidates.csv
    python -m eval.datasets audio    # tải audio theo CSV
    python -m eval.datasets align    # ước lượng độ lệch đầu bài, kiểm tra độ trôi đầu–cuối, cắt vùng chung, ghi ref/
    python -m eval.datasets all      # labels + audio + align
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from chordtrack.paths import EVAL_DIR
from chordtrack.types import ChordSegment, lab_to_segments, segments_to_lab
from eval.evaluate import _utf8_console

ISO_URL = "http://isophonics.net/files/annotations/The%20Beatles%20Annotations.tar.gz"
ISO_DIR = EVAL_DIR / "isophonics"
CHORDLAB_DIR = ISO_DIR / "chordlab"
AUDIO_DIR = ISO_DIR / "audio"
REF_DIR = ISO_DIR / "ref"
REPORT_PATH = ISO_DIR / "alignment.json"
MAPPING_CSV = Path(__file__).with_name("isophonics_youtube.csv")
CSV_FIELDS = ("name", "title", "lab_file", "video_id", "yt_title")
SEARCH_OUT = ISO_DIR / "youtube_candidates.csv"   # lệnh `search` ghi ra đây, không ghi đè bảng đã commit
N_LABELS = 180                                    # số file chordlab trong bộ Isophonics The Beatles

# Ngưỡng căn thời gian (xem README).
MAX_DURATION_DIFF = 15.0    # |độ dài audio - độ dài nhãn| lớn hơn thì loại bài (chắc chắn là bản khác)
MAX_DRIFT = 0.6             # |độ lệch ở cuối bài - độ lệch ở đầu bài| lớn hơn thì loại bài (bản khác tốc độ hoặc cắt ghép khác)
DRIFT_WINDOW = 60.0         # độ dài (giây) vùng đầu bài và cuối bài dùng để đo độ trôi
MAX_OFFSET = 2.0            # |độ lệch đầu bài ước lượng| lớn hơn thì loại bài
MIN_SHIFT = 0.03            # độ lệch nhỏ hơn thì coi là 0, không dịch nhãn
MIN_ALIGN_SCORE = 0.5       # cosine chroma trung bình thấp hơn thì coi là nhãn không khớp audio

# Bài ứng viên: (file nhãn, tên bài để tìm trên YouTube). Lệnh `search` dùng danh sách này.
CANDIDATES: list[tuple[str, str]] = [
    ("01_-_Please_Please_Me/01_-_I_Saw_Her_Standing_There.lab", "I Saw Her Standing There"),
    ("01_-_Please_Please_Me/07_-_Please_Please_Me.lab", "Please Please Me"),
    ("01_-_Please_Please_Me/08_-_Love_Me_Do.lab", "Love Me Do"),
    ("01_-_Please_Please_Me/14_-_Twist_And_Shout.lab", "Twist And Shout"),
    ("02_-_With_the_Beatles/03_-_All_My_Loving.lab", "All My Loving"),
    ("02_-_With_the_Beatles/01_-_It_Won't_Be_Long.lab", "It Won't Be Long"),
    ("03_-_A_Hard_Day's_Night/01_-_A_Hard_Day's_Night.lab", "A Hard Day's Night"),
    ("03_-_A_Hard_Day's_Night/05_-_And_I_Love_Her.lab", "And I Love Her"),
    ("03_-_A_Hard_Day's_Night/07_-_Can't_Buy_Me_Love.lab", "Can't Buy Me Love"),
    ("03_-_A_Hard_Day's_Night/03_-_If_I_Fell.lab", "If I Fell"),
    ("04_-_Beatles_for_Sale/08_-_Eight_Days_a_Week.lab", "Eight Days A Week"),
    ("04_-_Beatles_for_Sale/02_-_I'm_a_Loser.lab", "I'm A Loser"),
    ("05_-_Help!/01_-_Help!.lab", "Help!"),
    ("05_-_Help!/13_-_Yesterday.lab", "Yesterday"),
    ("05_-_Help!/07_-_Ticket_To_Ride.lab", "Ticket To Ride"),
    ("05_-_Help!/03_-_You've_Got_To_Hide_Your_Love_Away.lab", "You've Got To Hide Your Love Away"),
    ("06_-_Rubber_Soul/02_-_Norwegian_Wood_(This_Bird_Has_Flown).lab", "Norwegian Wood (This Bird Has Flown)"),
    ("06_-_Rubber_Soul/07_-_Michelle.lab", "Michelle"),
    ("06_-_Rubber_Soul/11_-_In_My_Life.lab", "In My Life"),
    ("06_-_Rubber_Soul/04_-_Nowhere_Man.lab", "Nowhere Man"),
    ("06_-_Rubber_Soul/01_-_Drive_My_Car.lab", "Drive My Car"),
    ("07_-_Revolver/02_-_Eleanor_Rigby.lab", "Eleanor Rigby"),
    ("07_-_Revolver/05_-_Here,_There_And_Everywhere.lab", "Here, There And Everywhere"),
    ("07_-_Revolver/06_-_Yellow_Submarine.lab", "Yellow Submarine"),
    ("07_-_Revolver/13_-_Got_To_Get_You_Into_My_Life.lab", "Got To Get You Into My Life"),
    ("08_-_Sgt._Pepper's_Lonely_Hearts_Club_Band/02_-_With_A_Little_Help_From_My_Friends.lab", "With A Little Help From My Friends"),
    ("08_-_Sgt._Pepper's_Lonely_Hearts_Club_Band/04_-_Getting_Better.lab", "Getting Better"),
    ("08_-_Sgt._Pepper's_Lonely_Hearts_Club_Band/09_-_When_I'm_Sixty-Four.lab", "When I'm Sixty-Four"),
    ("09_-_Magical_Mystery_Tour/09_-_Penny_Lane.lab", "Penny Lane"),
    ("09_-_Magical_Mystery_Tour/07_-_Hello_Goodbye.lab", "Hello, Goodbye"),
    ("09_-_Magical_Mystery_Tour/02_-_The_Fool_On_The_Hill.lab", "The Fool On The Hill"),
    ("09_-_Magical_Mystery_Tour/11_-_All_You_Need_Is_Love.lab", "All You Need Is Love"),
    ("10CD1_-_The_Beatles/CD1_-_04_-_Ob-La-Di,_Ob-La-Da.lab", "Ob-La-Di, Ob-La-Da"),
    ("10CD1_-_The_Beatles/CD1_-_07_-_While_My_Guitar_Gently_Weeps.lab", "While My Guitar Gently Weeps"),
    ("10CD1_-_The_Beatles/CD1_-_11_-_Black_Bird.lab", "Blackbird"),
    ("10CD1_-_The_Beatles/CD1_-_16_-_I_Will.lab", "I Will"),
    ("11_-_Abbey_Road/01_-_Come_Together.lab", "Come Together"),
    ("11_-_Abbey_Road/02_-_Something.lab", "Something"),
    ("11_-_Abbey_Road/07_-_Here_Comes_The_Sun.lab", "Here Comes The Sun"),
    ("11_-_Abbey_Road/05_-_Octopus's_Garden.lab", "Octopus's Garden"),
    ("12_-_Let_It_Be/06_-_Let_It_Be.lab", "Let It Be"),
    ("12_-_Let_It_Be/03_-_Across_the_Universe.lab", "Across The Universe"),
    ("12_-_Let_It_Be/10_-_The_Long_and_Winding_Road.lab", "The Long And Winding Road"),
    ("12_-_Let_It_Be/12_-_Get_Back.lab", "Get Back"),
    ("12_-_Let_It_Be/01_-_Two_of_Us.lab", "Two Of Us"),
]


def song_name(lab_file: str) -> str:
    """'11_-_Abbey_Road/02_-_Something.lab' -> '11_02_Something'. Dùng làm tên file audio và nhãn."""
    album, track = lab_file.split("/", 1)
    album_no = album.split("_-_", 1)[0]
    track = Path(track).stem.replace("CD1_-_", "").replace("CD2_-_", "")
    track = re.sub(r"[^A-Za-z0-9]+", "_", track.replace("_-_", "_", 1)).strip("_")
    return f"{album_no}_{track}"


def read_mapping(path: Path = MAPPING_CSV) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return [row for row in csv.DictReader(f) if row.get("video_id")]


# ---------------------------------------------------------------- nhãn

def download_labels(force: bool = False) -> int:
    """Tải bộ nhãn Isophonics, chỉ giữ các file chordlab/*.lab. Trả về số file."""
    have = len(list(CHORDLAB_DIR.rglob("*.lab"))) if CHORDLAB_DIR.exists() else 0
    if have >= N_LABELS and not force:
        return have
    if CHORDLAB_DIR.exists():  # lần trước tải dở hoặc force: xóa rồi tải lại
        shutil.rmtree(CHORDLAB_DIR)
    CHORDLAB_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tgz = Path(tmp) / "beatles.tar.gz"
        print(f"Tải {ISO_URL}")
        urllib.request.urlretrieve(ISO_URL, tgz)
        n = 0
        with tarfile.open(tgz) as tar:
            for m in tar.getmembers():
                parts = Path(m.name).parts
                # chordlab/The Beatles/<Album>/<Track>.lab
                if not m.isfile() or not m.name.endswith(".lab") or "chordlab" not in parts:
                    continue
                rel = parts[parts.index("chordlab") + 2:]
                if len(rel) != 2 or any(p in ("..", "") for p in rel):
                    continue
                dst = CHORDLAB_DIR.joinpath(*rel)
                dst.parent.mkdir(parents=True, exist_ok=True)
                src = tar.extractfile(m)
                assert src is not None
                dst.write_bytes(src.read())
                n += 1
    if n < N_LABELS:
        raise RuntimeError(f"Chỉ giải nén được {n}/{N_LABELS} file nhãn từ {ISO_URL}")
    return n


# Nhãn giọng và nhãn đoạn, cùng tarball với chordlab. Cấu trúc giống chordlab: <kind>/<Album>/<Track>.lab
KEYLAB_DIR = ISO_DIR / "keylab"
SEGLAB_DIR = ISO_DIR / "seglab"
N_KEYLABS = 179   # bộ Isophonics thiếu keylab của "Revolution 1" (chỉ có .ttl)
N_SEGLABS = 180


def download_key_seg_labels(force: bool = False) -> dict[str, int]:
    """Tải nhãn giọng (keylab) và nhãn đoạn (seglab) của Isophonics. Trả về số file mỗi loại."""
    dirs = {"keylab": (KEYLAB_DIR, N_KEYLABS), "seglab": (SEGLAB_DIR, N_SEGLABS)}
    have = {k: len(list(d.rglob("*.lab"))) if d.exists() else 0 for k, (d, _) in dirs.items()}
    if not force and all(have[k] >= need for k, (_, need) in dirs.items()):
        return have
    for d, _ in dirs.values():
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    counts = {k: 0 for k in dirs}
    with tempfile.TemporaryDirectory() as tmp:
        tgz = Path(tmp) / "beatles.tar.gz"
        print(f"Tải {ISO_URL}")
        urllib.request.urlretrieve(ISO_URL, tgz)
        with tarfile.open(tgz) as tar:
            for m in tar.getmembers():
                parts = Path(m.name).parts
                kind = next((k for k in dirs if k in parts), None)
                if kind is None or not m.isfile() or not m.name.endswith(".lab"):
                    continue
                rel = parts[parts.index(kind) + 2:]   # <kind>/The Beatles/<Album>/<Track>.lab
                if len(rel) != 2 or any(p in ("..", "") for p in rel):
                    continue
                dst = dirs[kind][0].joinpath(*rel)
                dst.parent.mkdir(parents=True, exist_ok=True)
                src = tar.extractfile(m)
                assert src is not None
                dst.write_bytes(src.read())
                counts[kind] += 1
    for k, (_, need) in dirs.items():
        if counts[k] < need:
            raise RuntimeError(f"Chỉ giải nén được {counts[k]}/{need} file {k} từ {ISO_URL}")
    return counts


# ---------------------------------------------------------------- tìm videoId

def _yt_dlp() -> list[str]:
    return [sys.executable, "-m", "yt_dlp"]


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _core_title(yt_title: str) -> str:
    """'The Beatles - Help! (Remastered 2009)' -> 'Help!'."""
    t = re.sub(r"^\s*the beatles\s*-\s*", "", yt_title, flags=re.I)
    t = re.sub(r"^\s*the beatles\s*-\s*", "", t, flags=re.I)  # có video lặp tên ban nhạc hai lần
    return re.split(r"\s[-–]\s|\(|\[", t)[0].strip()


def search_video(title: str, label_dur: float, n: int = 8) -> dict | None:
    """Tìm bản audio chính thức gần độ dài nhãn nhất. Ưu tiên tiêu đề có 'Remastered 2009'."""
    q = f"ytsearch{n}:The Beatles {title} Remastered 2009"
    res = subprocess.run(
        _yt_dlp() + ["--flat-playlist", "--no-warnings", "--print", "%(id)s\t%(duration)s\t%(channel)s\t%(title)s", q],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if res.returncode != 0:
        raise RuntimeError(f"yt-dlp lỗi khi tìm '{title}': {res.stderr.strip()[-500:]}")
    out = res.stdout
    best, best_key = None, None
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 4:
            continue
        try:
            dur = float(parts[1])
        except ValueError:  # "NA", "None": không biết độ dài
            continue
        vid, channel, yt_title = parts[0], parts[2], parts[3]
        low = yt_title.lower()
        if any(bad in low for bad in ("live", "cover", "karaoke", "instrumental", "lyrics", "mono", "demo", "take ")):
            continue
        if _norm(title) != _norm(_core_title(yt_title)):
            continue
        official = channel.strip().lower() == "the beatles"
        remaster = "remaster" in low and "2009" in low
        remix = re.search(r"(19|20)\d\d mix", low) is not None
        # Khóa xếp hạng: "Remastered 2009" (khớp nhãn nhất), không phải bản remix mới,
        # kênh chính thức, rồi lệch độ dài ít nhất.
        key = (not remaster, remix, not official, abs(dur - label_dur))
        if best_key is None or key < best_key:
            best, best_key = {"video_id": vid, "yt_title": yt_title, "yt_duration": dur, "channel": channel}, key
    return best


def build_mapping(out_csv: Path) -> None:
    rows = []
    for lab_file, title in CANDIDATES:
        lab_path = CHORDLAB_DIR / lab_file
        label_dur = lab_to_segments(lab_path)[-1].end
        hit = search_video(title, label_dur)
        if hit is None:
            print(f"[không tìm thấy] {title}")
            continue
        print(f"{title:40s} nhãn {label_dur:6.1f}s  yt {hit['yt_duration']:6.1f}s  {hit['video_id']}  {hit['yt_title']}")
        rows.append({"name": song_name(lab_file), "title": title, "lab_file": lab_file,
                     "video_id": hit["video_id"], "yt_title": hit["yt_title"]})
    if not rows:
        raise RuntimeError("Không tìm được bài nào, không ghi CSV")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- audio

def download_audio(rows: list[dict]) -> dict[str, str]:
    """Tải audio WAV cho từng bài. Trả về {name: lỗi} cho các bài tải hỏng."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    errors: dict[str, str] = {}
    for row in rows:
        dst = AUDIO_DIR / f"{row['name']}.wav"
        if dst.exists():
            continue
        cmd = _yt_dlp() + ["-q", "--no-warnings", "--no-playlist", "-f", "bestaudio", "-x", "--audio-format", "wav",
                           "-o", str(AUDIO_DIR / f"{row['name']}.%(ext)s"), f"https://www.youtube.com/watch?v={row['video_id']}"]
        print(f"Tải {row['name']} ({row['video_id']})")
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if res.returncode != 0 or not dst.exists():
            lines = res.stderr.strip().splitlines()
            errors[row["name"]] = lines[-1] if lines else "không có file wav"
    return errors


# ---------------------------------------------------------------- căn thời gian
#
# Độ lệch đầu bài ước lượng bằng tương quan chroma: dựng "chroma mẫu" từ nhãn (12 nốt, 1 ở
# các nốt thuộc hợp âm), tính chroma của audio, rồi trượt nhãn trong khoảng ±MAX_OFFSET giây
# để tìm độ trễ có độ tương đồng cosine trung bình cao nhất. Cách này không phụ thuộc vào
# khoảng lặng hay tiếng đếm nhịp ở đầu bài.

ALIGN_SR = 22050
ALIGN_HOP = 512             # khoảng 23 ms mỗi khung


def label_chroma(segments: list[ChordSegment], n_frames: int, hop_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Chroma mẫu (n_frames, 12) từ nhãn và mặt nạ các khung có hợp âm (không phải N/X)."""
    import mir_eval

    chroma = np.zeros((n_frames, 12), dtype=np.float32)
    mask = np.zeros(n_frames, dtype=bool)
    for s in segments:
        try:
            root, bitmap, _bass = mir_eval.chord.encode(s.label)
        except mir_eval.chord.InvalidChordException:
            continue
        if root < 0 or bitmap.sum() == 0:
            continue
        a, b = int(round(s.start / hop_s)), min(n_frames, int(round(s.end / hop_s)))
        if b <= a:
            continue
        chroma[a:b] = np.roll(bitmap, root)
        mask[a:b] = True
    return chroma, mask


def audio_chroma(y: np.ndarray, sr: int) -> np.ndarray:
    """Chroma CENS (n_frames, 12) chuẩn hóa L2 theo từng khung. CENS đã làm mượt theo thời gian
    nên ít bị ảnh hưởng bởi nốt giai điệu và cho đường cong điểm theo độ lệch rõ hơn chroma CQT."""
    import librosa

    if sr != ALIGN_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=ALIGN_SR)
    c = librosa.feature.chroma_cens(y=y, sr=ALIGN_SR, hop_length=ALIGN_HOP).T
    return c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-9)


def best_offset(ach: np.ndarray, lch: np.ndarray, mask: np.ndarray, max_lag: int) -> tuple[int, float, float]:
    """Tìm lag (khung) trong [-max_lag, max_lag] sao cho audio[t + lag] khớp nhãn[t] nhất.

    Trả về (lag, điểm ở lag tốt nhất, điểm ở lag 0). Điểm là cosine trung bình trên các khung có hợp âm.
    """
    lnorm = lch / (np.linalg.norm(lch, axis=1, keepdims=True) + 1e-9)
    # Chỉ dùng các khung hợp lệ ở MỌI lag, để điểm giữa các lag so trên cùng một tập khung.
    t = np.nonzero(mask)[0]
    t = t[(t - max_lag >= 0) & (t + max_lag < len(ach))]
    scores = {}
    for lag in range(-max_lag, max_lag + 1):
        scores[lag] = float(np.mean(np.sum(ach[t + lag] * lnorm[t], axis=1))) if len(t) else -1.0
    lag = max(scores, key=lambda k: (scores[k], -abs(k)))
    return lag, scores[lag], scores[0]


def shift_segments(segments: list[ChordSegment], offset: float, audio_dur: float) -> list[ChordSegment]:
    """Dịch nhãn theo offset (giây, dương = audio vào muộn hơn), cắt về [0, audio_dur], lấp khoảng trống bằng N."""
    out: list[ChordSegment] = []
    for s in segments:
        a, b = max(0.0, s.start + offset), min(audio_dur, s.end + offset)
        if b > a:
            out.append(ChordSegment(a, b, s.label))
    if out and out[0].start > 0:
        if out[0].label == "N":
            out[0] = ChordSegment(0.0, out[0].end, "N")
        else:
            out.insert(0, ChordSegment(0.0, out[0].start, "N"))
    return out


@dataclass
class AlignResult:
    name: str
    status: str            # "ok" hoặc "rejected"
    reason: str
    label_duration: float
    audio_duration: float
    duration_diff: float   # audio - nhãn
    offset: float          # đã cộng vào thời gian nhãn (giây)
    score: float           # cosine chroma trung bình ở độ lệch tốt nhất
    score_zero: float      # cosine chroma trung bình khi không dịch
    head_offset: float = 0.0   # độ lệch tốt nhất chỉ tính trên DRIFT_WINDOW giây đầu
    tail_offset: float = 0.0   # độ lệch tốt nhất chỉ tính trên DRIFT_WINDOW giây cuối của vùng chung
    common_end: float = 0.0    # cuối vùng chung giữa nhãn đã dịch và audio; nhãn chuẩn bị cắt tại đây


def window_mask(mask: np.ndarray, start: float, end: float, hop_s: float) -> np.ndarray:
    """Giữ các khung có hợp âm nằm trong [start, end) giây."""
    out = np.zeros_like(mask)
    a, b = max(0, int(start / hop_s)), min(len(mask), int(end / hop_s))
    out[a:b] = mask[a:b]
    return out


def check_alignment(diff: float, lag: int, max_lag: int, score: float, head: float, tail: float) -> str:
    """Lý do loại bài, hoặc "" nếu đạt. Không xét độ dài (đuôi fade-out khác nhau là chuyện thường),
    mà xét độ lệch có căn được không, chroma có khớp không, và độ lệch ở cuối bài có trùng đầu bài không."""
    if abs(diff) > MAX_DURATION_DIFF:
        return f"độ dài lệch {diff:+.2f}s (> {MAX_DURATION_DIFF}s): chắc chắn là bản khác"
    if abs(lag) >= max_lag:  # đỉnh nằm ở biên: độ lệch thật có thể còn lớn hơn
        return f"độ lệch tốt nhất chạm biên ±{MAX_OFFSET}s: không căn được"
    if score < MIN_ALIGN_SCORE:
        return f"chroma khớp kém (cosine {score:.2f} < {MIN_ALIGN_SCORE})"
    if abs(tail - head) > MAX_DRIFT:
        return (f"độ lệch đầu bài {head:+.2f}s khác cuối bài {tail:+.2f}s (> {MAX_DRIFT}s): "
                "bản khác tốc độ hoặc cắt ghép khác")
    return ""


def align_song(name: str, lab_path: Path, audio_path: Path) -> tuple[AlignResult, list[ChordSegment] | None]:
    import soundfile as sf

    segs = lab_to_segments(lab_path)
    y, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    audio_dur = len(y) / sr
    label_dur = segs[-1].end
    diff = audio_dur - label_dur

    hop_s = ALIGN_HOP / ALIGN_SR
    ach = audio_chroma(y, sr)
    lch, mask = label_chroma(segs, max(len(ach), int(label_dur / hop_s) + 1), hop_s)
    max_lag = int(round(MAX_OFFSET / hop_s))
    lag, score, score0 = best_offset(ach, lch, mask, max_lag)
    offset = lag * hop_s

    # Vùng chung (theo thời gian của nhãn): nhãn [0, label_dur] và audio [-offset, audio_dur - offset].
    common_end_label = min(label_dur, audio_dur - offset)
    head_lag = best_offset(ach, lch, window_mask(mask, 0.0, DRIFT_WINDOW, hop_s), max_lag)[0]
    tail_lag = best_offset(ach, lch, window_mask(mask, common_end_label - DRIFT_WINDOW, common_end_label, hop_s), max_lag)[0]
    head, tail = head_lag * hop_s, tail_lag * hop_s

    reason = check_alignment(diff, lag, max_lag, score, head, tail)
    if abs(offset) < MIN_SHIFT:
        offset = 0.0
    common_end = min(audio_dur, label_dur + offset)
    res = AlignResult(name, "rejected" if reason else "ok", reason, round(label_dur, 3), round(audio_dur, 3),
                      round(diff, 3), round(offset, 3), round(score, 4), round(score0, 4),
                      round(head, 3), round(tail, 3), round(common_end, 3))
    if reason:
        return res, None
    # shift_segments cắt nhãn tại cuối audio; phần audio sau cuối nhãn nằm ngoài nhãn chuẩn nên không được chấm.
    return res, shift_segments(segs, offset, common_end)


def align_all(rows: list[dict]) -> list[AlignResult]:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    for old in REF_DIR.glob("*.lab"):
        old.unlink()
    results: list[AlignResult] = []
    for row in rows:
        audio = AUDIO_DIR / f"{row['name']}.wav"
        lab = CHORDLAB_DIR / row["lab_file"]
        if not audio.exists() or not lab.exists():
            missing = "audio" if not audio.exists() else "nhãn"
            results.append(AlignResult(row["name"], "rejected", f"thiếu {missing}", 0, 0, 0, 0, 0, 0))
            continue
        try:
            res, segs = align_song(row["name"], lab, audio)
        except Exception as ex:  # file wav hỏng, nhãn rỗng...: loại bài, ghi lý do, chạy tiếp
            results.append(AlignResult(row["name"], "rejected", f"lỗi khi căn: {ex!r}", 0, 0, 0, 0, 0, 0))
            continue
        if segs is not None:
            segments_to_lab(segs, REF_DIR / f"{row['name']}.lab")
        results.append(res)
    REPORT_PATH.write_text(json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def print_report(results: list[AlignResult]) -> None:
    ok = [r for r in results if r.status == "ok"]
    bad = [r for r in results if r.status != "ok"]
    print(f"{'bài':42s} {'nhãn':>6} {'audio':>6} {'chênh':>6} {'lệch':>6} {'đầu':>6} {'cuối':>6} {'cos':>5} {'cos0':>5}  trạng thái")
    for r in results:
        print(f"{r.name:42s} {r.label_duration:6.1f} {r.audio_duration:6.1f} {r.duration_diff:+6.2f} {r.offset:+6.2f} "
              f"{r.head_offset:+6.2f} {r.tail_offset:+6.2f} "
              f"{r.score:5.2f} {r.score_zero:5.2f}  {'OK' if r.status == 'ok' else 'LOẠI: ' + r.reason}")
    print(f"\nĐạt: {len(ok)} bài. Bị loại: {len(bad)} bài. Nhãn đã căn ở {REF_DIR}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Dựng bộ đánh giá Isophonics The Beatles")
    ap.add_argument("cmd", choices=["labels", "search", "audio", "align", "all"])
    ap.add_argument("--csv", type=Path, default=MAPPING_CSV)
    args = ap.parse_args(argv)
    _utf8_console()

    if shutil.which("ffmpeg") is None and args.cmd in ("audio", "all"):
        print("Không thấy ffmpeg trong PATH (yt-dlp cần để đổi sang wav).", file=sys.stderr)
        return 1
    if args.cmd in ("labels", "all", "search"):
        print(f"Nhãn: {download_labels()} file trong {CHORDLAB_DIR}")
    if args.cmd == "search":
        out = args.csv if args.csv != MAPPING_CSV else SEARCH_OUT
        build_mapping(out)
        print(f"Đã ghi {out}. Kiểm tra rồi chép vào {MAPPING_CSV} nếu muốn đổi bảng ánh xạ.")
        return 0
    rows = read_mapping(args.csv)
    if args.cmd in ("audio", "all"):
        errors = download_audio(rows)
        for name, err in errors.items():
            print(f"[lỗi tải] {name}: {err}")
    if args.cmd in ("align", "all"):
        print_report(align_all(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
