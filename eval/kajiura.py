"""Bộ đánh giá nhạc Yuki Kajiura: bản hợp âm ChordWiki (không có mốc thời gian) + audio YouTube.

    python -m eval.kajiura parse [NAME ...]   # đọc bản hợp âm, in thống kê và ký hiệu không đọc được
    python -m eval.kajiura probe              # xem metadata (kênh, độ dài) của các videoId trong CSV
    python -m eval.kajiura search             # tìm ứng viên YouTube, ghi EVAL_DIR/kajiura/youtube_candidates.csv
    python -m eval.kajiura download           # tải audio theo kajiura_youtube.csv
    python -m eval.kajiura align              # căn bản hợp âm vào audio (DTW chroma), ghi ref/ + alignment.json

Dữ liệu (không commit) nằm ở `EVAL_DIR/kajiura/`:
  sheets/<name>.cho   bản hợp âm dạng ChordPro **chỉ gồm hợp âm** (không lưu lời bài hát)
  audio/<name>.wav    audio tải bằng yt-dlp
  ref/<name>.lab      bản hợp âm đã căn thời gian (mọi bài; `alignment.json` ghi bài nào căn tốt)
  alignment.json      độ dịch giọng, điểm căn của từng bài

ChordWiki chặn request tự động bằng Cloudflare, nên bản hợp âm được đọc bằng trình duyệt thật (người
dùng tự qua bước xác minh), mỗi trang cách nhau vài giây. Đoạn JS `EXTRACT_JS` chạy trên trang bài chỉ
lấy các `span.chord`, tên đoạn và dòng đổi giọng. Xem eval/README.md, mục "Bộ Kajiura".

Cách chấm khi đáp án không có thời gian (xem `score_sequence`, `align_sheet`):
- Theo chuỗi: rút gọn hợp âm (majmin hoặc sevenths), gộp hợp âm trùng liên tiếp, bỏ N, rồi so chuỗi
  đáp án với chuỗi dự đoán bằng LCS (Needleman–Wunsch không phạt khoảng trống, không cho thay thế).
  precision = LCS / len(dự đoán), recall = LCS / len(đáp án), F1. Kèm F1 trên các cặp chuyển hợp âm.
- Độ dịch giọng chọn bằng audio (DTW chroma, không dùng model), nên không thiên vị phương án nào.
  Vẫn ghi thêm độ dịch tốt nhất theo chuỗi cho từng phương án để đối chiếu.
- Theo thời gian: căn chuỗi đáp án vào chroma của audio bằng DTW, chấm bằng `evaluate_pair`.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from chordtrack.paths import CACHE_DIR, EVAL_DIR
from chordtrack.types import ChordSegment

KAJIURA_DIR = EVAL_DIR / "kajiura"
SHEETS_DIR = KAJIURA_DIR / "sheets"
AUDIO_DIR = KAJIURA_DIR / "audio"
REF_DIR = KAJIURA_DIR / "ref"
REPORT_PATH = KAJIURA_DIR / "alignment.json"
MAPPING_CSV = Path(__file__).with_name("kajiura_youtube.csv")
CSV_FIELDS = ("name", "title", "artist", "chordwiki_url", "video_id", "yt_title", "note")
SEARCH_OUT = KAJIURA_DIR / "youtube_candidates.csv"


# ---------------------------------------------------------------- đọc ký hiệu hợp âm kiểu Nhật -> Harte

NOTE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
SEMITONE_DEGREE = {1: "b2", 2: "2", 3: "b3", 4: "3", 5: "4", 6: "b5", 7: "5", 8: "#5", 9: "6", 10: "b7", 11: "7"}
PC_NAME = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# Phần "chất lượng" chính, thử theo thứ tự (chuỗi dài trước): (ký hiệu, shorthand Harte, bậc thêm).
# Ký hiệu đã qua `normalize_symbol`: △ -> M, φ -> ø, ° -> o.
_QUALITIES: list[tuple[str, str, tuple[str, ...]]] = [
    ("m7-5", "hdim7", ()), ("m7b5", "hdim7", ()), ("ø7", "hdim7", ()), ("ø", "hdim7", ()),
    ("mM7", "minmaj7", ()), ("mmaj7", "minmaj7", ()),
    ("M7", "maj7", ()), ("maj7", "maj7", ()), ("M9", "maj9", ()), ("maj9", "maj9", ()),
    ("maj", "maj", ()), ("M", "maj", ()),
    ("m69", "min6", ("9",)), ("m6", "min6", ()), ("m7", "min7", ()), ("m9", "min9", ()), ("m11", "min11", ()),
    ("madd9", "min", ("9",)), ("madd2", "min", ("2",)), ("madd11", "min", ("11",)), ("madd4", "min", ("4",)),
    ("min7", "min7", ()), ("min", "min", ()), ("msus4", "sus4", ()), ("m", "min", ()),
    ("dim7", "dim7", ()), ("dim", "dim", ()), ("o7", "dim7", ()), ("o", "dim", ()),
    ("aug7", "aug", ("b7",)), ("aug", "aug", ()), ("7+5", "aug", ("b7",)), ("7#5", "aug", ("b7",)),
    ("+7", "aug", ("b7",)), ("+5", "aug", ()), ("+", "aug", ()), ("#5", "aug", ()),
    ("7sus4", "sus4", ("b7",)), ("7sus2", "sus2", ("b7",)), ("9sus4", "sus4", ("b7", "9")),
    ("sus4", "sus4", ()), ("sus2", "sus2", ()), ("sus9", "sus2", ()), ("sus", "sus4", ()),
    ("add9", "maj", ("9",)), ("add2", "maj", ("2",)), ("add11", "maj", ("11",)), ("add4", "maj", ("4",)),
    ("69", "maj6", ("9",)), ("6", "maj6", ()), ("7", "7", ()), ("9", "9", ()), ("11", "11", ()), ("13", "13", ()),
    ("5", "5", ()), ("", "maj", ()),
]

# Một bậc thêm: (add)?(dấu)?(số). Ví dụ 9, b9, #11, -5, +5, add9.
_TENSION_RE = re.compile(r"(add)?([#b+\-]?)(2|4|5|6|9|11|13)")


class ChordSyntaxError(ValueError):
    pass


def normalize_symbol(s: str) -> str:
    """Chuẩn hóa ký tự toàn chiều rộng và các biến thể ký hiệu kiểu Nhật."""
    s = unicodedata.normalize("NFKC", s).strip()
    for a, b in {"♯": "#", "♭": "b", "△": "M", "Δ": "M", "∆": "M", "−": "-", "–": "-", "φ": "ø",
                 "Φ": "ø", "°": "o", "º": "o", " ": ""}.items():
        s = s.replace(a, b)
    # "ConE", "Am7onG" -> "/E", "/G". Chỉ khi "on" đứng ngay trước một nốt.
    s = re.sub(r"on(?=[A-G])", "/", s)
    # "A7(sus4)" -> "A7sus4"
    s = re.sub(r"\((sus[24]?)\)", r"\1", s)
    # "C6/9" là hợp âm 6 thêm 9, không phải bass
    s = s.replace("6/9", "69")
    return s


def pitch_class(note: str) -> int:
    pc = NOTE_PC[note[0]]
    for ch in note[1:]:
        pc += 1 if ch == "#" else -1
    return pc % 12


def parse_chord(symbol: str) -> str:
    """Một ký hiệu hợp âm (C, Am7, F#m7-5, ConE, C/E, Cadd9, Bbsus4, CM7(9), N.C.) -> nhãn Harte.

    Hợp âm có bass trùng nốt gốc thì bỏ phần bass. Ném ChordSyntaxError nếu không đọc được.
    Ký hiệu chỉ có bass (`/E`) không xử lý ở đây, xem `parse_chordpro`.
    """
    s = normalize_symbol(symbol).strip("()")
    if not s:
        raise ChordSyntaxError(symbol)
    if re.fullmatch(r"N\.?C\.?", s, flags=re.I):
        return "N"
    m = re.match(r"^([A-G][#b]?)(.*)$", s)
    if not m:
        raise ChordSyntaxError(symbol)
    root, rest = m.group(1), m.group(2)
    bass = None
    if "/" in rest:
        rest, bass_s = rest.split("/", 1)
        if not re.fullmatch(r"[A-G][#b]?", bass_s):
            raise ChordSyntaxError(symbol)
        bass = bass_s

    for key, short, adds in _QUALITIES:
        if rest.startswith(key):
            tail = rest[len(key):]
            break
    else:  # pragma: no cover - "" luôn khớp
        raise ChordSyntaxError(symbol)
    extra = list(adds)
    for part in filter(None, re.split(r"[(),]", tail)):
        pos = 0
        while pos < len(part):
            tm = _TENSION_RE.match(part, pos)
            if not tm:
                raise ChordSyntaxError(symbol)
            acc = {"+": "#", "-": "b"}.get(tm.group(2), tm.group(2))
            extra.append(acc + tm.group(3))
            pos = tm.end()
    short, extra = _combine(short, extra)
    label = f"{root}:{short}"
    if extra:
        label += "(" + ",".join(extra) + ")"
    if bass is not None:
        interval = (pitch_class(bass) - pitch_class(root)) % 12
        if interval:
            label += "/" + SEMITONE_DEGREE[interval]
    return label


def _combine(short: str, extra: list[str]) -> tuple[str, list[str]]:
    """Gộp bậc thêm vào shorthand khi có tên chuẩn (7 + 9 -> 9, M7 + 9 -> maj9, m7 + b5 -> hdim7...)."""
    extra = list(dict.fromkeys(extra))
    if short == "aug" and "#5" in extra:
        extra.remove("#5")
    if short in ("dim", "hdim7") and "b5" in extra:
        extra.remove("b5")
    if short in ("7", "maj7", "min7") and "9" in extra:
        extra.remove("9")
        short = {"7": "9", "maj7": "maj9", "min7": "min9"}[short]
    if short == "7" and "#5" in extra:
        extra.remove("#5")
        short, extra = "aug", ["b7", *extra]
    if short == "min7" and "b5" in extra:
        extra.remove("b5")
        short = "hdim7"
    if short in ("maj", "7", "9") and "b5" in extra:   # C(b5), C7-5: bỏ quãng 5 đúng
        extra.remove("b5")
        extra = ["*5", "b5", *extra]
    return short, extra


def transpose(label: str, k: int) -> str:
    """Dịch nhãn Harte lên k nửa cung (bậc bass tương đối nên giữ nguyên)."""
    m = re.match(r"^([A-G][#b]*)(.*)$", label)
    if not m or k % 12 == 0:
        return label
    return PC_NAME[(pitch_class(m.group(1)) + k) % 12] + m.group(2)


# ---------------------------------------------------------------- đọc bản hợp âm ChordPro (chỉ hợp âm)

_CHORD_IN_BRACKETS = re.compile(r"\[([^\]]*)\]")
_REPEAT_RE = re.compile(r"[x×✕](\d)\s*$", re.I)


@dataclass
class Sheet:
    name: str
    chords: list[str] = field(default_factory=list)                  # nhãn Harte (gồm "N")
    sections: list[tuple[int, str]] = field(default_factory=list)    # (vị trí trong chords, tên đoạn)
    keys: list[tuple[int, str]] = field(default_factory=list)        # (vị trí, giọng ghi trên trang)
    meta: dict = field(default_factory=dict)                         # yt, sub, note...
    unparsed: Counter = field(default_factory=Counter)
    n_tokens: int = 0

    @property
    def parse_rate(self) -> float:
        bad = sum(self.unparsed.values())
        return 1.0 - bad / self.n_tokens if self.n_tokens else 0.0


def split_tokens(text: str) -> list[str]:
    """Nội dung một `[...]`: '| Am | F G |' -> ['Am', 'F', 'G']; '( E /F# )' -> ['E', '/F#'].

    Bỏ vạch nhịp, gạch nối, mũi tên, ghi chú trong ngoặc không phải hợp âm ('(break)', '(Dr.fill)').
    """
    text = unicodedata.normalize("NFKC", text).replace("|", " ")
    out = []
    for tok in text.split():
        tok = tok.strip(",[]")
        if tok.startswith("("):
            tok = tok[1:]
        if tok.endswith(")") and tok.count(")") > tok.count("("):
            tok = tok[:-1]
        if not tok:
            continue
        # Chỉ nhận token trông như hợp âm: nốt (kèm dấu) rồi hết, hoặc số/ký hiệu/từ khóa hợp âm.
        # Bỏ vạch nhịp, mũi tên, '~', '>' và ghi chú chữ như '(Dr.fill)', '(break)'.
        if not re.match(r"^(/?[A-G][#b♯♭]?($|[^a-z]|m|maj|min|dim|aug|sus|add|on)|[Nn]\.?[Cc])", tok):
            continue
        # "C-G" (đổi nhanh trong một ô): tách theo '-' chỉ khi phía sau là nốt.
        out.extend(p for p in re.split(r"-(?=[A-G])", tok) if p)
    return out


def parse_chordpro(text: str, name: str = "") -> Sheet:
    """ChordPro kiểu ChordWiki (chỉ hợp âm) -> Sheet.

    - `{c:...}` là tên đoạn; `{key:...}` là đổi giọng (chỉ ghi lại; hợp âm vẫn là tuyệt đối);
      các chỉ thị khác (`{yt:...}`, `{sub:...}`, `{note:...}`, `{title:...}`) vào `meta`.
    - Hợp âm trong `[...]`; mỗi cặp ngoặc có thể chứa nhiều hợp âm cách nhau bằng dấu cách.
    - `onC` = `/C`; `N.C.` -> "N"; `(C)` vẫn tính là hợp âm.
    - `/E` đứng riêng: hợp âm trước đó với bass E (`C /B` = C, C/B).
    - Dòng kết thúc bằng `x2`, `×3` (ngoài ngoặc) thì lặp lại hợp âm của dòng đó.
    """
    sheet = Sheet(name=name)
    prev: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        d = re.match(r"^\{\s*([a-zA-Z_]+)\s*:?\s*(.*?)\s*\}$", line)
        if d:
            key, val = d.group(1).lower(), d.group(2)
            if key in ("c", "comment", "ci"):
                sheet.sections.append((len(sheet.chords), val))
            elif key == "key":
                sheet.keys.append((len(sheet.chords), val))
            else:
                sheet.meta[key] = val
            continue
        line_chords: list[str] = []
        for body in _CHORD_IN_BRACKETS.findall(line):
            for tok in split_tokens(body):
                sheet.n_tokens += 1
                try:
                    if tok.startswith("/"):
                        if prev is None or prev == "N":
                            raise ChordSyntaxError(tok)
                        base = re.sub(r"/[^/]*$", "", prev) if "/" in prev else prev
                        root = re.match(r"^([A-G][#b]*)", base).group(1)
                        bass = normalize_symbol(tok[1:])
                        if not re.fullmatch(r"[A-G][#b]?", bass):
                            raise ChordSyntaxError(tok)
                        iv = (pitch_class(bass) - pitch_class(root)) % 12
                        lab = base + (f"/{SEMITONE_DEGREE[iv]}" if iv else "")
                    else:
                        lab = parse_chord(tok)
                except ChordSyntaxError:
                    sheet.unparsed[tok] += 1
                    continue
                line_chords.append(lab)
                prev = lab
        rep = _REPEAT_RE.search(_CHORD_IN_BRACKETS.sub("", line))
        times = int(rep.group(1)) if rep else 1
        sheet.chords.extend(line_chords * max(1, min(times, 8)))
    return sheet


def load_sheet(name: str) -> Sheet:
    return parse_chordpro((SHEETS_DIR / f"{name}.cho").read_text(encoding="utf-8"), name)


# ---------------------------------------------------------------- rút gọn nhãn

def _encode(label: str):
    import mir_eval

    try:
        return mir_eval.chord.encode(label)
    except mir_eval.chord.InvalidChordException:
        return None


def reduce_label(label: str, level: str) -> str | None:
    """Rút gọn nhãn Harte về một nhãn so sánh được. None = không có hợp âm (N, X, nhãn hỏng).

    - `majmin`: nốt gốc + maj/min theo quãng ba (dim, hdim7 -> min; aug, 7 -> maj); không có quãng
      ba (sus, power) -> `sus`.
    - `sevenths`: thêm loại quãng bảy và quãng năm lệch: maj, min, 7, maj7, min7, dim, hdim7, dim7,
      aug, minmaj7, sus, sus7.
    Bỏ qua nốt bass (không chấm hợp âm đảo theo chuỗi).
    """
    enc = _encode(label)
    if enc is None:
        return None
    root, bitmap, _bass = enc
    if root < 0 or bitmap.sum() == 0:
        return None
    name = PC_NAME[root]
    third = "min" if bitmap[3] and not bitmap[4] else "maj" if bitmap[4] else "sus"
    if level == "majmin":
        return f"{name}:{third}"
    if level != "sevenths":
        raise ValueError(level)
    fifth = "dim" if bitmap[6] and not bitmap[7] else "aug" if bitmap[8] and not bitmap[7] and third == "maj" else ""
    sev = "b7" if bitmap[10] else "7" if bitmap[11] else "bb7" if bitmap[9] and fifth == "dim" else ""
    table = {
        ("maj", "", ""): "maj", ("maj", "", "b7"): "7", ("maj", "", "7"): "maj7",
        ("min", "", ""): "min", ("min", "", "b7"): "min7", ("min", "", "7"): "minmaj7",
        ("min", "dim", ""): "dim", ("min", "dim", "b7"): "hdim7", ("min", "dim", "bb7"): "dim7",
        ("maj", "aug", ""): "aug", ("maj", "aug", "b7"): "aug7", ("maj", "dim", ""): "maj",
        ("maj", "dim", "b7"): "7", ("sus", "", ""): "sus", ("sus", "", "b7"): "sus7", ("sus", "", "7"): "susmaj7",
    }
    q = table.get((third, fifth, sev), f"{third}{fifth}{sev}" or third)
    return f"{name}:{q}"


def chord_sequence(labels: list[str], level: str) -> list[str]:
    """Rút gọn, bỏ N, gộp nhãn trùng liên tiếp."""
    out: list[str] = []
    for lab in labels:
        r = reduce_label(lab, level)
        if r is not None and (not out or out[-1] != r):
            out.append(r)
    return out


def transpose_sequence(seq: list[str], k: int) -> list[str]:
    return [transpose(s, k) for s in seq]


# ---------------------------------------------------------------- chấm theo chuỗi

def lcs_length(a: list[str], b: list[str]) -> int:
    """Độ dài dãy con chung dài nhất (thuật toán bit-parallel của Hyyrö, O(len(a) * len(b) / 64))."""
    if not a or not b:
        return 0
    masks: dict[str, int] = {}
    for i, x in enumerate(a):
        masks[x] = masks.get(x, 0) | (1 << i)
    full = (1 << len(a)) - 1
    v = full
    for y in b:
        u = v & masks.get(y, 0)
        v = ((v + u) | (v - u)) & full
    return len(a) - bin(v).count("1")


def _f1(match: float, n_ref: int, n_est: int) -> dict:
    p = match / n_est if n_est else 0.0
    r = match / n_ref if n_ref else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def bigram_f1(ref: list[str], est: list[str]) -> float:
    rb = Counter(zip(ref, ref[1:]))
    eb = Counter(zip(est, est[1:]))
    match = sum((rb & eb).values())
    return _f1(match, sum(rb.values()), sum(eb.values()))["f1"]


def score_sequence(ref: list[str], est: list[str], shifts: range | list[int] = (0,)) -> dict:
    """So hai chuỗi hợp âm đã rút gọn và gộp. Thử các độ dịch giọng của đáp án trong `shifts`, chọn độ
    dịch có F1 (LCS) cao nhất. Trả về {"f1", "precision", "recall", "bigram_f1", "shift", "n_ref", "n_est"}.
    """
    best = None
    for k in shifts:
        r = transpose_sequence(ref, k)
        res = _f1(lcs_length(r, est), len(r), len(est))
        res.update(bigram_f1=bigram_f1(r, est), shift=int(k) % 12, n_ref=len(r), n_est=len(est))
        if best is None or (res["f1"], res["bigram_f1"]) > (best["f1"], best["bigram_f1"]):
            best = res
    return best


def segments_in_window(segments: list[ChordSegment], start: float, end: float) -> list[str]:
    """Nhãn của các đoạn giao với [start, end] (giữ thứ tự)."""
    return [s.label for s in segments if s.end > start and s.start < end]


# ---------------------------------------------------------------- căn bản hợp âm vào audio (DTW chroma)
#
# Đáp án là chuỗi hợp âm không có thời gian. Chroma mẫu của mỗi hợp âm (12 nốt, 1 ở các nốt của hợp
# âm) được so với chroma CENS của audio (giống `eval.datasets`). DTW chỉ cho phép đứng yên ở hợp âm
# hiện tại hoặc sang hợp âm kế tiếp, mỗi hợp âm ít nhất một khung. Thêm trạng thái "ngoài bản hợp âm"
# ở đầu và cuối với chi phí cố định, để phần intro/outro không có trong bản hợp âm (hoặc bản hợp âm
# chỉ ghi một phần bài) không bị kéo giãn. Không dùng xác suất của model nào nên không thiên vị.

DTW_HOP_FRAMES = 4          # gộp 4 khung CENS (23 ms) -> khoảng 93 ms
OUTSIDE_COST = 0.55         # chi phí mỗi khung nằm ngoài bản hợp âm (1 - cosine)
MIN_ALIGN_QUALITY = 0.60    # cosine trung bình trên đường căn thấp hơn thì không dùng để chấm theo thời gian
MIN_ALIGN_COVERAGE = 0.5    # phần bài được bản hợp âm phủ ít hơn thì không dùng để chấm theo thời gian
# Cosine tuyệt đối không phân biệt được căn đúng hay sai (độ dịch sai vẫn cho 0.60–0.70), nên dùng thêm:
MIN_SHIFT_MARGIN = 0.02     # cosine ở độ dịch tốt nhất phải hơn độ dịch thứ hai ít nhất chừng này
MIN_ORDER_GAIN = 0.02       # và hơn chuỗi cùng hợp âm nhưng xáo trộn thứ tự (thứ tự phải có ý nghĩa)
N_SHUFFLES = 3


def chord_templates(labels: list[str]) -> np.ndarray:
    """(n, 12) chroma mẫu chuẩn hóa L2. Nốt bass được cộng thêm 0.5."""
    out = np.zeros((len(labels), 12), dtype=np.float64)
    for i, lab in enumerate(labels):
        enc = _encode(lab)
        if enc is None or enc[0] < 0:
            continue
        root, bitmap, bass = enc
        v = np.roll(bitmap.astype(float), root)
        if bass >= 0:
            v[(root + bass) % 12] += 0.5
        out[i] = v / (np.linalg.norm(v) + 1e-9)
    return out


def dtw_align(cost: np.ndarray, outside: float = OUTSIDE_COST) -> tuple[float, np.ndarray]:
    """cost: (n, T) chi phí của hợp âm i ở khung t. Trạng thái 0 và n+1 là "ngoài bản hợp âm".

    Trả về (tổng chi phí, state[t] trong 0..n+1).
    """
    n, T = cost.shape
    big = np.full((n + 2, T), outside)
    big[1:n + 1] = cost
    S = n + 2
    D = np.full(S, np.inf)
    D[0] = big[0, 0]
    D[1] = big[1, 0]
    back = np.zeros((S, T), dtype=bool)   # True = đến từ trạng thái trước
    for t in range(1, T):
        stay = D
        move = np.concatenate(([np.inf], D[:-1]))
        take_move = move < stay
        back[:, t] = take_move
        D = np.where(take_move, move, stay) + big[:, t]
    # Kết thúc ở hợp âm cuối hoặc ở trạng thái "ngoài" cuối.
    s = n + 1 if D[n + 1] <= D[n] else n
    total = float(D[s])
    path = np.empty(T, dtype=int)
    for t in range(T - 1, -1, -1):
        path[t] = s
        if t > 0 and back[s, t]:
            s -= 1
    return total, path


@dataclass
class AlignResult:
    name: str
    shift: int = 0                  # dịch đáp án lên `shift` nửa cung để khớp audio
    quality: float = 0.0            # cosine trung bình trên các khung nằm trong bản hợp âm
    runner_up: float = 0.0          # quality của độ dịch tốt thứ hai (để biết độ dịch có chắc không)
    shuffled: float = 0.0           # quality trung bình khi xáo trộn thứ tự hợp âm (cùng độ dịch)
    seq_shift: int = 0              # độ dịch dùng khi chấm theo chuỗi: `shift` nếu chắc chắn, không thì 0
    coverage: float = 0.0           # phần thời lượng bài được bản hợp âm phủ
    start: float = 0.0
    end: float = 0.0
    n_chords: int = 0
    ok: bool = False
    reason: str = ""


def _audio_chroma_frames(audio: Path) -> tuple[np.ndarray, float]:
    import soundfile as sf

    from eval.datasets import ALIGN_HOP, ALIGN_SR, audio_chroma

    y, sr = sf.read(str(audio), dtype="float32", always_2d=True)
    ach = audio_chroma(y.mean(axis=1), sr)
    T = len(ach) // DTW_HOP_FRAMES
    ach = ach[: T * DTW_HOP_FRAMES].reshape(T, DTW_HOP_FRAMES, 12).mean(axis=1)
    ach /= np.linalg.norm(ach, axis=1, keepdims=True) + 1e-9
    return ach, ALIGN_HOP * DTW_HOP_FRAMES / ALIGN_SR


def align_sheet(sheet: Sheet, audio: Path) -> tuple[AlignResult, list[ChordSegment]]:
    """Căn bản hợp âm vào audio. Trả về (kết quả, nhãn có thời gian đã dịch giọng, phủ cả bài)."""
    labels = []
    for lab in sheet.chords:          # gộp trùng liên tiếp, bỏ N (giữ nốt bass để mẫu chroma đúng hơn)
        if lab != "N" and _encode(lab) is not None and (not labels or labels[-1] != lab):
            labels.append(lab)
    res = AlignResult(sheet.name, n_chords=len(labels))
    ach, hop = _audio_chroma_frames(audio)
    duration = len(ach) * hop
    if len(labels) < 4:
        res.reason = "bản hợp âm quá ngắn"
        return res, [ChordSegment(0.0, duration, "N")]
    tmpl = chord_templates(labels)
    trials = []
    for k in range(12):
        sim = np.roll(tmpl, k, axis=1) @ ach.T            # (n, T) cosine
        total, path = dtw_align(1.0 - sim)
        inside = (path >= 1) & (path <= len(labels))
        q = float(sim[path[inside] - 1, np.nonzero(inside)[0]].mean()) if inside.any() else 0.0
        trials.append((q, k, path, inside))
    trials.sort(key=lambda x: -x[0])
    q, k, path, inside = trials[0]
    if not inside.any():   # DTW không khả thi (audio ngắn hơn số hợp âm)
        res.reason = "không căn được (audio quá ngắn so với bản hợp âm)"
        return res, [ChordSegment(0.0, duration, "X")]
    res.shift, res.quality, res.runner_up = k, q, trials[1][0]
    res.seq_shift = k if q - res.runner_up >= MIN_SHIFT_MARGIN else 0
    rng = np.random.default_rng(0)
    shuffled = []
    for _ in range(N_SHUFFLES):
        sim = np.roll(tmpl[rng.permutation(len(labels))], k, axis=1) @ ach.T
        _, sp = dtw_align(1.0 - sim)
        ins = (sp >= 1) & (sp <= len(labels))
        shuffled.append(float(sim[sp[ins] - 1, np.nonzero(ins)[0]].mean()) if ins.any() else 0.0)
    res.shuffled = float(np.mean(shuffled))
    idx = np.nonzero(inside)[0]
    res.start, res.end = float(idx[0] * hop), float((idx[-1] + 1) * hop)
    res.coverage = float(inside.mean())
    segs: list[ChordSegment] = []
    t0 = 0
    for t in range(1, len(path) + 1):
        if t == len(path) or path[t] != path[t0]:
            st = path[t0]
            # Khung nằm ngoài bản hợp âm là "không có đáp án" (X, không tính điểm), không phải im lặng (N).
            lab = transpose(labels[st - 1], k) if 1 <= st <= len(labels) else "X"
            segs.append(ChordSegment(t0 * hop, t * hop, lab))
            t0 = t
    if res.quality < MIN_ALIGN_QUALITY:
        res.reason = f"căn kém (cosine {res.quality:.2f} < {MIN_ALIGN_QUALITY})"
    elif res.quality - res.runner_up < MIN_SHIFT_MARGIN:
        res.reason = f"độ dịch giọng không chắc (hơn độ dịch thứ hai {res.quality - res.runner_up:.3f})"
    elif res.quality - res.shuffled < MIN_ORDER_GAIN:
        res.reason = f"thứ tự hợp âm không giúp căn (hơn chuỗi xáo trộn {res.quality - res.shuffled:.3f})"
    elif res.coverage < MIN_ALIGN_COVERAGE:
        res.reason = f"bản hợp âm chỉ phủ {100 * res.coverage:.0f}% bài"
    else:
        res.ok = True
    return res, segs


def write_lab(path: Path, segments: list[ChordSegment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in segments:
            f.write(f"{s.start:.3f}\t{s.end:.3f}\t{s.label}\n")


def read_report() -> dict[str, dict]:
    if not REPORT_PATH.exists():
        return {}
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- bảng ánh xạ và audio

def read_mapping(path: Path = MAPPING_CSV) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("video_id")]


def _yt_dlp() -> list[str]:
    return [sys.executable, "-m", "yt_dlp"]


def _yt_lines(args: list[str]) -> list[list[str]]:
    res = subprocess.run(_yt_dlp() + ["--no-warnings", "--print",
                                      "%(id)s\t%(duration)s\t%(channel)s\t%(title)s", *args],
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    return [line.split("\t") for line in res.stdout.splitlines() if line.count("\t") == 3]


def probe(ids: list[str]) -> list[list[str]]:
    """Metadata (id, giây, kênh, tiêu đề) của các videoId."""
    return _yt_lines(["--skip-download", *[f"https://www.youtube.com/watch?v={i}" for i in ids]])


def search(query: str, n: int = 6) -> list[list[str]]:
    return _yt_lines(["--flat-playlist", f"ytsearch{n}:{query}"])


def stems_dir(name: str) -> Path:
    return CACHE_DIR / f"kajiura_{name}"


def download_audio(rows: list[dict], cookies_from_browser: str | None = None, pause: float = 5.0,
                   cookies: str | None = None) -> dict[str, str]:
    """Tải lần lượt, nghỉ `pause` giây giữa các bài.

    Khi YouTube đòi xác minh "not a bot": `cookies` là file cookie định dạng Netscape (mặc định đọc biến
    môi trường CHORDTRACK_YT_COOKIES), hoặc `cookies_from_browser` (ví dụ "firefox"). Cookie chỉ được
    yt-dlp đọc, không in ra và không ghi đi đâu. yt-dlp bản mới cần thêm JS runtime (Deno) và gói
    yt-dlp-ejs để giải signature của YouTube.
    """
    import os
    import time

    cookies = cookies or os.environ.get("CHORDTRACK_YT_COOKIES") or None

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    errors: dict[str, str] = {}
    first = True
    for row in rows:
        dst = AUDIO_DIR / f"{row['name']}.wav"
        if dst.exists():
            continue
        if not first:
            time.sleep(pause)
        first = False
        print(f"Tải {row['name']} ({row['video_id']})", flush=True)
        extra = ["--cookies", cookies] if cookies else []
        if cookies_from_browser:
            extra += ["--cookies-from-browser", cookies_from_browser]
        cmd = _yt_dlp() + extra + ["-q", "--no-warnings", "--no-playlist", "-f", "bestaudio", "-x", "--audio-format", "wav",
                           "-o", str(AUDIO_DIR / f"{row['name']}.%(ext)s"),
                           f"https://www.youtube.com/watch?v={row['video_id']}"]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if res.returncode != 0 or not dst.exists():
            lines = res.stderr.strip().splitlines()
            msg = lines[-1] if lines else "không có file wav"
            errors[row["name"]] = msg.replace(cookies, "<cookies>") if cookies else msg
    return errors


# ---------------------------------------------------------------- JS dùng trên trang ChordWiki

# Chạy trên trang bài của ChordWiki (đã mở bằng trình duyệt, sau khi người dùng qua bước xác minh).
# Chỉ lấy hợp âm (`span.chord`), tên đoạn và dòng đổi giọng; không lấy lời bài hát. Bỏ vạch nhịp, gạch
# nối; giữ dấu lặp cuối dòng (x2). Kết quả là nội dung file sheets/<name>.cho.
EXTRACT_JS = r"""
(() => {
  const out = [];
  const yt = [...document.querySelectorAll('a[href*="youtu"],iframe[src*="youtu"]')]
    .map(e => (e.href || e.src).match(/(?:v=|embed\/|youtu\.be\/)([\w-]{11})/)).filter(Boolean).map(m => m[1]);
  out.push('{yt:' + [...new Set(yt)].join(',') + '}');
  const h2 = document.querySelector('h2.subtitle');
  out.push('{sub:' + (h2 ? h2.textContent.trim().slice(0, 40) : '') + '}');
  for (const p of document.querySelectorAll('p.line, p.key')) {
    if (p.classList.contains('comment')) {
      const m = p.textContent.trim().match(/(intro|verse|chorus|bridge|interlude|outro|ending|[ABC]メロ|サビ|間奏|前奏|後奏|大サビ)/i);
      if (m) out.push('{c:' + m[0] + '}');
      continue;
    }
    if (p.classList.contains('key')) { out.push('{key:' + p.textContent.trim().replace(/^Key:\s*/, '').slice(0, 8) + '}'); continue; }
    const c = [...p.querySelectorAll('span.chord')].map(s => s.textContent.trim())
      .filter(t => !/^[\s|\-=>↓↑.:~〇○]*$/.test(t) && !/^\(\d\/\d\)$/.test(t));
    if (!c.length) continue;
    const rep = p.textContent.trim().match(/[x×✕]\s*(\d)\s*$/i);
    out.push('[' + c.join(' ') + ']' + (rep ? ' x' + rep[1] : ''));
  }
  return out.join('\n');
})()
"""


# ---------------------------------------------------------------- CLI

def _cmd_parse(names: list[str]) -> int:
    paths = sorted(SHEETS_DIR.glob("*.cho"))
    if names:
        paths = [p for p in paths if p.stem in names]
    bad: Counter = Counter()
    for p in paths:
        sh = load_sheet(p.stem)
        bad.update(sh.unparsed)
        seq = chord_sequence(sh.chords, "majmin")
        print(f"{p.stem:28s} {len(sh.chords):4d} hợp âm, {len(seq):4d} sau gộp, đọc được {100 * sh.parse_rate:5.1f}%, "
              f"{len(sh.keys)} lần ghi giọng, yt={sh.meta.get('yt', '')}")
    if bad:
        print("Không đọc được:", dict(bad.most_common()))
    return 0


def _cmd_probe() -> int:
    ids = []
    for p in sorted(SHEETS_DIR.glob("*.cho")):
        ids += [i for i in load_sheet(p.stem).meta.get("yt", "").split(",") if i]
    for row in probe(ids):
        print("\t".join(row))
    return 0


def _cmd_search(queries: list[str]) -> int:
    rows = []
    for q in queries:
        for vid, dur, ch, title in search(q):
            print(f"{q[:30]:30s} {vid} {dur:>6s} {ch[:25]:25s} {title}")
            rows.append({"query": q, "video_id": vid, "duration": dur, "channel": ch, "title": title})
    if rows:
        SEARCH_OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(SEARCH_OUT, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return 0


def _cmd_download(cookies_from_browser: str | None, cookies: str | None) -> int:
    errors = download_audio(read_mapping(), cookies_from_browser, cookies=cookies)
    for k, v in errors.items():
        print(f"[lỗi] {k}: {v}")
    return 1 if errors else 0


def _cmd_align(names: list[str]) -> int:
    report = read_report()
    for row in read_mapping():
        name = row["name"]
        if names and name not in names:
            continue
        audio = AUDIO_DIR / f"{name}.wav"
        sheet_path = SHEETS_DIR / f"{name}.cho"
        if not audio.exists() or not sheet_path.exists():
            print(f"{name}: thiếu {'audio' if not audio.exists() else 'bản hợp âm'}")
            continue
        res, segs = align_sheet(load_sheet(name), audio)
        write_lab(REF_DIR / f"{name}.lab", segs)
        report[name] = asdict(res)
        # Ghi sau mỗi bài, để lỗi ở bài sau không làm mất kết quả các bài trước.
        REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{name:28s} dịch {res.shift:2d}  cosine {res.quality:.3f} (thứ hai {res.runner_up:.3f}, "
              f"xáo trộn {res.shuffled:.3f})  "
              f"phủ {100 * res.coverage:3.0f}%  {res.start:6.1f}–{res.end:6.1f}s  "
              f"{'OK' if res.ok else 'loại: ' + res.reason}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    from eval.evaluate import _utf8_console

    _utf8_console()
    ap = argparse.ArgumentParser(description="Bộ đánh giá Kajiura (ChordWiki + YouTube)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("parse").add_argument("names", nargs="*")
    sub.add_parser("probe")
    sub.add_parser("search").add_argument("queries", nargs="+")
    dl = sub.add_parser("download")
    dl.add_argument("--cookies-from-browser", default=None, help="ví dụ firefox, khi YouTube đòi đăng nhập")
    dl.add_argument("--cookies", default=None, help="file cookie Netscape (hoặc biến môi trường CHORDTRACK_YT_COOKIES)")
    sub.add_parser("align").add_argument("names", nargs="*")
    args = ap.parse_args(argv)
    if args.cmd == "parse":
        return _cmd_parse(args.names)
    if args.cmd == "probe":
        return _cmd_probe()
    if args.cmd == "search":
        return _cmd_search(args.queries)
    if args.cmd == "download":
        return _cmd_download(args.cookies_from_browser, args.cookies)
    return _cmd_align(args.names)


if __name__ == "__main__":
    raise SystemExit(main())
