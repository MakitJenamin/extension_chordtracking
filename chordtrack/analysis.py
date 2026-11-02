"""Bước 5 và 9: nhận diện giọng, bậc La Mã, chia đoạn và tìm vòng hợp âm.

Chỉ dùng CPU. Đầu vào là chuỗi hợp âm (nhãn Harte) và, nếu có, lưới phách.
Chữ ký ba hàm công khai là hợp đồng với pipeline.py, không được đổi.

Tóm tắt thuật toán:
- `estimate_key`: so phân bố cao độ của các hợp âm (gia trọng theo thời lượng) với profile
  Temperley cho 24 giọng, cộng thêm điểm cho hợp âm chủ (thời lượng, hợp âm đầu và cuối bài)
  và cho quãng V–I. Độ tin cậy là xác suất softmax của giọng thắng.
- `analyze_structure`: dựng chuỗi hợp âm theo phách, tính "structure features" kiểu
  Serrà và cộng sự (2012) trên ma trận lặp (recurrence) theo độ trễ, lấy đỉnh novelty làm
  ranh giới, gom các đoạn giống nhau thành nhãn A/B/C, rồi đặt tên intro/verse/chorus...
  bằng heuristic. Vòng hợp âm của mỗi đoạn là n-gram (2–8 hợp âm) phủ được nhiều nhất.
"""
from __future__ import annotations

from collections import Counter

import mir_eval.chord as mchord
import numpy as np

from chordtrack.types import BeatGrid, ChordSegment, KeyEstimate, Section

# ---------------------------------------------------------------- nốt và giọng

_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
# Tên nốt chủ theo quy ước thường dùng của từng giọng (Bb major, không A# major).
MAJOR_TONIC_NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MINOR_TONIC_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]

# Profile Temperley (Kostka–Payne), index 0 là nốt chủ.
_MAJOR_PROFILE = np.array([0.748, 0.060, 0.488, 0.082, 0.670, 0.460, 0.096, 0.715, 0.104, 0.366, 0.057, 0.400])
_MINOR_PROFILE = np.array([0.712, 0.084, 0.474, 0.618, 0.049, 0.460, 0.105, 0.747, 0.404, 0.067, 0.133, 0.330])

# Trọng số các thành phần điểm giọng (chỉnh trên bộ Isophonics, xem PLAN.md mục "Kết quả analysis").
W_TONIC_DUR = 2.0     # tỉ lệ thời lượng hợp âm chủ (đúng nốt gốc và đúng quãng ba của mode)
W_FIRST = 0.3         # hợp âm đầu bài là hợp âm chủ
W_LAST = 0.25         # hợp âm cuối bài là hợp âm chủ
W_CADENCE = 0.5       # tỉ lệ bước chuyển hợp âm là V(trưởng) -> I
KEY_TEMPERATURE = 0.05


def pitch_class(note: str) -> int:
    """'C' -> 0, 'Bb' -> 10, 'F#' -> 6, 'Cbb' -> 10."""
    pc = _PC[note[0].upper()]
    for ch in note[1:]:
        pc += 1 if ch == "#" else -1 if ch == "b" else 0
    return pc % 12


def key_tonic_pc(key: KeyEstimate) -> int:
    return pitch_class(key.tonic)


def key_to_mir_eval(key: KeyEstimate | None) -> str:
    """KeyEstimate -> chuỗi mir_eval.key, ví dụ 'C major'. None -> 'X'."""
    if key is None:
        return "X"
    return f"{key.tonic} {key.mode}"


class _Chord:
    """Nhãn Harte đã phân tích: nốt gốc, các nốt (bitmap tương đối với nốt gốc), bass."""

    __slots__ = ("label", "root", "quality", "bitmap", "bass_degree", "bass_pc")

    def __init__(self, label: str):
        self.label = label
        root, bitmap, bass = mchord.encode(label)
        self.root = int(root)                       # -1 nếu N hoặc X
        self.bitmap = np.asarray(bitmap, dtype=float)
        self.quality = ""
        self.bass_degree = "1"
        self.bass_pc = -1
        if self.root >= 0:
            _, quality, _, bass_degree = mchord.split(label)
            self.quality = quality
            self.bass_degree = bass_degree or "1"
            self.bass_pc = (self.root + int(bass)) % 12

    @property
    def is_chord(self) -> bool:
        return self.root >= 0

    def has(self, interval: int) -> bool:
        return self.is_chord and self.bitmap[interval % 12] > 0

    def triad_class(self) -> str:
        """Kiểu hợp âm ba rút gọn: maj, min, dim, aug, sus, 5 (dùng để so khớp vòng hợp âm)."""
        if not self.is_chord:
            return ""
        if self.has(3) and self.has(6) and not self.has(7):
            return "dim"
        if self.has(4) and self.has(8) and not self.has(7):
            return "aug"
        if self.has(3) and not self.has(4):
            return "min"
        if self.has(4):
            return "maj"
        if self.has(5) or self.has(2):
            return "sus"
        return "5" if self.has(7) else "1"

    def pitch_vector(self) -> np.ndarray:
        """Vector 12 nốt tuyệt đối, nốt gốc và bass được nhấn thêm."""
        v = np.zeros(12)
        if not self.is_chord:
            return v
        v += np.roll(self.bitmap, self.root)
        v[self.root] += 0.5
        v[self.bass_pc] += 0.25
        return v


_CHORD_CACHE: dict[str, _Chord] = {}


def _chord(label: str) -> _Chord:
    c = _CHORD_CACHE.get(label)
    if c is None:
        try:
            c = _Chord(label)
        except Exception:   # nhãn không đọc được: coi như không có hợp âm
            c = _Chord("X")
            c.label = label
        _CHORD_CACHE[label] = c
    return c


def simplify_label(label: str) -> str:
    """Rút gọn nhãn về nốt gốc + kiểu hợp âm ba Harte: 'A:min/b7' -> 'A:min', 'F:maj7' -> 'F:maj'.

    'sus' -> ':sus4' hoặc ':sus2', '5' -> ':5', chỉ có nốt gốc -> ':1'. N và X giữ nguyên.
    Nốt gốc viết theo một cách cố định cho mỗi cao độ ('C#:maj' và 'Db:maj' đều thành 'Db:maj'),
    để cùng một hợp âm viết khác nhau vẫn khớp khi so vòng và so đoạn.
    """
    c = _chord(label)
    if not c.is_chord:
        return label.split("/")[0] if label else "N"
    root_name = MAJOR_TONIC_NAMES[c.root]
    t = c.triad_class()
    if t == "sus":
        t = "sus4" if c.has(5) else "sus2"
    return f"{root_name}:{t}"


# ---------------------------------------------------------------- 1. giọng

def _key_scores(segments: list[ChordSegment]) -> np.ndarray | None:
    """Điểm của 24 giọng: index 0–11 là major (theo nốt chủ), 12–23 là minor."""
    chords = [(s, _chord(s.label)) for s in segments if s.end > s.start]
    chords = [(s, c) for s, c in chords if c.is_chord]
    if not chords:
        return None
    durs = np.array([s.end - s.start for s, _ in chords])
    total = durs.sum()
    if total <= 0:
        return None

    pv = np.zeros(12)
    for (s, c), d in zip(chords, durs):
        pv += d * c.pitch_vector()
    if pv.std() == 0:
        return None

    roots = np.array([c.root for _, c in chords])
    is_maj = np.array([c.has(4) and not c.has(3) for _, c in chords])
    is_min = np.array([c.has(3) and not c.has(4) for _, c in chords])
    # Bước chuyển hợp âm (bỏ hợp âm trùng liên tiếp)
    trans: list[tuple[_Chord, _Chord]] = []
    for (_, a), (_, b) in zip(chords, chords[1:]):
        if a.root != b.root or a.triad_class() != b.triad_class():
            trans.append((a, b))
    first, last = chords[0][1], chords[-1][1]

    scores = np.zeros(24)
    for mode_i, (profile, third_mask) in enumerate(((_MAJOR_PROFILE, is_maj), (_MINOR_PROFILE, is_min))):
        for t in range(12):
            corr = np.corrcoef(pv, np.roll(profile, t))[0, 1]
            tonic = (roots == t) & third_mask
            tonic_frac = durs[tonic].sum() / total

            def is_tonic(c: _Chord) -> bool:
                return c.root == t and (c.has(4) if mode_i == 0 else c.has(3)) and not (c.has(3) and c.has(4))

            cad = 0.0
            if trans:
                v = (t + 7) % 12
                cad = sum(1 for a, b in trans if a.root == v and a.has(4) and b.root == t) / len(trans)
            scores[mode_i * 12 + t] = (corr + W_TONIC_DUR * tonic_frac + W_FIRST * is_tonic(first)
                                       + W_LAST * is_tonic(last) + W_CADENCE * cad)
    return scores


def estimate_key(segments: list[ChordSegment]) -> KeyEstimate | None:
    """Ước lượng giọng chính của bài từ chuỗi hợp âm. Trả về None nếu không có hợp âm nào."""
    scores = _key_scores(segments)
    if scores is None:
        return None
    best = int(np.argmax(scores))
    p = np.exp((scores - scores.max()) / KEY_TEMPERATURE)
    conf = float(np.clip(p[best] / p.sum(), 0.0, 1.0))
    if best < 12:
        return KeyEstimate(MAJOR_TONIC_NAMES[best], "major", conf)
    return KeyEstimate(MINOR_TONIC_NAMES[best - 12], "minor", conf)


# ---------------------------------------------------------------- 2. bậc La Mã

# Tên bậc theo khoảng cách (nửa cung) từ nốt chủ. Giọng thứ đặt tên theo âm giai thứ tự nhiên
# (III, VI, VII không có dấu), bậc 6 và 7 nâng lên ghi "#VI", "#VII".
_MAJOR_DEGREES = ["I", "bII", "II", "bIII", "III", "IV", "#IV", "V", "bVI", "VI", "bVII", "VII"]
_MINOR_DEGREES = ["I", "bII", "II", "III", "#III", "IV", "#IV", "V", "VI", "#VI", "VII", "#VII"]

# Hậu tố theo quality Harte. Chữ hoa/thường quyết định bởi quãng ba (xem to_roman).
_SUFFIX = {
    "maj": "", "min": "", "dim": "°", "aug": "+", "5": "5", "1": "1",
    "maj7": "maj7", "min7": "7", "7": "7", "dim7": "°7", "hdim7": "ø7", "minmaj7": "(maj7)",
    "maj6": "6", "min6": "6", "9": "9", "maj9": "maj9", "min9": "9",
    "11": "11", "min11": "11", "13": "13", "maj13": "maj13", "min13": "13",
    "sus2": "sus2", "sus4": "sus4", "aug7": "+7",
}


def to_roman(label: str, key: KeyEstimate) -> str:
    """Đổi một nhãn Harte sang bậc La Mã theo giọng.

    Quy ước:
    - Chữ hoa nếu hợp âm có quãng ba trưởng (hoặc không có quãng ba: sus, 5), chữ thường nếu có
      quãng ba thứ. Ví dụ ("A:min", C major) -> "vi", ("G:7", C major) -> "V7".
    - Bậc ngoài âm giai có dấu đặt trước: ("Bb:maj", C major) -> "bVII".
    - Giọng thứ đặt tên theo âm giai thứ tự nhiên: ("C:maj", A minor) -> "III",
      ("G:maj", A minor) -> "VII", ("E:maj", A minor) -> "V"; bậc 7 nâng: ("G#:dim", A minor) -> "#vii°".
    - Hậu tố: ° (dim), ø7 (hdim7), °7 (dim7), + (aug), maj7, 7, 6, 9, sus4...
      Phần mở rộng trong ngoặc của Harte (ví dụ "(9)") bị bỏ qua.
    - Hợp âm đảo **giữ bậc của bass theo Harte sau dấu "/"**, không dùng ký hiệu figured bass:
      ("C:maj/5", C major) -> "I/5", ("C:maj/3", C major) -> "I/3", ("G:7/b7", C major) -> "V7/b7".
    - "N" -> "N.C.", "X" hoặc nhãn không đọc được -> "?".
    """
    if label in ("N", ""):
        return "N.C."
    c = _chord(label)
    if not c.is_chord:
        return "?"
    deg = (c.root - key_tonic_pc(key)) % 12
    numeral = (_MAJOR_DEGREES if key.mode == "major" else _MINOR_DEGREES)[deg]
    accidental = numeral.rstrip("IV")
    base = numeral[len(accidental):]
    minor_third = c.has(3) and not c.has(4)
    if minor_third:
        base = base.lower()
    suffix = _SUFFIX.get(c.quality)
    if suffix is None:   # quality rỗng (chỉ có ngoặc) hoặc lạ: suy từ các nốt
        t = c.triad_class()
        suffix = {"dim": "°", "aug": "+", "sus": "sus4" if c.has(5) else "sus2", "5": "5", "1": "1"}.get(t, "")
        if c.has(10) and t in ("maj", "min"):
            suffix += "7"
        elif c.has(11) and t == "maj":
            suffix += "maj7"
    out = accidental + base + suffix
    if c.bass_degree not in ("1", ""):
        out += "/" + c.bass_degree
    return out


# ---------------------------------------------------------------- 3. chia đoạn

MIN_SECTION_BEATS = 8       # đoạn ngắn hơn thì gộp vào đoạn bên cạnh
EMBED_BEATS = 8             # cửa sổ "nhớ" khi so hai vị trí (time-delay embedding)
NOVELTY_SMOOTH_BEATS = 8    # độ rộng Gauss làm mượt structure features theo thời gian
LAG_SMOOTH = 1              # làm mượt theo trục độ trễ
PEAK_WINDOW_BEATS = 12      # đỉnh novelty phải lớn nhất trong cửa sổ ± này
PEAK_THRESHOLD = 0.2        # ngưỡng novelty tương đối (so với trung vị và max)
SAME_SECTION_SIM = 0.5      # hai đoạn có độ giống >= ngưỡng thì cùng nhãn
REC_THRESHOLD = 0.75        # hai vị trí coi là lặp lại nếu tỉ lệ ô khớp trong cửa sổ embed >= ngưỡng
SNAP_SECONDS = 1.0          # kéo ranh giới về điểm đổi hợp âm gần nhất trong khoảng này
FALLBACK_BEAT = 0.5         # bước lưới (giây) khi không có phách


def _beat_grid(beats: BeatGrid | None, duration: float) -> np.ndarray:
    """Các mốc thời gian của lưới (đầu mỗi ô), có 0 ở đầu và duration ở cuối."""
    if beats is not None and len(beats.beats) >= 8:
        b = np.unique(np.asarray(beats.beats, dtype=float))
        b = b[(b > 0) & (b < duration)]
        if len(b) >= 8 and np.median(np.diff(b)) > 0:
            # ô trước phách đầu và sau phách cuối: chia theo khoảng phách trung vị
            step = float(np.median(np.diff(b)))
            head = np.arange(b[0] - step, 0, -step)[::-1] if b[0] > step * 1.5 else np.array([])
            tail = np.arange(b[-1] + step, duration - step * 0.5, step)
            grid = np.concatenate([[0.0], head[head > step * 0.5], b, tail, [duration]])
            return np.unique(grid)
    n = max(1, int(round(duration / FALLBACK_BEAT)))
    return np.linspace(0.0, duration, n + 1)


def _frame_labels(segments: list[ChordSegment], grid: np.ndarray) -> list[str]:
    """Nhãn rút gọn chiếm nhiều thời gian nhất trong mỗi ô của lưới."""
    out: list[str] = []
    j = 0
    segs = sorted(segments, key=lambda s: s.start)
    for a, b in zip(grid[:-1], grid[1:]):
        while j < len(segs) and segs[j].end <= a:
            j += 1
        acc: dict[str, float] = {}
        k = j
        while k < len(segs) and segs[k].start < b:
            ov = min(b, segs[k].end) - max(a, segs[k].start)
            if ov > 0:
                lab = simplify_label(segs[k].label)
                acc[lab] = acc.get(lab, 0.0) + ov
            k += 1
        out.append(max(acc, key=acc.get) if acc else "N")
    return out


def _gauss1d(x: np.ndarray, sigma: float, axis: int) -> np.ndarray:
    if sigma <= 0:
        return x
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.apply_along_axis(lambda v: np.convolve(np.pad(v, r, mode="edge"), k, mode="valid"), axis, x)


def _recurrence(codes: np.ndarray, embed: int) -> np.ndarray:
    """R[i, j] = tỉ lệ ô khớp hợp âm trong cửa sổ embed ô bắt đầu tại i và j (bỏ ô N)."""
    n = len(codes)
    eq = (codes[:, None] == codes[None, :]) & (codes[:, None] >= 0)
    # cộng dồn theo đường chéo: R[i,j] = mean(eq[i+t, j+t], t=0..embed-1)
    pad = np.zeros((n + embed, n + embed), dtype=float)
    pad[:n, :n] = eq
    acc = np.zeros((n, n))
    for t in range(embed):
        acc += pad[t:t + n, t:t + n]
    return acc / embed


def _novelty_boundaries(codes: np.ndarray) -> list[int]:
    """Ranh giới (chỉ số ô) theo structure features của Serrà và cộng sự (2012)."""
    n = len(codes)
    if n < 2 * MIN_SECTION_BEATS:
        return []
    embed = min(EMBED_BEATS, max(1, n // 4))
    R = _recurrence(codes, embed)
    # R dùng cửa sổ nhìn về phía trước [i, i+embed), nên mẫu lặp thay đổi trên cả dải
    # [b-embed, b] quanh ranh giới b. Dời R đi nửa cửa sổ để cửa sổ thành đối xứng quanh i.
    h = embed // 2
    if h:
        R = np.pad(R, ((h, 0), (h, 0)))[:n, :n]
    R = (R >= REC_THRESHOLD).astype(float)
    np.fill_diagonal(R, 0.0)
    # ma trận độ trễ: L[i, l] = R[i, (i - l) mod n] (đường chéo thành đường ngang)
    idx = (np.arange(n)[:, None] - np.arange(n)[None, :]) % n
    L = R[np.arange(n)[:, None], idx]
    L = _gauss1d(L, NOVELTY_SMOOTH_BEATS / 2, axis=0)
    L = _gauss1d(L, LAG_SMOOTH, axis=1)
    nov = np.r_[0.0, np.sqrt(((L[1:] - L[:-1]) ** 2).sum(axis=1))]
    if nov.max() <= 0:
        return []
    nov = nov / nov.max()
    w = PEAK_WINDOW_BEATS
    peaks = []
    for i in range(1, n):
        lo, hi = max(0, i - w), min(n, i + w + 1)
        if nov[i] >= nov[lo:hi].max() and nov[i] > PEAK_THRESHOLD + np.median(nov[lo:hi]) * 0.5:
            peaks.append(i)
    return peaks


def _align_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Độ giống hai đoạn: tỉ lệ ô khớp tốt nhất khi trượt đoạn ngắn dọc đoạn dài, chia theo đoạn dài."""
    if len(a) > len(b):
        a, b = b, a
    if len(a) == 0:
        return 0.0
    best = 0
    for off in range(len(b) - len(a) + 1):
        m = int(np.sum((a == b[off:off + len(a)]) & (a >= 0)))
        best = max(best, m)
    return best / len(b)


def _collapse(seq: list[str]) -> list[str]:
    out: list[str] = []
    for s in seq:
        if s in ("N", "X"):
            continue
        if not out or out[-1] != s:
            out.append(s)
    return out


MIN_LOOP_COVERAGE = 0.6     # vòng phải phủ ít nhất tỉ lệ này của chuỗi hợp âm, nếu không thì trả cả chuỗi


def normalize_label(label: str) -> str:
    """Viết nhãn Harte ở dạng đầy đủ: 'C' -> 'C:maj', 'C/5' -> 'C:maj/5'. Nhãn khác giữ nguyên."""
    if ":" in label or label in ("N", "X", ""):
        return label
    root, sep, bass = label.partition("/")
    return f"{root}:maj{sep}{bass}"


def find_progression(sequences: list[list[str]] | list[str], min_len: int = 2,
                     max_len: int = 8) -> list[str]:
    """Vòng hợp âm lặp lại: n-gram (min_len..max_len hợp âm) phủ được nhiều hợp âm nhất.

    `sequences` là một chuỗi hợp âm, hoặc nhiều chuỗi (mỗi lần xuất hiện của cùng một loại đoạn);
    n-gram không vượt qua ranh giới giữa các chuỗi. Bỏ N và hợp âm trùng liên tiếp trước khi đếm.
    Điểm = số lần xuất hiện không chồng nhau x độ dài; hòa thì lấy vòng ngắn hơn, rồi cách xoay vòng
    bắt đầu bằng hợp âm mở đầu nhiều chuỗi nhất, rồi vòng xuất hiện sớm hơn. Nếu không có n-gram nào lặp ít nhất hai lần và phủ được MIN_LOOP_COVERAGE của
    tổng số hợp âm, trả về chuỗi dài nhất (đã gộp, cắt còn max_len hợp âm).
    """
    if sequences and isinstance(sequences[0], str):
        sequences = [sequences]  # type: ignore[list-item]
    seqs = [_collapse(list(q)) for q in sequences]
    seqs = [q for q in seqs if q]
    if not seqs:
        return []
    total = sum(len(q) for q in seqs)
    openers = Counter(q[0] for q in seqs)       # hợp âm mở đầu mỗi chuỗi: ưu tiên vòng bắt đầu bằng nó
    best: tuple[int, int, int, int] | None = None   # (điểm, -độ dài, số chuỗi mở đầu bằng g[0], -vị trí đầu)
    best_gram: tuple[str, ...] = ()
    for n in range(min_len, max_len + 1):
        first_pos: dict[tuple[str, ...], int] = {}
        counts: Counter = Counter()
        offset = 0
        for q in seqs:
            last_end: dict[tuple[str, ...], int] = {}
            for i in range(len(q) - n + 1):
                g = tuple(q[i:i + n])
                if len(set(g)) < 2:
                    continue
                first_pos.setdefault(g, offset + i)
                if i >= last_end.get(g, 0):
                    counts[g] += 1
                    last_end[g] = i + n
            offset += len(q)
        for g, cnt in counts.items():
            if cnt < 2 or cnt * n < MIN_LOOP_COVERAGE * total:
                continue
            key = (cnt * n, -n, openers[g[0]], -first_pos[g])
            if best is None or key > best:
                best, best_gram = key, g
    if best is None:
        return max(seqs, key=len)[:max_len]
    return list(best_gram)


def _chord_runs(segments: list[ChordSegment], start: float, end: float) -> list[tuple[str, str]]:
    """Chuỗi (nhãn rút gọn, nhãn đầy đủ đại diện) trong [start, end): gộp các hợp âm liên tiếp
    cùng nhãn rút gọn, nhãn đại diện là nhãn chiếm thời lượng lớn nhất trong cụm. Bỏ N/X."""
    # Bỏ N/X trước khi gộp, giống _collapse, để "C N C G" thành "C G" như lúc tìm vòng.
    inside = [s for s in segments if min(s.end, end) - max(s.start, start) > 0.05 and _chord(s.label).is_chord]
    runs: list[tuple[str, str]] = []
    i = 0
    simple = [simplify_label(s.label) for s in inside]
    while i < len(inside):
        j = i
        dur: dict[str, float] = {}
        while j < len(inside) and simple[j] == simple[i]:
            ov = min(inside[j].end, end) - max(inside[j].start, start)
            dur[inside[j].label] = dur.get(inside[j].label, 0.0) + ov
            j += 1
        if _chord(simple[i]).is_chord:
            runs.append((simple[i], normalize_label(max(dur, key=dur.get))))
        i = j
    return runs


def _cluster_progression(runs_list: list[list[tuple[str, str]]]) -> list[str]:
    """Vòng hợp âm chung của mọi đoạn cùng nhãn, ở dạng nhãn Harte đầy đủ hay gặp nhất mỗi vị trí."""
    prog = find_progression([[s for s, _ in runs] for runs in runs_list])
    if not prog:
        return []
    n = len(prog)
    votes: list[Counter] = [Counter() for _ in range(n)]
    for runs in runs_list:
        simples = [s for s, _ in runs]
        for i in range(len(simples) - n + 1):
            if simples[i:i + n] == prog:
                for k in range(n):
                    votes[k][runs[i + k][1]] += 1
    return [votes[k].most_common(1)[0][0] if votes[k] else normalize_label(prog[k]) for k in range(n)]


def _name_sections(cluster_ids: list[int], durations: list[float]) -> list[str]:
    """Đặt tên đoạn bằng heuristic; đoạn không đặt tên được thì giữ chữ cái A/B/C."""
    n = len(cluster_ids)
    letters = {}
    for c in cluster_ids:
        if c not in letters:
            letters[c] = chr(ord("A") + len(letters)) if len(letters) < 26 else f"S{len(letters)}"
    names = [letters[c] for c in cluster_ids]
    count = Counter(cluster_ids)
    if n == 1:
        return names
    total = sum(durations)
    # intro, outro: đoạn đầu và cuối không lặp lại, ngắn
    if count[cluster_ids[0]] == 1 and durations[0] < 0.2 * total:
        names[0] = "intro"
    if count[cluster_ids[-1]] == 1 and durations[-1] < 0.2 * total:
        names[-1] = "outro"
    repeated = [c for c in dict.fromkeys(cluster_ids) if count[c] >= 2]
    if len(repeated) >= 2:
        verse = repeated[0]
        # chorus: cụm lặp nhiều nhất trong số còn lại (hòa thì cụm có tổng thời lượng lớn hơn)
        dur_of = Counter()
        for c, d in zip(cluster_ids, durations):
            dur_of[c] += d
        chorus = max(repeated[1:], key=lambda c: (count[c], dur_of[c]))
        for i, c in enumerate(cluster_ids):
            if names[i] in ("intro", "outro"):
                continue
            if c == verse:
                names[i] = "verse"
            elif c == chorus:
                names[i] = "chorus"
    # đoạn không lặp nằm giữa bài, sau ít nhất một chorus: bridge
    seen_chorus = False
    for i, c in enumerate(cluster_ids):
        if names[i] == "chorus":
            seen_chorus = True
        elif seen_chorus and count[c] == 1 and 0 < i < n - 1 and len(names[i]) == 1:
            names[i] = "bridge"
    return names


def analyze_structure(
    segments: list[ChordSegment],
    beats: BeatGrid | None,
    duration: float,
    key: KeyEstimate | None,
) -> list[Section]:
    """Chia bài thành các đoạn phủ liên tục [0, duration] và tìm vòng hợp âm của mỗi đoạn.

    `progression` là nhãn Harte dạng đầy đủ ("C:maj", không viết "C"), lấy nhãn hay gặp nhất ở mỗi
    vị trí của vòng; các đoạn cùng nhãn dùng chung một vòng (đếm n-gram gộp trên mọi đoạn đó).
    `roman` là bậc La Mã theo `key` (list rỗng nếu key là None).
    """
    if duration <= 0:
        return []
    segs = [s for s in segments if s.end > s.start]
    grid = _beat_grid(beats, duration)
    frames = _frame_labels(segs, grid)
    vocab: dict[str, int] = {}
    codes = np.array([-1 if f in ("N", "X") else vocab.setdefault(f, len(vocab)) for f in frames])

    bounds = _novelty_boundaries(codes) if segs else []
    # gộp đoạn quá ngắn
    cuts = [0] + bounds + [len(codes)]
    merged = [0]
    for c in cuts[1:-1]:
        if c - merged[-1] >= MIN_SECTION_BEATS:
            merged.append(c)
    if len(merged) > 1 and len(codes) - merged[-1] < MIN_SECTION_BEATS:
        merged.pop()
    cuts = merged + [len(codes)]

    # gom cụm theo độ giống chuỗi hợp âm
    parts = [codes[a:b] for a, b in zip(cuts[:-1], cuts[1:])]
    cluster_ids: list[int] = []
    reps: list[int] = []   # đoạn đại diện của mỗi cụm
    for i, p in enumerate(parts):
        best_c, best_s = -1, 0.0
        for cid, r in enumerate(reps):
            ratio = len(p) / max(1, len(parts[r]))
            if not 0.5 <= ratio <= 2.0:
                continue
            s = _align_sim(p, parts[r])
            if s > best_s:
                best_c, best_s = cid, s
        if best_s >= SAME_SECTION_SIM:
            cluster_ids.append(best_c)
        else:
            cluster_ids.append(len(reps))
            reps.append(i)

    times = [float(grid[c]) for c in cuts]
    times[0], times[-1] = 0.0, float(duration)
    # Ranh giới đoạn thường trùng điểm đổi hợp âm: kéo về điểm đổi gần nhất (lưới phách hoặc
    # lưới cố định có thể lệch nửa phách so với chỗ đổi thật).
    changes = np.array(sorted({s.start for s in segs if 0 < s.start < duration}))
    if len(changes):
        for i in range(1, len(times) - 1):
            j = int(np.argmin(np.abs(changes - times[i])))
            if abs(changes[j] - times[i]) <= SNAP_SECONDS and times[i - 1] < changes[j] < times[i + 1]:
                times[i] = float(changes[j])
    durations = [b - a for a, b in zip(times[:-1], times[1:])]
    names = _name_sections(cluster_ids, durations)

    # vòng hợp âm: tính chung cho mọi đoạn cùng cụm, để cùng một loại đoạn có cùng một vòng
    spans = list(zip(times[:-1], times[1:]))
    runs = [_chord_runs(segs, a, b) for a, b in spans]
    progs: dict[int, list[str]] = {}
    for cid in set(cluster_ids):
        progs[cid] = _cluster_progression([r for r, c in zip(runs, cluster_ids) if c == cid])

    out: list[Section] = []
    for (a, b), name, cid in zip(spans, names, cluster_ids):
        prog = progs[cid]
        roman = [to_roman(p, key) for p in prog] if key is not None else []
        out.append(Section(start=a, end=b, label=name, progression=list(prog), roman=roman))
    return out
