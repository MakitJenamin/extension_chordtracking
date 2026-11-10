"""Chọn một dòng giai điệu đơn âm từ đầu ra thô của model melody.

Hai loại đầu vào (xem `MelodyModelOutput`):

- **Nốt đa âm** (YourMT3+, MuScriptor, Basic Pitch): `notes_to_line`. Trục thời gian được chia thành các
  khoảng sơ cấp, trong mỗi khoảng tập nốt đang vang không đổi (ranh giới là mọi điểm bắt đầu và kết thúc
  nốt), nên kết quả giữ đúng thời điểm nốt chứ không bị làm tròn theo khung. Mỗi khoảng chọn một nốt
  (hoặc nghỉ) theo một trong ba chiến lược:
    * `skyline`: nốt cao nhất.
    * `salience`: nốt có độ nổi cao nhất. Độ nổi gộp confidence, độ dài nốt, độ cao và ưu tiên nguồn.
    * `viterbi`: như `salience` nhưng tối ưu trên cả bài bằng quy hoạch động: phạt nhảy cao độ xa, phạt
      cắt ngang một nốt đang vang hoặc nhảy vào giữa một nốt đã bắt đầu.
  Lọc chung trước khi chọn: bỏ nốt ngoài khoảng cao độ, nốt quá ngắn, nguồn đệm (bass, trống, pad) nếu
  `exclude_accomp`. `source_mode="strict"` chỉ giữ các nốt có mức ưu tiên nguồn cao nhất trong từng
  khoảng (giọng hát > nhạc cụ giai điệu > không rõ > nhạc cụ đệm như piano, guitar).

- **Đường f0** (RMVPE): `f0_to_line`. Ngưỡng voicing, bù lệch dây (tuning) cả bài, lọc trung vị, rồi tách
  nốt có trễ (hysteresis): cao độ lệch khỏi nốt hiện tại quá `split_semitones` liên tục ít nhất
  `split_min_dur` giây thì sang nốt mới, nên vibrato quanh một nốt vẫn là một nốt.

`combine_lines` ghép hai dòng (vd RMVPE ở đoạn có giọng, model nhạc cụ lấp chỗ trống).
`quantize_to_beats` căn đầu và cuối nốt về lưới phách chia nhỏ (tùy chọn).

Mỗi nốt đầu ra có `confidence` trong [0, 1] để giao diện hiện mờ nốt không chắc. Kết quả luôn sắp theo
thời gian và không chồng nhau.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, replace

import numpy as np

from chordtrack.types import BeatGrid, MelodyModelOutput, MelodyNote, PitchTrack

# --------------------------------------------------------------------------- nguồn (nhạc cụ)

VOICE, MELODIC, UNKNOWN, ACCOMP, EXCLUDED = 3, 2, 1, 0, -1

_VOICE_WORDS = ("voice", "vocal", "sing", "choir", "soprano", "alto", "tenor", "baritone", "aah", "ooh")
_EXCLUDED_WORDS = ("drum", "percussion", "timpani", "cymbal", "snare", "kick", "hihat", "hi-hat", "pad", "fx",
                   "effect", "noise")
_MELODIC_WORDS = ("violin", "viola", "cello", "string", "flute", "piccolo", "oboe", "clarinet", "bassoon", "sax",
                  "trumpet", "trombone", "horn", "tuba", "brass", "recorder", "whistle", "ocarina", "pipe",
                  "lead", "erhu", "shakuhachi", "fiddle", "harmonica", "accordion", "reed", "wind")


def _gm_class(program: int) -> int:
    """Lớp ưu tiên theo số chương trình General MIDI (0–127), 128 = bộ trống."""
    if program >= 128:
        return EXCLUDED
    if 32 <= program <= 39 or 88 <= program <= 103 or 112 <= program <= 127:
        return EXCLUDED          # bass, pad và hiệu ứng synth, bộ gõ, tiếng động
    if program in (47, 55):
        return EXCLUDED          # timpani, orchestra hit
    if 52 <= program <= 54:
        return VOICE             # choir aahs, voice oohs, synth voice
    if 40 <= program <= 87:
        return MELODIC           # dây, kèn đồng, kèn gỗ, sáo, synth lead (gồm cả bè dây 48-51)
    return ACCOMP                # piano, phím gõ, organ, guitar, nhạc cụ dân tộc...


def source_class(source: str | None, chorus_as_voice: bool = True) -> int:
    """Phân loại nhãn nguồn của model: VOICE, MELODIC, UNKNOWN (không có nhãn), ACCOMP, EXCLUDED.
    `chorus_as_voice=False`: bè hát / hợp xướng ("voice_chorus", "choir...") chỉ tính là nhạc cụ giai điệu."""
    if source is None or not str(source).strip():
        return UNKNOWN
    s = str(source).strip().lower()
    m = re.fullmatch(r"(?:program[:_ ]?)?(\d+)", s)
    if m:
        return _gm_class(int(m.group(1)))
    if "sax" in s:               # alto/tenor/soprano/baritone sax không phải giọng hát
        return MELODIC
    if any(w in s for w in _VOICE_WORDS):
        if not chorus_as_voice and ("chorus" in s or "choir" in s):
            return MELODIC
        return VOICE
    if "bass" in s and "bassoon" not in s:
        return EXCLUDED
    if any(w in s for w in _EXCLUDED_WORDS):
        return EXCLUDED
    if any(w in s for w in _MELODIC_WORDS):
        return MELODIC
    return ACCOMP


# --------------------------------------------------------------------------- tham số


@dataclass(frozen=True)
class LineParams:
    """Tham số chọn dòng giai điệu. Mặc định là cấu hình hợp lý chưa tinh chỉnh; `eval/melody/score.py`
    chọn lại bằng cross-validation."""

    # --- nốt đa âm
    strategy: str = "viterbi"         # "skyline" | "salience" | "viterbi"
    pitch_min: int = 40               # E2
    pitch_max: int = 100              # E7
    min_input_dur: float = 0.03       # bỏ nốt đầu vào ngắn hơn (giây)
    exclude_accomp: bool = True       # bỏ nguồn EXCLUDED (bass, trống, pad)
    source_mode: str = "bonus"        # "ignore" | "bonus" (cộng vào độ nổi) | "strict" (chỉ giữ lớp cao nhất)
    w_source: float = 0.3             # độ nổi cộng thêm mỗi bậc ưu tiên nguồn
    w_conf: float = 1.0               # độ nổi theo confidence (velocity) của nốt
    w_dur: float = 0.3                # độ nổi theo log độ dài nốt
    w_pitch: float = 0.5              # độ nổi cộng thêm mỗi quãng tám cao hơn C4
    chorus_as_voice: bool = True      # bè hát / hợp xướng cùng mức ưu tiên với giọng chính
    rest_score: float = 0.0           # độ nổi của trạng thái nghỉ (> 0 thì bỏ được nốt yếu)
    jump_penalty: float = 0.5         # viterbi: phạt mỗi quãng tám nhảy (sau phần miễn phạt)
    jump_free: float = 2.0            # viterbi: số bán cung nhảy không bị phạt
    phrase_gap: float = 1.0           # viterbi: nghỉ lâu hơn (giây) thì coi là câu mới, không phạt nhảy
    cut_penalty: float = 0.15         # viterbi: phạt cắt ngang nốt đang vang / vào giữa nốt đã bắt đầu
    # --- đường f0
    voicing_threshold: float = 0.0    # khung có confidence thấp hơn coi là không có giai điệu
    median_frames: int = 5            # cửa sổ lọc trung vị (khung, số lẻ; 1 = không lọc)
    fill_gap: float = 0.03            # lấp khoảng lặng ngắn hơn (giây) giữa hai đoạn có giai điệu
    split_semitones: float = 0.6      # lệch quá ngưỡng này (bán cung) thì có thể sang nốt mới
    split_min_dur: float = 0.08       # ... nếu lệch liên tục ít nhất ngần này giây
    tuning: bool = True               # bù lệch dây chung của cả bài trước khi làm tròn
    # --- chung cho đầu ra
    min_dur: float = 0.06             # bỏ nốt đầu ra ngắn hơn (giây)

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_PARAMS = LineParams()

# --------------------------------------------------------------------------- tiện ích


def hz_to_midi(f0_hz: np.ndarray) -> np.ndarray:
    f = np.asarray(f0_hz, dtype=np.float64)
    out = np.full(f.shape, np.nan)
    pos = f > 0
    out[pos] = 69.0 + 12.0 * np.log2(f[pos] / 440.0)
    return out


def midi_to_hz(midi) -> np.ndarray:
    return 440.0 * 2.0 ** ((np.asarray(midi, dtype=np.float64) - 69.0) / 12.0)


def make_monophonic(notes: list[MelodyNote], min_dur: float = 0.0) -> list[MelodyNote]:
    """Sắp theo thời gian, cắt đuôi nốt trước tại đầu nốt sau, bỏ nốt ngắn hơn `min_dur`."""
    notes = sorted((n for n in notes if n.end > n.start), key=lambda n: (n.start, -n.pitch))
    out: list[MelodyNote] = []
    for n in notes:
        if out and out[-1].end > n.start:
            if n.start <= out[-1].start:     # cùng điểm bắt đầu: giữ nốt đã có (cao hơn nhờ thứ tự sắp)
                continue
            out[-1] = replace(out[-1], end=n.start)
        out.append(n)
    return [n for n in out if n.end - n.start >= min_dur - 1e-9]


def notes_to_f0(notes: list[MelodyNote], times: np.ndarray) -> np.ndarray:
    """Đổi dòng nốt đơn âm ra f0 (Hz) tại `times`; 0 ở chỗ không có nốt. Nốt phủ [start, end)."""
    times = np.asarray(times, dtype=np.float64)
    f0 = np.zeros(len(times))
    if not notes:
        return f0
    mono = make_monophonic(notes)
    starts = np.array([n.start for n in mono])
    ends = np.array([n.end for n in mono])
    hz = midi_to_hz([n.pitch for n in mono])
    i = np.searchsorted(starts, times, side="right") - 1
    ok = (i >= 0)
    ok[ok] &= times[ok] < ends[i[ok]]
    f0[ok] = hz[i[ok]]
    return f0


# --------------------------------------------------------------------------- nốt đa âm


def _salience(n: MelodyNote, cls: int, p: LineParams) -> float:
    conf = 0.5 if n.confidence is None else float(n.confidence)
    s = 1.0 + p.w_conf * (conf - 0.5) + p.w_dur * math.log(max(n.end - n.start, 1e-3) / 0.25)
    s += p.w_pitch * (n.pitch - 60) / 12.0
    if p.source_mode != "ignore":
        s += p.w_source * max(cls, 0)
    return s


def _elementary(notes: list[MelodyNote]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Ranh giới các khoảng sơ cấp và danh sách chỉ số nốt vang trong mỗi khoảng."""
    starts = np.array([n.start for n in notes])
    ends = np.array([n.end for n in notes])
    bounds = np.unique(np.concatenate([starts, ends]))
    # Nốt i vang ở các khoảng k thuộc [chỉ số của start_i, chỉ số của end_i): tổng chi phí tuyến tính theo
    # kích thước đầu ra, không quét lại mọi nốt ở mỗi khoảng.
    si = np.searchsorted(bounds, starts)
    ei = np.searchsorted(bounds, ends)
    lists: list[list[int]] = [[] for _ in range(len(bounds) - 1)]
    for i in np.argsort(starts, kind="stable"):
        for k in range(si[i], ei[i]):
            lists[k].append(int(i))
    return bounds, [np.array(x, dtype=int) for x in lists]


def notes_to_line(notes: list[MelodyNote], params: LineParams = DEFAULT_PARAMS) -> list[MelodyNote]:
    """Chọn dòng giai điệu đơn âm từ các nốt đa âm (xem docstring module)."""
    p = params
    cls_all = [source_class(n.source, p.chorus_as_voice) for n in notes]
    keep = [i for i, n in enumerate(notes)
            if p.pitch_min <= n.pitch <= p.pitch_max and n.end - n.start >= p.min_input_dur
            and not (p.exclude_accomp and cls_all[i] == EXCLUDED)]
    if not keep:
        return []
    cand = [notes[i] for i in keep]
    cls = np.array([cls_all[i] for i in keep])
    sal = np.array([_salience(n, c, p) for n, c in zip(cand, cls)])
    pitch = np.array([n.pitch for n in cand], dtype=np.float64)
    starts = np.array([n.start for n in cand])
    ends = np.array([n.end for n in cand])
    bounds, active = _elementary(cand)
    if p.source_mode == "strict":
        active = [a[cls[a] == cls[a].max()] if len(a) else a for a in active]
    dur = np.diff(bounds)

    if p.strategy == "skyline":
        choice = [int(a[np.argmax(pitch[a])]) if len(a) else -1 for a in active]
    elif p.strategy == "salience":
        choice = [int(a[np.argmax(sal[a])]) if len(a) and sal[a].max() > p.rest_score else -1 for a in active]
    elif p.strategy == "viterbi":
        choice = _viterbi(bounds, dur, active, sal, pitch, starts, ends, p)
    else:
        raise ValueError(f"strategy không hợp lệ: {p.strategy}")

    # Gộp các khoảng liền nhau chọn cùng một nốt.
    out: list[MelodyNote] = []
    k = 0
    while k < len(choice):
        c = choice[k]
        if c < 0:
            k += 1
            continue
        j = k
        while j + 1 < len(choice) and choice[j + 1] == c:
            j += 1
        n = cand[c]
        t0, t1 = float(bounds[k]), float(bounds[j + 1])
        coverage = (t1 - t0) / max(n.end - n.start, 1e-6)
        base = 0.8 if n.confidence is None else float(n.confidence)
        conf = float(np.clip(base * (0.6 + 0.4 * min(1.0, coverage)), 0.0, 1.0))
        out.append(MelodyNote(t0, t1, int(n.pitch), conf, n.source))
        k = j + 1
    return make_monophonic(out, p.min_dur)


def _viterbi(bounds, dur, active, sal, pitch, starts, ends, p: LineParams) -> list[int]:
    """Quy hoạch động trên các khoảng sơ cấp. Trạng thái: một nốt đang vang hoặc nghỉ (-1)."""
    K = len(active)
    if K == 0:
        return []
    eps = 1e-9
    prev_states = np.array([-1])
    prev_score = np.array([0.0])
    # Cao độ và thời điểm kết thúc của nốt cuối trên đường tốt nhất tới mỗi trạng thái. Trạng thái nghỉ
    # mang theo nốt trước nó (xấp xỉ), để đi vòng qua chỗ nghỉ không né được phạt nhảy xa.
    prev_last_pitch = np.array([np.nan])
    prev_last_end = np.array([-np.inf])
    backs: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(K):
        st = np.concatenate([[-1], active[k]]).astype(int)
        emit = np.where(st >= 0, sal[np.maximum(st, 0)], p.rest_score) * dur[k]
        b = bounds[k]
        # chi phí chuyển từ prev (hàng) sang st (cột)
        a = prev_states[:, None]
        c = st[None, :]
        same = a == c
        a_note = a >= 0
        c_note = c >= 0
        a_cont = a_note & (ends[np.maximum(a, 0)] > b + eps)          # nốt trước còn vang qua ranh giới
        c_late = c_note & (starts[np.maximum(c, 0)] < b - eps)         # nốt sau đã bắt đầu từ trước
        lp = prev_last_pitch[:, None]
        near = np.isfinite(lp) & ((b - prev_last_end[:, None]) <= p.phrase_gap)
        jump = np.abs(np.where(np.isfinite(lp), lp, 0.0) - pitch[np.maximum(c, 0)])
        cost = np.where(near & c_note, p.jump_penalty * np.maximum(jump - p.jump_free, 0.0) / 12.0, 0.0)
        cost = cost + p.cut_penalty * (a_cont.astype(float) + c_late.astype(float))
        cost = np.where(same, 0.0, cost)
        tot = prev_score[:, None] - cost
        arg = np.argmax(tot, axis=0)
        prev_score = tot[arg, np.arange(len(st))] + emit
        backs.append((prev_states[arg], st))
        is_note = st >= 0
        prev_last_pitch = np.where(is_note, pitch[np.maximum(st, 0)], prev_last_pitch[arg])
        prev_last_end = np.where(is_note, bounds[k + 1], prev_last_end[arg])
        prev_states = st
    # truy vết
    choice = [0] * K
    cur = int(prev_states[int(np.argmax(prev_score))])
    for k in range(K - 1, -1, -1):
        choice[k] = cur
        from_states, st = backs[k]
        idx = int(np.nonzero(st == cur)[0][0])
        cur = int(from_states[idx])
    return choice


# --------------------------------------------------------------------------- đường f0


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Các đoạn liên tiếp True: danh sách (bắt đầu, kết thúc) kiểu nửa mở."""
    m = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.nonzero(d == 1)[0].tolist(), np.nonzero(d == -1)[0].tolist()))


def estimate_tuning(midi: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Lệch dây chung (bán cung, trong [-0.5, 0.5)) theo trung bình vòng của phần lẻ cao độ."""
    v = np.isfinite(midi)
    if not v.any():
        return 0.0
    w = np.ones(v.sum()) if weights is None else np.asarray(weights, dtype=np.float64)[v]
    ang = 2 * np.pi * midi[v]
    z = np.sum(w * np.exp(1j * ang))
    # Phần lẻ phân tán (vibrato rộng, glissando) thì trung bình vòng không đáng tin: không bù.
    if abs(z) < 0.3 * max(float(np.sum(w)), 1e-9):
        return 0.0
    off = float(np.angle(z) / (2 * np.pi))
    return ((off + 0.5) % 1.0) - 0.5


def f0_to_line(track: PitchTrack, params: LineParams = DEFAULT_PARAMS) -> list[MelodyNote]:
    """Đổi đường f0 theo khung ra dòng nốt (xem docstring module)."""
    from scipy.ndimage import median_filter

    p = params
    times = np.asarray(track.times, dtype=np.float64)
    if len(times) == 0:
        return []
    midi = hz_to_midi(track.f0_hz)
    # Luôn sao chép: phần lấp khoảng ghi vào conf, không được sửa mảng của người gọi.
    conf = None if track.confidence is None else np.array(track.confidence, dtype=np.float64)
    voiced = np.isfinite(midi)
    if conf is not None and p.voicing_threshold > 0:
        voiced &= conf >= p.voicing_threshold
    voiced &= (midi >= p.pitch_min - 0.5) & (midi <= p.pitch_max + 0.5)
    midi = np.where(voiced, midi, np.nan)
    hop = float(np.median(np.diff(times))) if len(times) > 1 else 0.01

    # Lấp khoảng lặng ngắn bằng nội suy tuyến tính (chỉ ở giữa hai đoạn có giai điệu).
    if p.fill_gap > 0:
        for a, b in _runs(~voiced):
            if a > 0 and b < len(midi) and (b - a) * hop <= p.fill_gap + 1e-9:
                midi[a:b] = np.linspace(midi[a - 1], midi[b], b - a + 2)[1:-1]
                voiced[a:b] = True
                if conf is not None:
                    conf[a:b] = np.minimum(conf[a - 1], conf[b])

    if p.tuning:
        # Ước lượng trên cao độ đã lọc trung vị khoảng 0.2 s để vibrato không làm lệch trung bình vòng.
        smooth = np.full_like(midi, np.nan)
        win = max(1, int(round(0.2 / hop))) | 1
        for a, b in _runs(voiced):
            smooth[a:b] = median_filter(midi[a:b], size=min(win, (b - a) | 1), mode="nearest")
        midi = midi - estimate_tuning(smooth, conf)

    notes: list[MelodyNote] = []
    split_frames = max(1, int(round(p.split_min_dur / hop)))
    for a, b in _runs(voiced):
        seg = midi[a:b]
        if p.median_frames > 1 and len(seg) > 1:
            seg = median_filter(seg, size=min(p.median_frames | 1, len(seg) | 1), mode="nearest")
        # tách nốt có trễ
        bounds = [0]
        cur = int(np.round(np.median(seg[: max(1, split_frames)])))
        dev_start = None
        for j in range(len(seg)):
            if abs(seg[j] - cur) > p.split_semitones:
                if dev_start is None:
                    dev_start = j
                if j - dev_start + 1 >= split_frames:
                    dev = seg[dev_start: j + 1]
                    new = int(np.round(np.median(dev)))
                    if new != cur:
                        if dev_start > bounds[-1]:
                            bounds.append(dev_start)
                        cur = new
                    dev_start = None
            else:
                dev_start = None
        bounds.append(len(seg))
        for s, e in zip(bounds[:-1], bounds[1:]):
            vals = seg[s:e]
            med = float(np.median(vals))
            pitch = int(np.round(med))
            t0 = float(times[a + s]) - hop / 2
            t1 = float(times[a + e - 1]) + hop / 2
            c = 1.0 if conf is None else float(np.mean(conf[a + s: a + e]))
            in_tune = 1.0 - min(0.5, abs(med - pitch))
            notes.append(MelodyNote(max(0.0, t0), t1, pitch, float(np.clip(c * in_tune, 0, 1)), "f0"))
    return make_monophonic(notes, p.min_dur)


# --------------------------------------------------------------------------- ghép, căn phách


def raw_pitch_track(out: MelodyModelOutput) -> PitchTrack | None:
    """Đường f0 chưa cắt ngưỡng nếu model lưu (RMVPE: `extras` có `f0_raw_hz` và `salience_max`), để tự chọn
    ngưỡng voicing qua `LineParams.voicing_threshold`. Không có thì trả `out.pitch_track`."""
    pt = out.pitch_track
    if pt is None or "f0_raw_hz" not in out.extras or "salience_max" not in out.extras:
        return pt
    raw = np.asarray(out.extras["f0_raw_hz"], dtype=np.float64)
    sal = np.clip(np.asarray(out.extras["salience_max"], dtype=np.float64), 0.0, 1.0)
    if len(raw) != len(pt.times) or len(sal) != len(pt.times):
        return pt
    return PitchTrack(pt.times, raw, sal)


def extract_line(out: MelodyModelOutput, params: LineParams = DEFAULT_PARAMS) -> list[MelodyNote]:
    """Dòng giai điệu từ một đầu ra model: dùng `notes` nếu có, ngược lại đường f0 (bản thô nếu có và
    `voicing_threshold` > 0, ngược lại bản model đã cắt ngưỡng)."""
    if out.notes:
        return notes_to_line(out.notes, params)
    if out.pitch_track is not None:
        pt = raw_pitch_track(out) if params.voicing_threshold > 0 else out.pitch_track
        return f0_to_line(pt, params)
    return []


def combine_lines(primary: list[MelodyNote], secondary: list[MelodyNote], min_gap: float = 0.3,
                  min_dur: float = 0.06) -> list[MelodyNote]:
    """Giữ nguyên `primary`; lấp các khoảng trống dài ít nhất `min_gap` giây bằng các phần của
    `secondary` nằm trong khoảng trống đó (vd RMVPE khi có giọng, model nhạc cụ ở đoạn còn lại)."""
    prim = make_monophonic(primary)
    gaps: list[tuple[float, float]] = []
    t = -math.inf
    for n in prim:
        if n.start - t >= min_gap:
            gaps.append((t, n.start))
        t = max(t, n.end)
    gaps.append((t, math.inf))
    gaps = [g for g in gaps if g[1] - g[0] >= min_gap]
    fill: list[MelodyNote] = []
    for n in make_monophonic(secondary):
        for g0, g1 in gaps:
            s, e = max(n.start, g0), min(n.end, g1)
            if e - s >= min_dur:
                fill.append(replace(n, start=s, end=e))
    return make_monophonic(prim + fill)


def beat_subdivisions(grid: BeatGrid, subdivisions: int = 4) -> np.ndarray:
    """Lưới phách chia đều mỗi phách thành `subdivisions` phần; kéo dài một phách ở hai đầu."""
    b = np.unique(np.asarray(grid.beats, dtype=np.float64))
    if len(b) < 2:
        return b
    step0, step1 = b[1] - b[0], b[-1] - b[-2]
    b = np.concatenate([[b[0] - step0], b, [b[-1] + step1]])
    frac = np.arange(subdivisions) / subdivisions
    pts = (b[:-1, None] + np.diff(b)[:, None] * frac[None, :]).ravel()
    return np.concatenate([pts, b[-1:]])


def quantize_to_beats(notes: list[MelodyNote], grid: BeatGrid, subdivisions: int = 4,
                      max_shift: float = 0.08) -> list[MelodyNote]:
    """Căn đầu và cuối nốt về điểm gần nhất của lưới phách chia nhỏ nếu lệch không quá `max_shift`.
    Nốt bị căn thành độ dài 0 thì kéo cuối tới điểm lưới kế tiếp sau đầu nốt."""
    pts = beat_subdivisions(grid, subdivisions)
    if len(pts) == 0:
        return list(notes)

    def snap(t: float) -> float:
        i = int(np.clip(np.searchsorted(pts, t), 1, len(pts) - 1))
        q = pts[i - 1] if t - pts[i - 1] <= pts[i] - t else pts[i]
        return float(q) if abs(q - t) <= max_shift else t

    out = []
    prev_s = -math.inf
    for n in sorted(notes, key=lambda x: x.start):
        s, e = snap(n.start), snap(n.end)
        if s <= prev_s:          # hai nốt nhanh bị căn về cùng một điểm: giữ đầu gốc để không mất nốt
            s = n.start
        prev_s = s
        if e <= s:
            later = pts[pts > s + 1e-9]
            e = float(later[0]) if len(later) else max(n.end, s + 1e-3)
        out.append(replace(n, start=s, end=e))
    return make_monophonic(out)


def voice_intervals(notes: list[MelodyNote], pad: float = 0.2, chorus_as_voice: bool = True) -> list[tuple[float, float]]:
    """Các khoảng có nốt giọng hát (theo nhãn nguồn), nới mỗi bên `pad` giây và gộp chỗ chồng nhau."""
    iv = sorted((n.start - pad, n.end + pad) for n in notes
                if source_class(n.source, chorus_as_voice) == VOICE)
    out: list[tuple[float, float]] = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((max(a, 0.0), b))
    return out


def clip_to(notes: list[MelodyNote], intervals: list[tuple[float, float]], inside: bool = True,
            min_dur: float = 0.06) -> list[MelodyNote]:
    """Giữ phần các nốt nằm trong (`inside`) hoặc ngoài các khoảng (đã sắp, không chồng nhau)."""
    if not inside:
        comp, t = [], -math.inf
        for a, b in intervals:
            if a > t:
                comp.append((t, a))
            t = max(t, b)
        comp.append((t, math.inf))
        intervals = comp
    out = []
    for n in notes:
        for a, b in intervals:
            s, e = max(n.start, a), min(n.end, b)
            if e - s >= min_dur:
                out.append(replace(n, start=s, end=e))
    return out


def combine_gated(primary: list[MelodyNote], secondary: list[MelodyNote], gate: list[tuple[float, float]],
                  min_dur: float = 0.06) -> list[MelodyNote]:
    """`primary` trong các khoảng `gate` (vd RMVPE ở chỗ model nốt thấy giọng hát), `secondary` ở phần còn lại."""
    return make_monophonic(clip_to(primary, gate, True, min_dur) + clip_to(secondary, gate, False, min_dur), min_dur)
