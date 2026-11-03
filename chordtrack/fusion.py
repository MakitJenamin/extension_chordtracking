"""Bước 6–8: kết hợp nhiều model hợp âm, hậu xử lý và gán nốt bass (hợp âm đảo).

Luồng xử lý của `fuse()`:

1. **Lưới khung chung** (`FRAME_HOP` ≈ 0.093 s, bằng khung của ChordMini) phủ [0, duration].
   Xác suất của từng model được lấy trung bình về lưới này (`resample_probs`).
2. **Kết hợp model hợp âm:** 301 nhãn của CNN-LSTM được ánh xạ về 170 nhãn của ChordMini
   (`cnnlstm_to_chordmini`: bỏ nốt bass, 9/11/13 → 7, maj9 → maj7...), rồi trộn với ChordMini
   bằng trung bình log có trọng số `cnn_weight` (0 = chỉ dùng ChordMini). `mix_level="family"`
   trộn riêng phân bố họ (nốt gốc, họ maj/min/dim/aug/sus) với trọng số `cnn_weight` và phân bố
   loại hợp âm trong họ (C so với Cmaj7, C7) với trọng số `quality_weight`.
3. **Làm mượt** (`smoothing`):
   - `"frame"`: argmax từng khung;
   - `"beat"`: bỏ phiếu theo phách (trung bình log-xác suất trong mỗi phách, lấy argmax);
   - `"hmm"`: Viterbi theo khung, xác suất giữ nguyên hợp âm `self_prob`;
   - `"beat_hmm"`: Viterbi trên quan sát theo phách (hợp âm chỉ đổi ở phách).
   Sau đó gộp đoạn ngắn hơn `min_duration` vào đoạn bên cạnh, thay N.C. giữa bài kém tin cậy bằng hợp âm
   tốt thứ hai (`replace_inner_nc`) và gộp đoạn vừa ngắn vừa kém tin cậy (`merge_weak`).
4. **Gán bass:** phân bố nốt bass (12 nốt) trong mỗi đoạn lấy từ `bass_probs` của CNN-LSTM, từ
   chroma dải thấp của stem bass, hoặc trộn cả hai (`bass_source`, `stem_weight`). Chỉ thêm `/x`
   khi nốt bass nổi bật nhất thuộc hợp âm, khác nốt gốc và có xác suất ≥ `bass_threshold`.
   Nhãn ghi theo Harte: `C:maj/5`, `A:min/b3`.
5. **Độ tin cậy:** trung bình xác suất (đã kết hợp) của nhãn được chọn trong đoạn, khoảng [0, 1].

Giá trị mặc định của `FusionParams` là bộ tốt nhất đo bằng `eval/refit_fusion.py` trên cả Beatles và
Kajiura, chỉ dùng mix, không tách nguồn (xem PLAN.md, mục "Chọn lại tham số fuse trên Beatles + Kajiura").
Bộ cũ (chọn chỉ trên Beatles) nằm ở `eval.ablation_kajiura.OLD_DEFAULTS`.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from chordtrack.types import BeatGrid, ChordModelOutput, ChordSegment

FRAME_HOP = 2048 / 22050          # khung của ChordMini, khoảng 0.0929 s
EPS = 1e-6

# Bậc (Harte) của nốt bass theo số nửa cung tính từ nốt gốc. 8 là #5 (hợp âm aug), 9 là 6 (maj6, dim7).
SEMITONE_TO_DEGREE = {1: "b2", 2: "2", 3: "b3", 4: "3", 5: "4", 6: "b5", 7: "5", 8: "#5", 9: "6", 10: "b7", 11: "7"}

# Rút gọn loại hợp âm của CNN-LSTM về loại có trong từ vựng ChordMini.
QUALITY_REDUCTION = {"maj9": "maj7", "min9": "min7", "9": "7", "11": "7", "13": "7",
                     "min11": "min7", "min13": "min7", "maj11": "maj7", "maj13": "maj7"}


@dataclass(frozen=True)
class FusionParams:
    cnn_weight: float = 0.6          # trọng số CNN-LSTM khi trộn xác suất hợp âm (0 = chỉ ChordMini)
    mix_level: str = "family"        # "full": trộn trên cả 170 nhãn; "family": trộn riêng phân bố họ (gốc, họ
                                     # maj/min/dim/aug/sus) và phân bố loại hợp âm 7, 6 trong họ
    quality_weight: float = 1.0      # "family": trọng số CNN-LSTM khi trộn phân bố loại hợp âm trong họ
                                     # (0 = giữ loại hợp âm của ChordMini, 1 = lấy của CNN-LSTM; loại mà
                                     # CNN-LSTM không có như maj6, min6 vẫn theo ChordMini)
    smoothing: str = "hmm"           # "frame" | "beat" | "hmm" | "beat_hmm"
    self_prob: float = 0.7           # HMM: xác suất giữ nguyên hợp âm sang khung/phách sau
    min_duration: float = 0.25       # gộp đoạn ngắn hơn (giây)
    bass_source: str = "cnn"         # "none" | "cnn" | "stem" | "both" (stem cần tách nguồn, tắt mặc định)
    stem_weight: float = 0.3         # "both": trọng số của chroma stem bass
    bass_threshold: float = 0.5      # xác suất tối thiểu của nốt bass để thêm /x
    bass_per_beat: bool = False      # True: quyết định bass theo từng phách, đổi bass thì tách đoạn
    nc_min_prob: float = 0.6         # N.C. giữa bài chỉ giữ khi P(N) trung bình của đoạn ≥ mức này (0 = tắt)
    nc_min_duration: float = 4.0     # ... và đoạn dài ít nhất chừng này giây; đầu/cuối bài luôn giữ
    weak_max_beats: float = 1.5      # gộp đoạn ngắn hơn chừng này phách (0 = tắt) ...
    weak_max_conf: float = 0.25      # ... và có confidence nhỏ hơn mức này vào đoạn bên cạnh


DEFAULT_PARAMS = FusionParams()


def fuse(
    models: dict[str, ChordModelOutput],
    beats: BeatGrid | None,
    bass_audio: Path | None,
    duration: float,
    params: FusionParams | None = None,
) -> list[ChordSegment]:
    """Trả về chuỗi hợp âm phủ liên tục từ 0 tới `duration`, mỗi đoạn có confidence.

    models: các khóa đã biết là "chordmini" (BTC-PL) và "cnnlstm"; có thể thiếu khóa.
    beats: lưới phách để bỏ phiếu theo phách; None nếu không có.
    bass_audio: file WAV chỉ có bass (Stems.bass); None nếu không có.
    """
    p = params or DEFAULT_PARAMS
    if not models or duration <= 0:
        return [ChordSegment(0.0, max(duration, 0.0), "N", None)] if duration > 0 else []
    if not any(m.frame_probs is not None for m in models.values()):
        # Không có xác suất theo khung: chỉ lấp khoảng trống cho đoạn của model chính.
        base = models["chordmini"] if "chordmini" in models else next(iter(models.values()))
        return fill_gaps(base.segments, duration)

    times = frame_grid(duration)
    probs, vocab = chord_probs(models, times, p.cnn_weight, p.mix_level, p.quality_weight)
    stem_chroma = None
    if bass_audio is not None and p.bass_source in ("stem", "both"):
        stem_chroma = resample_probs(*bass_chroma(Path(bass_audio)), times)
    cnn_bass = None
    if "cnnlstm" in models and "bass_probs" in models["cnnlstm"].extras and p.bass_source in ("cnn", "both"):
        cnn_bass = cnnlstm_bass(models["cnnlstm"], times)
    return fuse_probs(probs, vocab, times, beats, duration, p, cnn_bass=cnn_bass, stem_chroma=stem_chroma)


def fuse_probs(
    probs: np.ndarray,
    vocab: list[str],
    times: np.ndarray,
    beats: BeatGrid | None,
    duration: float,
    p: FusionParams,
    cnn_bass: np.ndarray | None = None,
    stem_chroma: np.ndarray | None = None,
) -> list[ChordSegment]:
    """Phần lõi của `fuse` trên dữ liệu đã đưa về lưới khung chung (dùng lại trong eval/ablation.py)."""
    labels = decode(probs, times, beats, duration, p)
    segs = frames_to_segments(labels, times, duration)
    segs = merge_short(segs, probs, times, p.min_duration)
    replaced: set[tuple[float, float]] = set()
    if p.nc_min_prob > 0:
        segs, replaced = replace_inner_nc(segs, probs, vocab, times, duration, p.nc_min_prob, p.nc_min_duration)
    if p.weak_max_beats > 0:
        segs = merge_weak(segs, probs, times, beats, p.weak_max_beats, p.weak_max_conf)
    bass = bass_distribution(cnn_bass, stem_chroma, p)
    if bass is not None:
        intervals = _beat_intervals(beats, duration) if p.bass_per_beat else None
        labeled = assign_bass(segs, vocab, bass, times, p.bass_threshold, intervals)
    else:
        labeled = [(a, b, idx, vocab[idx]) for a, b, idx in segs]
    out = []
    for a, b, idx, lab in labeled:
        conf = _mean_prob(probs, times, a, b, idx)
        if any(ra <= a + 1e-9 and b <= rb + 1e-9 for ra, rb in replaced):
            conf = min(conf, REPLACED_NC_CONF)
        out.append(ChordSegment(a, b, lab, conf))
    return _merge_same(out)


# ---------------------------------------------------------------- lưới khung và xác suất

def frame_grid(duration: float, hop: float = FRAME_HOP) -> np.ndarray:
    """Tâm các khung [k*hop, (k+1)*hop) phủ [0, duration]."""
    n = max(1, int(np.ceil(duration / hop)))
    return (np.arange(n) + 0.5) * hop


def resample_probs(src_times: np.ndarray, src: np.ndarray, dst_times: np.ndarray) -> np.ndarray:
    """Trung bình các khung nguồn có tâm nằm trong mỗi khung đích; khung đích không có khung
    nguồn nào thì lấy khung nguồn gần nhất."""
    src_times = np.asarray(src_times, dtype=np.float64)
    src = np.asarray(src, dtype=np.float64)
    dst_times = np.asarray(dst_times, dtype=np.float64)
    if len(dst_times) > 1:
        half = np.diff(dst_times) / 2
        edges = np.concatenate([[dst_times[0] - half[0]], dst_times[:-1] + half, [dst_times[-1] + half[-1]]])
    else:
        edges = np.array([-np.inf, np.inf])
    lo = np.searchsorted(src_times, edges[:-1], side="left")
    hi = np.searchsorted(src_times, edges[1:], side="left")
    csum = np.concatenate([np.zeros((1, src.shape[1])), np.cumsum(src, axis=0)])
    count = (hi - lo)[:, None]
    out = np.empty((len(dst_times), src.shape[1]))
    has = count[:, 0] > 0
    out[has] = (csum[hi[has]] - csum[lo[has]]) / count[has]
    if (~has).any():
        nearest = np.clip(np.searchsorted(src_times, dst_times[~has]), 0, len(src_times) - 1)
        prev = np.clip(nearest - 1, 0, len(src_times) - 1)
        use_prev = np.abs(src_times[prev] - dst_times[~has]) < np.abs(src_times[nearest] - dst_times[~has])
        out[~has] = src[np.where(use_prev, prev, nearest)]
    return out


def _normalize(x: np.ndarray) -> np.ndarray:
    s = x.sum(axis=1, keepdims=True)
    return np.where(s > 0, x / np.maximum(s, 1e-12), 1.0 / x.shape[1])


def _parse(label: str) -> tuple[int, str, str | None]:
    """Nhãn Harte -> (nốt gốc 0..11, loại, bậc bass hoặc None). N/X trả về (-1, label, None)."""
    if label in ("N", "X"):
        return -1, label, None
    import mir_eval

    root, quality, _ext, bass = mir_eval.chord.split(label, reduce_extended_chords=False)
    return mir_eval.chord.pitch_class_to_semitone(root), quality, (bass if bass and bass != "1" else None)


@lru_cache(maxsize=8)
def _mapping_matrix(src_vocab: tuple[str, ...], dst_vocab: tuple[str, ...]) -> np.ndarray:
    dst_index = {}
    for j, lab in enumerate(dst_vocab):
        r, q, _ = _parse(lab)
        dst_index[(r, q)] = j
    m = np.zeros((len(src_vocab), len(dst_vocab)))
    for i, lab in enumerate(src_vocab):
        r, q, _ = _parse(lab)
        q = QUALITY_REDUCTION.get(q, q)
        j = dst_index.get((r, q))
        if j is None and r >= 0:
            j = dst_index.get((r, _triad_family(lab)))
        if j is None:
            j = dst_index.get((-1, "N"), 0)
        m[i, j] = 1.0
    return m


def _triad_family(label: str) -> str:
    """Họ bộ ba của nhãn lạ, theo các nốt: có b3 mà không có 3 -> min (b5 nữa thì dim), còn lại maj."""
    import mir_eval

    _root, bitmap, _bass = mir_eval.chord.encode(label)
    if bitmap[3] and not bitmap[4]:
        return "dim" if bitmap[6] and not bitmap[7] else "min"
    return "maj"


def cnnlstm_to_chordmini(probs: np.ndarray, src_vocab: list[str], dst_vocab: list[str]) -> np.ndarray:
    """Cộng xác suất của các nhãn CNN-LSTM vào nhãn ChordMini tương ứng (bỏ bass, rút gọn loại)."""
    return np.asarray(probs, dtype=np.float64) @ _mapping_matrix(tuple(src_vocab), tuple(dst_vocab))


def chord_probs(models: dict[str, ChordModelOutput], times: np.ndarray, cnn_weight: float,
                mix_level: str = "full", quality_weight: float = 0.0) -> tuple[np.ndarray, list[str]]:
    """Xác suất hợp âm đã kết hợp (T, V) trên lưới `times`, cùng từ vựng V.

    Có ChordMini: từ vựng ChordMini, trộn log với CNN-LSTM đã ánh xạ. Chỉ có CNN-LSTM: dùng thẳng
    từ vựng của nó (đã có hợp âm đảo)."""
    cm = models.get("chordmini")
    cn = models.get("cnnlstm")
    if cm is None or cm.frame_probs is None:
        m = cn if cn is not None and cn.frame_probs is not None else next(m for m in models.values() if m.frame_probs is not None)
        return _normalize(resample_probs(m.frame_times, m.frame_probs, times)), list(m.vocabulary)
    p_cm = _normalize(resample_probs(cm.frame_times, cm.frame_probs, times))
    if cn is None or cn.frame_probs is None or (cnn_weight <= 0 and (mix_level != "family" or quality_weight <= 0)):
        return p_cm, list(cm.vocabulary)
    p_cn = cnnlstm_on_chordmini(cn, cm, times, p_cm)
    abstain = cnnlstm_abstain(list(cn.vocabulary), list(cm.vocabulary))
    return (mix_models(p_cm, p_cn, list(cm.vocabulary), cnn_weight, mix_level, quality_weight, abstain),
            list(cm.vocabulary))


def cnnlstm_abstain(cn_vocab: list[str], cm_vocab: list[str]) -> np.ndarray:
    """Mặt nạ (V,) các nhãn ChordMini mà CNN-LSTM không đoán được (maj6, min6, minmaj7, X...)."""
    return _mapping_matrix(tuple(cn_vocab), tuple(cm_vocab)).sum(axis=0) == 0


def cnnlstm_on_chordmini(cn: ChordModelOutput, cm: ChordModelOutput, times: np.ndarray, p_cm: np.ndarray) -> np.ndarray:
    """Xác suất CNN-LSTM trên từ vựng ChordMini, trên lưới `times`.

    Nhãn ChordMini mà CNN-LSTM không có (maj6, min6, minmaj7, X với từ điển "submission") thì CNN-LSTM
    "không có ý kiến": cột đó lấy đúng xác suất của ChordMini, các cột còn lại co lại cho tổng bằng 1.
    Nếu không, log(0) khi trộn sẽ loại hẳn các nhãn đó."""
    m = _mapping_matrix(tuple(cn.vocabulary), tuple(cm.vocabulary))
    covered = m.sum(axis=0) > 0
    p_cn = _normalize(resample_probs(cn.frame_times, np.asarray(cn.frame_probs, dtype=np.float64) @ m, times))
    if (~covered).any():
        rest = p_cm[:, ~covered].sum(axis=1, keepdims=True)
        p_cn = p_cn * (1 - rest)
        p_cn[:, ~covered] = p_cm[:, ~covered]
    return p_cn


def cnnlstm_bass(cn: ChordModelOutput, times: np.ndarray) -> np.ndarray:
    """Xác suất 12 nốt bass của CNN-LSTM trên lưới `times`. Bỏ cột "không có bass" nhưng **không**
    chuẩn hóa lại: tổng mỗi hàng là 1 − P(không có bass), nên ngưỡng bass vẫn lọc được khung không chắc."""
    return resample_probs(cn.frame_times, np.asarray(cn.extras["bass_probs"], dtype=np.float64)[:, 1:], times)


# Họ hợp âm (theo bộ ba nốt): loại hợp âm -> họ. Loại không có trong bảng là họ của chính nó.
FAMILY = {"maj": "maj", "maj6": "maj", "maj7": "maj", "7": "maj",
          "min": "min", "min6": "min", "min7": "min", "minmaj7": "min",
          "dim": "dim", "dim7": "dim", "hdim7": "dim"}


@lru_cache(maxsize=8)
def _family_matrix(vocab: tuple[str, ...]) -> np.ndarray:
    """Ma trận (V, số họ) gán mỗi nhãn vào họ (gốc, họ)."""
    keys: dict[tuple, int] = {}
    rows = []
    for lab in vocab:
        r, q, _ = _parse(lab)
        rows.append(keys.setdefault((r, FAMILY.get(q, q)), len(keys)))
    m = np.zeros((len(vocab), len(keys)))
    m[np.arange(len(vocab)), rows] = 1.0
    return m


def mix_models(p_cm: np.ndarray, p_cn: np.ndarray, vocab: list[str], w: float, level: str = "full",
               quality_weight: float = 0.0, abstain: np.ndarray | None = None) -> np.ndarray:
    """Trộn xác suất ChordMini với CNN-LSTM (đã ánh xạ về cùng từ vựng), trọng số `w` cho CNN-LSTM.

    `level="family"`: P(nhãn) = P(họ) · P(nhãn | họ). P(họ) trộn log với trọng số `w`; P(nhãn | họ)
    (C so với Cmaj7, C7) trộn log với trọng số `quality_weight` rồi chuẩn hóa lại trong từng họ.
    `abstain` (V,): nhãn CNN-LSTM không có ý kiến; trong họ, các nhãn này giữ đúng tỉ lệ của ChordMini và
    CNN-LSTM chỉ chia phần còn lại giữa các nhãn nó đoán được."""
    if level == "full":
        return mix_log(p_cm, p_cn, w)
    if level != "family":
        raise ValueError(f"mix_level không hợp lệ: {level!r}")
    a = _family_matrix(tuple(vocab))
    fam_cm, fam_cn = p_cm @ a, p_cn @ a
    fam = mix_log(fam_cm, fam_cn, w)
    within = p_cm / np.maximum(fam_cm @ a.T, 1e-12)   # P(nhãn | họ) theo ChordMini
    if quality_weight > 0:
        if abstain is None:
            within_cn = p_cn / np.maximum(fam_cn @ a.T, 1e-12)
        else:
            cov = ~np.asarray(abstain, dtype=bool)
            cov_mass = ((p_cn * cov) @ a) @ a.T                # khối CNN-LSTM của các nhãn nó đoán được, theo họ
            free = 1.0 - ((within * ~cov) @ a) @ a.T           # phần trong họ không thuộc nhãn "không ý kiến"
            within_cn = np.where(cov, p_cn / np.maximum(cov_mass, 1e-12) * free, within)
        within = np.exp((1 - quality_weight) * np.log(within + EPS) + quality_weight * np.log(within_cn + EPS))
        within = within / np.maximum((within @ a) @ a.T, 1e-300)
    return _normalize(within * (fam @ a.T))


def mix_log(a: np.ndarray, b: np.ndarray, w_b: float) -> np.ndarray:
    """Trung bình hình học có trọng số (trung bình log), chuẩn hóa lại theo từng khung."""
    logp = (1 - w_b) * np.log(a + EPS) + w_b * np.log(b + EPS)
    logp -= logp.max(axis=1, keepdims=True)
    return _normalize(np.exp(logp))


# ---------------------------------------------------------------- làm mượt

def _beat_intervals(beats: BeatGrid | None, duration: float) -> list[tuple[float, float]] | None:
    if beats is None or len(beats.beats) == 0:
        return None
    from chordtrack.beats import beat_synchronous_segments

    intervals = beat_synchronous_segments(beats, duration)
    # Không có phách nào nằm trong bài (ví dụ chỉ có phách ở 0): coi như không có phách.
    return intervals if len(intervals) >= 2 else None


def _interval_frames(intervals: list[tuple[float, float]], times: np.ndarray) -> list[np.ndarray]:
    """Chỉ số các khung có tâm nằm trong mỗi khoảng; khoảng không chứa tâm nào thì lấy khung gần nhất."""
    out = []
    for a, b in intervals:
        lo, hi = np.searchsorted(times, a, side="left"), np.searchsorted(times, b, side="left")
        if hi > lo:
            out.append(np.arange(lo, hi))
        else:
            out.append(np.array([int(np.clip(np.searchsorted(times, (a + b) / 2), 0, len(times) - 1))]))
    return out


def viterbi_uniform(logobs: np.ndarray, self_prob: float) -> np.ndarray:
    """Viterbi với ma trận chuyển: giữ nguyên `self_prob`, đổi sang mỗi trạng thái khác như nhau.
    Chạy O(T·V) nhờ ma trận chuyển đều. Trả về chỉ số trạng thái theo thời gian."""
    T, V = logobs.shape
    if T == 0:
        return np.zeros(0, dtype=np.int64)
    if V == 1:
        return np.zeros(T, dtype=np.int64)
    ls = np.log(self_prob)
    lc = np.log((1 - self_prob) / (V - 1))
    delta = logobs[0].copy()
    back_best = np.zeros(T, dtype=np.int64)     # trạng thái tốt nhất ở bước t-1 (khi đổi)
    stay = np.zeros((T, V), dtype=bool)         # True: ở bước t giữ nguyên trạng thái của bước t-1
    for t in range(1, T):
        best = int(np.argmax(delta))
        change = delta[best] + lc
        keep = delta + ls
        stay[t] = keep >= change
        back_best[t] = best
        delta = np.where(stay[t], keep, change) + logobs[t]
    path = np.empty(T, dtype=np.int64)
    path[-1] = int(np.argmax(delta))
    for t in range(T - 1, 0, -1):
        path[t - 1] = path[t] if stay[t, path[t]] else back_best[t]
    return path


def decode(probs: np.ndarray, times: np.ndarray, beats: BeatGrid | None, duration: float, p: FusionParams) -> np.ndarray:
    """Nhãn (chỉ số từ vựng) cho từng khung theo `p.smoothing`. Thiếu phách thì phương pháp theo
    phách quay về phương pháp theo khung tương ứng."""
    logp = np.log(probs + EPS)
    method = p.smoothing
    intervals = _beat_intervals(beats, duration) if method in ("beat", "beat_hmm") else None
    if method in ("beat", "beat_hmm") and intervals is None:
        method = "frame" if method == "beat" else "hmm"
    if method == "frame":
        return np.argmax(probs, axis=1)
    if method == "hmm":
        return viterbi_uniform(logp, p.self_prob)
    if method not in ("beat", "beat_hmm"):
        raise ValueError(f"smoothing không hợp lệ: {method!r}")
    assert intervals is not None
    groups = _interval_frames(intervals, times)
    obs = np.stack([logp[g].mean(axis=0) for g in groups])
    if method == "beat":
        beat_lab = np.argmax(obs, axis=1)
    else:
        # Quan sát của phách là trung bình log theo khung; nhân với số khung để phách dài nặng hơn.
        weight = np.array([len(g) for g in groups], dtype=np.float64)[:, None]
        beat_lab = viterbi_uniform(obs * np.maximum(weight, 1.0), p.self_prob)
    return beat_labels_to_frames(intervals, beat_lab, times)


def beat_labels_to_frames(intervals: list[tuple[float, float]], beat_lab: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Mỗi khung nhận nhãn của khoảng phách chứa tâm khung."""
    starts = np.array([a for a, _ in intervals])
    idx = np.clip(np.searchsorted(starts, times, side="right") - 1, 0, len(intervals) - 1)
    return np.asarray(beat_lab)[idx]


# ---------------------------------------------------------------- đoạn

Seg = tuple[float, float, int, str]   # (bắt đầu, kết thúc, chỉ số nhãn, nhãn Harte)


def frames_to_segments(labels: np.ndarray, times: np.ndarray, duration: float) -> list[tuple[float, float, int]]:
    """Gộp khung liên tiếp cùng nhãn; biên đoạn là biên khung, đoạn cuối kéo tới `duration`."""
    if len(labels) == 0:
        return []
    hop = times[1] - times[0] if len(times) > 1 else duration
    change = np.nonzero(np.diff(labels))[0] + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(labels)]])
    out = []
    for s, e in zip(starts, ends):
        a = 0.0 if s == 0 else float(times[s] - hop / 2)
        b = float(duration) if e == len(labels) else float(times[e] - hop / 2)
        if b > a:
            out.append((a, b, int(labels[s])))
    if out:
        out[-1] = (out[-1][0], float(duration), out[-1][2])
    return out


def _mean_prob(probs: np.ndarray, times: np.ndarray, a: float, b: float, idx: int) -> float:
    lo, hi = np.searchsorted(times, a), np.searchsorted(times, b)
    if hi <= lo:
        hi = min(lo + 1, len(times))
        lo = max(0, hi - 1)
    return float(np.clip(probs[lo:hi, idx].mean(), 0.0, 1.0))


def merge_short(segs: list[tuple[float, float, int]], probs: np.ndarray, times: np.ndarray, min_duration: float) -> list[tuple[float, float, int]]:
    """Lần lượt gộp đoạn ngắn nhất (< min_duration) vào đoạn bên cạnh mà nhãn của nó có xác suất
    trung bình cao hơn trên chính đoạn ngắn đó."""
    segs = list(segs)
    while len(segs) > 1:
        lens = [b - a for a, b, _ in segs]
        i = int(np.argmin(lens))
        if lens[i] >= min_duration:
            break
        a, b, _ = segs[i]
        cands = []
        if i > 0:
            cands.append((_mean_prob(probs, times, a, b, segs[i - 1][2]), i - 1))
        if i + 1 < len(segs):
            cands.append((_mean_prob(probs, times, a, b, segs[i + 1][2]), i + 1))
        _, j = max(cands)
        lab = segs[j][2]
        segs[i] = (a, b, lab)
        # gộp các đoạn liền nhau cùng nhãn
        merged: list[tuple[float, float, int]] = []
        for s in segs:
            if merged and merged[-1][2] == s[2]:
                merged[-1] = (merged[-1][0], s[1], s[2])
            else:
                merged.append(s)
        segs = merged
    return segs


# Hợp âm thay cho N.C. giữa bài luôn có confidence dưới mức này, để pipeline đánh dấu "uncertain"
# (pipeline.UNCERTAIN_THRESHOLD = 0.5).
REPLACED_NC_CONF = 0.49


def _merge_adjacent(segs: list[tuple[float, float, int]]) -> list[tuple[float, float, int]]:
    out: list[tuple[float, float, int]] = []
    for s in segs:
        if out and out[-1][2] == s[2]:
            out[-1] = (out[-1][0], s[1], s[2])
        else:
            out.append(s)
    return out


def replace_inner_nc(segs: list[tuple[float, float, int]], probs: np.ndarray, vocab: list[str], times: np.ndarray,
                     duration: float, min_prob: float, min_duration: float
                     ) -> tuple[list[tuple[float, float, int]], set[tuple[float, float]]]:
    """Đoạn N.C. ở giữa bài (không chạm đầu/cuối) mà P(N) trung bình < `min_prob` hoặc ngắn hơn
    `min_duration` thì lấy nhãn tốt thứ hai: nhãn không phải N/X có xác suất trung bình cao nhất trong đoạn.
    Nhạc vẫn chơi mà model thấy "không có hợp âm" thường là đoạn hòa âm mỏng (chỉ giai điệu, nốt ngân),
    hiện hợp âm gần đúng kèm dấu không chắc có ích hơn một ô N.C. dài.

    Trả về (đoạn mới đã gộp đoạn liền nhau cùng nhãn, tập (đầu, cuối) của các đoạn bị thay)."""
    if "N" not in vocab:
        return segs, set()
    n_idx = vocab.index("N")
    no_chord = [i for i, lab in enumerate(vocab) if lab in ("N", "X")]
    mask = np.ones(len(vocab), dtype=bool)
    mask[no_chord] = False
    out = []
    replaced: set[tuple[float, float]] = set()
    for a, b, idx in segs:
        inner = a > 1e-9 and b < duration - 1e-9
        if idx == n_idx and inner:
            mean = _mean_rows(probs, times, a, b)
            if mean[n_idx] < min_prob or b - a < min_duration:
                alt = int(np.argmax(np.where(mask, mean, -1.0)))
                out.append((a, b, alt))
                replaced.add((a, b))
                continue
        out.append((a, b, idx))
    return _merge_adjacent(out), replaced


def merge_weak(segs: list[tuple[float, float, int]], probs: np.ndarray, times: np.ndarray, beats: BeatGrid | None,
               max_beats: float, max_conf: float) -> list[tuple[float, float, int]]:
    """Gộp đoạn vừa ngắn (< `max_beats` khoảng phách tại đó) vừa kém tin cậy (xác suất trung bình của nhãn
    < `max_conf`) vào đoạn bên cạnh có xác suất trung bình trên đoạn đó cao hơn. Không có phách thì giữ nguyên.
    Gộp lần lượt đoạn yếu nhất trước, giống `merge_short`."""
    if beats is None or len(beats.beats) < 2:
        return segs
    bt = np.asarray(beats.beats, dtype=np.float64)
    ibi = np.diff(bt)

    def beat_len(a: float, b: float) -> float:
        k = int(np.clip(np.searchsorted(bt, (a + b) / 2) - 1, 0, len(ibi) - 1))
        return float(ibi[k])

    segs = list(segs)
    while len(segs) > 1:
        weak = []
        for i, (a, b, idx) in enumerate(segs):
            if b - a < max_beats * beat_len(a, b):
                conf = _mean_prob(probs, times, a, b, idx)
                if conf < max_conf:
                    weak.append((conf, i))
        if not weak:
            break
        _, i = min(weak)
        a, b, _ = segs[i]
        cands = []
        if i > 0:
            cands.append((_mean_prob(probs, times, a, b, segs[i - 1][2]), i - 1))
        if i + 1 < len(segs):
            cands.append((_mean_prob(probs, times, a, b, segs[i + 1][2]), i + 1))
        _, j = max(cands)
        segs[i] = (a, b, segs[j][2])
        segs = _merge_adjacent(segs)
    return segs


def _merge_same(segs: list[ChordSegment]) -> list[ChordSegment]:
    out: list[ChordSegment] = []
    for s in segs:
        if out and out[-1].label == s.label and abs(out[-1].end - s.start) < 1e-9:
            prev = out[-1]
            w1, w2 = prev.end - prev.start, s.end - s.start
            conf = None
            if prev.confidence is not None and s.confidence is not None:
                conf = (prev.confidence * w1 + s.confidence * w2) / (w1 + w2)
            out[-1] = ChordSegment(prev.start, s.end, s.label, conf)
        else:
            out.append(s)
    return out


# ---------------------------------------------------------------- bass

BASS_HARMONICS = ((1, 1.0), (2, 0.5), (3, 0.33), (4, 0.25))   # (bậc bồi âm, trọng số)


def bass_chroma(path: Path, sr: int = 11025, hop: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Độ nổi bật của nốt bass theo 12 nốt, từ stem bass.

    CQT 36 bin/quãng tám từ E1 (41 Hz) lên 5 quãng tám, mỗi nửa cung lấy bin lớn nhất trong 3 bin
    (chịu được lệch chuẩn âm). Độ nổi bật của nốt cơ bản p (E1..D#3) = tổng có trọng số biên độ ở
    các bồi âm 1–4 của nó, để bồi âm bậc 3 (quãng 5) không lấn nốt cơ bản. Trả về (thời điểm khung,
    (T, 12)) chưa chuẩn hóa: tổng mỗi hàng tỉ lệ với năng lượng, khung im lặng có tổng gần 0.

    Lưu ý: stem của HTDemucs chỉ có bass guitar. Đoạn không có bass guitar (ví dụ piano tự đệm)
    thì không có thông tin, khi đó `stem_bass_probs` trả về phân bố đều."""
    import librosa
    import soundfile as sf

    y, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    n_semi = 60
    fmin = librosa.note_to_hz("E1") * 2 ** (-1 / 36)   # bin giữa của mỗi bộ 3 rơi đúng nốt
    c = np.abs(librosa.cqt(y, sr=sr, hop_length=hop, fmin=fmin, n_bins=3 * n_semi, bins_per_octave=36))
    c = c.reshape(n_semi, 3, -1).max(axis=1)           # (60 nửa cung từ E1, T)
    n_fund = 24
    sal = np.zeros((n_fund, c.shape[1]))
    for h, w in BASS_HARMONICS:
        off = int(round(12 * np.log2(h)))
        sal += w * c[off:off + n_fund]
    first_pc = 4  # E
    chroma = np.zeros((12, c.shape[1]))
    for k in range(n_fund):
        chroma[(first_pc + k) % 12] += sal[k]
    times = (np.arange(c.shape[1]) + 0.5) * hop / sr
    return times, chroma.T


def stem_bass_probs(chroma: np.ndarray, power: float = 2.0, silence: float = 0.05) -> np.ndarray:
    """Đổi chroma bass (T, 12) thành phân bố: lũy thừa `power` rồi chuẩn hóa. Khung có năng lượng
    dưới `silence` × trung vị năng lượng của bài thì phân bố đều (không có thông tin)."""
    e = chroma.sum(axis=1)
    ref = np.median(e[e > 0]) if (e > 0).any() else 1.0
    p = _normalize(np.maximum(chroma, 0) ** power)
    p[e < silence * ref] = 1.0 / 12
    return p


def bass_distribution(cnn_bass: np.ndarray | None, stem_chroma: np.ndarray | None, p: FusionParams) -> np.ndarray | None:
    """Phân bố 12 nốt bass theo khung từ các nguồn được chọn. `stem_chroma` là chroma chưa chuẩn hóa."""
    if p.bass_source not in ("none", "cnn", "stem", "both"):
        raise ValueError(f"bass_source không hợp lệ: {p.bass_source!r}")
    stem = stem_bass_probs(stem_chroma) if stem_chroma is not None else None
    if p.bass_source == "none":
        return None
    if p.bass_source == "cnn":
        return cnn_bass
    if p.bass_source == "stem":
        return stem
    if cnn_bass is None:
        return stem
    if stem is None:
        return cnn_bass
    return (1 - p.stem_weight) * cnn_bass + p.stem_weight * stem


def chord_tones(label: str) -> tuple[int, set[int]]:
    """(nốt gốc, tập số nửa cung các nốt thuộc hợp âm tính từ nốt gốc). N/X: (-1, ∅)."""
    if label in ("N", "X"):
        return -1, set()
    import mir_eval

    root, bitmap, _ = mir_eval.chord.encode(label)
    return int(root), {int(i) for i in np.nonzero(bitmap)[0]}


def with_bass(label: str, bass_pc: int) -> str:
    """Thêm bậc bass vào nhãn nếu nốt bass thuộc hợp âm và khác nốt gốc; ngược lại giữ nguyên."""
    if "/" in label:
        return label
    root, tones = chord_tones(label)
    if root < 0:
        return label
    iv = (bass_pc - root) % 12
    if iv == 0 or iv not in tones:
        return label
    return f"{label}/{SEMITONE_TO_DEGREE[iv]}"


def pick_bass(dist: np.ndarray, label: str, threshold: float) -> str:
    """Áp quy tắc gán bass cho một phân bố 12 nốt (đã lấy trung bình trên đoạn)."""
    pc = int(np.argmax(dist))
    if dist[pc] < threshold:
        return label
    return with_bass(label, pc)


def assign_bass(
    segs: list[tuple[float, float, int]],
    vocab: list[str],
    bass: np.ndarray,
    times: np.ndarray,
    threshold: float,
    beat_intervals: list[tuple[float, float]] | None = None,
) -> list[Seg]:
    """Gán bass cho từng đoạn. Có `beat_intervals` thì quyết định theo từng phách trong đoạn và
    tách đoạn khi bass đổi (ví dụ C giữ hai ô nhịp, bass đi C → E thành C rồi C/E)."""
    out: list[Seg] = []
    for a, b, idx in segs:
        label = vocab[idx]
        if beat_intervals is None:
            out.append((a, b, idx, pick_bass(_mean_rows(bass, times, a, b), label, threshold)))
            continue
        cuts = [a] + [s for s, _ in beat_intervals if a < s < b] + [b]
        for s, e in zip(cuts[:-1], cuts[1:]):
            lab = pick_bass(_mean_rows(bass, times, s, e), label, threshold)
            if out and out[-1][3] == lab and out[-1][2] == idx and abs(out[-1][1] - s) < 1e-9:
                out[-1] = (out[-1][0], e, idx, lab)
            else:
                out.append((s, e, idx, lab))
    return out


def _mean_rows(x: np.ndarray, times: np.ndarray, a: float, b: float) -> np.ndarray:
    lo, hi = np.searchsorted(times, a), np.searchsorted(times, b)
    if hi <= lo:
        hi = min(lo + 1, len(times))
        lo = max(0, hi - 1)
    return x[lo:hi].mean(axis=0)


# ---------------------------------------------------------------- lấp khoảng trống

def fill_gaps(segments: list[ChordSegment], duration: float) -> list[ChordSegment]:
    """Lấp khoảng trống: kéo dài đoạn trước tới đầu đoạn sau, thêm "N" ở đầu và cuối nếu thiếu.
    Đoạn vượt quá `duration` bị cắt."""
    out: list[ChordSegment] = []
    t = 0.0
    for s in sorted(segments, key=lambda x: x.start):
        end = min(s.end, duration)
        if end <= t:
            continue
        start = max(s.start, t)
        if start > t:
            if out:
                out[-1] = ChordSegment(out[-1].start, start, out[-1].label, out[-1].confidence)
            else:
                out.append(ChordSegment(0.0, start, "N", None))
        out.append(ChordSegment(start, end, s.label, s.confidence))
        t = end
    if t < duration:
        out.append(ChordSegment(t, duration, "N", None))
    return out
