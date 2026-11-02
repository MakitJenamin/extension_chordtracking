"""CLI: python -m chordtrack.cli <videoId|url|file> [-o out.json] [--force] [--separation]

Mặc định không tách nguồn (mọi model chạy trên mix); `--separation` bật BS-Roformer + HTDemucs.
`--no-separation` vẫn nhận để tương thích (giờ là mặc định).

In bảng tóm tắt (giọng, các vùng giọng và chỗ đổi giọng kèm gợi ý transpose phím trắng, tempo,
20 hợp âm đầu, các đoạn, dòng hợp âm phím trắng, thời gian từng bước) và ghi JSON.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from chordtrack.pipeline import STEM_NAMES, analyze

N_CHORDS_SHOWN = 20


def _fmt_time(t: float) -> str:
    cs = int(round(float(t) * 100))  # làm tròn trước: 59.999 -> 1:00.00, không phải 0:60.00
    m, cs = divmod(cs, 6000)
    return f"{m}:{cs // 100:02d}.{cs % 100:02d}"


def _fmt_semitones(k: int) -> str:
    return f"{k:+d}" if k else "0"


def white_key_lines(result: dict, per_line: int = 16) -> list[str]:
    """Chuỗi hợp âm phím trắng, bỏ hợp âm trùng liên tiếp và N.C., xuống dòng ở mỗi vùng giọng."""
    keys = result.get("keys") or []
    lines: list[str] = []
    cur_idx, row = None, []

    def flush() -> None:
        if row:
            lines.append("    " + " ".join(row))

    for c in result["chords"]:
        sym, idx = c.get("white_key_symbol"), c.get("key_index")
        if sym is None or sym == "N.C.":
            continue
        if idx != cur_idx:
            flush()
            row = []
            cur_idx = idx
            if idx is not None:
                k = keys[idx]
                lines.append(f"  {_fmt_time(k['start'])} {k['tonic']} {k['mode']} "
                             f"[transpose {_fmt_semitones(k['white_key_transpose'])}]")
        if row and row[-1] == sym:
            continue
        if len(row) >= per_line:
            flush()
            row = []
        row.append(sym)
    flush()
    return lines


def summarize(result: dict, n_chords: int = N_CHORDS_SHOWN) -> str:
    lines = []
    src = result.get("source") or {}
    title = src.get("title") or result["song_id"]
    lines.append(f"Bài: {title}  [{result['song_id']}]  dài {_fmt_time(result['duration'])}")
    key = result.get("key")
    key_s = f"{key['tonic']} {key['mode']}" + (f" (tin cậy {key['confidence']:.2f})" if key.get("confidence") is not None else "") if key else "chưa có"
    tempo = result.get("tempo")
    lines.append(f"Giọng: {key_s}   Tempo: {tempo if tempo is not None else '?'} BPM   "
                 f"Phách: {len(result['beats'])}   Phách mạnh: {len(result['downbeats'])}")

    keys = result.get("keys") or []
    if keys:
        lines.append("")
        lines.append("Các vùng giọng (transpose đàn để chơi phím trắng: trưởng như C, thứ như A):")
        for k in keys:
            conf = "" if k.get("confidence") is None else f"  tin cậy {k['confidence']:.2f}"
            lines.append(f"  {_fmt_time(k['start']):>8}–{_fmt_time(k['end']):<8} {k['tonic']} {k['mode']:<6} "
                         f"transpose {_fmt_semitones(k['white_key_transpose'])}{conf}")
        mods = result.get("modulations") or []
        if mods:
            lines.append("Chỗ đổi giọng:")
            for m in mods:
                lines.append(f"  {_fmt_time(m['time']):>8}  {m['from']} -> {m['to']} ({_fmt_semitones(m['semitones'])} nửa cung): "
                             f"đặt transpose đàn: {_fmt_semitones(m['transpose_to'])}"
                             f" (đang {_fmt_semitones(m['transpose_from'])})")

    chords = result["chords"]
    lines.append("")
    lines.append(f"{min(n_chords, len(chords))} hợp âm đầu (trên tổng {len(chords)}):")
    lines.append(f"  {'bắt đầu':>8}  {'kết thúc':>8}  {'hợp âm':<9} {'bậc':<6} {'phím trắng':<10} {'tin cậy':>7}  nhãn Harte")
    for c in chords[:n_chords]:
        conf = "" if c["confidence"] is None else f"{c['confidence']:.2f}"
        flag = " ?" if c["uncertain"] else ""
        lines.append(f"  {_fmt_time(c['start']):>8}  {_fmt_time(c['end']):>8}  {c['symbol']:<9} "
                     f"{(c['roman'] or ''):<6} {(c.get('white_key_symbol') or ''):<10} {conf:>7}{flag:<2} {c['label']}")
    if keys:
        lines.append("")
        lines.append("Hợp âm phím trắng (theo từng vùng giọng, [transpose] ở đầu mỗi vùng):")
        lines.extend(white_key_lines(result))

    lines.append("")
    if result["sections"]:
        lines.append("Các đoạn:")
        for s in result["sections"]:
            prog = " – ".join(s["progression"])
            roman = f"  ({'–'.join(s['roman'])})" if s.get("roman") else ""
            extra = ""
            if s.get("key_index") is not None and keys:
                k = keys[s["key_index"]]
                extra = f"  [{k['tonic']} {k['mode']}]"
            if s.get("transposed_from") is not None:
                extra += f"  = đoạn {s['transposed_from'] + 1} dịch {_fmt_semitones(s['semitones'])}"
            lines.append(f"  {_fmt_time(s['start']):>8}–{_fmt_time(s['end']):<8} {s['label']:<8} {prog}{roman}{extra}")
    else:
        lines.append("Các đoạn: chưa có")

    lines.append("")
    timing = result["timing"]
    lines.append("Thời gian (giây): " + ", ".join(f"{k} {v:.2f}" for k, v in timing.items()))
    vram = (result.get("pipeline") or {}).get("peak_vram_mb") or {}
    if any(v is not None for v in vram.values()):
        lines.append("VRAM đỉnh (MiB): " + ", ".join(f"{k} {v}" for k, v in vram.items() if v is not None))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m chordtrack.cli", description="Nhận diện hợp âm cho một bài hát.")
    parser.add_argument("source", help="videoId, URL YouTube hoặc đường dẫn file audio")
    parser.add_argument("-o", "--output", type=Path, help="ghi JSON ra file này")
    parser.add_argument("--force", action="store_true", help="bỏ qua kết quả đã cache, chạy lại")
    sep = parser.add_mutually_exclusive_group()
    sep.add_argument("--separation", dest="separation", action="store_true",
                     help="bật tách nguồn (BS-Roformer + HTDemucs), trộn stem bass vào phần gán bass")
    sep.add_argument("--no-separation", dest="separation", action="store_false",
                     help="không tách nguồn (mặc định, giữ để tương thích)")
    parser.set_defaults(separation=False)
    parser.add_argument("--chord-stem", default="mix", choices=STEM_NAMES, help="stem chạy model hợp âm")
    parser.add_argument("--beat-stem", default="mix", choices=STEM_NAMES, help="stem chạy bắt nhịp")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    if not args.separation and (args.chord_stem != "mix" or args.beat_stem != "mix"):
        parser.error("--chord-stem/--beat-stem khác mix cần --separation (mặc định không tách nguồn)")

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    result = analyze(args.source, force=args.force, use_separation=args.separation,
                     chord_stem=args.chord_stem, beat_stem=args.beat_stem)
    print(summarize(result))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nĐã ghi {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
