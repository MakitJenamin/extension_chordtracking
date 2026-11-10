# Đánh giá dò giai điệu

Hợp đồng chung giữa phần dữ liệu, phần model và phần chấm điểm.

## Dữ liệu: `data/eval/melody/<dataset>/manifest.jsonl`

Mỗi dòng một bài:

```json
{"track_id": "...", "dataset": "mir_st500", "audio": "<đường dẫn WAV tuyệt đối>",
 "ref_notes": "<csv: onset,offset,midi>" | null, "ref_f0": "<csv: time,hz (0 = không có giai điệu)>" | null,
 "melody_type": "vocal" | "instrumental" | "mixed"}
```

Bộ không có đáp án (Kajiura) để `ref_notes` và `ref_f0` là null, chỉ chấm các số không cần đáp án.
Audio và đáp án không commit (repo public). Script tải và chuẩn bị nằm trong `eval/melody/`.

## Kết quả model: cache `.npz`

`chordtrack.melody.io.cache_path(model, dataset, track_id)`, lưu bằng `save_output`. Mỗi model có
`chordtrack/melody/<model>.py` với `transcribe(audio_path) -> MelodyModelOutput`, và script
`python -m eval.melody.cache_<model>` chạy hết mọi manifest (bỏ qua bài đã có cache).

Trường tùy chọn của manifest mà phần chấm dùng: `duration` (giây), `offset` (giây, cộng vào mọi thời điểm của
đáp án để khớp audio), `chord_ref` (file .lab nhãn Harte, bỏ qua nếu `chord_ref_ok` là `false`).

## Chọn dòng giai điệu và chấm điểm

`chordtrack/melody/line.py` biến đầu ra thô thành một dòng nốt đơn âm (`extract_line`, `notes_to_line`,
`f0_to_line`, `combine_lines`, `quantize_to_beats`). `python -m eval.melody.score` chấm mọi
(model × chiến lược × bộ), chọn tham số bằng cross-validation theo bài, ghi `EVAL_DIR/results/melody/`
(`per_track.csv`, `summary.csv`, `cv_choices.json`) và in bảng markdown. Chi tiết chỉ số ở docstring của
`eval/melody/score.py`. Lưới phách tùy chọn: `CACHE_DIR/melody/_beats/<dataset>/<track_id>.npz` (mảng `beats`,
`downbeats`); có thì chấm thêm bản căn phách.
