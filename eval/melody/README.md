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
