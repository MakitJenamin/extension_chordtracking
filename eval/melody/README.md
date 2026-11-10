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

## Chuẩn bị dữ liệu: `python -m eval.melody.datasets <bộ>`

Gọi bằng python của venv chung. Mỗi bộ ghi vào `data/eval/melody/<bộ>/`: `manifest.jsonl`, `ref/<track_id>.csv`
(đáp án đã đổi sang csv), `raw/` (file tải về gốc), `audio/` (nếu phải chuyển định dạng). Chạy lại được nhiều
lần, bỏ qua phần đã tải. Đọc bằng `eval.melody.datasets.read_manifest(<bộ>)` và `load_reference(entry)`
(đã cộng `offset`). `python -m eval.melody.datasets summary` in số bài và tổng thời lượng.

Trường thêm trong manifest: `sr`, `channels`, `duration` (đọc từ WAV); `offset` (giây, **cộng vào thời gian
đáp án**, chỉ MIR-ST500); `chord_ref`, `chord_ref_ok` (Kajiura); `url`, `video_id`.

| bộ | lệnh | nguồn | bài | audio | đáp án | melody_type |
|---|---|---|---|---|---|---|
| `kajiura` | `kajiura` | audio có sẵn ở `data/eval/kajiura/audio` | 32 (150 phút) | WAV gốc (48 kHz stereo) | không; `chord_ref` = `data/eval/kajiura/ref/<tên>.lab` | mixed |
| `sanity` | `sanity` | `data/audio/QDYfEBY9NM4.wav` ("Let It Be") | 1 | 48 kHz stereo | không | vocal |
| `mir_st500` | `mir_st500 [--cookies FILE]` | nhãn: github.com/york135/singing_transcription_ICASSP2021, audio YouTube | 82 (test 401–500; 18 video mất, 2026-11-10), 337 phút | 44.1 kHz stereo 16-bit (ffmpeg) | `ref_notes` | vocal |
| `orchset` | `orchset` | Zenodo 1289786 (CC BY-NC-SA 4.0) | 64 (23,6 phút) | bản mono 44.1 kHz | `ref_f0`, bước 10 ms | instrumental |
| `adc2004` | `adc2004` | labrosa.ee.columbia.edu/projects/melody | 20 (6,1 phút) | mono 44.1 kHz | `ref_f0`, bước 5.8 ms | vocal: daisy, opera, pop (12); instrumental: jazz, midi (8) |
| `mirex05` | `mirex05` | như trên, `mirex05TrainFiles.zip` | 13 (6,5 phút) | mono 44.1 kHz | `ref_f0`, bước 10 ms | vocal: train01–09; instrumental: train10–13 |

- **MIR-ST500**: YouTube đòi xác minh "not a bot" trên máy này; script thử `player_client=web_embedded` trước
  (tải được không cần cookie), rồi mặc định, rồi `mweb`. Nếu vẫn bị chặn: `--cookies` hoặc `CHORDTRACK_YT_COOKIES`.
  Kết quả từng bài (mất, lệch, điểm) ghi ở `data/eval/melody/mir_st500/offsets.json`.
- **Kiểm tra lệch thời gian MIR-ST500** (`estimate_offset`, CPU, khoảng 1 s/bài): salience cao độ từ CQT
  (1 bin/nửa cung, trừ trung vị theo thời gian, cộng điều hòa, chuẩn hóa z theo khung), so với nhãn nốt ở
  mọi độ lệch trong ±2 s (nếu không khớp thì ±10 s). Giữ bài khi `score` ≥ 0.8 và `margin` (đỉnh trừ điểm cao
  nhất ở độ lệch cách đỉnh > 0.25 s) ≥ 0.15; lệch < 0.05 s coi là 0. Bài khớp có score 1.4–1.9, margin ≥ 0.45;
  nhãn của bài khác cho score ≈ 0, margin < 0.05. Chọn lại ngưỡng không cần tính lại:
  `mir_st500 --redecide MIN_SCORE MIN_MARGIN`.
- **MIREX05**: bộ test 25 bài của MIREX không công bố; bản công khai là 13 bài LabROSA, cũng là bản các bài báo
  (MSNet, FTANet, TONet) dùng. ADC2004 và MIREX05 thường chấm riêng phần vocal (12 và 9 bài).
- **MedleyDB**: bỏ, Zenodo để "restricted" (phải xin phép).
- **Kết quả kiểm tra lệch MIR-ST500 (2026-11-10)**: cả 82 bài tải được đều khớp nhãn (score 0,97–2,2, trung vị
  1,75; margin 0,23–0,9). Độ lệch: 69 bài trong 0–0,02 s (1–2 khung, coi như 0), 411 +0,058 s, 474 −0,081 s,
  **500 +0,859 s** (score tại 0 chỉ 0,51). Kiểm tra chéo bằng `librosa.pyin` + `mir_eval` trên 90 s: bài 500 có
  Raw Chroma Accuracy 0,18 ở offset 0, 0,39 ở offset 0,859; bài 401 cao nhất ở offset 0.
