# Hợp đồng API dò giai điệu (backend ↔ extension)

Người dùng đã chốt (2026-11-11): dùng cách ghép **YourMT3+ → MuScriptor small** (`yourmt3>muscriptor_small`,
xem `data/eval/results/melody/report.md` và PLAN.md mục 11). Dò giai điệu là **job riêng, bấm mới chạy**, không chạy
cùng phân tích hợp âm.

## Endpoint

### `POST /melody`

Body JSON: `{"source": "<videoId hoặc URL YouTube>", "force": false}`. Giống `POST /analyze`:

- Đã có kết quả (đúng `code_revision`) và không `force`: `{"status": "done", "song_id": "...", "melody": <Melody>}`.
- Ngược lại tạo (hoặc trả lại) job: `{"status": "queued" | "running", "job_id": "...", "song_id": "..."}`.

Theo dõi job bằng `GET /jobs/{job_id}` như cũ (cùng dạng `status`, `step`, `progress`, `eta_seconds`, `message`,
`error`). Job melody có `"kind": "melody"`, job hợp âm có `"kind": "analyze"`. Các bước của job melody:
`fetch` (dùng lại audio đã tải nếu có), `melody_voice` (YourMT3+), `melody_fill` (MuScriptor small), `melody_line`
(ghép và tạo dòng nốt).

### `GET /melody/{song_id}`

Trả `<Melody>`, 404 nếu chưa có. Kết quả mã cũ trả kèm header `X-Chordtrack-Stale: 1` như `/result`.

### `GET /health`

Thêm trường `"melody": true | false` (đã cài đủ model dò giai điệu chưa).

## `<Melody>`

```json
{
  "song_id": "QDYfEBY9NM4",
  "model": "yourmt3>muscriptor_small",
  "code_revision": "...",
  "duration": 243.1,
  "notes": [
    {"start": 13.02, "end": 13.41, "pitch": 55, "confidence": 0.82, "uncertain": false, "source": "voice"}
  ]
}
```

- `notes` là **một dòng đơn âm**, sắp theo `start`, không chồng nhau. Thời gian tính bằng giây từ đầu bài.
- `pitch`: số MIDI (60 = C4). Backend luôn giữ quãng tám đầy đủ.
- `confidence` trong [0, 1]; `uncertain = confidence < ngưỡng` (backend chọn ngưỡng, ghi trong code).
- `source`: `"voice"` hoặc tên nhóm nhạc cụ, chỉ để tham khảo.

## Hiển thị trong extension (người dùng đã chốt)

- Một dòng nốt **dưới thanh hợp âm**, chạy cùng thời gian với video.
- Tên nốt kiểu C D E F G A B (thăng/giáng theo cùng quy ước với hợp âm), **không ghi số quãng tám**.
- Mỗi nốt có **mũi tên lên/xuống** so với nốt trước (`pitch` lớn hơn: ↑, nhỏ hơn: ↓, bằng: không mũi tên).
- Nốt `uncertain` hiện **mờ**.
- Dịch giọng (`transpose` và chế độ hiển thị của hợp âm) áp dụng cho cả nốt giai điệu: tên nốt tính từ `pitch + transpose`.
- Có nút **"Dò giai điệu"**: bấm mới gửi `POST /melody`, hiện tiến độ job; đã có kết quả thì tự tải bằng `GET /melody/{id}`.
