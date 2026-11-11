# Hợp đồng API dò giai điệu (backend ↔ extension)

Người dùng đã chốt (2026-11-11): dùng cách ghép **YourMT3+ → MuScriptor small** (`yourmt3>muscriptor_small`,
xem `data/eval/results/melody/report.md` và PLAN.md mục 11). Dò giai điệu là **job riêng, bấm mới chạy**, không chạy
cùng phân tích hợp âm.

Thêm (2026-11-11): tùy chọn **chỉ YourMT3+** (`"model": "yourmt3"`), bỏ bước MuScriptor small nên nhanh hơn khoảng
3-4 lần. Theo `report.md`, YourMT3+ một mình hơn cách ghép ở F1 nốt giọng hát (MIR-ST500 53.1 so với 49.0) nhưng kém
khoảng 10 điểm OA ở đoạn dàn nhạc / nhạc không lời (ORCHSET 43.0 so với 53.2). Mặc định vẫn là ghép.

Thêm (2026-11-11): cờ **tách giọng** `"vocals": true`, kết hợp được với cả hai `model`. BS-Roformer
(BS-Roformer-Viperx-1297, như bước tách nguồn của pipeline hợp âm) tách stem giọng `CACHE_DIR/<song_id>/vocals.wav`
(dùng lại nếu đã có), YourMT3+ chạy trên stem giọng; đoạn có giọng (nốt `voice` của YourMT3+ trên stem, nới 0,6 s) lấy
dòng đó, đoạn không giọng lấy YourMT3+ trên mix (`model: "yourmt3"`) hoặc MuScriptor small trên mix (`model: "combo"`).
Chưa chấm trên bộ eval, người dùng tự nghe thử. Thêm khoảng 75 giây cho bài 4 phút ở lần đầu (tách giọng ~50 s,
YourMT3+ trên stem ~23 s).

## Biến thể (`model` + `vocals`)

| `model`, `vocals` (request) | khóa biến thể | `<Melody>.model` | File cache | `kind` của job | Các bước |
|---|---|---|---|---|---|
| `"combo"` (mặc định), `false` | `combo` | `"yourmt3>muscriptor_small"` | `melody.json` | `"melody"` | `fetch`, `melody_voice`, `melody_fill`, `melody_line` |
| `"yourmt3"`, `false` | `yourmt3` | `"yourmt3"` | `melody_yourmt3.json` | `"melody_yourmt3"` | `fetch`, `melody_voice`, `melody_line` |
| `"yourmt3"`, `true` | `yourmt3_vocals` | `"yourmt3[vocals]"` | `melody_yourmt3_vocals.json` | `"melody_yourmt3_vocals"` | `fetch`, `melody_separate`, `melody_voice`, `melody_voice_mix`, `melody_line` |
| `"combo"`, `true` | `combo_vocals` | `"yourmt3[vocals]>muscriptor_small"` | `melody_vocals.json` | `"melody_combo_vocals"` | `fetch`, `melody_separate`, `melody_voice`, `melody_fill`, `melody_line` |

Ở bản tách giọng, `melody_voice` là YourMT3+ trên stem giọng, `melody_voice_mix` là YourMT3+ trên mix. Đầu ra thô
dùng lại tối đa giữa các biến thể: `melody_raw/yourmt3.npz` (YourMT3+ mix), `yourmt3_vocals.npz` (YourMT3+ stem),
`muscriptor_small.npz`, và `vocals.wav` (đầu ra thô trên stem ghi dấu file `vocals.wav`; stem bị tách lại thì chạy
lại). `force` chạy lại các model của biến thể được gửi, bản tách giọng tách lại cả stem (bass của pipeline hợp âm làm
từ instrumental cũ bị xóa để lần phân tích sau tách lại);
đầu ra thô nào chạy lại thì kết quả các biến thể khác dùng nó bị đánh dấu cũ (`"raw-changed"`).

- Mỗi biến thể có cache và job riêng: job theo (song_id, biến thể), nên hai biến thể không chặn nhau; gửi trùng cùng
  biến thể thì trả lại job đang chờ/chạy đó. Hai job vẫn chạy lần lượt trên cùng một worker (GPU 8 GB).
- Hai biến thể dùng chung đầu ra thô của YourMT3+ (`melody_raw/yourmt3.npz`): bài đã dò bằng biến thể này thì
  biến thể kia không chạy lại YourMT3+ (bản YourMT3+ khi đó mất khoảng 1 giây). `force` chạy lại các model của
  biến thể được gửi (biến thể `"yourmt3"` chỉ chạy lại YourMT3+). Khi YourMT3+ chạy lại, kết quả đã lưu của biến
  thể kia (dựng từ đầu ra thô cũ) bị đánh dấu cũ (`code_revision` = `"raw-changed"`): `GET` vẫn trả kèm
  `X-Chordtrack-Stale: 1`, `POST` không `force` thì dựng lại từ đầu ra thô mới (khoảng 1 giây với bản YourMT3+).
- `code_revision` tính riêng theo biến thể (bản YourMT3+ không phụ thuộc mã MuScriptor).
- Extension hỏi `GET /health` trước khi gửi `model` khác `"combo"`: không có `melody_models` (server cũ, sẽ bỏ qua
  `model` và chạy cả job ghép) thì báo cần cập nhật server, không gửi job.
- Bản YourMT3+ dùng cấu hình tốt nhất của YourMT3+ một mình theo CV (`source_viterbi`, `chorus_as_voice=False`),
  vẫn qua bước sửa nhảy quãng tám (không làm giảm điểm eval).
- `model` lạ: 422.

## Endpoint

### `POST /melody`

Body JSON: `{"source": "<videoId hoặc URL YouTube>", "force": false, "model": "combo", "vocals": false}` (`model`
tùy chọn, mặc định `"combo"`; `vocals` tùy chọn, mặc định `false`; body không có hai trường này là cách ghép như
trước). Giống `POST /analyze`:

- Đã có kết quả (đúng `code_revision`) và không `force`: `{"status": "done", "song_id": "...", "melody": <Melody>}`.
- Ngược lại tạo (hoặc trả lại) job: `{"status": "queued" | "running", "job_id": "...", "song_id": "..."}`.

Theo dõi job bằng `GET /jobs/{job_id}` như cũ (cùng dạng `status`, `step`, `progress`, `eta_seconds`, `message`,
`error`). Job hợp âm có `"kind": "analyze"`; job dò giai điệu có `"kind": "melody"` (ghép) hoặc `"melody_yourmt3"`
(chỉ YourMT3+), `"melody_yourmt3_vocals"`, `"melody_combo_vocals"` (tách giọng), kèm `"model": "combo" | "yourmt3"`
và `"vocals": bool`. Các bước: `fetch` (dùng lại audio đã tải nếu có), `melody_separate` (BS-Roformer, chỉ bản tách
giọng), `melody_voice` (YourMT3+, trên stem giọng nếu tách giọng), `melody_voice_mix` (YourMT3+ trên mix, chỉ bản
YourMT3+ tách giọng), `melody_fill` (MuScriptor small, chỉ bản ghép), `melody_line` (ghép / tạo dòng nốt).

### `GET /melody/{song_id}?model=combo|yourmt3&vocals=0|1`

Trả `<Melody>` của biến thể (`model`, `vocals`) (mặc định `combo`, không tách giọng), 404 nếu biến thể đó chưa có. Kết quả mã cũ trả kèm header `X-Chordtrack-Stale: 1` như `/result`.

### `GET /health`

Thêm trường `"melody": true | false` (đã cài đủ model cho biến thể ghép chưa) và
`"melody_models": {"combo": bool, "yourmt3": bool, "yourmt3_vocals": bool, "combo_vocals": bool}` (từng biến thể
theo khóa; `yourmt3` chỉ cần YourMT3+, bản `*_vocals` cần thêm gói `audio-separator`). Extension gửi `vocals` chỉ khi
khóa biến thể có trong `melody_models` (server cũ không có thì báo cần cập nhật server).

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
- Công tắc "Tách giọng" (`CT.settings.melodyVocals`, mặc định tắt, nhớ lựa chọn) cạnh nút phân đoạn: chọn bản tách
  giọng của model đang chọn; như đổi model, đã có thì hiện, chưa có thì hiện nút dò (không tự chạy job).
- Công tắc "Nốt khớp phách" (`CT.settings.melodySnap`, mặc định tắt): căn đầu/cuối nốt về lưới phách `song.beats` của
  kết quả hợp âm (`theory.snapNotes`), chỉ ở extension, không gọi server. Mức chia mỗi phách (2 hoặc 4) **tự chọn**
  theo tempo và độ dày giai điệu (`theory.autoSnapDiv`), người dùng không phải chọn.
- Nút phân đoạn cạnh nút dò: "♪ YourMT3+ | Ghép", cùng kiểu với Gốc/Phím trắng/Bậc La Mã (nhớ lựa chọn,
  `CT.settings.melodyModel`, **mặc định `yourmt3`**, người dùng chốt 2026-11-11). Extension luôn gửi `model`. Nút dò và
  lần tự tải dùng biến thể đang chọn; đổi lựa chọn thì tải biến thể đó nếu đã có, chưa có thì hiện trạng thái chưa dò
  (không tự chạy job).
