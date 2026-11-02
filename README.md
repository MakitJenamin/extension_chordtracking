# extension_chordtracking

Chrome extension hiện hợp âm chạy theo video YouTube. Phần phân tích AI (bắt nhịp, nhận diện hợp âm và hợp âm đảo, giọng theo vùng, chia đoạn, vòng hợp âm) chạy hoàn toàn local trên GPU. Tách giọng và tách bass vẫn có nhưng tắt mặc định (`--separation`).

Trạng thái: **bản đầu tiên dùng được.** Kế hoạch, quyết định và số đo chất lượng: [PLAN.md](PLAN.md).

## Dùng nhanh

1. Bật backend chạy nền (mỗi lần mở máy):
   `powershell -ExecutionPolicy Bypass -File scripts\start_server.ps1 -Background`
2. Nạp extension một lần: `chrome://extensions` → bật Developer mode → Load unpacked → chọn thư mục `extension/`.
3. Mở video YouTube bất kỳ. Bài mới mất khoảng 10–20 giây để phân tích, lần sau hiện ngay.

Hướng dẫn chi tiết các nút, chế độ 🎹 Phím trắng, metronome và toàn màn hình: [extension/README.md](extension/README.md).

## Chạy backend local

Backend là server FastAPI, chỉ lắng nghe `127.0.0.1`. Extension gọi vào server này.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_server.ps1            # cổng mặc định 8765, giữ cửa sổ mở
powershell -ExecutionPolicy Bypass -File scripts\start_server.ps1 -Background # chạy ẩn ở nền, log ở data\cache\server.log
powershell -ExecutionPolicy Bypass -File scripts\start_server.ps1 -Port 8766
powershell -ExecutionPolicy Bypass -File scripts\start_server.ps1 -Cookies C:\duong\dan\cookies.txt
```

Script nạp lại PATH (để thấy `ffmpeg` và `deno` vừa cài), kích hoạt venv rồi chạy `python -m chordtrack.server`.
Cần `ffmpeg` và Deno (yt-dlp cần để tải YouTube). Cài bằng `winget install Gyan.FFmpeg DenoLand.Deno`.

| Endpoint | Mô tả |
|---|---|
| `GET /health` | `status`, `version`, `gpu`, `queue` (số job đang chờ), `deno`, `ffmpeg`, `cookies`, `worker_alive` |
| `POST /analyze` | Body `{"source": "<videoId hoặc URL YouTube>", "force": false}`. Có cache thì trả `{"status": "done", "result": {...}}`, chưa có thì `{"status": "queued", "job_id", "song_id"}`. Gửi trùng bài đang chờ hoặc đang chạy thì trả lại job đó |
| `GET /jobs/{job_id}` | `status` (queued, running, done, error), `progress` 0–1, `step`, `message`, `eta_seconds`, `error` |
| `GET /result/{song_id}` | JSON kết quả, hoặc 404. Kết quả do mã cũ tạo vẫn được trả, kèm header `X-Chordtrack-Stale: 1` |
| `GET /songs` | Các bài đã phân tích: `song_id`, `title`, `duration`, `key`, `analyzed_at` |

```powershell
Invoke-RestMethod http://127.0.0.1:8765/health
Invoke-RestMethod -Method Post http://127.0.0.1:8765/analyze -ContentType application/json -Body '{"source": "QDYfEBY9NM4"}'
Invoke-RestMethod http://127.0.0.1:8765/jobs/<job_id>
```

- Hàng đợi job lưu ở `data/cache/jobs.sqlite`. Server có **một** worker xử lý lần lượt từng bài. Job đang chạy dở
  khi server tắt sẽ được chạy lại ở lần khởi động sau.
- **YouTube đòi xác minh "not a bot":** xuất cookie YouTube của trình duyệt đã đăng nhập ra file `cookies.txt`
  (định dạng Netscape, ví dụ bằng extension "Get cookies.txt LOCALLY"). Sau đó đặt biến môi trường
  `CHORDTRACK_YT_COOKIES=<đường dẫn>` hoặc chạy script với `-Cookies`. Server chỉ chuyển đường dẫn file cho yt-dlp
  và không ghi nội dung cookie ra log.
- CORS chỉ cho phép `chrome-extension://*` và `https://www.youtube.com`.
