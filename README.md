# extension_chordtracking

Chrome extension hiện hợp âm chạy theo video YouTube. Phần phân tích AI (tách giọng, tách bass, bắt nhịp, nhận diện hợp âm, tìm vòng hợp âm) chạy hoàn toàn local trên GPU.

Trạng thái: đang ở giai đoạn lên kế hoạch. Xem chi tiết ở [PLAN.md](PLAN.md).

## Chạy backend local

Backend là server FastAPI, chỉ lắng nghe `127.0.0.1`. Extension gọi vào server này.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_server.ps1            # cổng mặc định 8765
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
