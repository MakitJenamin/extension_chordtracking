# ChordTrack — hợp âm chạy theo video YouTube

Chrome extension hiện hợp âm chạy theo video YouTube, kiểu karaoke: hợp âm đang phát, vài hợp âm tiếp theo, vạch
phách, metronome, cấu trúc bài (intro, verse, chorus) và vòng hợp âm, chế độ 🎹 Phím trắng cho người chơi piano
(tự transpose theo vùng giọng), toàn màn hình. Phần phân tích AI (bắt nhịp, nhận hợp âm kể cả hợp âm 7, sus, dim và
hợp âm đảo, giọng theo vùng) chạy **hoàn toàn trên máy bạn**, không gửi dữ liệu đi đâu.

*A Chrome extension that shows chords in sync with YouTube videos. All analysis (beats, chords incl. inversions,
key regions, song sections) runs locally on your machine. Windows setup: see "Cài đặt" below — three commands.*

Kế hoạch, các quyết định và số đo chất lượng: [PLAN.md](PLAN.md). Hướng dẫn các nút trong extension:
[extension/README.md](extension/README.md).

> ⚠️ **Tải audio YouTube có thể trái điều khoản của YouTube. Bạn tự chịu trách nhiệm.**
> Server dùng [yt-dlp](https://github.com/yt-dlp/yt-dlp) tải riêng phần âm thanh của bài bạn bấm phân tích, giữ trên
> máy (`data/audio/`) để phân tích. Đừng chia sẻ lại các file này.
>
> *Downloading YouTube audio may violate [YouTube's Terms of Service](https://www.youtube.com/t/terms). Use at your
> own risk. Audio stays on your machine; don't redistribute it.*

## Cài đặt (Windows)

Cần Windows 10/11, Chrome, khoảng 10 GB trống. GPU NVIDIA không bắt buộc nhưng nhanh hơn (10–20 giây một bài,
CPU khoảng 1 phút).

```powershell
git clone https://github.com/MakitJenamin/extension_chordtracking.git
cd extension_chordtracking
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

Script tự cài mọi thứ còn thiếu (Python, PyTorch, model, ffmpeg, Deno), mất 10–20 phút lần đầu.
Tùy chọn: `-Cpu`, `-Full` (thêm công cụ đánh giá), `-NoTools`.

> ⚠️ **Script tải và chạy mã, weights của bên thứ ba ([THIRD_PARTY.md](THIRD_PARTY.md)). Chỉ chạy nếu bạn tin các repo đó.**
> Weights đã được ghim commit và kiểm tra SHA-256; đừng thay bằng file lạ.

## Dùng

1. Bật server (mỗi lần mở máy): `powershell -ExecutionPolicy Bypass -File scripts\start_server.ps1 -Background`
2. Nạp extension (một lần): `chrome://extensions` → bật **Developer mode** → **Load unpacked** → chọn thư mục `extension`.
3. Mở video YouTube, bấm nút ChordTrack trên player (Alt+Shift+C).

**Lỗi "not a bot":** xuất cookie YouTube ra `cookies.txt` (extension "Get cookies.txt LOCALLY"), chạy server thêm
`-Cookies <đường dẫn>`.

> ⚠️ **`cookies.txt` là phiên đăng nhập Google của bạn.** Không gửi cho ai, để ngoài thư mục repo, nên dùng tài khoản phụ.

**Cập nhật:** `git pull`, chạy lại `setup.ps1`, tải lại extension. **Gỡ:** xóa extension và thư mục repo.

## License

Mã của dự án: [MIT](LICENSE). Các model hợp âm, bắt nhịp là của tác giả gốc, mỗi model một license riêng, xem
[THIRD_PARTY.md](THIRD_PARTY.md). ⚠️ ChordFormer không có license; THIRD_PARTY.md hướng dẫn cách tắt nó.

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
