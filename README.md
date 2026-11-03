# ChordTrack — hợp âm chạy theo video YouTube

Chrome extension hiện hợp âm chạy theo video YouTube, kiểu karaoke: hợp âm đang phát, vài hợp âm tiếp theo, vạch
phách, metronome, cấu trúc bài (intro, verse, chorus) và vòng hợp âm, chế độ 🎹 Phím trắng cho người chơi piano
(tự transpose theo vùng giọng), toàn màn hình. Phần phân tích AI (bắt nhịp, nhận hợp âm kể cả hợp âm 7, sus, dim và
hợp âm đảo, giọng theo vùng) chạy **hoàn toàn trên máy bạn**, không gửi dữ liệu đi đâu.

*A Chrome extension that shows chords in sync with YouTube videos. All analysis (beats, chords incl. inversions,
key regions, song sections) runs locally on your machine. Windows setup: see "Cài đặt" below — three commands.*

Kế hoạch, các quyết định và số đo chất lượng: [PLAN.md](PLAN.md). Hướng dẫn các nút trong extension:
[extension/README.md](extension/README.md).

> ⚠️ **Chỉ dùng cho mục đích cá nhân, học tập.** Server tải audio YouTube bằng [yt-dlp](https://github.com/yt-dlp/yt-dlp)
> để phân tích, việc này có thể trái [Điều khoản dịch vụ của YouTube](https://www.youtube.com/t/terms). Bạn tự chịu
> trách nhiệm khi dùng. Audio tải về chỉ nằm trên máy bạn (`data/`), không được chia sẻ.
>
> *Personal and educational use only. The local server downloads YouTube audio with yt-dlp, which may violate
> YouTube's Terms of Service. You are responsible for how you use it.*

## Cài đặt (Windows)

**Yêu cầu:** Windows 10/11, Google Chrome, khoảng 10 GB trống. Nên có GPU NVIDIA (khoảng 1 GB VRAM là đủ):
mỗi bài mất 10–20 giây. Không có GPU vẫn chạy được trên CPU, chậm hơn. Script tự cài Python 3.11, git, ffmpeg và
Deno bằng `winget` nếu máy chưa có.

```powershell
git clone https://github.com/MakitJenamin/extension_chordtracking.git
cd extension_chordtracking
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

`setup.ps1` tạo `.venv`, cài PyTorch (bản CUDA nếu có GPU NVIDIA, không thì bản CPU) và các thư viện, tải mã và
weights các model hợp âm đúng phiên bản đã kiểm chứng (kiểm tra SHA-256, xem [THIRD_PARTY.md](THIRD_PARTY.md)),
rồi chạy thử. Lần đầu mất khoảng 10–20 phút tùy mạng. Chạy lại nhiều lần không sao. Tùy chọn: `-Cpu` (ép bản CPU),
`-Full` (thêm tách nguồn và công cụ đánh giá), `-NoTools` (không tự cài ffmpeg, Deno).

## Dùng

1. **Bật server** chạy nền (mỗi lần mở máy):
   `powershell -ExecutionPolicy Bypass -File scripts\start_server.ps1 -Background`
2. **Nạp extension** một lần: mở `chrome://extensions` → bật **Developer mode** → **Load unpacked** → chọn thư mục
   `extension` trong repo.
3. Mở một video YouTube, bấm nút ChordTrack trên thanh điều khiển của player (hoặc Alt+Shift+C). Bài mới mất
   10–20 giây để phân tích, lần sau hiện ngay.

**YouTube đòi xác minh "not a bot"?** Xuất cookie YouTube của trình duyệt đã đăng nhập ra file `cookies.txt`
(định dạng Netscape, ví dụ bằng extension "Get cookies.txt LOCALLY"), rồi chạy server với
`-Cookies C:\duong\dan\cookies.txt`. Chi tiết ở mục dưới.

**Cập nhật:** `git pull` rồi chạy lại `scripts\setup.ps1`, bấm nút tải lại extension ở `chrome://extensions`.

**Gỡ:** xóa extension ở `chrome://extensions` và xóa thư mục repo (mọi thứ, kể cả model và audio đã tải, nằm trong đó).

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
