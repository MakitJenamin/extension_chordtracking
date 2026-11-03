<div align="center">

<img src="extension/icons/icon.svg" width="112" alt="ChordTrack logo">

# ChordTrack

**Hợp âm chạy theo video YouTube, kiểu karaoke. AI chạy ngay trên máy bạn.**

*Live chords for any YouTube video, analyzed 100% locally.*

![Chrome](https://img.shields.io/badge/Chrome-extension-4285F4?logo=googlechrome&logoColor=white)
![Windows](https://img.shields.io/badge/Windows-10%20%7C%2011-0078D6?logo=windows&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Offline](https://img.shields.io/badge/AI-local%20only-ff4081)
![License](https://img.shields.io/badge/license-MIT-yellow)

</div>

---

<p align="center">
  <img src="docs/img/chord-bar.png" alt="Thanh hợp âm ChordTrack dưới video YouTube">
  <br><sub>Thanh hợp âm chạy theo vạch phát: giọng, BPM, nhịp, chế độ 🎹 Phím trắng, hợp âm kèm bậc La Mã</sub>
</p>

<p align="center">
  <img src="docs/img/song-structure.png" width="520" alt="Khung cấu trúc bài">
  <br><sub>Cấu trúc bài: từng đoạn, vòng hợp âm, giọng và nút transpose. Bấm vào đoạn để tua tới đó</sub>
</p>

## ✨ Có gì

| | |
|---|---|
| 🎼 **Hợp âm theo thời gian thực** | Hợp âm đang phát + các hợp âm sắp tới, nhận cả 7, sus, dim và hợp âm đảo |
| 🥁 **Nhịp** | Vạch phách và metronome khớp với bài |
| 🗺️ **Cấu trúc bài** | Intro, verse, chorus, vòng hợp âm, giọng theo từng đoạn |
| 🎹 **Phím trắng** | Tự transpose về giọng dễ đánh nhất cho người chơi piano |
| 🖥️ **Toàn màn hình** | Hợp âm vẫn hiện khi xem fullscreen |
| 🔒 **Riêng tư** | Không gửi dữ liệu đi đâu |

## 🚀 Cài đặt (Windows)

> Cần Chrome, ~10 GB trống. Có GPU NVIDIA thì 10–20 giây/bài, CPU ~1 phút.

```powershell
git clone https://github.com/MakitJenamin/extension_chordtracking.git
cd extension_chordtracking
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1   # 10–20 phút lần đầu
```

Script tự cài Python, PyTorch, model, ffmpeg, Deno. Tùy chọn: `-Cpu`, `-Full`, `-NoTools`.

> [!WARNING]
> Script tải mã và weights của bên thứ ba ([THIRD_PARTY.md](THIRD_PARTY.md)). Weights đã ghim commit và kiểm SHA-256, đừng thay bằng file lạ.

## 🎶 Dùng

1. **Bật server** (mỗi lần mở máy)
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\start_server.ps1 -Background
   ```
2. **Nạp extension** (một lần): `chrome://extensions` → bật *Developer mode* → *Load unpacked* → chọn thư mục `extension`.
3. **Mở YouTube**, bấm nút ChordTrack trên player hoặc <kbd>Alt</kbd>+<kbd>Shift</kbd>+<kbd>C</kbd>.

Các nút trong extension: [extension/README.md](extension/README.md).

> [!CAUTION]
> Server dùng [yt-dlp](https://github.com/yt-dlp/yt-dlp) tải audio về `data/audio/`, việc này có thể trái [điều khoản YouTube](https://www.youtube.com/t/terms). Bạn tự chịu trách nhiệm, đừng chia sẻ lại các file đó.

**🔄 Cập nhật:** `git pull` → chạy lại `setup.ps1` → tải lại extension.  **🗑️ Gỡ:** xóa extension và thư mục repo.

## 🛠️ Sự cố

**YouTube báo "not a bot"** → xuất cookie ra `cookies.txt` (extension *Get cookies.txt LOCALLY*), chạy server thêm `-Cookies <đường dẫn>`.

> [!CAUTION]
> `cookies.txt` là phiên đăng nhập Google của bạn. Không gửi cho ai, để ngoài thư mục repo, nên dùng tài khoản phụ.

<details>
<summary><b>📡 API backend</b> (cho dev)</summary>

Server FastAPI, chỉ nghe `127.0.0.1:8765` (đổi bằng `-Port`). CORS chỉ cho `chrome-extension://*` và `https://www.youtube.com`. Một worker xử lý lần lượt, hàng đợi ở `data/cache/jobs.sqlite`, log ở `data/cache/server.log`.

| Endpoint | Mô tả |
|---|---|
| `GET /health` | Trạng thái server, GPU, hàng đợi, ffmpeg, Deno, cookies |
| `POST /analyze` | `{"source": "<videoId hoặc URL>", "force": false}` → kết quả nếu đã cache, không thì `job_id` |
| `GET /jobs/{job_id}` | `status`, `progress`, `step`, `eta_seconds`, `error` |
| `GET /result/{song_id}` | JSON kết quả (header `X-Chordtrack-Stale: 1` nếu do mã cũ tạo) |
| `GET /songs` | Danh sách bài đã phân tích |

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8765/analyze -ContentType application/json -Body '{"source": "QDYfEBY9NM4"}'
```

</details>

## 📜 License

Mã dự án: [MIT](LICENSE). Model hợp âm và bắt nhịp thuộc tác giả gốc, xem [THIRD_PARTY.md](THIRD_PARTY.md).
⚠️ ChordFormer không có license, THIRD_PARTY.md hướng dẫn cách tắt.

<div align="center"><sub>Kế hoạch, quyết định và số đo chất lượng: <a href="PLAN.md">PLAN.md</a></sub></div>
