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
  <img src="docs/img/fullscreen.jpg" alt="ChordTrack khi xem toàn màn hình">
  <br><sub>🖥️ Toàn màn hình: dải hợp âm đè lên video, ngay trên thanh điều khiển, chữ luôn rõ để vừa xem vừa đàn</sub>
</p>

## ✨ Có gì

<table>
<tr>
<td align="center" width="33%">🎼<br><b>Hợp âm thời gian thực</b><br><sub>Nhận cả 7, sus, dim và hợp âm đảo</sub></td>
<td align="center" width="33%">🥁<br><b>Nhịp</b><br><sub>Vạch phách, BPM, metronome khớp bài</sub></td>
<td align="center" width="33%">🗺️<br><b>Cấu trúc bài</b><br><sub>Từng đoạn, vòng hợp âm, giọng</sub></td>
</tr>
<tr>
<td align="center">🎹<br><b>Phím trắng</b><br><sub>Tự transpose cho người chơi piano</sub></td>
<td align="center">🖥️<br><b>Toàn màn hình &amp; rạp phim</b><br><sub>Hợp âm hiện ở mọi chế độ xem</sub></td>
<td align="center">🔒<br><b>Riêng tư</b><br><sub>AI chạy trên máy, không gửi gì đi</sub></td>
</tr>
<tr>
<td align="center">🎵<br><b>Dòng giai điệu</b><br><sub>Bấm "Dò giai điệu": tên nốt và ↑/↓ chạy dưới hợp âm, dịch giọng như hợp âm. Hai cách dò: Bình thường (YourMT3+) và Kỹ (thêm MuScriptor)</sub></td>
<td align="center">🎤<br><b>Tách giọng</b><br><sub>BS-Roformer tách giọng hát trước khi dò, ít lẫn nhạc đệm</sub></td>
<td align="center">📏<br><b>Nốt khớp phách</b><br><sub>Nốt tự căn về lưới phách của bài, dễ đọc nhịp</sub></td>
</tr>
</table>

## 📸 Giao diện

### Ba cách đọc hợp âm

**🎹 Phím trắng** (mặc định): đổi hợp âm về giọng chỉ dùng phím trắng, kèm số nút transpose cần bấm trên đàn. Thanh vàng đếm ngược trước chỗ đổi giọng.

<img src="docs/img/chord-bar.png" alt="Chế độ Phím trắng">

**Gốc**: hợp âm đúng như bản thu, nút Tông −/+ để dịch giọng tự do.

<img src="docs/img/mode-goc.png" alt="Chế độ Gốc">

**Bậc La Mã**: hợp âm theo bậc trong giọng (I, IV, V, vi…), hợp để học vòng hợp âm và chơi ở giọng bất kỳ.

<img src="docs/img/mode-la-ma.png" alt="Chế độ Bậc La Mã">

> Vạch đỏ là chỗ đang phát · vạch vàng đứt là chỗ đổi giọng · ⚠ là hợp âm model chưa chắc · bấm vào hợp âm để tua tới đó.

### Cấu trúc bài

<table>
<tr>
<td width="45%"><img src="docs/img/song-structure.png" alt="Cấu trúc bài ở cột phải"></td>
<td>

Cột phải liệt kê **từng đoạn** của bài: vòng hợp âm, bậc La Mã, giọng và nút transpose.

🎨 Cùng màu là cùng một đoạn nhạc lặp lại.

⏩ Bấm vào đoạn để tua tới đó.

🎬 Ở **chế độ rạp phim**, các đoạn xếp ngang ngay dưới dải hợp âm (ảnh dưới).

</td>
</tr>
</table>

<img src="docs/img/theater.png" alt="Chế độ rạp phim: dải hợp âm và cấu trúc bài dưới video">

## 🚀 Cài đặt (Windows)

### Cần có

| | |
|---|---|
| 💻 Hệ điều hành | Windows 10 hoặc 11 (64-bit) |
| 🎮 GPU | NVIDIA khuyến nghị (đã thử RTX 3060 Ti 8 GB). Không có GPU vẫn chạy được bằng CPU nhưng chậm hơn nhiều |
| 💾 Ổ đĩa | Khoảng 10 GB trống (cài xong chiếm 7,3 GB, PyTorch bản CUDA đã 4 GB) |
| 🌐 Mạng | Tải khoảng 5 GB ở lần cài đầu |
| 🧭 Trình duyệt | Google Chrome (hoặc trình duyệt nhân Chromium cho nạp extension) |

Python 3.11, git, ffmpeg và Deno: chưa có thì script tự cài bằng `winget`.

### 1. Tải mã nguồn và cài

Mở **PowerShell** ở thư mục muốn đặt dự án, chạy:

```powershell
git clone https://github.com/MakitJenamin/extension_chordtracking.git
cd extension_chordtracking
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

Script tự làm hết: Python 3.11 → `.venv` → PyTorch (bản CUDA nếu thấy GPU NVIDIA) → thư viện → model (kiểm SHA-256) →
ffmpeg, Deno → chạy thử. Cuối cùng in **tóm tắt**: thời gian từng bước, có GPU không, và dò giai điệu bật được những gì.
Chạy lại nhiều lần không sao, bước nào xong rồi sẽ bỏ qua.

Đo thật (2026-11-11): clone mới, máy RTX 3060 Ti, mạng khoảng 12 MB/s, không dùng cache pip, không đăng nhập Hugging
Face. **Tổng khoảng 14 phút.**

| Phần | Tải về | Trên đĩa | Thời gian |
|---|---|---|---|
| PyTorch 2.14 bản CUDA | 2,6 GB | 4,0 GB | phần lớn của ~11 phút đầu (cùng thư viện và model hợp âm) |
| Thư viện Python khác (gồm tách nguồn audio-separator) | ~0,4 GB | ~0,9 GB | |
| Model hợp âm, nhịp (ChordMini, Chord-CNN-LSTM, ChordFormer, beat_this) | ~0,9 GB | 0,96 GB | |
| Dò giai điệu: BS-Roformer (tách giọng) | 0,64 GB | 0,61 GB | ~1 phút |
| Dò giai điệu: YourMT3+ (mã, checkpoint, transformers riêng) | 0,6 GB | 0,64 GB | ~1,5 phút |
| Chạy thử (hợp âm C – G và một chuỗi nốt qua YourMT3+) | | | ~40 giây |

Tùy chọn: `-Cpu` (cài PyTorch bản CPU), `-NoMelody` (bỏ dò giai điệu), `-Full` (thêm thư viện đánh giá cho dev),
`-NoTools` (không tự cài ffmpeg, Deno).

### 2. Bật server

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_server.ps1 -Background
```

Server chạy ẩn ở `http://127.0.0.1:8765` (chỉ máy bạn gọi được), log ở `data\cache\server.log`. Mỗi lần mở máy chạy lại
lệnh này. Bỏ `-Background` để xem log ngay trong cửa sổ (Ctrl+C để tắt).

### 3. Nạp extension vào Chrome (một lần)

1. Mở `chrome://extensions`, bật **Developer mode** (góc trên bên phải).
2. Bấm **Load unpacked**, chọn thư mục `extension` trong thư mục dự án.
3. Ghim biểu tượng ChordTrack lên thanh công cụ nếu muốn: bấm vào là thấy server có chạy không.

### 4. Dùng

Mở một video YouTube, bấm nút **ChordTrack** trên thanh điều khiển của player (hoặc <kbd>Alt</kbd>+<kbd>Shift</kbd>+<kbd>C</kbd>).
Lần đầu mỗi bài server tải audio và phân tích khoảng 10–20 giây trên GPU (CPU khoảng 1 phút), lần sau có ngay.

Các nút trong extension: [extension/README.md](extension/README.md).

**🔄 Cập nhật:** `git pull` → chạy lại `setup.ps1` → khởi động lại server → bấm ↻ ở extension trong `chrome://extensions`.
**🗑️ Gỡ:** xóa extension và thư mục dự án.

> [!WARNING]
> Script tải mã và weights của bên thứ ba ([THIRD_PARTY.md](THIRD_PARTY.md)). Weights đã ghim commit và kiểm SHA-256, đừng thay bằng file lạ.

## 🎵 Dò giai điệu

Bấm **♪ Dò giai điệu** trong khung ChordTrack (sau khi hợp âm đã hiện): tên nốt và mũi tên ↑/↓ chạy dưới dải hợp âm.
Có hai cách dò, chọn ở nút **Giai điệu: Bình thường | Kỹ** (nhớ lựa chọn):

| Cách | Model | Bài 4 phút, lần đầu | Cần tài khoản? |
|---|---|---|---|
| **Bình thường** (mặc định) | YourMT3+ | khoảng 2 phút | Không, `setup.ps1` cài sẵn |
| **Kỹ** | YourMT3+ ở đoạn hát, MuScriptor small ở phần còn lại | khoảng 3 phút | Có, tài khoản Hugging Face (miễn phí) |

Bình thường tốt nhất với bài có lời; Kỹ bắt giai điệu nhạc cụ ở đoạn không lời tốt hơn. Cả hai luôn **tách giọng** hát bằng
BS-Roformer trước khi dò và **căn nốt về phách** của bài cho dễ đọc nhịp. Mỗi lần bấm chỉ chạy cách đang chọn.

**Bật cách Kỹ** (tùy chọn). Weights MuScriptor theo license **CC BY-NC 4.0, chỉ dùng phi thương mại**, và để ở repo
gated, nên `setup.ps1` không tự tải được. Chưa làm thì nút Kỹ mờ đi, rê chuột vào để xem hướng dẫn. Ba bước:

1. Tạo tài khoản ở [huggingface.co](https://huggingface.co), vào [Settings → Access Tokens](https://huggingface.co/settings/tokens) tạo token loại **Read**.
2. Mở [huggingface.co/MuScriptor/muscriptor-small](https://huggingface.co/MuScriptor/muscriptor-small), bấm đồng ý điều khoản.
3. Trong thư mục dự án:
   ```powershell
   .venv\Scripts\hf.exe auth login                          # dán token ở bước 1
   .venv\Scripts\python.exe scripts\fetch_models.py --muscriptor
   ```
   rồi khởi động lại server (tắt tiến trình cũ, chạy lại `start_server.ps1`).

   Weights MuScriptor small tải thêm khoảng 0,4 GB.

## 🛠️ Gặp lỗi thường gặp

<details>
<summary><b>Extension báo "Chưa kết nối được server" / bấm không có gì</b></summary>

Server chưa chạy. Chạy lại `scripts\start_server.ps1 -Background`, rồi bấm biểu tượng ChordTrack trên thanh công cụ
Chrome để kiểm tra (phải thấy "Đang chạy"). Vẫn lỗi thì xem `data\cache\server.log`. Vừa cập nhật extension thì tải lại
trang YouTube (F5).
</details>

<details>
<summary><b>YouTube báo "not a bot" / "Sign in to confirm"</b></summary>

YouTube đòi cookie đăng nhập. Xuất cookie ra `cookies.txt` (extension *Get cookies.txt LOCALLY*), để **ngoài** thư mục
dự án, rồi chạy server kèm file đó:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_server.ps1 -Background -Cookies C:\duong\dan\cookies.txt
```

> [!CAUTION]
> `cookies.txt` là phiên đăng nhập Google của bạn. Không gửi cho ai, nên dùng tài khoản phụ.
</details>

<details>
<summary><b>Thiếu ffmpeg hoặc Deno</b></summary>

`setup.ps1` cài bằng winget; máy không có winget thì cài tay: `winget install Gyan.FFmpeg` và
`winget install DenoLand.Deno` (hoặc tải từ trang chủ), rồi **mở PowerShell mới** và chạy lại server. Popup của extension
có chip ffmpeg / Deno cho biết server có thấy hai công cụ này không.
</details>

<details>
<summary><b>Hết VRAM (CUDA out of memory)</b></summary>

GPU 8 GB là đủ khi chỉ ChordTrack dùng GPU. Đóng game, trình chỉnh video hay app AI khác đang giữ VRAM rồi bấm Thử lại.
Server chạy từng job một nên không tự tranh VRAM với chính nó. Máy GPU nhỏ hơn: cài lại với `setup.ps1 -Cpu` (chậm hơn).
</details>

<details>
<summary><b>Nút "Kỹ" mờ / setup báo "chưa có MuScriptor small"</b></summary>

Đây là chuyện bình thường khi chưa có tài khoản Hugging Face: cách Bình thường vẫn dùng được. Muốn bật Kỹ, làm ba bước ở mục
[Dò giai điệu](#-dò-giai-điệu). `fetch_models.py --muscriptor` báo 401/403 nghĩa là chưa đăng nhập (`hf auth login`)
hoặc chưa bấm đồng ý điều khoản trên trang model.
</details>

<details>
<summary><b>Setup báo "Filename too long" / lỗi khi tải YourMT3+</b></summary>

Windows giới hạn đường dẫn 260 ký tự. Script đã bật `core.longpaths` cho git, nhưng nếu vẫn lỗi thì clone lại vào thư
mục ngắn (ví dụ `C:\chordtrack`) hoặc bật *Enable Win32 long paths* (Group Policy) / `LongPathsEnabled` rồi chạy lại
`setup.ps1`.
</details>

<details>
<summary><b>Nút "Dò giai điệu" mờ</b></summary>

Server chưa cài dò giai điệu (cài với `-NoMelody` hoặc bước Model bị lỗi). Chạy lại `setup.ps1`, xem tóm tắt cuối, rồi
khởi động lại server.
</details>

> [!CAUTION]
> Server dùng [yt-dlp](https://github.com/yt-dlp/yt-dlp) tải audio về `data/audio/`, việc này có thể trái [điều khoản YouTube](https://www.youtube.com/t/terms). Chỉ dùng cho cá nhân, bạn tự chịu trách nhiệm, đừng chia sẻ lại các file đó.

<details>
<summary><b>📡 API backend</b> (cho dev)</summary>

Server FastAPI, chỉ nghe `127.0.0.1:8765` (đổi bằng `-Port`). CORS chỉ cho `chrome-extension://*` và `https://www.youtube.com`. Một worker xử lý lần lượt, hàng đợi ở `data/cache/jobs.sqlite`, log ở `data/cache/server.log`.

| Endpoint | Mô tả |
|---|---|
| `GET /health` | Trạng thái server, GPU, hàng đợi, ffmpeg, Deno, cookies; `melody` (có cách dò giai điệu nào chạy được) và `melody_models` (từng biến thể) |
| `POST /analyze` | `{"source": "<videoId hoặc URL>", "force": false}` → kết quả nếu đã cache, không thì `job_id` |
| `GET /jobs/{job_id}` | `status`, `progress`, `step`, `eta_seconds`, `error` |
| `GET /result/{song_id}` | JSON kết quả (header `X-Chordtrack-Stale: 1` nếu do mã cũ tạo) |
| `GET /songs` | Danh sách bài đã phân tích |
| `POST /melody` | Dò giai điệu, giống `/analyze`; `"model": "combo"` (Kỹ, mặc định của API) hoặc `"yourmt3"` (Bình thường), `"vocals": true` để tách giọng BS-Roformer trước (extension luôn gửi). Biến thể chưa đủ model (vd thiếu MuScriptor): 409 kèm hướng dẫn. Hợp đồng ở [docs/melody_api.md](docs/melody_api.md) |
| `GET /melody/{song_id}?model=combo\|yourmt3&vocals=0\|1` | Dòng nốt giai điệu (`notes`: `start`, `end`, `pitch` MIDI, `confidence`, `uncertain`, `source`) |

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8765/analyze -ContentType application/json -Body '{"source": "QDYfEBY9NM4"}'
```

</details>

## 📜 License

Mã dự án: [MIT](LICENSE). Model hợp âm, bắt nhịp và dò giai điệu thuộc tác giả gốc, xem [THIRD_PARTY.md](THIRD_PARTY.md).
YourMT3+ (dò giai điệu, cách Bình thường): Apache-2.0. BS-Roformer (tách giọng): qua gói audio-separator (MIT).
⚠️ Weights MuScriptor (cách dò Kỹ, tùy chọn) theo CC BY-NC 4.0: chỉ dùng phi thương mại.
⚠️ ChordFormer không có license, THIRD_PARTY.md hướng dẫn cách tắt.
⚠️ Tải audio bằng yt-dlp chỉ nên dùng cho cá nhân (xem cảnh báo ở mục Gặp lỗi thường gặp).

<div align="center"><sub>Kế hoạch, quyết định và số đo chất lượng: <a href="PLAN.md">PLAN.md</a></sub></div>
