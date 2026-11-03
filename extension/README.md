# ChordTrack – extension Chrome

Hiện hợp âm chạy theo video YouTube: dải hợp âm trượt ngang kiểu karaoke ngay dưới player, bảng "Cấu trúc bài"
ở cột phải, và chính dải đó đè lên video khi toàn màn hình. Phân tích chạy bằng backend local trên máy bạn.

## Cài vào Chrome thật (chế độ nhà phát triển)
1. Chạy backend: `powershell -ExecutionPolicy Bypass -File scripts\start_server.ps1 -Background` (chạy ẩn), hoặc bỏ `-Background` để giữ cửa sổ.
   Kiểm tra: mở http://127.0.0.1:8765/health, phải thấy `"status": "ok"`.
2. Mở `chrome://extensions`, bật **Developer mode** (góc phải trên).
3. Bấm **Load unpacked**, chọn thư mục `extension/` của repo này.
4. Mở một video YouTube (`https://www.youtube.com/watch?v=...`). Trên thanh điều khiển của player (bên phải, cạnh nút
   phụ đề) có nút ChordTrack (logo bàn phím piano nghiêng). **Bấm nút (hoặc Alt+Shift+C) thì mới tải audio và phân tích**; mở video không tự phân tích.
   Bài chưa phân tích thì có thanh tiến độ (lần đầu mất khoảng 1–2 phút, video vẫn xem bình thường); lần sau có ngay.
5. Sửa mã xong: bấm nút ↻ của extension ở `chrome://extensions`, rồi tải lại tab YouTube (F5).

Nếu YouTube đòi xác minh "không phải bot": xuất cookie youtube.com (đã đăng nhập) ra `cookies.txt` (ví dụ bằng
extension "Get cookies.txt LOCALLY"), rồi chạy server kèm `-Cookies C:\duong\dan\cookies.txt`. Panel sẽ hướng dẫn.

### Nạp lại khi phát triển (không cần mở chrome://extensions)
Sau khi `git pull` ở thư mục đã nạp, chạy trong console (hoặc Claude in Chrome) của một tab YouTube:
```js
window.postMessage({ type: "ct:dev-reload" }, "*")
```
Content script chuyển tin cho service worker, service worker gọi `chrome.runtime.reload()`. Sau đó tải lại
tab YouTube (F5) để nạp content script mới. Cơ chế này chỉ để phát triển và vô hại: nó chỉ nạp lại chính
extension, không đọc hay gửi dữ liệu gì. Lần đầu (bản đang nạp chưa có cơ chế này) vẫn phải bấm reload tay.

## Dùng
| Điều khiển | Ý nghĩa | Phím tắt |
|---|---|---|
| Nút ChordTrack trên thanh điều khiển player | Bật/tắt ChordTrack cho video này. Chỉ khi bật mới phân tích (POST /analyze) | Alt+Shift+C |
| Gốc / 🎹 Phím trắng / Bậc La Mã | Cách hiện hợp âm. Mặc định Phím trắng | Alt+Shift+1 / 2 / 3 |
| Tông − / + | Dịch tông ở chế độ Gốc (−6..+6) | Alt+Shift+↓ / ↑ |
| Ô "Đặt transpose đàn" | Giá trị nút transpose của đàn cho vùng giọng đang phát; chuyển vàng và đếm ngược 8 giây trước chỗ đổi giọng | |
| Metronome + thanh âm lượng | Tiếng phách theo `beats` của bài (src/metronome.js) | Alt+Shift+M |
| ▴ / ▾ | Thu gọn khung (chỉ còn hợp âm hiện tại và chấm phách) | Alt+Shift+H |
| Bấm hợp âm / đoạn | Tua video tới đó | |
| Kéo ngang dải hợp âm | Xem trước/sau (chuột, cảm ứng, cuộn ngang trackpad hoặc Shift + lăn chuột). Lúc kéo, thanh mỏng ở đáy dải cho biết khung đang xem nằm đâu trong bài (vệt sáng), chỗ đang phát (vạch đỏ) và thời điểm đang xem, lệch bao nhiêu giây. Thả tay 2,5 giây thì dải tự trượt về vạch đỏ; video không bị tua | |

**Nút ChordTrack trên thanh điều khiển:** trắng là đang tắt; có **chấm xanh** là bài đã có kết quả trên server (bấm là hiện
ngay, biết nhờ `GET /result/{videoId}` chỉ đọc cache, không tải audio, không dùng GPU); vòng tròn là đang phân tích
(vòng trắng: đang chạy ngầm trong khi panel đã ẩn); xanh sáng là đang hiện; chấm đỏ là lỗi. Bấm lần nữa thì ẩn panel và
lớp phủ, job đang chạy vẫn chạy tiếp. Đã bật cho video nào thì quay lại video đó (trong cùng tab) vẫn bật; video mới
mặc định tắt. Popup có tùy chọn **"Tự hiện cho bài đã có kết quả"** (mặc định tắt, `CT.settings.autoShowCached`):
bật thì bài có sẵn hiện luôn, nhưng bài mới vẫn không bao giờ tự phân tích.

Phím tắt dùng Alt+Shift để không đụng phím của YouTube (k, j, l, f, số…). Trên dải: vạch đỏ là thời điểm đang phát,
vạch cao kèm số là phách 1 của ô nhịp, vạch thấp là phách nhẹ, vạch vàng đứt là chỗ đổi giọng, ⚠ là hợp âm model không chắc.

**Toàn màn hình:** dải hợp âm chuyển vào lớp phủ trong `#movie_player`, nằm ngay trên thanh điều khiển YouTube. Khi
YouTube tự ẩn thanh điều khiển, lớp phủ hạ xuống sát đáy và nền trong hơn nhưng **không mờ chữ hợp âm**, vì người
dùng vừa xem vừa đàn và cần đọc hợp âm liên tục. Phụ đề YouTube được đẩy lên trên lớp phủ.

## Cấu trúc
Không có bước build: JavaScript thuần, nạp theo thứ tự trong `manifest.json`. Mọi module gắn vào `window.CT`
và giao tiếp qua `CT.bus` (xem đầu `src/core.js`). Class CSS có tiền tố `ct-`, màu lấy từ biến `--yt-spec-*`
nên theo theme sáng/tối của YouTube.

| File | Việc |
|---|---|
| `src/background.js` | Service worker: chuyển `{type:"ct:fetch", path, method, body}` tới `CT.settings.serverUrl`, trả `{ok, status, data \| error}` |
| `src/api.js` | `CT.api.loadSong`: `POST /analyze`, poll `/jobs/{id}` mỗi giây, `GET /result/{song_id}` |
| `src/theory.js` | Dịch giọng theo ký hiệu (kể cả hợp âm đảo), đơn giản hóa, ẩn bass, nhãn theo chế độ, số chỉ nhịp |
| `src/sync.js` | Vòng `requestAnimationFrame` đọc `video.currentTime` (phát `time`), `CT.seekTo`, toàn màn hình |
| `src/content.js` | Điều phối: videoId khi YouTube điều hướng SPA, chỉ phân tích khi người dùng bật, nhớ video đã bật trong phiên, tự thử lại khi server tắt, dọn khi rời trang video |
| `src/ui/icons.js` | Bộ icon SVG riêng (`CT.icons`): logo và avatar piano cách điệu, metronome có quả lắc xoay theo phách, phím đàn |
| `src/ui/controls_button.js` | Nút ChordTrack trong `.ytp-right-controls` (trạng thái tắt/đang phân tích/bật/lỗi, tooltip), gắn lại khi YouTube vẽ lại thanh điều khiển |
| `src/ui/strip.js` | Dải karaoke: dựng DOM một lần mỗi bài; mỗi khung hình chỉ đặt `transform` và đổi class khi hợp âm đổi |
| `src/ui/panel.js` | Panel dưới player, trạng thái đang phân tích/lỗi, phím tắt, gắn vào DOM YouTube (`MutationObserver`) |
| `src/ui/sections.js` | Bảng cấu trúc: đầu `#secondary`; chế độ rạp hát hoặc trang một cột thì xếp dưới panel |
| `src/ui/fullscreen.js` | Lớp phủ toàn màn hình dùng chính phần tử dải hợp âm |

## Kiểm thử không cần cài extension
`extension/dev/harness.html` giả lập DOM tối thiểu của trang xem video YouTube (`#movie_player` có `<video>` thật phát
audio local, `#below` với `ytd-watch-metadata`, `#secondary`), nạp `background.js` rồi các content script **đúng thứ
tự trong manifest.json**, với shim `chrome.runtime` và `chrome.storage` (lưu vào `localStorage`).

```powershell
# 1. backend (cổng 8765)
powershell -ExecutionPolicy Bypass -File scripts\start_server.ps1
# 2. server tĩnh + proxy /api tới backend (CORS của backend chỉ cho chrome-extension:// và youtube.com)
.venv\Scripts\python.exe extension\dev\serve.py
# 3. mở http://127.0.0.1:8770/watch?v=QDYfEBY9NM4
```

- Thanh trên cùng của harness có sẵn: Let It Be (một giọng), Magia (Kajiura, 5 chỗ đổi giọng), "Đang phân tích"
  (job giả ~15 giây), "Lỗi YouTube" (job giả đòi cookie), "Server tắt" (proxy đóng kết nối), "Trang chủ" (không
  phải trang video), nút đổi theme và chế độ rạp hát.
- Tham số URL: `dark=0|1`, `theater=1`, `down=1`, `w=900` (bó trang vào 900 px để thử bố cục một cột khi không đổi
  được kích thước cửa sổ), `measure=1` (xem dưới).
- Trình duyệt kiểm thử chạy ở máy khác hoặc không vào được 127.0.0.1 (ví dụ Claude in Chrome trên máy này):
  `serve.py --host <IP LAN> --port 8771`. Lệnh này để proxy lộ ra mạng LAN, chỉ dùng lúc thử rồi tắt.
- Nút ⛶ của player giả gọi `requestFullscreen`; nếu trình duyệt từ chối (trình duyệt điều khiển tự động hay bị),
  harness giả lập toàn màn hình: player phủ kín cửa sổ và phát sự kiện `fullscreen` như `sync.js`.

**Đo độ lệch đồng bộ:** phát nhạc, rồi gõ `ctSync.report()` trong console. Mỗi lần hợp âm hiện tại đổi, harness lấy
`video.currentTime` lúc đổi trừ mốc `start` của hợp âm. Cửa sổ bị che hoặc không có focus thì Chrome bóp
`requestAnimationFrame` xuống 1–3 khung/giây và số đo vô nghĩa; khi đó mở với `&measure=1` để gọi thêm chính
`CT.sync.tick` (thân vòng rAF) ở 60 Hz.

**Test đơn vị** (`theory.js`, dùng thêm `result.json` thật trong `data/cache` nếu có):
`node --test extension/dev/theory.test.js`
