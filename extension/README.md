# ChordTrack – extension Chrome

## Cài (chế độ nhà phát triển)
1. Chạy backend: `powershell -ExecutionPolicy Bypass -File scripts\start_server.ps1`
2. Mở `chrome://extensions`, bật **Developer mode**, bấm **Load unpacked**, chọn thư mục `extension/` này.
3. Mở một video YouTube.

## Cấu trúc
Không có bước build: JavaScript thuần, nạp theo thứ tự trong `manifest.json`. Mọi module gắn vào `window.CT`
và giao tiếp qua `CT.bus` (xem đầu `src/core.js`). Class CSS có tiền tố `ct-`.

## Phát triển và kiểm thử không cần cài extension
`extension/dev/harness.html`: trang giả lập YouTube (có `<video>` thật với audio local) nạp đúng các script của
content script, gọi backend thật. Xem hướng dẫn trong file.
