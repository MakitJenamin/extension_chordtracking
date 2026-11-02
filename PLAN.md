# Kế hoạch: Extension nhận diện hợp âm cho YouTube

Cập nhật lần cuối: 2026-11-02
Trạng thái: **đã chốt kế hoạch, chưa bắt đầu code.** Việc tiếp theo là cài môi trường rồi làm bước 1.

---

## 1. Mục tiêu

Làm Chrome extension hiện hợp âm chạy theo video YouTube. Ưu tiên số một là **chất lượng hợp âm**:
- Nhận đầy đủ hợp âm 7, sus, dim và **hợp âm đảo** (C/E, G/B…).
- Chia đoạn (intro, verse, chorus) và tìm vòng hợp âm của từng đoạn.
- Chế độ "đơn giản hóa hợp âm" (Cmaj7 → C) để làm sau, dưới dạng tùy chọn.

## 2. Các quyết định đã chốt

| Vấn đề | Quyết định |
|---|---|
| Nguồn audio | `yt-dlp` cho server tự tải audio. **Chỉ dùng cá nhân.** Nếu phát hành công khai thì phải xem lại, có thể chuyển sang `tabCapture`, vì tự tải audio vi phạm điều khoản của YouTube |
| Phần cứng | Chạy local trên **RTX 3060 Ti, 8 GB VRAM**. Sau này cần thì đưa lên cloud |
| Mức chi tiết hợp âm | Đầy đủ: 7, sus, dim, hợp âm đảo. Đơn giản hóa làm sau |
| Tách bass (bước 3) | **Giữ, bật mặc định.** Bắt buộc phải có để nhận hợp âm đảo |
| Nơi đặt code | `C:\Users\MakitJenamin\Documents\extension_chordtracking` |
| Hàng đợi job | Bản local dùng hàng đợi lưu bằng SQLite, vì Redis chạy trên Windows khá phiền. Khi lên cloud mới đổi sang Redis |
| VRAM 8 GB | Chạy từng model một và giải phóng VRAM sau mỗi model. BS-Roformer xử lý audio theo từng đoạn ngắn |

## 3. Kiến trúc

```
Extension (Chrome MV3) ──HTTP──► Backend Python (FastAPI) ──► Hàng đợi job ──► Worker GPU
        ▲                              │                                      (pipeline AI)
        └──── JSON hợp âm + vòng ◄─────┴── Cache theo videoId ◄───────────────┘
```

- Pipeline AI viết thành **package Python độc lập**, để sau này bọc vào Docker là đưa lên cloud được.
- Phân tích **cả bài một lần**, không xử lý theo thời gian thực. Có cả bài thì mới tìm được giọng, cấu trúc, vòng hợp âm và mới làm mượt được kết quả.

## 4. Pipeline AI

| # | Bước | Công cụ | Mục đích |
|---|---|---|---|
| 0 | Tải audio | `yt-dlp` | Lấy audio từ `videoId` |
| 1 | Chuẩn hóa audio | ffmpeg, librosa | Đưa mọi bài về cùng sample rate và độ lớn |
| 2 | Tách giọng | **BS-Roformer-Viperx-1297** | Bỏ giọng hát. Giai điệu hát là nguồn gây sai hợp âm lớn nhất |
| 3 | Tách bass | **HTDemucs**, chạy trên phần instrumental | Lấy nốt bass để nhận hợp âm đảo và xác định đúng nốt gốc |
| 4 | Bắt nhịp và phách mạnh | **beat_this** (dự phòng: madmom DBN) | Dựng lưới ô nhịp để hợp âm đổi đúng chỗ |
| 5 | Nhận diện giọng (key) | Suy ra từ kết quả hợp âm, hoặc dùng essentia/madmom | Đổi sang bậc La Mã, gợi ý capo, làm cơ sở cho bước làm mượt |
| 6 | **Nhận diện hợp âm** | Kết hợp nhiều model, xem mục 5 | Đoán hợp âm theo thời gian |
| 7 | Hậu xử lý | Bỏ phiếu theo phách, Viterbi có ưu tiên theo giọng, độ dài tối thiểu | Loại hợp âm chớp nháy và các hợp âm ngoài giọng khó xảy ra |
| 8 | Độ tin cậy | Xác suất từ model | Đánh dấu hợp âm "không chắc" |
| 9 | Cấu trúc và vòng hợp âm | Self-similarity matrix, đếm n-gram trên bậc La Mã | Chia đoạn và tìm vòng lặp lại của từng đoạn |

## 5. Model nhận diện hợp âm (ACR, Automatic Chord Recognition)

### Các model đã tìm hiểu

| Model | Tác giả, năm | Nhận hợp âm đảo | Có weights | Ghi chú |
|---|---|---|---|---|
| **ChordMini (BTC-PL)** | Nghia Phan và cộng sự, 2026 (DAFx26) | ❌ (170 loại hợp âm, không có nốt bass) | ✅ MIT | BTC được huấn luyện lại bằng pseudo-label trên hơn 1.000 giờ nhạc. Vượt BTC gốc khoảng 1–3%, rõ nhất ở hợp âm 7 và hợp âm 4 nốt |
| **Chord-CNN-LSTM** | Junyan Jiang và cộng sự (Music X Lab), ISMIR 2019 | ✅ (đoán riêng nốt bass) | ✅ (nằm trong repo ChordMiniApp) | Đoán riêng từng phần của hợp âm: nốt gốc, loại, bass, 7… |
| BTC | Park và cộng sự, 2019 | ❌ | ✅ | Đã có ChordMini là bản tốt hơn nên không dùng làm model chính |
| ChordFormer | Akram và cộng sự, 2025 | ✅ | ❌ **không có code/weights** | Muốn dùng thì phải tự huấn luyện, quá tốn công. Để sau |

Kết quả của ChordMini (bộ test trong bài báo của họ):

| Model | Root | MajMin | Sevenths | Tetrads |
|---|---|---|---|---|
| BTC gốc | 81.9 | 79.0 | 66.3 | 63.7 |
| ChordMini (BTC-PL) | 83.0 | 80.2 | 69.4 | 67.0 |

⚠️ Mỗi bài báo đo trên một bộ test khác nhau, nên không so trực tiếp con số giữa các bài báo được. Phải tự đo bằng script đánh giá của dự án.

### Cách kết hợp đã chốt

1. **ChordMini (BTC-PL)**: model chính, đoán loại hợp âm (C, Am7, Gsus4…).
2. **Chord-CNN-LSTM**: đoán nốt bass và hợp âm đảo.
3. **Phần bass tách ở bước 3**: kiểm tra chéo nốt bass.
4. Ghép kết quả. Ví dụ: ChordMini đoán `C`, bass đoán `E`, ra `C/E`.
5. Chọn cách kết hợp tốt nhất **dựa trên số đo**, không theo cảm tính.

Có thể tham khảo app **ChordMiniApp** (MIT): nó dùng `yt-dlp` cùng các model nhận hợp âm và bắt nhịp, gần giống ý tưởng của dự án này. Nhưng nó không tách giọng bằng BS-Roformer, không tách bass bằng Demucs. Đây là lợi thế chất lượng của dự án.

## 6. Backend

| Phần | Mục đích |
|---|---|
| FastAPI: `POST /analyze`, `GET /result/{videoId}` | Cổng duy nhất cho extension gọi vào |
| Hàng đợi job và trạng thái tiến độ | Mỗi bài mất khoảng 1 phút nên không xử lý trực tiếp trong request. Local: SQLite. Cloud: Redis kèm RQ hoặc Celery |
| Cache theo `videoId` | Mỗi bài chỉ phân tích một lần. Local: SQLite. Cloud: Postgres |

## 7. Extension (Chrome Manifest V3)

| Phần | Mục đích |
|---|---|
| Content script trên trang YouTube | Lấy `videoId` và đọc `video.currentTime` |
| Khung hợp âm dưới video | Làm nổi hợp âm đang phát, hiện vài hợp âm tiếp theo |
| Bảng "Vòng hợp âm" | Hiện vòng của từng đoạn, ví dụ Chorus: C – G – Am – F (I–V–vi–IV) |
| Transpose, capo | Đổi giọng cho dễ đánh |
| Nút "đơn giản hóa" | Làm sau |
| Bấm vào hợp âm để tua video | Nhảy video tới đúng chỗ đó để luyện |

## 8. Đo chất lượng (quan trọng nhất)

- Dữ liệu có sẵn đáp án: Isophonics/Beatles, McGill Billboard, cùng vài bài Việt tự ghi hợp âm.
- Chấm bằng `mir_eval` theo nhiều mức: root, majmin, sevenths, tetrads, mirex, và **có tính hợp âm đảo**.
- Mỗi thay đổi (model tách nguồn, cách kết hợp model, tham số làm mượt) đều phải có số đo cụ thể.

## 9. Thứ tự làm

1. **Bản thử nghiệm Python dạng CLI:** đưa vào file audio hoặc `videoId`, ra JSON hợp âm. Kèm script đánh giá. ← **làm tiếp theo**
2. **Tinh chỉnh chất lượng** dựa trên số đo: kết hợp model, làm mượt, nhận hợp âm đảo.
3. **Backend API:** hàng đợi và cache.
4. **Extension:** giao diện và đồng bộ theo video.
5. **Chế độ dự phòng** khi server bận hoặc không có GPU.
6. (Sau này) Đơn giản hóa hợp âm, deploy lên cloud, cân nhắc ChordFormer nếu có weights.

## 10. Môi trường máy (kiểm tra ngày 2026-11-02)

| Thành phần | Thông số | Đánh giá |
|---|---|---|
| GPU | RTX 3060 Ti, 8 GB VRAM, driver 595.79 (CUDA 12) | ✅ Đủ, nếu chạy từng model một |
| RAM | 32 GB | ✅ Dư |
| CPU | i5-12400F, 6 nhân / 12 luồng | ✅ |
| Ổ C còn trống | 64 GB | ✅ Cần khoảng 15–25 GB, nên để ý dung lượng |

- Thời gian ước tính: khoảng 40–70 giây mỗi bài 4 phút, chỉ lần đầu (bài đã phân tích thì lấy từ cache). Bước 1 sẽ đo con số thật.
- Mỗi lần chỉ phân tích một bài vì VRAM có 8 GB. Hàng đợi job xếp lần lượt.
- **Tất cả chạy local.** Internet chỉ dùng để `yt-dlp` tải audio và tải weights hoặc thư viện lúc cài đặt. Không dùng API cloud. Không dùng phần Firebase và Gemini của ChordMiniApp, chỉ lấy weights model.

Đã cài (2026-11-02):
- ✅ Python 3.11.9, gọi bằng `py -3.11`. Lệnh `python` có thể vẫn trỏ tới bản giả của Microsoft Store, nên luôn dùng venv.
- ✅ ffmpeg 9.0.2 (Gyan.FFmpeg, qua `winget`).
- ✅ git 2.49, gh 2.103 (đã đăng nhập tài khoản `MakitJenamin`).
- ✅ venv `.venv` (Python 3.11), PyTorch 2.14.1+cu126 nhận GPU, `yt-dlp`, các thư viện trong `requirements.txt`.
- ⚠️ ffmpeg được thêm vào PATH của user, nên shell cũ phải nạp lại PATH thì `yt-dlp` mới tìm thấy ffmpeg.
- ⚠️ Phải giữ **numpy 1.26.4**. Cài `seaborn` sẽ kéo numpy lên 2.x và pandas lên 3.x, gây xung đột (đã ghim `pandas<3`).

## Quy ước làm việc

- Repo GitHub **public**: `MakitJenamin/extension_chordtracking`.
- **Làm xong việc gì thì commit và push lên đó.**
- Không commit audio, weights model, dữ liệu đánh giá có bản quyền hay cache (xem `.gitignore`).

## 11. Việc cần kiểm tra khi bắt đầu

- [x] Lấy weights. Không cần ChordMiniApp: clone thẳng 2 repo model vào `third_party_src/` (đã gitignore). Weights có sẵn trong repo, không cần LFS.
  - `https://github.com/ptnghia-j/ChordMini`, `checkpoints/`: `btc_model_best.pth` (BTC-PL, model chính), `2e1d_model_best.pth` (ChordNet 2E1D), `btc_model_large_voca.pt` (BTC gốc).
  - `https://github.com/ptnghia-j/chord-cnn-lstm-model`, `cache_data/`: 5 file `.sdict`, là ensemble 5 model chạy rồi lấy trung bình.
- [x] Chạy thử cả hai model trên "Let It Be" (videoId `QDYfEBY9NM4`, dài 243 giây). **Cả hai ra đúng vòng C – G – Am – F.**
  - ChordMini BTC-PL: khoảng **12 giây** mỗi bài trên GPU, bắt được `F:maj7`. Lệnh chạy: `src/evaluation/test.py --model_type BTC --checkpoint checkpoints/btc_model_best.pth --config config/ChordMini.yaml --smooth_logits --use_overlap --use_gaussian --kernel_size 9 --vote_aggregation logit --min_segment_duration 0.5 --smooth_predictions`
  - Chord-CNN-LSTM: `chord_recognition.py <audio> <out.lab> [submission|ismir2017|full]`.
- [x] Định dạng nốt bass của Chord-CNN-LSTM: ký hiệu Harte có **bậc của bass**, ví dụ `C:maj/5` = C/G, `C:maj/3` = C/E, `F:maj/2` = F/G. Đã bắt đúng **C/G** ở đoạn điệp khúc Am – C/G – F – C.
- [x] Lấy xác suất từng khung: `chordtrack/chords/chordmini.py` và `cnnlstm.py` gọi model trực tiếp trong Python, `segments` trùng file `.lab` của script gốc.
  - ChordMini: `frame_probs` (T, 170), khung 0.0929 s, là softmax của trung bình logits đã làm mượt. Script gốc **bỏ đoạn ngắn hơn 0.5 s** nên giữa các đoạn có khoảng trống. Nhãn trưởng của ChordMini ghi tắt (`C`), wrapper đổi sang `C:maj`.
  - CNN-LSTM: `frame_probs` (T, 301) trên từ điển của HMM (N + 25 mẫu `submission` x 12 nốt gốc), khung 0.0232 s, là tích xác suất các thành phần chuẩn hóa lại (không có hệ số giảm N 0.5 của bản gốc). `extras` có `bass_probs`, `root_probs` (T, 13), `triad_probs` và 7th/9th/11th/13th.
  - Khi đã nạp sẵn: BTC 0.6 s, VRAM đỉnh 79 MiB; 2E1D 0.5 s, 35 MiB; CNN-LSTM 1.9 s, 509 MiB (lần đầu mỗi tiến trình 3–5 s).
  - Mã CNN-LSTM đọc file theo đường dẫn tương đối và **dùng weights ngẫu nhiên mà không báo lỗi** nếu không thấy file `.sdict`. Wrapper chdir vào repo khi import và kiểm tra weights đã nạp.
- [ ] Kiểm tra BS-Roformer-Viperx-1297 chạy vừa 8 GB VRAM (qua `audio-separator`).
- [ ] Tìm nguồn dữ liệu đánh giá: file nhãn hợp âm cùng audio tương ứng.
- [x] Bắt nhịp bằng beat_this (checkpoint `final0`, hậu xử lý minimal, không cần madmom) chạy được trên Windows, Python 3.11, numpy 1.26. "Let It Be": 279 phách, 78 phách mạnh (tỉ lệ 0,28), tempo trung vị 69,8 BPM, khoảng 1 giây trên GPU, VRAM đỉnh khoảng 270 MB.
  - Tempo trôi từ khoảng 73 BPM (phút đầu) xuống khoảng 68 BPM (cuối bài). Đầu ra theo khung 50 fps nên khoảng cách phách bị lượng tử 20 ms, tempo lệch theo bước khoảng 1,5 BPM.
  - ⚠️ Đoạn 95–118 giây, beat_this đặt phách mạnh cách 2 phách thay vì 4. Bước hậu xử lý hợp âm không nên tin tuyệt đối vào phách mạnh, có thể ép lại chu kỳ 4 phách.

## Nguồn tham khảo

- ChordMini: https://arxiv.org/abs/2602.19778
- ChordMiniApp: https://github.com/ptnghia-j/ChordMiniApp
- ChordMini ONNX: https://huggingface.co/musetric/chordmini-onnx
- ChordFormer: https://arxiv.org/abs/2502.11840
