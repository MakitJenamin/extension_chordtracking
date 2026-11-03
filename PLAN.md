# Kế hoạch: Extension nhận diện hợp âm cho YouTube

Cập nhật lần cuối: 2026-11-03
Trạng thái: **bản đầu tiên đã hoàn thiện, đang dùng được (2026-11-03).** Pipeline local (ChordMini + Chord-CNN-LSTM, không tách nguồn mặc định, khoảng 10 giây mỗi bài), backend FastAPI ở 127.0.0.1:8765, extension Chrome đã thử trên YouTube thật (PR #1–#24). Việc để sau: xem mục 9 bước 6.

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
| Tách bass (bước 3) | **Tắt mặc định, giữ tùy chọn** (`--separation`), từ 2026-11-03. Hợp âm đảo đến từ `bass_probs` của CNN-LSTM; stem bass chỉ thêm +0.1 điểm mà tốn 84/89 giây mỗi bài (mục "Kết quả đánh giá Kajiura") |
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
| ChordFormer | Akram và cộng sự, 2025 (IEEE TASLP 2026) | ✅ | ⚠️ repo không license, tác giả đã xóa checkpoint; lấy lại từ lịch sử git, chỉ để trong máy | Đã thử 2026-11-04: trên Kajiura ngang CNN-LSTM, không đưa vào mặc định (mục "Thử ChordFormer và consonance-ACE") |
| consonance-ACE | Poltronieri và cộng sự, ISMIR 2025 | ✅ (đoán riêng bass) | ✅ MIT | Đã thử 2026-11-04: kém hơn hẳn trên cả hai bộ, không dùng |

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

**Đã làm (2026-11-03):** `chordtrack/server.py` (FastAPI, `python -m chordtrack.server` hoặc
`scripts/start_server.ps1`) và `chordtrack/jobs.py`. Cách dùng ở `README.md`.

- Chỉ bind `127.0.0.1:8765`. Chỉ nhận Host `127.0.0.1`/`localhost` (chống DNS rebinding). CORS chỉ cho
  `chrome-extension://<id>` và `https://www.youtube.com`, có `Access-Control-Allow-Private-Network` cho Chrome.
  `POST /analyze` bắt buộc JSON, nên trang web lạ không tạo được job.
- Endpoint: `GET /health`, `POST /analyze`, `GET /jobs/{id}`, `GET /result/{song_id}`, `GET /songs`.
  Chỉ nhận videoId/URL YouTube, không nhận đường dẫn file.
- Cache kết quả vẫn là file `data/cache/<song_id>/result.json` (đúng `code_revision`). SQLite
  (`data/cache/jobs.sqlite`) chỉ lưu job và thời gian từng bước, dùng ước tính `progress` và `eta_seconds`.
- Một worker thread. Job `running` khi server tắt thì quay về `queued` lúc khởi động lại. Lỗi được đổi thành
  câu tiếng Việt (`jobs.friendly_error`): đòi cookie, video không tồn tại, thiếu Deno/ffmpeg, hết VRAM…
- Model hợp âm và beat_this nạp một lần mỗi tiến trình. Model giữ trên CPU, chỉ chuyển lên GPU trong
  `gpu_lock()` rồi trả về CPU, nên server không chiếm VRAM lúc rảnh. Riêng tách nguồn (audio-separator,
  HTDemucs) vẫn nạp lại mỗi bài.

## 7. Extension (Chrome Manifest V3)

| Phần | Mục đích |
|---|---|
| Content script trên trang YouTube | Lấy `videoId` và đọc `video.currentTime` |
| Khung hợp âm dưới video | Làm nổi hợp âm đang phát, hiện vài hợp âm tiếp theo |
| Bảng "Vòng hợp âm" | Hiện vòng của từng đoạn, ví dụ Chorus: C – G – Am – F (I–V–vi–IV) |
| Transpose, capo | Đổi giọng cho dễ đánh |
| Nút "đơn giản hóa" | Làm sau |
| Bấm vào hợp âm để tua video | Nhảy video tới đúng chỗ đó để luyện |

### Giao diện đã chốt với người dùng (2026-11-03, theo mockup)

Mockup tham chiếu: dải hợp âm trượt ngang kiểu karaoke dưới video (có vạch đỏ cố định ở khoảng 28% chiều rộng), bảng "Cấu trúc bài" bên phải, thanh tua có dải màu theo đoạn và vạch vàng ở chỗ đổi giọng.

- **Ba chế độ hiển thị:** Gốc (có Tông −/+), **🎹 Phím trắng** (mặc định cho người dùng, vì người dùng chơi piano), Bậc La Mã. Tùy chọn: ẩn hợp âm đảo, đơn giản hóa.
- **Phím trắng:** ô "Đặt transpose đàn: +k" theo vùng giọng đang phát. Báo trước 8 giây khi sắp đổi giọng. Trên dải hợp âm có vạch "⬆ transpose +a → +b".
- **Nhịp:** hiện số chỉ nhịp (ví dụ "Nhịp 4/4", suy ra từ số phách giữa các phách mạnh) và các chấm 1-2-3-4 sáng theo phách, phách 1 đỏ và to hơn. Trên dải hợp âm có **vạch dọc ở mỗi phách**: vạch cao kèm số ô nhịp ở phách mạnh, vạch thấp ở phách nhẹ. JSON có `meter` (số nguyên, số phách mỗi ô nhịp; xem mục 11, "Metronome, popup và `meter`").
- **Metronome:** tiếng phách đồng bộ với `beats` của video. Phách mạnh cao và to hơn. Có nút bật/tắt và **thanh chỉnh âm lượng**. Tiếng tạo bằng Web Audio. Phải theo đúng tốc độ phát của video (0.5×, 0.75×…) và khi tua.
- **Toàn màn hình:** khi video YouTube ở chế độ toàn màn hình, đưa **chính dải hợp âm karaoke như lúc bình thường** (cùng vạch phách, vạch đổi giọng, vạch đỏ đang phát) đè lên phía dưới video, nền bán trong suốt. Phía trên dải có chấm phách, giọng hoặc transpose, và cảnh báo sắp đổi giọng. Phải gắn vào phần tử fullscreen của YouTube thì mới hiện được.
- ⚠ đánh dấu hợp âm có `uncertain = true`.

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
   - **Tạm gác (người dùng quyết định 2026-11-03):** thêm ChordMini 2E1D làm model thứ ba vào bộ kết hợp. Chỉ làm lại khi người dùng yêu cầu.

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
- [x] Kiểm tra BS-Roformer-Viperx-1297 chạy vừa 8 GB VRAM (qua `audio-separator`). Trên "Let It Be" (243 giây): BS-Roformer 50 giây, VRAM đỉnh 1,6 GB; `htdemucs_ft` trên instrumental 29 giây, VRAM đỉnh 0,55 GB (`torch.cuda.max_memory_allocated`).
  - `audio-separator` từ 0.31 đòi numpy 2, nên ghim **0.30.2**. Bản này ghi file theo cwd, bỏ qua `output_dir`, nên phải truyền đường dẫn tuyệt đối trong `custom_output_names`.
  - `no_bass.wav` = drums + other của HTDemucs, bỏ stem vocals (giọng còn sót sau Roformer).
- [x] Tìm nguồn dữ liệu đánh giá: file nhãn hợp âm cùng audio tương ứng. Chi tiết ở `eval/README.md`.
  - Nhãn **Isophonics The Beatles** (180 bài) cộng audio "Remastered 2009" tải từ YouTube. Bảng ánh xạ ở `eval/isophonics_youtube.csv` (43 bài).
  - **Phải căn thời gian.** Audio YouTube lệch nhãn tới 0.8 s ở vài bài. Không căn thì điểm majmin của ChordMini tụt từ 86.3 xuống 80.9, riêng "Eight Days a Week" tụt 31 điểm. Căn bằng tương quan chroma CENS giữa nhãn và audio.
  - Đạt **31 bài** (88 phút). Loại 12 bài vì độ dài lệch quá 2 s, phần lớn do đuôi fade-out khác nhau.
- [x] **Baseline ChordMini BTC-PL** trên 31 bài Isophonics (2026-11-02), chạy `python -m eval.run_chordmini`, WCSR gia quyền theo thời lượng (%):

  | root | majmin | majmin_inv | thirds | triads | sevenths | sevenths_inv | tetrads | mirex | seg |
  |---|---|---|---|---|---|---|---|---|---|
  | 86.79 | 86.30 | 81.35 | 85.15 | 83.14 | 79.27 | 76.26 | 74.20 | 86.64 | 82.02 |

  - Chạy 31 bài trong 14 s trên GPU, kể cả lúc nạp model. VRAM của cả GPU tăng từ 1.1 GB lên 1.4 GB.
  - Mức có hợp âm đảo thấp hơn mức không đảo 5 điểm (majmin 86.3, majmin_inv 81.4), do BTC-PL không đoán nốt bass. Đây là phần Chord-CNN-LSTM và tách bass cần bù.
  - ⚠️ Beatles có thể nằm trong dữ liệu huấn luyện của ChordMini, nên điểm này có thể cao hơn thực tế. Dùng để so sánh giữa các biến thể, không dùng làm điểm tuyệt đối.
- [x] Bắt nhịp bằng beat_this (checkpoint `final0`, hậu xử lý minimal, không cần madmom) chạy được trên Windows, Python 3.11, numpy 1.26. "Let It Be": 279 phách, 78 phách mạnh (tỉ lệ 0,28), tempo trung vị 69,8 BPM, khoảng 1 giây trên GPU, VRAM đỉnh khoảng 270 MB.
  - Tempo trôi từ khoảng 73 BPM (phút đầu) xuống khoảng 68 BPM (cuối bài). Đầu ra theo khung 50 fps nên khoảng cách phách bị lượng tử 20 ms, tempo lệch theo bước khoảng 1,5 BPM.
  - ⚠️ Đoạn 95–118 giây, beat_this đặt phách mạnh cách 2 phách thay vì 4. Bước hậu xử lý hợp âm không nên tin tuyệt đối vào phách mạnh, có thể ép lại chu kỳ 4 phách.
- [x] Pipeline end-to-end và CLI (2026-11-03): `python -m chordtrack.cli <videoId|url|file> [-o out.json] [--force] [--separation] [--chord-stem ...] [--beat-stem ...]` (từ 2026-11-03 mặc định không tách nguồn; `--no-separation` vẫn nhận). Kết quả cache ở `data/cache/<song_id>/result.json`; dạng JSON kiểm tra trong `tests/test_pipeline.py::check_schema`.
  - Bài mới hoàn toàn "Nơi này có anh" (`FN7ALfpGxiI`, 278 giây), tính cả thời gian chờ khóa GPU do agent khác: tải 9,6 s, tách nguồn 178 s (Roformer 58 s + Demucs 47 s chạy thật, chờ khóa 57 s), bắt nhịp 17 s, hợp âm 6,8 s, tổng 211 s. VRAM đỉnh 1,6 GB (Roformer).
  - "Let It Be" khi đã có stem: tổng 44 s, phần lớn là bắt nhịp (27 s, gồm cả chờ khóa và nạp model).
  - `audio-separator` chạy được khi không có mạng: nạp BS-Roformer thẳng từ ckpt và yaml trong `data/models/audio-separator`, không đọc `download_checks.json`.
- [x] **Kết quả analysis** (bước 5 và 9, `chordtrack/analysis.py`, 2026-11-03). Chấm bằng `python -m eval.eval_analysis` trên nhãn chuẩn Isophonics (chordlab làm đầu vào, keylab và seglab làm đáp án; script tự tải keylab và seglab). Chỉ dùng CPU: 180 bài mất khoảng 4 giây.
  - Giọng: so phân bố cao độ của hợp âm với profile Temperley, cộng điểm cho hợp âm chủ (thời lượng, đầu và cuối bài) và quãng V–I. Giọng chuẩn là giọng chiếm thời lượng lớn nhất trong keylab (178 bài, bỏ 2 bài "modal" hoặc thiếu nhãn).

    | Đầu vào | Số bài | weighted (`mir_eval.key`) | Đúng tuyệt đối | Quãng 5 | Tương đối | Song song | Khác |
    |---|---|---|---|---|---|---|---|
    | chordlab | 178 | **0.870** | **84.8%** | 5 | 2 | 4 | 16 |
    | ChordMini BTC-PL (31 bài có audio) | 31 | 0.819 | 80.6% | 0 | 0 | 2 | 4 |
    | chordlab, cùng 31 bài | 31 | 0.819 | 80.6% | 0 | 0 | 2 | 4 |

    Trên 31 bài, giọng đoán từ hợp âm của ChordMini đúng y như từ nhãn chuẩn. Lỗi còn lại chủ yếu là: bài mixolydian hoặc blues bị nhận thành giọng bậc IV (Drive My Car, Get Back, The Word), và vài bài có nhãn hợp âm lệch nửa cung so với keylab (Yellow Submarine, Ob-La-Di, Piggies, Across the Universe), có thể do bản thu bị lệch cao độ. Trọng số được chỉnh bằng dò lưới trên chính bộ này, mặt điểm khá phẳng (0.84–0.87) nên ít nguy cơ overfit.
  - Đoạn: structure features (Serrà và cộng sự, 2012) trên chuỗi hợp âm theo phách, ranh giới kéo về điểm đổi hợp âm gần nhất, gom nhãn A/B/C theo độ giống chuỗi hợp âm. Chấm `mir_eval.segment`, detection có `trim=True` (không tính ranh giới đầu và cuối bài). Đánh giá dùng lưới cố định 0,5 s vì không có phách (bộ dữ liệu không có audio cho phần lớn bài).

    | Đầu vào | F@0.5 | P@3 | R@3 | **F@3** | **pair F** | S_F (NCE) | Số đoạn đoán / chuẩn |
    |---|---|---|---|---|---|---|---|
    | chordlab, 180 bài | 0.217 | 0.512 | 0.510 | **0.495** | **0.583** | 0.604 | 10.7 / 10.0 |
    | Baseline chia đều 15 s | 0.047 | 0.298 | 0.303 | 0.292 | 0.355 | 0.526 | 10.8 / 10.0 |
    | ChordMini BTC-PL, 31 bài | 0.207 | 0.494 | 0.462 | 0.466 | 0.568 | 0.610 | 10.8 / 10.7 |
    | chordlab, cùng 31 bài | 0.237 | 0.508 | 0.487 | 0.481 | 0.580 | 0.617 | 10.7 / 10.7 |

    - ⚠️ Cửa sổ so sánh nhìn về phía trước làm ranh giới sớm hơn đáp án 2–3 s một cách hệ thống. Đã dời ma trận lặp nửa cửa sổ, F@3 tăng từ 0.445 lên 0.495.
    - Chỉ dùng hợp âm nên không tách được intro với verse khi chúng cùng vòng (ví dụ "Let It Be"). Muốn tốt hơn phải thêm đặc trưng âm thanh (năng lượng, âm sắc, có giọng hát hay không).
    - Tên đoạn (intro, verse, chorus, bridge, outro) là heuristic, `mir_eval` không chấm phần này. Đoạn không đặt tên được thì giữ chữ cái.
  - Vòng hợp âm: n-gram 2–8 hợp âm (đã rút gọn về hợp âm ba và bỏ hợp âm trùng liên tiếp), đếm gộp trên mọi đoạn cùng nhãn, phải phủ ít nhất 60% chuỗi. Bậc La Mã: hợp âm đảo ghi theo bậc bass của Harte (`I/5`), không dùng figured bass.
  - "Let It Be": chordlab cho giọng C major, verse `I – V – vi – IVmaj7 – I – V – IV`. Nhãn của ChordMini trên audio thật kèm phách beat_this cũng ra C major, verse `I – V – vi – IV – I`, analysis mất 16 ms.
  - **Giọng theo vùng** (đổi giọng, nâng tông; `estimate_key_regions`, 2026-11-03). Chấm 24 giọng trên cửa sổ trượt 16 s
    căn giữa mỗi ô nhịp (điểm `_key_scores` bỏ phần hợp âm đầu/cuối, cộng 2 x tỉ lệ thời lượng hợp âm nằm trọn trong
    âm giai), giải Viterbi với phạt đổi giọng 12 (đổi sang giọng tương đối/song song x0.7), gộp vùng < 4 ô và < 8 s,
    tinh chỉnh chỗ đổi trong ± 2 ô (ít hợp âm ngoài âm giai nhất, ưu tiên chỗ hợp âm chủ mới bắt đầu, rồi ranh giới
    đoạn), rồi chấm lại giọng từng vùng. Bài một vùng thì giọng đúng bằng `estimate_key`.
    Chấm bằng `python -m eval.eval_analysis --key-regions`: tỉ lệ thời gian đúng giọng, `mir_eval.key.weighted_score`
    theo từng khúc, gia quyền thời lượng, trên nhãn hợp âm chuẩn. "tr" là tỉ lệ thời gian đúng giá trị transpose phím
    trắng (đúng giọng hoặc chỉ nhầm sang giọng tương đối).

    | Bộ | Nhóm | Số bài | 1 giọng: weighted / tr | Theo vùng: weighted / tr | Số vùng (đoán / chuẩn) |
    |---|---|---|---|---|---|
    | Isophonics (keylab) | cả bộ | 178 | 0.852 / 0.847 | 0.841 / 0.842 | 1.33 / 1.14 |
    | | **có đổi giọng** | 21 | 0.600 / 0.635 | **0.683 / 0.740** | 2.24 / 2.19 |
    | | một giọng | 157 | 0.888 / 0.877 | 0.863 / 0.857 | 1.21 / 1.00 |
    | Kajiura (ChordWiki `{key:}`) | cả bộ | 17 | 0.384 / 0.427 | **0.644 / 0.726** | 5.35 / 2.82 |
    | | **có đổi giọng** | 12 | 0.368 / 0.398 | **0.704 / 0.797** | 6.33 / 3.58 |
    | | một giọng | 5 | 0.425 / 0.496 | 0.499 / 0.554 | 3.00 / 1.00 |

    - Giọng chuẩn của Kajiura: dấu `{key:...}` trong `data/eval/kajiura/sheets/*.cho`, gán cho từng hợp âm rồi căn với
      `ref/*.lab` bằng difflib (`kajiura_key_reference`; bản hợp âm đọc bằng `eval.kajiura.load_sheet`). 15 bài không ghi `{key:}` nên không chấm. Bài chỉ ghi một
      giọng có thể do người chép lười ghi đổi giọng, nên nhóm "một giọng" của Kajiura không đáng tin.
    - Bài Kajiura tốt: akeboshi 0.50 → 0.99, magia 0.73 → 0.99, heavenly_blue 0.25 → 0.94, "to the beginning"
      0.00 → 0.69 (D#m, Cm, Am, F#m, Em, Am, F#m, C, F#m, D#m: tìm ra 7 vùng, gộp mất đoạn Cm 12 s và Em 12 s).
    - Đánh đổi: bài Beatles một giọng mất 0.025 (13% bài bị tách thêm một vùng, thường là đoạn bridge sang giọng IV
      hoặc tương đối). Phạt cao hơn (24) giữ được Beatles (0.849) nhưng Kajiura chỉ còn 0.630 và bỏ sót vùng nâng
      tông ngắn hơn 32 s. Chọn bộ hiện tại vì nhạc người dùng nghe đổi giọng nhiều.
    - ⚠️ Không có thành phần "âm giai", vòng I–V–vi–IV nâng +2 bị nhận thành một giọng "dung hòa" (C → D thành G cả
      bài), vì giọng bậc IV chỉ thua giọng thật 0.1–0.2 điểm mỗi ô. Thành phần âm giai sửa được lỗi này.
    - Giọng chính (`key`) vẫn là `estimate_key` trên cả bài: 0.870 trên giọng chính của keylab, hơn lấy vùng dài nhất
      (0.860). Riêng khi bài có nhiều vùng mà giọng đó không thuộc vùng nào (giọng "dung hòa") thì lấy giọng có tổng
      thời lượng lớn nhất (`pipeline.main_key`).
    - Chạy CLI đầy đủ trên audio thật (YouTube đang đòi đăng nhập "not a bot" nên dùng audio có sẵn):
      - "And I Love Her" (keylab: E, lên F ở 88,8 s): vùng E 0–88,98 s, rồi D thứ/F trưởng (cùng transpose +5); in
        "1:28.98 E major -> D minor: đặt transpose đàn: +5 (đang +4)", hai đoạn solo ghi "= đoạn 2 dịch +1".
        Tổng 8,5 s khi stem đã có.
      - "magia" (ChordWiki: Cm, Dm, Cm, Dm): Cm 0–51 s, Dm 51–95 s, Cm 95–123 s, Dm 123–183 s, D trưởng 183–206 s
        (sai, đoạn ngắn dùng hợp âm trưởng), Dm đến hết. 106 s (tách nguồn 99 s), VRAM đỉnh 1,6 GB (Roformer).
      - "to the beginning": 6 vùng, đúng các đoạn Am và F#m; đoạn đầu D#m/Cm bị đọc thành G thứ, đoạn cuối D#m
        thành Eb trưởng. 92 s, VRAM đỉnh 1,6 GB.
      - "Let It Be": một vùng C trưởng, không bị chia sai.
    - Đoạn nâng tông nhận bằng cách so vòng hợp âm ba đã dịch k nửa cung (kể cả xoay vòng) giữa hai vùng khác nhau;
      khi cùng mode thì tương đương so bậc La Mã, nhưng vẫn nhận ra khi một vùng bị đọc thành giọng tương đối.
    - JSON thêm `keys[]` (vùng giọng kèm `white_key_transpose`), `modulations[]` (`time`, `from`, `to`, `semitones`,
      `transpose_from`, `transpose_to`); `chords[]` thêm `white_key_symbol`, `key_index`, và `roman` tính theo giọng
      của vùng; `sections[]` thêm `key_index`, và `transposed_from`, `semitones` khi đoạn là bản nâng tông của đoạn
      trước; đoạn đó (và cả cụm A/B/C của nó) lấy nhãn của đoạn gốc.
    - Chế độ phím trắng (`chordtrack/transpose.py`): transpose +k nghĩa là bấm C đàn phát C+k; hợp âm để bấm = hợp âm
      gốc dịch -k. Giọng trưởng về C, thứ về A, k trong [-6, +5] (F# trưởng, D#/Eb thứ: -6).

- [x] **Kết hợp model và hậu xử lý (bước 6–8), 2026-11-03.** Chi tiết ở mục "Kết quả đánh giá đợt 2" bên dưới.
  - Bộ đánh giá từ 31 lên **42 bài**: bỏ quy tắc loại theo độ dài, thay bằng độ trôi đầu–cuối bài (`eval/README.md`).
  - `fuse()` mặc định tăng majmin_inv **+2.8**, sevenths_inv **+2.1** so với ChordMini (điểm cross-validation).
  - **Chạy model hợp âm trên mix, không trên instrumental:** tách giọng làm ChordMini giảm 3 điểm sevenths, CNN-LSTM giảm 3–4 điểm.
  - Stem bass của HTDemucs chỉ có bass guitar (đoạn piano tự đệm như "Let It Be" giây 36–44 không có gì), và
    bass của McCartney đi nốt lướt nhiều. Nốt nổi bật nhất trong stem chỉ trùng bass của nhãn ở khoảng 49% số khung,
    còn `bass_probs` của CNN-LSTM trùng ở 86% (đo trên 8–10 bài đầu). Stem chỉ có ích khi trộn với trọng số nhỏ (0.3).
- [x] **Bộ đánh giá Kajiura** (2026-11-03, `eval/kajiura.py`, `eval/ablation_kajiura.py`). Chi tiết ở mục
  "Kết quả đánh giá Kajiura" bên dưới và `eval/README.md` mục 6.
  - ChordWiki chặn script bằng Cloudflare: bản hợp âm phải đọc bằng trình duyệt thật (người dùng tự xác minh).
  - yt-dlp bản mới cần **Deno** (JS runtime) và gói `yt-dlp-ejs` (`yt-dlp[default]`) để tải được YouTube. Khi
    YouTube đòi xác minh "not a bot" thì truyền file cookie (`--cookies` hoặc `CHORDTRACK_YT_COOKIES`).
  - **Stem bass của HTDemucs gần như không giúp gì trên cả hai bộ** (+0.10 majmin_inv trên Kajiura, +0.09 trên
    Beatles), trong khi tách nguồn chiếm 84 trên 89 giây xử lý mỗi bài. Đề xuất tắt tách nguồn mặc định.
  - Trên Kajiura, CNN-LSTM một mình tốt hơn `fuse()` mặc định (sevenths +3.9). Bộ tham số chọn trên Beatles
    có thể đã thiên về ChordMini (Beatles có thể nằm trong dữ liệu huấn luyện của ChordMini).
- [x] **Tắt tách nguồn mặc định và chọn lại `FusionParams` trên Beatles + Kajiura** (2026-11-03,
  `eval/refit_fusion.py`). Chi tiết ở mục "Chọn lại tham số fuse trên Beatles + Kajiura" bên dưới.
  - Pipeline mặc định: chuẩn hóa ffmpeg → mix.wav, bắt nhịp và 2 model hợp âm trên mix, `fuse()` không có stem.
    Khoảng **10 giây mỗi bài** (trước đây khoảng 90 giây), VRAM đỉnh khoảng 0.6 GB (CNN-LSTM).
  - Thêm cách trộn loại hợp âm 7 (`quality_weight`): bộ mới lấy loại hợp âm trong họ (C so với C7, Cmaj7) hoàn
    toàn từ CNN-LSTM, gốc + họ trộn 0.6 CNN-LSTM. `fuse` mới gần ngang CNN-LSTM trên Kajiura (CV −0.5),
    Beatles giảm 1.45 (CV) so với `fuse` cũ: chưa đạt trọn mục tiêu, xem đánh đổi ở mục bên dưới.

- [x] **Backend local** (bước 3 của mục 9, 2026-11-03). "Let It Be" khi đã có stem: lần đầu trong tiến trình
  16,5 s (hợp âm 13,0 s, gồm nạp model), lần sau 7,6 s (hợp âm 4,6 s, bắt nhịp 1,3 s). Bài mới "akeboshi" (273 s,
  có tách nguồn): 112 s, trong đó tách nguồn 93 s. VRAM đỉnh 1,6 GB (Roformer), CNN-LSTM 0,58 GB.
  - ⚠️ YouTube vẫn đòi xác minh "not a bot" khi tải bằng yt-dlp từ máy này, nên cần cookie
    (`CHORDTRACK_YT_COOKIES`) để extension phân tích bài mới.
  - `eta_seconds` lấy trung bình 20 lần chạy gần nhất của từng bước, chưa tính độ dài bài. Bật hay tắt tách nguồn
    làm bước `separation` dao động từ 0 đến 100 s, nên ETA cần vài bài để ổn định.
- [x] **Metronome, popup và `meter`** (2026-11-03).
  - JSON thêm `"meter": <int>` ngay sau `tempo` (`pipeline.estimate_meter`): số phách nằm giữa hai phách mạnh liên
    tiếp, lấy giá trị phổ biến nhất (hòa thì lấy số nhỏ hơn); dưới hai phách mạnh thì 4. Không bump
    `RESULT_VERSION`: `code_revision` đổi nên cache cũ tự chạy lại. "Let It Be" và mọi bài trong cache ra 4.
  - `extension/src/metronome.js`: lên lịch trước 100 ms, tick 25 ms, tiếng "woodblock" tổng hợp (sine có pitch rơi
    nhanh, phách mạnh 1250 Hz to hơn, phách nhẹ 880 Hz), có kiểu `sine` dự phòng (`CT.metronome.sound`). Mốc
    đồng bộ dùng `AudioContext.getOutputTimestamp()` (đồng hồ của mẫu đang ra loa, đã gồm `outputLatency`).
  - Đo bằng `extension/dev/metronome_test.html` ("Let It Be", Chrome, 48 kHz, baseLatency 10 ms, outputLatency 40 ms),
    độ lệch giữa lúc tiếng gõ được lên lịch và phách trong JSON, theo đường khớp đồng hồ media (|lệch| TB / max):
    1×: 0,6 / 6,5 ms (20 tiếng, max là tiếng đầu chặng); 0,75×: 0,05 / 0,14 ms (15); 1× sau khi tua tới 120 s:
    0,05 / 0,10 ms (18); 0,75× sau khi tua lùi tới 60 s: 0,03 / 0,05 ms (14). Đạt mục tiêu < 15 ms. (Bản đầu chưa
    giới hạn bước chỉnh mốc mỗi khung hình: max 4–11 ms.)
  - Kết quả cũ (`X-Chordtrack-Stale`) không có `meter`: phía extension dùng `result.meter ?? 4`. ⚠️ Cách đo này không thấy được độ trễ
    khác nhau giữa trình phát video và AudioContext trên thiết bị thật; muốn chắc cần thu loopback.
  - Popup: trạng thái `/health`, bật/tắt, chế độ mặc định, âm lượng metronome, địa chỉ server, danh sách `/songs`.
  - ⚠️ Trình duyệt Claude in Chrome trên máy này không vào được `127.0.0.1` của shell (ERR_CONNECTION_REFUSED), phải
    phục vụ trang thử qua IP LAN và proxy `/api/` tới backend. Extension thật không bị ảnh hưởng.
- [x] Extension lõi (2026-11-03): dải karaoke, 3 chế độ, toàn màn hình, bảng cấu trúc chạy trên harness `extension/dev/`. Đồng bộ: hợp âm đổi lệch 6,8 ms trung bình, tối đa 17 ms (1×), 25,7 ms (1.5×). Lưu ý: dải 46 px/s quá chật với nhạc Kajiura (hợp âm ~1 s bị cắt chữ), nên panel nhân 1,5 và toàn màn hình nhân 2. Claude in Chrome trên máy này không vào được 127.0.0.1 và bị bóp `requestAnimationFrame` (1–3 khung/giây): harness phục vụ qua IP LAN, đo với `&measure=1`. Chưa thử trên trang YouTube thật (selector `#below`, `#secondary`, `ytp-autohide`, phần tử fullscreen) vì không được dùng tab YouTube của người dùng.
- [x] Thử extension trên YouTube thật (2026-11-03). Toàn màn hình mới của YouTube có cụm nút like/bình luận/chia sẻ
  (`.ytp-fullscreen-quick-actions`) ngay trên thanh tua, đè lên dải: lớp phủ giờ đo và nằm trên cả cụm này. Vạch đỏ
  không thể nằm "dưới" ô hiện tại bằng z-index (`.ct-strip` có `will-change: transform` là stacking context riêng), nên
  vạch chỉ vẽ ở làn trên và làn thước phách, tiến độ trong ô hiện bằng thanh đỏ ở đáy ô. Chrome bản chính thức
  (137+) bỏ `--load-extension`, Playwright `--channel chrome` không nạp được extension. Người dùng chốt: kiểm thử
  trình duyệt chỉ bằng Claude in Chrome; nạp lại bản mới bằng `window.postMessage({type:"ct:dev-reload"})` (README).
- [x] Mép hợp âm lệch vạch phách (2026-11-03): không phải lỗi vẽ (sai số 0,02 px) mà do model nhận hợp âm theo
  khung ~0,093 s (lệch trung bình 0,12 s, 7% lệch > 0,25 s). Extension kéo mép về phách gần nhất (`snapToBeats`).
- [x] Vùng giọng quá ngắn (2026-11-03): intro một hợp âm Ab của "Nơi này có anh" thành vùng "Ab trưởng". Thêm bước 6
  ở `estimate_key_regions`: gộp vùng < 12 s hoặc < 2 hợp âm khác nhau. Giọng theo vùng (w): Isophonics 0,841 → 0,852
  (đổi giọng 0,683 → 0,683), Kajiura 0,644 → 0,641. Ngưỡng 3 hợp âm làm nhóm đổi giọng Isophonics tụt 2,2 điểm.
- [x] Chỉ phân tích khi người dùng bấm (2026-11-03): nút ♫ trong `.ytp-right-controls` (Alt+Shift+C). Mở video chỉ gọi
  `GET /result/{id}` (đọc cache, không tạo job) để hiện chấm "có sẵn". YouTube bật Trusted Types: không dùng `innerHTML`,
  dựng SVG bằng `createElementNS`. YouTube không hiện tooltip cho nút chèn thêm nên extension tự vẽ tooltip.
- [x] **Thử và bỏ: thống nhất hợp âm theo các đoạn lặp lại** (2026-11-03, không merge, người dùng đã test tay và
  chốt không lấy). Cách làm: chroma CQT của mix ghép 2–4 s ngữ cảnh, ma trận lặp của librosa (k láng giềng, cách nhau
  ≥ 4 s), trộn xác suất hợp âm mỗi khung với trung bình các khung lặp với nó (mức trộn 0,2–1,0) trước HMM. Lưới 49
  cấu hình, CV 3 phần, cùng hàm mục tiêu với `eval/refit_fusion.py`:
  - Kajiura mean4 72,34 → 72,24 (CV −0,10, 9/16 bài thắng/thua, CI chứa 0); F1 chuỗi sevenths 73,91 → 74,16.
  - Beatles mean4 82,98 → 83,59 (CV +0,61, khoảng 31/11, CI không chứa 0).
  - Thêm khoảng 2,5 s mỗi bài (chroma + ma trận lặp).
  - Bài Madoka (kvrJ2pOv3ww): không đổi. Đoạn 0:29–0:51 (giai điệu trên nốt pedal C–G, tầng bass vẫn có C và G)
    chỉ khớp với các đoạn cũng là pedal C (0:55–1:05, 2:30), nên không có lần lặp nào "chữa" được.
  - Kết luận: với Kajiura, model sai giống nhau ở mọi lần lặp nên gộp không sửa được. Muốn tăng tiếp phải có thông
    tin mới (model khác: consonance-ACE MIT có checkpoint; ChordFormer có weights trong lịch sử repo nhưng tác giả đã
    xóa và repo không có license), không phải hậu xử lý thêm trên hai model hiện có.
- [x] Thử ChordFormer và consonance-ACE (2026-11-04): trên Kajiura không hơn bộ hiện tại (CI chứa 0), Beatles tăng do
  ChordFormer đã học chính bộ này. Giữ nguyên mặc định. Chi tiết ở mục "Thử ChordFormer và consonance-ACE".

## Kết quả đánh giá đợt 2: kết hợp model và hậu xử lý (2026-11-03)

42 bài Isophonics The Beatles (117 phút), `python -m eval.ablation`. WCSR gia quyền theo thời lượng (%).
Tham số chọn bằng **cross-validation 3 phần**: chọn trên 28 bài, chấm 14 bài còn lại, ghép 3 phần.
Hàm mục tiêu là trung bình majmin, majmin_inv, sevenths, sevenths_inv. Các dòng 1–3 không có tham số.

| biến thể | root | majmin | majmin_inv | sevenths | sevenths_inv | tetrads | mirex | seg |
|---|---|---|---|---|---|---|---|---|
| 1 ChordMini mix (baseline) | 87.19 | 86.69 | 82.50 | 79.16 | 76.63 | 74.80 | 86.86 | 81.95 |
| 2 ChordMini instrumental | 86.57 | 85.61 | 81.82 | 76.16 | 73.87 | 72.12 | 85.61 | 82.21 |
| 3a CNN-LSTM mix | 85.36 | 84.72 | 82.33 | 76.65 | 75.52 | 72.14 | 84.36 | 83.09 |
| 3b CNN-LSTM instrumental | 82.68 | 81.61 | 79.43 | 72.29 | 71.32 | 68.00 | 80.98 | 83.03 |
| 4 ChordMini + bass CNN-LSTM | 87.19 | 86.69 | 83.58 | 79.16 | 77.62 | 74.80 | 86.86 | 81.95 |
| 5 ChordMini + bass stem | 87.19 | 86.69 | 82.75 | 79.16 | 76.84 | 74.80 | 86.86 | 81.95 |
| 6 ChordMini + bass cả hai | 87.19 | 86.69 | 83.52 | 79.16 | 77.57 | 74.80 | 86.86 | 81.95 |
| 7a Trộn 2 model (cả 170 nhãn), HMM | 89.12 | 89.01 | 84.56 | 80.60 | 78.08 | 76.02 | 88.46 | 81.58 |
| 7b Trộn 2 model (theo họ hợp âm), HMM | 88.98 | 88.76 | 84.35 | 81.00 | 78.48 | 76.45 | 88.23 | 81.38 |
| 8 Làm mượt (phách/HMM), chỉ ChordMini | 87.76 | 87.42 | 83.11 | 79.87 | 77.31 | 75.37 | 87.38 | 80.89 |
| **9 Bộ tốt nhất (CV)** | **88.66** | **88.65** | **85.29** | **80.29** | **78.75** | **75.75** | **88.05** | 81.03 |
| 9 Bộ tốt nhất, chọn trên cả bộ (mặc định của `fuse`, lạc quan) | 89.02 | 88.84 | 85.60 | 81.13 | 79.64 | 76.57 | 88.32 | 81.85 |

- Bộ mặc định của `FusionParams` (chọn trên cả 42 bài): ChordMini và CNN-LSTM trên mix. Trộn log theo họ hợp âm
  (nốt gốc + maj/min/dim/aug/sus) với trọng số CNN-LSTM 0.35, giữ phân bố loại hợp âm 7/6 trong họ của ChordMini.
  Viterbi theo khung với `self_prob` 0.8, gộp đoạn < 0.25 s. Bass = 0.7 × `bass_probs` của CNN-LSTM (giữ phần xác suất
  "không có bass") + 0.3 × chroma stem bass. Chỉ thêm `/x` khi nốt bass thuộc hợp âm, khác nốt gốc và có xác suất ≥ 0.4.
  Ba phần CV chọn các bộ gần giống nhau (trọng số 0.35–0.5, `self_prob` 0.8–0.95, cả hai cách trộn), nên kết quả khá ổn định.
  Trộn trên cả 170 nhãn hơn majmin, trộn theo họ hơn sevenths; tổng thể hai cách gần như ngang nhau.
- So với baseline: majmin +1.96, **majmin_inv +2.79**, sevenths +1.13, **sevenths_inv +2.12** (điểm CV).
  Khoảng cách majmin → majmin_inv giảm từ 4.2 xuống 3.4 điểm. Phần lớn điểm tăng ở majmin đến từ việc trộn hai model.
- Trên 31 bài cũ, baseline cho đúng số cũ (86.30 / 81.35 / 79.27 / 76.26). `fuse()` mặc định cho majmin 88.69,
  majmin_inv 84.79, sevenths 81.22, sevenths_inv 79.46 (bộ tham số chọn trên cả 42 bài, có 31 bài này).
- Bỏ phiếu theo phách và HMM theo phách không hơn HMM theo khung. Quyết định bass theo từng phách cũng không được chọn.
- `fuse()` mất khoảng 0.35 s mỗi bài trên CPU, phần lớn là CQT của stem bass. Không dùng GPU.
- "Let It Be" (chạy thật trên `data/audio/QDYfEBY9NM4.wav`): ra `C:maj/5` ở 40.50–42.07 s, vòng C – G – Am – F,
  4 đoạn C/G trong cả bài. Chạy 2 model và bắt nhịp mất 6–8 s (stem đã có sẵn), VRAM đỉnh 509 MiB.
- ⚠️ Beatles có thể nằm trong dữ liệu huấn luyện của cả ChordMini và CNN-LSTM, nên điểm tuyệt đối có thể cao hơn thực tế.
  Bộ dữ liệu chỉ có 42 bài của một ban nhạc. Cần thêm bài khác (Billboard, bài Việt) để kiểm tra bộ tham số.
- Việc còn mở: căn nhãn có co giãn tuyến tính (vài bài trôi 0.3–0.56 s); thêm ưu tiên theo giọng vào HMM; thử model
  2E1D của ChordMini trong bộ trộn.

## Kết quả đánh giá Kajiura: có giữ bước tách nguồn không? (2026-11-03)

**Bộ dữ liệu:** 32 bài Yuki Kajiura (Kalafina, FictionJunction, See-Saw, LiSA, Aimer...), 150 phút. Đáp án là
bản hợp âm ChordWiki (**không có mốc thời gian**, do người dùng ghi). Audio YouTube, phần lớn là bản chính thức.
Bảng ánh xạ: `eval/kajiura_youtube.csv`. Lệnh: `python -m eval.ablation_kajiura run --beatles`.

**Cách chấm** (chi tiết ở `eval/README.md`, mục 6):
- Theo chuỗi (32 bài): rút gọn, gộp hợp âm trùng liên tiếp, so bằng LCS, lấy F1. Độ dịch giọng của đáp án
  lấy từ audio (DTW chroma), không phụ thuộc phương án.
- Theo thời gian (25 bài căn tốt): căn bản hợp âm vào chroma của audio bằng DTW, không dùng model, rồi
  chấm WCSR như bộ Beatles.
- Đáp án có nhiễu và mốc thời gian chỉ gần đúng, nên **chỉ dùng để so sánh giữa các phương án**.
- Mọi phương án dùng tham số mặc định của `FusionParams` (chọn trên Beatles), không chỉnh lại trên Kajiura.

| phương án | tách nguồn | Kajiura F1 chuỗi majmin | Kajiura F1 chuỗi sevenths | Kajiura majmin | Kajiura majmin_inv | Kajiura sevenths | Kajiura sevenths_inv | Beatles majmin | Beatles majmin_inv | Beatles sevenths_inv |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 ChordMini mix | không | 81.9 | 69.2 | 75.16 | 72.05 | 61.54 | 59.20 | 86.69 | 82.50 | 76.63 |
| 2 ChordMini instrumental | có | 80.4 | 68.3 | 72.58 | 69.52 | 59.44 | 57.05 | 85.61 | 81.82 | 73.87 |
| 3a CNN-LSTM mix | không | 85.4 | 73.7 | **79.12** | **77.05** | **67.62** | **66.13** | 84.72 | 82.33 | 75.52 |
| 3b CNN-LSTM instrumental | có | 84.6 | 73.2 | 77.60 | 75.83 | 66.03 | 64.75 | 81.61 | 79.43 | 71.32 |
| **4 fuse mix, bass CNN-LSTM** | **không** | 84.4 | 71.4 | 78.05 | 75.85 | 63.72 | 62.33 | 88.84 | 85.51 | 79.56 |
| 4b fuse mix, không bass | không | 84.4 | 71.4 | 78.05 | 74.78 | 63.72 | 61.22 | 88.84 | 84.48 | 78.61 |
| 5 fuse mix, bass CNN-LSTM + stem (mặc định hiện tại) | có | 84.4 | 71.4 | 78.05 | 75.95 | 63.72 | 62.39 | 88.84 | 85.60 | 79.64 |
| 6 fuse instrumental, bass cả hai | có | 82.6 | 70.2 | 75.34 | 73.37 | 61.60 | 60.36 | 87.87 | 84.69 | 76.40 |
| 7 fuse trộn mix + instrumental | có | 84.4 | 72.3 | 77.32 | 75.23 | 63.70 | 62.43 | 89.13 | 86.02 | 79.28 |
| 8 thăm dò: như 4, CNN-LSTM 0.65, trộn cả 170 nhãn | không | 85.2 | 74.4 | 79.06 | 76.92 | 66.88 | 65.53 | 88.05 | 84.70 | 77.95 |

Kajiura theo thời gian: WCSR gia quyền thời lượng (%) trên 25 bài. Beatles: 42 bài, cùng phương án. Cột F1 chuỗi
là trung bình theo bài trên 32 bài.

**So sánh theo từng bài** (chênh lệch a − b, khoảng tin cậy bootstrap 95% theo bài, số bài thắng/thua):

| so sánh | chỉ số | chênh | CI 95% | thắng/thua |
|---|---|---|---|---|
| 5 (có stem bass) − 4 (không) | majmin_inv | +0.10 | [+0.01, +0.21] | 7/1, 17 bài bằng nhau |
| 5 − 4 | sevenths_inv | +0.06 | [−0.01, +0.15] | 5/1 |
| 5 − 4 | F1 chuỗi majmin, sevenths | 0 | 0 | không bài nào khác (bass không đổi hợp âm gốc) |
| 4 − 4b (bass CNN-LSTM) | majmin_inv | +1.07 | [−0.54, +2.78] | 12/8 |
| 1 − 2 (ChordMini mix − inst) | majmin | +2.58 | [+1.09, +4.19] | 18/7; F1 chuỗi +1.54, p Wilcoxon 0.017 |
| 3a − 3b (CNN-LSTM mix − inst) | majmin | +1.52 | [+0.46, +2.90] | 19/6 |
| 5 − 6 (fuse mix − inst) | majmin | +2.71 | [+0.84, +4.82] | 15/10; F1 chuỗi +1.75, p 0.015 |
| 7 (trộn mix + inst) − 5 | majmin | −0.73 | [−2.02, +0.39] | không khác có ý nghĩa |
| 4 − 1 (fuse − ChordMini) | majmin_inv | +3.80 | [+1.90, +5.63] | 23/2; F1 chuỗi +2.44, p < 0.001 |
| 3a − 4 (CNN-LSTM một mình − fuse) | sevenths | +3.91 | [+1.84, +5.96] | 21/4; F1 chuỗi sevenths +2.22, p 0.017 |
| 8 − 4 (CNN-LSTM 0.65 − mặc định) | sevenths | +3.16 | [+1.86, +4.59] | 22/3; F1 chuỗi sevenths +2.91, p < 0.001 |

**Thời gian mỗi bài** (32 bài, trung bình 282 giây audio; không tính thời gian chờ khóa GPU):

| bước | giây | VRAM đỉnh |
|---|---|---|
| Chuẩn hóa audio | 0.2 | |
| BS-Roformer (tách giọng) | 53.8 | 1 617 MiB |
| HTDemucs (tách bass) | 29.9 | 552 MiB |
| Bắt nhịp (beat_this) | 1.1 | |
| ChordMini + CNN-LSTM trên mix | 0.7 + 2.4 | |
| `fuse()` có stem bass / không stem | 0.63 / 0.10 | CPU |
| **Tổng có tách nguồn (phương án 5)** | **88.7** | 1.6 GB |
| **Tổng không tách nguồn (phương án 4)** | **4.3** | khoảng 0.5 GB (CNN-LSTM) |

(Chưa tính tải audio và thời gian nạp model lần đầu của mỗi tiến trình.)

**Khuyến nghị: tắt tách nguồn mặc định**, tức là dùng phương án 4: `fuse()` trên mix với
`bass_source="cnn"`, không chạy BS-Roformer và HTDemucs. Có thể giữ tách nguồn làm tùy chọn cho người muốn thử.
Lý do bằng số đo:
1. **Stem bass gần như không đóng góp.** Trên Kajiura chỉ +0.10 điểm majmin_inv (CI [+0.01, +0.21], 17/25
   bài không đổi gì) và +0.06 sevenths_inv (CI chứa 0). Trên Beatles là +0.09 và +0.08. Mức này nhỏ hơn nhiều
   so với độ nhiễu của đáp án. Hợp âm đảo chủ yếu đến từ `bass_probs` của CNN-LSTM (4 − 4b: +1.07 trên
   Kajiura, +1.03 trên Beatles), và phần này không cần tách nguồn.
2. **Chạy model trên instrumental luôn kém hơn mix**, với cả ChordMini, CNN-LSTM và `fuse()`, trên cả hai bộ,
   kể cả với nhạc Kajiura có giọng và hợp xướng rất nổi. Trên Kajiura, mức giảm 1.5–2.7 điểm majmin có ý nghĩa
   (CI không chứa 0). Trộn mix với instrumental (7) cũng không hơn mix một mình. Vậy tách giọng không giúp nhận
   hợp âm ở bất kỳ cách dùng nào đã thử.
3. **Tiết kiệm khoảng 95% thời gian xử lý**: từ 88.7 xuống 4.3 giây mỗi bài. VRAM đỉnh giảm từ 1.6 GB xuống
   khoảng 0.5 GB.
4. **Không cần dùng có điều kiện** (ví dụ chỉ tách khi giọng hát lấn át): bộ Kajiura toàn bài giọng và hợp xướng
   nổi mà instrumental vẫn thua mix, nên không có nhóm bài nào cho thấy tách nguồn có lợi.

⚠️ Khuyến nghị này trái với quyết định đã chốt ở mục 2 ("Tách bass: giữ, bật mặc định, bắt buộc để nhận hợp âm
đảo"). Số đo cho thấy hợp âm đảo đến từ CNN-LSTM chứ không từ stem bass. **Chưa đổi giá trị mặc định nào trong
code; người dùng quyết định.**

**Phát hiện phụ:** trên Kajiura, CNN-LSTM một mình (3a) hơn `fuse()` mặc định 3.9 điểm sevenths, và tăng trọng số
CNN-LSTM lên 0.65 (8) hơn mặc định 3.2 điểm (22/25 bài). Trên Beatles thì ngược lại: 8 kém mặc định 1.7 điểm
sevenths_inv. Bộ tham số của `FusionParams` có thể đã thiên về ChordMini, vì Beatles có thể nằm trong dữ liệu
huấn luyện của ChordMini. Việc nên làm tiếp: chọn tham số trên cả hai bộ (Beatles + Kajiura) bằng
cross-validation, thay vì chỉ trên Beatles.

**Kiểm tra bằng mắt** (đáp án đã căn so với phương án 5):
- "to the beginning" (đổi giọng 8 lần): dự đoán khớp đáp án phần lớn thời gian. Mốc chuyển hợp âm của đáp án
  lệch khoảng 1–2 phách do DTW. Sheet ghi hợp âm phong phú hơn model (G6, Dm7, CM7, Bm7-5), nên điểm sevenths
  bị kéo xuống.
- "夏の林檎": bản ChordWiki ghi thấp hơn bản thu 5 nửa cung. Bước căn tìm đúng độ dịch +5 (sheet Bm – GM7 – A – D
  thành Em – CM7 – D – G, khớp audio). Mốc căn ở intro lệch khoảng 4 giây.
- "空は高く風は歌う": bản thu cao hơn sheet 1 nửa cung, căn tìm đúng +1. Model và đáp án khớp hợp âm, chỉ lệch
  thời điểm chuyển ở vài chỗ.

**Giới hạn:** đáp án do người dùng ghi, có thể đơn giản hóa hoặc bỏ đoạn lặp. Mốc thời gian sau khi căn chỉ gần
đúng, nên điểm theo thời gian thấp hơn Beatles khoảng 10 điểm. 7 bài chỉ lặp một vòng 2–3 hợp âm không căn được
theo thời gian. Bộ chỉ có 32 bài.

## Chọn lại tham số fuse trên Beatles + Kajiura (2026-11-03)

Lệnh: `python -m eval.refit_fusion` (khoảng 40 phút, 6 tiến trình). Chỉ dùng đặc trưng của **mix**, không có stem
(tách nguồn đã tắt mặc định), nên mọi cấu hình dùng `bass_source="cnn"`.

- **Hàm mục tiêu:** `0.6 × mean4(Kajiura theo thời gian, 25 bài căn tốt) + 0.4 × mean4(Beatles, 42 bài)`, với
  mean4 = trung bình WCSR gia quyền thời lượng của majmin, majmin_inv, sevenths, sevenths_inv. Kajiura nặng hơn vì
  người dùng chủ yếu nghe Kajiura và Beatles có thể nằm trong dữ liệu huấn luyện của các model.
- **Lưới** (2016 cấu hình, HMM theo khung): `cnn_weight` 0.35, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0; `mix_level` "full" hoặc
  "family" với `quality_weight` 0, 0.35, 0.5, 0.7, 1.0 (trọng số CNN-LSTM cho **loại hợp âm trong họ**, tham số mới);
  `self_prob` 0.7, 0.8, 0.9, 0.95; `min_duration` 0.25, 0.5, 1.0; `bass_threshold` 0.3–0.6.
- **Cross-validation 3 phần**, chia theo bài và riêng từng bộ (bài thứ i vào phần i mod 3), nên phần nào cũng có
  14 bài Beatles và 8–9 bài Kajiura căn tốt (10–11 bài Kajiura tính F1 chuỗi). Ba phần chọn: (0.5, family,
  q 1.0, sp 0.8, bass 0.5), (0.7, family, q 0.5, sp 0.7, bass 0.5), (0.6, family, q 1.0, sp 0.7, bass 0.5).
- Khi trộn loại hợp âm, nhãn CNN-LSTM không đoán được (maj6, min6, minmaj7) giữ đúng tỉ lệ trong họ của ChordMini
  (`fusion.cnnlstm_abstain`); thiếu bước này thì maj6 có thể thắng sai khi CNN-LSTM gần như không đặt khối nào vào họ.

| phương án | K majmin | K majmin_inv | K sevenths | K sevenths_inv | **K mean4** | K F1 chuỗi majmin | K F1 chuỗi sevenths | B majmin | B majmin_inv | B sevenths | B sevenths_inv | **B mean4** | mục tiêu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CNN-LSTM một mình | 79.12 | 77.05 | 67.62 | 66.13 | **72.48** | 85.43 | 73.67 | 84.72 | 82.33 | 76.65 | 75.52 | 79.81 | 75.41 |
| ChordMini một mình | 75.16 | 72.05 | 61.54 | 59.20 | 66.99 | 81.92 | 69.16 | 86.69 | 82.50 | 79.16 | 76.63 | 81.25 | 72.69 |
| fuse cũ (mặc định cũ, có stem bass) | 78.05 | 75.95 | 63.72 | 62.39 | 70.03 | 84.36 | 71.44 | 88.84 | 85.60 | 81.13 | 79.64 | **83.80** | 75.54 |
| fuse cũ, bass CNN-LSTM (không tách nguồn) | 78.05 | 75.85 | 63.72 | 62.33 | 69.99 | 84.36 | 71.44 | 88.84 | 85.51 | 81.13 | 79.56 | 83.76 | 75.50 |
| **fuse mới (CV 3 phần)** | 78.85 | 76.69 | 66.90 | 65.56 | **72.00** | 85.05 | 73.50 | 87.87 | 84.61 | 79.18 | 77.72 | **82.35** | 76.14 |
| fuse mới (chọn trên cả bộ, mặc định mới, lạc quan) | 79.01 | 76.89 | 67.38 | 66.07 | 72.34 | 85.18 | **73.91** | 88.58 | 85.31 | 79.73 | 78.29 | 82.98 | 76.59 |

K = Kajiura theo thời gian (25 bài), F1 chuỗi trên 32 bài; B = Beatles (42 bài). Hai dòng "cũ" khớp số ở các mục trước.

**Mặc định mới của `FusionParams`:** `cnn_weight=0.6, mix_level="family", quality_weight=1.0, smoothing="hmm",
self_prob=0.7, min_duration=0.25, bass_source="cnn", bass_threshold=0.5`. Khi bật `--separation`, pipeline đổi
`bass_source` thành "both" (trộn 0.3 stem bass).

**Đối chiếu với mục tiêu:**
- *Không kém CNN-LSTM một mình trên Kajiura:* **chưa đạt, kém ít.** Điểm CV kém 0.48 mean4 (majmin −0.27,
  majmin_inv −0.37, sevenths −0.72, sevenths_inv −0.57); CI bootstrap 95% theo bài chứa 0 trừ sevenths
  ([−1.49, −0.03], 9/16 bài thắng/thua). Bộ chọn trên cả bộ kém 0.14 mean4 (sevenths_inv −0.05, CI chứa 0). F1 chuỗi
  sevenths của bộ mặc định hơn CNN-LSTM (73.91 so với 73.67). So với `fuse` cũ trên Kajiura: **+2.0 mean4**, sevenths +3.2.
- *Không kém `fuse` cũ trên Beatles quá 1 điểm:* **CV chưa đạt** (−1.45 mean4: majmin −1.0, majmin_inv −1.0,
  sevenths −2.0, sevenths_inv −1.9), vì một phần CV chọn bộ nghiêng hẳn về CNN-LSTM (0.7, q 0.5). Bộ mặc định chọn
  trên cả bộ kém 0.82 mean4 (sevenths −1.4). Vẫn hơn CNN-LSTM một mình trên Beatles 2.5–3 điểm.
- **Đánh đổi:** mặt mục tiêu rất phẳng ở đỉnh; các bộ gần đỉnh chỉ đổi Kajiura lấy Beatles. Ví dụ (chọn trên cả bộ)
  `cnn_weight=0.5, family, q 1.0`: Kajiura 71.98, Beatles 83.31; `cnn_weight=0.6, "full"`: Kajiura 72.07, Beatles
  83.14 (mục tiêu đều khoảng 76.5). Không bộ nào trong lưới vừa hơn CNN-LSTM trên Kajiura vừa giữ Beatles trong 1 điểm.
  Muốn đạt cả hai chắc cần thêm thông tin mới (model khác), không chỉ chỉnh trọng số.
- Phần lớn lợi ích trên Kajiura đến từ việc lấy **loại hợp âm 7 từ CNN-LSTM** (`quality_weight` 1.0 so với 0 ở cùng
  `cnn_weight` 0.6: Kajiura +1.5 mean4, Beatles −0.5).
- Thăm dò thêm (không nằm trong lưới CV, chọn trên cả bộ): `self_prob` 0.3–0.7 gần như bằng nhau (±0.03),
  `beat_hmm` kém HMM theo khung 1.2 điểm mục tiêu.
- Thời gian end-to-end mặc định (CLI, `--force`, máy rảnh GPU): "Let It Be" 9.6 s (bắt nhịp 1.4, hợp âm 6.1 gồm nạp
  model, fuse 0.12, analysis 0.16), "空は高く風は歌う" (280 s audio) 10.0 s. VRAM đỉnh: CNN-LSTM 509–586 MiB, beat_this
  263–291 MiB, ChordMini 80 MiB. "Let It Be" vẫn ra `C:maj/5` ở 40.50–42.08 s, vòng C – G – Am – F, 4 đoạn C/G.
- Việc còn mở: model thứ ba hoặc model mới (ví dụ 2E1D của ChordMini) có thể phá thế đánh đổi; thêm bài Kajiura để
  CI hẹp hơn.

## Thử ChordFormer và consonance-ACE (2026-11-04)

Người dùng yêu cầu thử hai model mới nhất có weights. Không model nào trong các bài báo 2025–2026 (BMACE, event-based
ICASSP 2026...) công bố weights, trừ hai model này.

- **ChordFormer** (`chordtrack/chords/chordformer.py`): repo `github.com/mwaseemrandhawa/ChordFormer` **không có license**;
  tác giả đã xóa `cache_data/` (5 checkpoint) ở commit 77bec51. Người dùng quyết định lấy lại từ commit 0d1071a để
  dùng cá nhân. File chỉ nằm ở `third_party_src/ChordFormer/cache_data/`, **không commit**. Wrapper dựng lại mạng bằng
  `torchaudio.models.conformer.ConformerLayer` (khớp khóa trọng số), không import `confor_head16.py` vì file đó gọi
  `set_specific_gpu(5)` lúc import. CQT như CNN-LSTM nhưng đổi sang dB (`ref=max`). Cả bài một lượt: khoảng 6 s,
  VRAM đỉnh 154 MiB (attention dùng SDPA nên không nổ bộ nhớ với ~10 000 khung).
- **consonance-ACE** (`chordtrack/chords/consonance.py`, MIT): dựng lại `forward` bằng torch thuần, không cần lightning,
  gin hay harte. Model ra gốc, bass (softmax 13) và 12 nốt (sigmoid). `frame_probs` trên 170 nhãn ChordMini lấy từ
  log P(gốc) + Σ log P(nốt có/không). Cách này hơn cách ngưỡng 0.5 của tác giả (12 bài Beatles: majmin 76.6 so với
  73.5), nên điểm thấp không phải do wrapper. Khoảng 1 s mỗi bài.
- Không cài thêm gói nào. Lệnh: `python -m eval.cache_new_models` (75 bài, 5.4 phút), `python -m eval.new_models_eval`.

**Kết quả** (lưới `--quick`: trọng số họ bước 0.25 cho 4 model, 5 cách lấy loại hợp âm, rồi chỉnh bass/làm mượt
cho các bộ tốt nhất; CV 3 phần như `refit_fusion`; mục tiêu 0.6 Kajiura + 0.4 Beatles):

| phương án | K mean4 | K F1 chuỗi majmin | K F1 chuỗi sevenths | B mean4 | mục tiêu |
|---|---|---|---|---|---|
| CNN-LSTM một mình | 72.48 | 85.43 | 73.67 | 79.81 | 75.41 |
| ChordFormer một mình | 72.43 | 86.01 | 74.16 | 86.10 | 77.90 |
| consonance-ACE + HMM | 62.75 | 75.99 | 64.36 | 71.52 | 66.26 |
| **mặc định hiện tại** (cm + cn) | 72.34 | 85.18 | 73.91 | 82.98 | 76.59 |
| thay cn bằng ChordFormer | 71.92 | 85.37 | 73.70 | 85.23 | 77.24 |
| bộ trộn mới, CV 3 phần | 72.17 | 85.73 | 73.89 | 84.84 | 77.24 |
| bộ trộn mới, chọn trên cả bộ (cm 0.25, cn 0.25, cf 0.5) | 72.72 | 86.01 | 74.25 | 85.14 | 77.69 |

- **Trên Kajiura (nhạc model chưa từng học) không có cải thiện thật.** ChordFormer gốc − CNN-LSTM gốc: majmin −0.14
  [−1.02, +0.81], sevenths +0.02 [−1.57, +1.49], 12/13 bài thắng/thua. Bộ trộn mới (CV) − mặc định: majmin +0.27
  [−0.38, +1.10], sevenths −0.44 [−1.72, +0.83]; mọi CI chứa 0.
- **Trên Beatles tăng mạnh nhưng không tin được:** ChordFormer gốc hơn CNN-LSTM gốc +6.8 majmin. ChordFormer huấn luyện
  5 phần trên đúng bộ 1 217 bài có Isophonics (mỗi bài Beatles nằm trong tập học của 4/5 model), nên đây chủ yếu là
  nhớ bài. Điểm Kajiura ngang nhau xác nhận điều đó.
- consonance-ACE kém 10 điểm trên Kajiura và không bộ trộn tốt nào dùng nó (thêm vào chỉ làm giảm điểm).
- **Quyết định: giữ nguyên mặc định** (ChordMini + CNN-LSTM). Wrapper và script giữ lại để thử tiếp. Lưới đầy đủ
  (bước 0.1, khoảng 2–3 giờ) chưa chạy: mặt mục tiêu trên Kajiura phẳng, khó đổi kết luận.
- Đúng như dự đoán: ChordFormer gần như "bản nâng cấp" của CNN-LSTM nên sai cùng chỗ. Muốn tăng điểm trên nhạc mới
  cần thông tin khác loại (ví dụ giọng trong HMM; thống nhất đoạn lặp đã thử và bỏ, xem mục 11), không phải thêm
  model học trên cùng dữ liệu.

## Nguồn tham khảo

- ChordMini: https://arxiv.org/abs/2602.19778
- ChordMiniApp: https://github.com/ptnghia-j/ChordMiniApp
- ChordMini ONNX: https://huggingface.co/musetric/chordmini-onnx
- ChordFormer: https://arxiv.org/abs/2502.11840
