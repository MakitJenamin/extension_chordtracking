# Đánh giá chất lượng nhận diện hợp âm

Thư mục này gồm script chấm điểm, bộ dữ liệu đánh giá Isophonics The Beatles và baseline ChordMini.
Mọi biến thể pipeline đều so sánh bằng cùng một cách tính trong `evaluate.py`.

## 1. Chấm điểm: `evaluate.py`

```sh
python -m eval.evaluate --ref-dir <thư mục .lab chuẩn> --est-dir <thư mục .lab đoán> [--out results.json]
```

- Ghép file theo tên (`<name>.lab` ở cả hai thư mục). Bài thiếu nhãn đoán được liệt kê riêng, không tính điểm.
- `evaluate_pair(ref, est)` nhận đường dẫn `.lab`, `list[ChordSegment]` hoặc `(intervals, labels)`.
- Cách tính giống `mir_eval.chord.evaluate`: kéo nhãn đoán về đúng khoảng thời gian của nhãn chuẩn
  (phần thiếu là `N`), rồi tính WCSR (tỉ lệ thời lượng đoán đúng) ở các mức
  `root, majmin, majmin_inv, thirds, triads, sevenths, sevenths_inv, tetrads, mirex`.
  Khoảng thời gian mà nhãn chuẩn nằm ngoài từ vựng của mức đó (ví dụ `X`, hoặc hợp âm 4 nốt ở mức
  `majmin`) không tính vào mẫu số.
- Độ phân mảnh: `underseg`, `overseg`, `seg = min(underseg, overseg)`.
- **Trung bình nhiều bài là gia quyền theo thời lượng**: cộng thời lượng đúng của mọi bài rồi chia cho
  tổng thời lượng hợp lệ. Bài dài có trọng số lớn hơn.
- Mức có hợp âm đảo (`majmin_inv`, `sevenths_inv`) là mức cần theo dõi để đo việc nhận C/E, G/B.

## 2. Bộ đánh giá: `datasets.py`

```sh
python -m eval.datasets all      # tải nhãn + tải audio theo CSV + căn thời gian
```

Kết quả nằm trong `EVAL_DIR/isophonics/` (`chordtrack.paths.EVAL_DIR`, không commit):

| Đường dẫn | Nội dung |
|---|---|
| `chordlab/<Album>/<Track>.lab` | 180 file nhãn gốc Isophonics, tải từ isophonics.net |
| `audio/<name>.wav` | audio tải bằng `yt-dlp` |
| `ref/<name>.lab` | nhãn đã căn theo audio, **chỉ gồm bài đạt**. Dùng làm `--ref-dir` |
| `alignment.json` | độ lệch, điểm căn và lý do loại của từng bài |

Bảng ánh xạ `isophonics_youtube.csv` (có commit) gồm: `name, title, lab_file, video_id, yt_title`.
Bảng được dựng bằng `python -m eval.datasets search`: với mỗi bài trong `CANDIDATES`, tìm trên YouTube
`The Beatles <tên bài> Remastered 2009`, chỉ nhận video có tên bài khớp đúng, rồi xếp hạng theo:
có "Remastered 2009" > không phải bản remix mới (2015/2019/2026 Mix) > kênh chính thức > lệch độ dài ít nhất.
Lệnh này ghi ra `EVAL_DIR/isophonics/youtube_candidates.csv`, không ghi đè bảng đã commit. Kiểm tra xong mới chép sang.

### Căn thời gian

Nhãn Isophonics khớp bản CD gốc. Audio YouTube có thể là bản master khác: khoảng lặng đầu bài khác
nhau, đuôi fade-out dài ngắn khác nhau, đôi khi là bản mix khác hẳn. Cách xử lý:

1. **Ước lượng độ lệch đầu bài bằng tương quan chroma.** Dựng "chroma mẫu" từ nhãn (vector 12 nốt,
   bằng 1 ở các nốt của hợp âm, giải mã bằng `mir_eval.chord.encode`), tính chroma CENS của audio
   (22050 Hz, hop 512 ≈ 23 ms), rồi trượt nhãn trong khoảng ±2 s, chọn độ lệch có cosine trung bình
   cao nhất trên các khung có hợp âm. Cách này không phụ thuộc khoảng lặng hay tiếng đếm nhịp ở đầu bài.
2. **Đo độ trôi:** ước lượng lại độ lệch chỉ trên 60 s đầu và 60 s cuối của vùng chung (`head_offset`,
   `tail_offset` trong `alignment.json`).
3. **Loại bài** (`check_alignment`) khi: đỉnh nằm ở biên ±2 s, cosine tốt nhất < 0.5, độ lệch cuối bài
   khác đầu bài quá `MAX_DRIFT` = 0.6 s (bản khác tốc độ hoặc cắt ghép khác), hoặc độ dài lệch quá 15 s
   (chắc chắn là bản khác). **Không loại bài chỉ vì độ dài lệch vài giây**: phần lớn là do đuôi fade-out.
   Mọi lag được chấm trên cùng một tập khung, để lag lớn không được lợi vì bỏ bớt khung.
4. **Dịch nhãn** theo độ lệch tìm được (bỏ qua nếu < 0.03 s), **cắt về vùng chung**
   `[0, min(độ dài audio, cuối nhãn + độ lệch)]` (`common_end`), lấp đầu bài bằng `N`, ghi vào `ref/`.
   Phần audio sau cuối nhãn không được chấm vì `evaluate_pair` chỉ chấm trong khoảng của nhãn chuẩn.

Đã thử và bỏ:
- Dò khoảng lặng đầu bài (RMS): sai với bài có tiếng đếm nhịp hoặc tiếng nói ở đầu
  ("I Saw Her Standing There", "Two of Us").
- Chroma CQT thay cho CENS: dịch nhầm "Love Me Do" 0.5 s, làm điểm majmin của bài đó tụt từ 89.9 xuống 73.9.

Kiểm chứng (2026-11-02, 31 bài đạt): với mỗi bài, quét độ lệch trong ±1.5 s (bước 0.05 s) và lấy độ lệch
làm điểm majmin của ChordMini cao nhất ("oracle"; chỉ để kiểm tra, không dùng để căn).
- Độ lệch chroma khớp oracle trong ±0.15 s ở 28/31 bài.
- majmin trung bình: không căn 80.9, căn bằng chroma 86.3, oracle 87.2.
- Thử bỏ qua độ lệch nhỏ (chỉ dịch khi cosine tăng ≥ 0.01): điểm còn 85.7 và chỉ 24/31 bài khớp oracle, nên không dùng.
- Độ lệch lớn nhất: "Eight Days a Week" +0.72 s, "Yellow Submarine" +0.77 s,
  "While My Guitar Gently Weeps" +0.67 s. Không căn thì điểm các bài này tụt 20–30 điểm.

Kết quả bộ dữ liệu:
- 2026-11-02 (quy tắc cũ, loại khi độ dài lệch > 2 s): 43 bài có videoId, 31 bài đạt, 12 bài bị loại.
- **2026-11-03 (quy tắc mới): 42 bài đạt (117 phút), 1 bài bị loại**: All You Need Is Love (độ lệch đầu
  bài −0.12 s, cuối bài +1.58 s; đuôi bài khác hẳn). 11 bài lấy lại: Please Please Me, Twist and Shout,
  Can't Buy Me Love, If I Fell, I'm a Loser, Ticket to Ride (audio ngắn hơn nhãn 7.9 s, nhưng độ lệch đầu
  và cuối đều +0.21 s), You've Got to Hide Your Love Away, Here, There and Everywhere, Hello Goodbye,
  Ob-La-Di, Ob-La-Da, Get Back. Nhãn của 31 bài cũ không đổi.
- Độ trôi đầu–cuối của các bài đạt phần lớn dưới 0.3 s; vài bài 0.3–0.56 s (Love Me Do, Yellow Submarine,
  Eight Days a Week, Come Together, Penny Lane), có thể do nhãn gốc trôi nhẹ. Chưa căn co giãn tuyến tính.
- Hai bài không tìm được video phù hợp: Getting Better, Norwegian Wood.

## 3. Baseline ChordMini: `run_chordmini.py`

```sh
python -m eval.run_chordmini     # kết quả ở EVAL_DIR/results/chordmini_btc_pl/{lab/, results.json}
```

Chạy `third_party_src/ChordMini/src/evaluation/test.py` bằng subprocess, với checkpoint
`btc_model_best.pth` và cờ ở PLAN.md mục 11. Khóa `gpu_lock()` được giữ trong suốt subprocess.
VRAM đỉnh đo bằng `nvidia-smi`, là số của cả GPU, không riêng tiến trình.

Baseline trên bộ 42 bài (2026-11-03): root 87.19, majmin 86.69, majmin_inv 82.50, sevenths 79.16,
sevenths_inv 76.63, tetrads 74.80, mirex 86.86, seg 82.54. 18 s cho 42 bài, VRAM cả GPU đỉnh 1439 MiB.
WCSR trùng đúng với `fill_gaps(run_chordmini(...).segments)` của wrapper (seg lệch 0.6 vì wrapper lấp khoảng trống).

## 4. Cache đặc trưng: `cache_features.py`

```sh
python -m eval.cache_features [--only NAME ...] [--force]
```

Với mỗi bài trong CSV có audio (kể cả bài bị loại, để đổi quy tắc loại bài không phải chạy lại): tách nguồn
(`CACHE_DIR/eval_<tên>/`), bắt nhịp trên mix, chạy ChordMini BTC-PL và CNN-LSTM trên **mix và instrumental**.
Ghi `EVAL_DIR/features/<tên>.pkl` (đọc bằng `load_features`). Mỗi bước GPU tự lấy `gpu_lock()` riêng.
Chạy 43 bài mất 43.5 phút (tách nguồn 40–80 s mỗi bài, các model hợp âm 0.5–2 s). Stem chiếm khoảng 300 MB mỗi bài.
`ablation.py` cache thêm chroma stem bass ở `<tên>.bass_chroma.npz`.

## 5. Ablation: `ablation.py`

```sh
python -m eval.ablation [--folds 3] [--jobs 8] [--only 1 9]
```

So sánh các biến thể ở docstring của file, mỗi biến thể một dòng. Biến thể có tham số được chọn bằng
cross-validation 3 phần (bài thứ i vào phần i mod 3, chọn trên 2 phần, chấm phần còn lại). Hàm mục tiêu là
trung bình majmin, majmin_inv, sevenths, sevenths_inv. In thêm bảng chọn tham số trên cả bộ (lạc quan),
là bộ mặc định của `chordtrack.fusion.FusionParams`. Kết quả ở `EVAL_DIR/results/ablation.json`.
Chạy hết trong khoảng 2.5 phút trên CPU (8 tiến trình), không cần GPU. Bảng kết quả: PLAN.md, mục
"Kết quả đánh giá đợt 2".

## 6. Bộ Kajiura: `kajiura.py`, `ablation_kajiura.py`

Bộ đánh giá nhạc Yuki Kajiura (Kalafina, FictionJunction, See-Saw, nhạc anime): giọng hát và hợp xướng
nổi, phối dàn nhạc dày, đổi giọng nhiều. Đáp án là bản hợp âm **ChordWiki**, không có mốc thời gian.

```sh
python -m eval.kajiura parse                      # đọc bản hợp âm, báo ký hiệu không đọc được
python -m eval.kajiura download [--cookies FILE]  # tải audio theo kajiura_youtube.csv
python -m eval.kajiura align                      # căn bản hợp âm vào audio, ghi ref/ và alignment.json
python -m eval.ablation_kajiura cache             # tách nguồn + chạy model (GPU), cache đặc trưng
python -m eval.ablation_kajiura run --beatles     # bảng so sánh, kiểm định theo bài, thời gian xử lý
```

Dữ liệu nằm ở `EVAL_DIR/kajiura/` (không commit): `sheets/<tên>.cho`, `audio/<tên>.wav`, `ref/<tên>.lab`,
`alignment.json`, `features/`. Chỉ commit bảng ánh xạ `kajiura_youtube.csv`
(`name, title, artist, chordwiki_url, video_id, yt_title, note`).

### Lấy bản hợp âm

- Nguồn: tag `梶浦由記` (30 bài) và tag `kalafina` (11 bài, trùng 8) trên ja.chordwiki.org. "焔の扉" không
  có trang, còn **32 bài**.
- ChordWiki chặn request tự động bằng Cloudflare (curl và script đều nhận 403). Bản hợp âm được đọc bằng
  trình duyệt thật sau khi người dùng tự qua bước xác minh, mỗi trang cách nhau vài giây. Đoạn JS
  `kajiura.EXTRACT_JS` chạy trên trang bài **chỉ lấy hợp âm** (`span.chord`), tên đoạn và dòng đổi giọng;
  không lấy lời bài hát. Kết quả là file ChordPro rút gọn, một dòng nhạc một cặp ngoặc:
  `[Am G6 F G] x2`, `{key:F#m}`, `{c:Chorus}`, `{yt:<videoId gắn trên trang>}`.
- Hai trang ("炎", "ユメノツバサ") có thêm một bản gọn lặp lại ở cuối; chỉ giữ bản đầu.

### Đọc ký hiệu (`parse_chord`, `parse_chordpro`)

- `onC` = `/C`; `N.C.` -> `N`; `(C)` vẫn là hợp âm; `/E` đứng riêng = hợp âm trước đó với bass E.
- Chất lượng: `M7`/`maj7`/`△7`, `m7-5`/`m7b5`/`ø` (hdim7), `mM7`, `dim`, `dim7`, `aug`/`+5`, `sus4`, `sus2`,
  `7sus4`, `7(sus4)`, `add9`/`(9)`, `6`, `69`, `9`, `7(9)` = `9`, `M7(9)` = `maj9`, `7(b9)`, `7#9`, `7-5`, `5`.
  Ký tự toàn chiều rộng chuẩn hóa bằng NFKC.
- Bỏ vạch nhịp, gạch nối, mũi tên, ghi chú chữ (`(break)`, `(Dr.fill)`). Dấu lặp `x2` cuối dòng được nhân.
- Kết quả: đọc được toàn bộ 5 255 ký hiệu hợp âm của 32 bài.

### Chấm điểm khi đáp án không có thời gian

Bản hợp âm do người dùng tự ghi: có thể đơn giản hóa, sai, ghi theo capo, hoặc bỏ đoạn lặp. Điểm chỉ dùng
để **so sánh giữa các phương án**, không phải điểm tuyệt đối.

1. **Căn vào audio (DTW chroma, không dùng model).** Gộp hợp âm trùng liên tiếp, dựng chroma mẫu (nốt của
   hợp âm, cộng 0.5 cho nốt bass), so cosine với chroma CENS của mix (khung ~93 ms). DTW chỉ cho đứng yên
   hoặc sang hợp âm kế tiếp; thêm trạng thái "ngoài bản hợp âm" ở đầu và cuối với chi phí cố định 0.55,
   để intro/outro không ghi trong bản hợp âm không bị kéo giãn; các khung này ghi nhãn `X` (không tính
   điểm), không phải `N`. Thử 12 độ dịch giọng, chọn độ dịch có
   cosine trung bình cao nhất: đó là **độ dịch của đáp án** (khác 0 thì có thể đáp án ghi theo capo).
   Độ dịch này lấy từ audio nên không thiên vị phương án nào.
2. **Theo chuỗi (mọi bài):** rút gọn về `majmin` (theo quãng ba: dim -> min, aug/7 -> maj, sus riêng) hoặc
   `sevenths`, bỏ N, gộp trùng liên tiếp. Chuỗi dự đoán chỉ lấy trong vùng thời gian bản hợp âm phủ.
   So bằng LCS (Needleman–Wunsch không phạt khoảng trống, không cho thay thế):
   precision = LCS / độ dài dự đoán, recall = LCS / độ dài đáp án, **F1**. Kèm F1 trên multiset cặp chuyển
   hợp âm (bigram). Ghi thêm F1 khi tự chọn độ dịch tốt nhất cho từng phương án (`seq_*_free`).
3. **Theo thời gian (chỉ bài căn tốt):** nhãn đã căn chấm bằng `evaluate_pair` như bộ Beatles. Bài bị loại
   khi cosine trung bình trên đường căn < 0.60 hoặc bản hợp âm phủ < 50% bài. Độ chính xác của mốc thời
   gian bị giới hạn bởi DTW nên điểm thấp hơn bộ Beatles là bình thường.
4. **Kiểm định:** chênh lệch theo từng bài giữa hai phương án, khoảng tin cậy bootstrap 95% (2000 lần lấy
   mẫu lại bài) và kiểm định dấu hạng Wilcoxon. Điểm theo thời gian bootstrap trên tỉ số gia quyền.

### Phương án so sánh

Mọi phương án dùng tham số mặc định của `FusionParams` (chọn trên Beatles), không chỉnh trên Kajiura.
Danh sách ở docstring của `ablation_kajiura.py`. Cùng các phương án đó được chạy lại trên 42 bài Beatles
(`--beatles`) để đặt cạnh nhau. Kết quả ở `EVAL_DIR/results/ablation_kajiura.json`; bảng ở PLAN.md,
mục "Kết quả đánh giá Kajiura".

### Kết quả dựng bộ

- 32 bài, 150 phút audio. Bài bị loại khi dựng: "焔の扉" (tag có nhưng trang không tồn tại).
- Audio: 29 bài có bản chính thức (kênh nghệ sĩ, kênh Topic, hãng đĩa); "アイスクリイム" và
  "静寂はヘッドフォンの中" chỉ có bản studio do người khác đăng; "瞳の欠片" dùng video gắn trên trang
  ChordWiki. Ghi chú từng bài trong CSV.
- Tải audio: YouTube có thể đòi xác minh "not a bot". Khi đó truyền file cookie bằng `--cookies` hoặc biến
  môi trường `CHORDTRACK_YT_COOKIES`. yt-dlp bản mới cần thêm Deno và gói `yt-dlp-ejs` (`yt-dlp[default]`).
- Căn: 25/32 bài căn tốt. 7 bài bị loại khỏi phần chấm theo thời gian (vẫn chấm theo chuỗi): 5 bài độ
  dịch giọng không chắc (i_beg_you, life_goes_on, magia, oblivious, ongaku) và 2 bài thứ tự hợp âm không
  giúp căn (nowhere, silly_go_round). Phần lớn là bài chỉ lặp một vòng 2–3 hợp âm hoặc gần như một hợp âm,
  nên vị trí thời gian không xác định được. Bài chưa chắc độ dịch thì chấm theo chuỗi với độ dịch 0.
- Độ dịch khác 0: natsu_no_ringo (+5, bản ChordWiki ghi thấp hơn bản thu 5 nửa cung), sora_wa_takaku (+1).
  Đã kiểm tra bằng mắt, cả hai đều đúng.
- Cosine tuyệt đối không phân biệt được căn đúng hay sai (độ dịch sai vẫn cho 0.60–0.70), nên tiêu chí
  căn tốt là: hơn độ dịch thứ hai ≥ 0.02 **và** hơn chuỗi cùng hợp âm nhưng xáo trộn thứ tự ≥ 0.02.
