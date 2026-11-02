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
