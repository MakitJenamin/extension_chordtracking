# Quy ước dự án chordtrack

Đọc `PLAN.md` trước để nắm mục tiêu và các quyết định đã chốt. Trả lời người dùng bằng tiếng Việt có dấu.

## Môi trường

- Python 3.11. **Dùng venv chung ở checkout chính:** `C:\Users\MakitJenamin\Documents\extension_chordtracking\.venv\Scripts\python.exe`. Worktree không có venv riêng.
- Lệnh `python` trong shell có thể trỏ tới bản giả của Microsoft Store, nên luôn gọi python của venv bằng đường dẫn đầy đủ.
- Chạy test: `<venv python> -m pytest -q` (bỏ test GPU: `-m "not gpu"`).
- ffmpeg nằm trong PATH của user. Shell cũ phải nạp lại PATH thì mới thấy ffmpeg.
- **Không được nâng numpy lên 2.x.** Nếu một thư viện mới đòi numpy 2, dừng lại và báo coordinator.
- Thư viện mới ghi vào `requirements/<module>.txt` của module mình, không sửa file của module khác.
- Venv dùng chung giữa các agent: chỉ `pip install` gói mới, không gỡ và không nâng cấp gói có sẵn.

## Tài nguyên không nằm trong git

Đường dẫn khai báo ở `chordtrack/paths.py`, luôn trỏ vào checkout chính:
- `third_party_src/ChordMini`, `third_party_src/chord-cnn-lstm-model`: mã nguồn và weights hai model hợp âm. Chỉ đọc, không sửa.
- `data/audio/QDYfEBY9NM4.wav`: bài thử "Let It Be", vòng hợp âm chuẩn C – G – Am – F, có C/G ở điệp khúc.
- `data/models/`: weights tải thêm. `data/cache/`: kết quả trung gian. `data/eval/`: dữ liệu đánh giá.

**Repo là public:** không commit audio, weights hay dữ liệu có bản quyền.

## GPU

RTX 3060 Ti, chỉ có 8 GB VRAM, và nhiều agent dùng chung máy. **Mọi đoạn code chạy model trên GPU, kể cả trong test, phải nằm trong `with chordtrack.gpu.gpu_lock():`** để các agent chạy lần lượt. Test cần GPU thì đánh dấu `@pytest.mark.gpu`.

## Kiểu dữ liệu chung

Dùng các kiểu trong `chordtrack/types.py` (Stems, BeatGrid, ChordSegment, ChordModelOutput). Không tự định nghĩa kiểu trùng lặp. Nếu cần đổi thì nêu rõ trong mô tả PR.

## Quy trình git

- Mỗi việc làm trên một nhánh riêng, mở PR vào `main` bằng `gh pr create`.
- Tự review diff (dùng skill `code-review`), sửa các lỗi tìm được, chạy lại test.
- Trước khi merge: `git fetch origin && git rebase origin/main`, xử lý xung đột, chạy lại test.
- Merge bằng `gh pr merge --squash --delete-branch`.
- Commit message và mô tả PR viết bằng tiếng Việt.
- Cập nhật `PLAN.md` (mục 11, "Việc cần kiểm tra") khi có phát hiện quan trọng.
