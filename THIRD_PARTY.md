# Third-party models / Model bên thứ ba

Repo này **không chứa** mã nguồn hay weights của các model dưới đây. `scripts/fetch_models.py` tải chúng từ repo
gốc của tác giả vào `third_party_src/`, đúng commit ghi trong bảng, và kiểm tra SHA-256 của từng file weights.
License của LICENSE (MIT) chỉ áp dụng cho mã của dự án này, không áp dụng cho các model bên thứ ba.

This repository does **not** include code or weights of the models below. `scripts/fetch_models.py` downloads them
from the authors' repositories at the pinned commits. The MIT license of this project does not cover them.

| Model | Dùng để | Nguồn | Commit | License |
|---|---|---|---|---|
| ChordMini (BTC-PL) — Nghia Phan et al., DAFx 2026 | Nhận hợp âm (170 loại) | [ptnghia-j/ChordMini](https://github.com/ptnghia-j/ChordMini) | `aa6e3a8` | MIT |
| Chord-CNN-LSTM — Junyan Jiang et al. (Music X Lab), ISMIR 2019 | Nhận hợp âm, nốt bass (hợp âm đảo) | [ptnghia-j/chord-cnn-lstm-model](https://github.com/ptnghia-j/chord-cnn-lstm-model) | `35619f7` | MIT |
| ChordFormer — Akram, Dettori, Colla, Buttazzo, IEEE TASLP 2026 | Nhận hợp âm, nốt bass | [mwaseemrandhawa/ChordFormer](https://github.com/mwaseemrandhawa/ChordFormer) | mã `77bec51`, weights `0d1071a` | ⚠️ không có license |
| Beat This! — Foscarin, Schlüter, Widmer (CPJKU), ISMIR 2024 | Bắt nhịp, phách mạnh | [CPJKU/beat_this](https://github.com/CPJKU/beat_this) | `b95c8ab` | MIT |
| consonance-ACE — Poltronieri, Serra, Rocamora, ISMIR 2025 | Chỉ dùng cho đánh giá (`--extra`) | [andreamust/consonance-ACE](https://github.com/andreamust/consonance-ACE) | `d17633a` | MIT |
| YourMT3+ (`YPTF.MoE+Multi (noPS)`) — Chang, Dixon, Benetos, MLSP 2024 | Dò giai điệu, cách "Bình thường" và phần hát của "Kỹ" (cài mặc định, không cần tài khoản) | [HF Space mimbres/YourMT3](https://huggingface.co/spaces/mimbres/YourMT3); transformers 4.45.1 cài riêng ở `third_party_src/YourMT3_pydeps` | `5e66c1e` | Apache-2.0 (mã và checkpoint trong Space) |
| MuScriptor small — Kyutai + Mirelo, 2026 | Dò giai điệu nhạc cụ ngoài đoạn hát, chỉ cách "Kỹ" (tùy chọn, cần tài khoản Hugging Face) | mã [muscriptor/muscriptor](https://github.com/muscriptor/muscriptor), weights [HF MuScriptor/muscriptor-small](https://huggingface.co/MuScriptor/muscriptor-small) (gated) | mã `8a1ef88` | mã MIT; ⚠️ weights **CC BY-NC 4.0** |
| BS-Roformer, HTDemucs (qua audio-separator, demucs) | BS-Roformer: tách giọng trước khi dò giai điệu (cài cùng dò giai điệu); tách nguồn cho hợp âm tắt mặc định | PyPI / Hugging Face | | xem từng gói |

## ⚠️ MuScriptor: CC BY-NC 4.0, cần đăng nhập Hugging Face

Weights MuScriptor theo **CC BY-NC 4.0**: được dùng, chia sẻ, chỉnh sửa nếu ghi công tác giả và **không dùng vào mục
đích thương mại**. Repo trên Hugging Face là gated: phải đăng nhập (`hf auth login`) và bấm đồng ý điều khoản ở trang
model thì mới tải được. MuScriptor là **tùy chọn**: `setup.ps1` / `fetch_models.py --melody` không tải được thì chỉ in
cảnh báo kèm cách bật, dò giai điệu vẫn chạy bằng YourMT3+ (cách "Bình thường"), chỉ nút "Kỹ" bị mờ. Muốn bật: xem README,
bước 5 "(Tùy chọn) Bật cách dò Kỹ", rồi `fetch_models.py --muscriptor`.

MuScriptor weights are CC BY-NC 4.0 (non-commercial use only) and gated on Hugging Face. They are optional: without
them melody detection still works with YourMT3+ ("Bình thường", the default mode); log in, accept the terms and run
`fetch_models.py --muscriptor` to enable the "Kỹ" mode.

## ⚠️ ChordFormer

Repo ChordFormer **không có file license**, và tác giả đã xóa thư mục checkpoint ở commit `77bec51`.
`fetch_models.py` lấy weights từ commit `0d1071a` (lần phát hành đầu) trong lịch sử repo. Không có license nghĩa là
về pháp lý tác giả giữ mọi quyền; bạn tự chịu trách nhiệm khi dùng. Muốn chạy không có ChordFormer, đặt
`cf_weight=0.0, bass_model="cnnlstm", cnn_weight=0.6` trong `chordtrack/fusion.py` (`FusionParams`), tức bộ trộn
ChordMini + CNN-LSTM cũ (kém khoảng 0,5 điểm trên bộ đánh giá Kajiura, xem PLAN.md).

ChordFormer has no license and its authors removed the checkpoints from their repository. The setup script fetches
them from the repository history. Use at your own risk, or disable ChordFormer as described above.
