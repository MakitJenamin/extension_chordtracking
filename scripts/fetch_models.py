"""Tải mã nguồn và weights của các model hợp âm vào `third_party_src/`, đúng commit đã kiểm chứng.

    python scripts/fetch_models.py            # model pipeline cần: ChordMini, Chord-CNN-LSTM, ChordFormer, beat_this
    python scripts/fetch_models.py --extra    # thêm consonance-ACE (chỉ dùng cho script đánh giá)
    python scripts/fetch_models.py --check    # chỉ kiểm tra, không tải

Mọi weights đều nằm trong repo gốc của tác giả trên GitHub, nên chỉ cần `git clone` rồi checkout đúng commit.
Sau khi tải, từng file weights được kiểm tra SHA-256.

ChordFormer (github.com/mwaseemrandhawa/ChordFormer): repo không có license, và tác giả đã xóa thư mục checkpoint
ở commit 77bec51. Weights lấy từ commit 0d1071a (lần phát hành đầu), như trên máy của tác giả dự án này.
Bạn tự chịu trách nhiệm khi dùng; muốn bỏ, đặt `cf_weight=0` và `bass_model="cnnlstm"` trong `FusionParams`.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@dataclass
class Repo:
    name: str                       # thư mục trong third_party_src
    url: str
    commit: str                     # commit của mã nguồn
    weights: dict[str, str]         # đường dẫn tương đối -> SHA-256
    weights_commit: str | None = None   # lấy thư mục weights từ commit khác (ChordFormer)
    weights_dir: str | None = None
    extra: bool = False
    note: str = ""
    keep: list[str] = field(default_factory=list)


REPOS = [
    Repo(
        "ChordMini", "https://github.com/ptnghia-j/ChordMini.git", "aa6e3a8d7b017f082fd2aaff9329d5c26af49c03",
        {"checkpoints/btc_model_best.pth": "e0a12ca6d881f81e01dfb459b4836ec7dac8d8f87b0170a412b5807256b3a1fc"},
        note="MIT",
    ),
    Repo(
        "chord-cnn-lstm-model", "https://github.com/ptnghia-j/chord-cnn-lstm-model.git",
        "35619f70b920996746125fd14f2661467349a80c",
        {f"cache_data/joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s{i}.best.sdict": h for i, h in enumerate([
            "921b42d5d1cf9ce1c0c0e45a74d409b8066e0acec46058ef74e24ee0fb540761",
            "bcb75859e0efa256696cf5da396b320093317b9b1d9560c304f46c25fe1f8b17",
            "acddf85c3fff29954c4877021177d72e2cba9f729ce80c1010f054c477bf3f61",
            "65d81a3ab73435aaaade586981b4cabdf57b8953d76052703e6968c32ef8421c",
            "5ff6b0ec85640e17a09a9b3de68c93fdd45adc24488e8fa9be5715c28d561122",
        ])},
        note="MIT (Music X Lab)",
    ),
    Repo(
        "ChordFormer", "https://github.com/mwaseemrandhawa/ChordFormer.git", "77bec5136503c406e34f223f2c9b013f24c80862",
        {f"cache_data/chordformer_head16(1.0,1.0)_s{i}.best.sdict": h for i, h in enumerate([
            "2c496ad69036bd58e59a78ec1f42f60c23840c20553f08c8dd8f93b0711f5c57",
            "c6072f4237bc64472bb0efc5e6ced5a8e76b15fd5a171d6b250fe88029d6647d",
            "239a96e76da24c0d654739d2657e1f8341dd649cb052ce90437b4b22a8ff6e51",
            "0e04b933293ad7cb6895c91a63aa6940832adb46a1b895e77afd3d837a24af1b",
            "e46e18ff3f2b3f8bef8446b744f4b544d4e6d98cf937ae2b98d42bb22b3da906",
        ])},
        weights_commit="0d1071a130c3afbcc757d6d4b136d342d3746ea0", weights_dir="cache_data",
        note="không có license; weights lấy từ lịch sử repo (tác giả đã xóa)",
    ),
    Repo(
        "consonance-ACE", "https://github.com/andreamust/consonance-ACE.git", "d17633aea4e68e616e735d09df97b97ae3428e71",
        {}, extra=True, note="MIT, chỉ dùng cho eval",
    ),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True)


def verify(repo: Repo, dest: Path) -> list[str]:
    """Danh sách lỗi (thiếu file, sai checksum); rỗng là ổn."""
    errors = []
    if not dest.exists():
        return [f"chưa có {dest}"]
    for rel, expected in repo.weights.items():
        f = dest / rel
        if not f.exists():
            errors.append(f"thiếu {rel}")
        elif sha256(f) != expected:
            errors.append(f"sai SHA-256: {rel}")
    return errors


def fetch(repo: Repo, dest: Path) -> None:
    if dest.exists() and not (dest / ".git").exists():
        # Có thư mục nhưng không phải repo git (ví dụ đã xóa .git để tiết kiệm chỗ): chỉ chấp nhận khi đủ weights.
        if not verify(repo, dest):
            return
        raise SystemExit(f"{dest} có sẵn nhưng thiếu hoặc sai weights. Xóa thư mục này rồi chạy lại.")
    if not dest.exists():
        git("clone", "--no-checkout", repo.url, str(dest))
    git("fetch", "--quiet", "origin", repo.commit, cwd=dest)
    git("checkout", "--quiet", "--force", repo.commit, cwd=dest)
    if repo.weights_commit:
        git("fetch", "--quiet", "origin", repo.weights_commit, cwd=dest)
        git("checkout", "--quiet", repo.weights_commit, "--", repo.weights_dir, cwd=dest)
        git("reset", "--quiet", cwd=dest)   # bỏ khỏi index, giữ file trong thư mục


def fetch_beat_this() -> None:
    from chordtrack.beats import _checkpoint_path

    print(f"beat_this: {_checkpoint_path()}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--extra", action="store_true", help="tải thêm consonance-ACE (chỉ dùng cho đánh giá)")
    ap.add_argument("--check", action="store_true", help="chỉ kiểm tra, không tải")
    ap.add_argument("--no-beat-this", action="store_true", help="không tải checkpoint beat_this")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from chordtrack.paths import THIRD_PARTY

    if not args.check and shutil.which("git") is None:
        print("Cần git: winget install Git.Git", file=sys.stderr)
        return 1
    THIRD_PARTY.mkdir(parents=True, exist_ok=True)
    failed = []
    for repo in REPOS:
        if repo.extra and not args.extra:
            continue
        dest = THIRD_PARTY / repo.name
        if not args.check:
            print(f"== {repo.name} ({repo.note})", flush=True)
            fetch(repo, dest)
        errors = verify(repo, dest)
        print(f"   {repo.name}: {'OK' if not errors else '; '.join(errors)}", flush=True)
        if errors:
            failed.append(repo.name)
    if not args.check and not args.no_beat_this and not failed:
        fetch_beat_this()
    if failed:
        print(f"Lỗi: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("Đủ model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
