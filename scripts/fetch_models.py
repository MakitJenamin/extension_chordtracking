"""Tải mã nguồn và weights của các model hợp âm vào `third_party_src/`, đúng commit đã kiểm chứng.

    python scripts/fetch_models.py            # model pipeline cần: ChordMini, Chord-CNN-LSTM, ChordFormer, beat_this
    python scripts/fetch_models.py --extra    # thêm consonance-ACE (chỉ dùng cho script đánh giá)
    python scripts/fetch_models.py --check    # chỉ kiểm tra, không tải
    python scripts/fetch_models.py --melody   # thêm model dò giai điệu: YourMT3+ (+ MuScriptor small nếu tải được)
    python scripts/fetch_models.py --muscriptor   # chỉ tải MuScriptor small (sau `hf auth login`)

Dò giai điệu (`--melody`, xem README, mục "Dò giai điệu"):
- YourMT3+: mã của Hugging Face Space `mimbres/YourMT3` (Apache-2.0) clone vào `third_party_src/YourMT3` (bỏ qua file
  LFS), checkpoint `YPTF.MoE+Multi (noPS)` tải riêng về `data/models/yourmt3/` và kiểm SHA-256; transformers 4.45.1
  cùng các gói đi kèm cài riêng vào `third_party_src/YourMT3_pydeps` (không đụng venv, xem
  `requirements/melody_yourmt3.txt`).
- MuScriptor small (tùy chọn, chỉ cách dò "Kỹ" cần): weights CC BY-NC 4.0, repo gated. Phải `hf auth login` (hoặc
  đặt HF_TOKEN) và bấm đồng ý license ở https://huggingface.co/MuScriptor/muscriptor-small. Chưa làm thì `--melody`
  chỉ in cảnh báo kèm cách bật, mã thoát vẫn 0 (YourMT3+ đủ để dò giai điệu); `--muscriptor` thì báo lỗi.

Mọi weights đều nằm trong repo gốc của tác giả trên GitHub, nên chỉ cần `git clone` rồi checkout đúng commit.
Sau khi tải, từng file weights được kiểm tra SHA-256.

ChordFormer (github.com/mwaseemrandhawa/ChordFormer): repo không có license, và tác giả đã xóa thư mục checkpoint
ở commit 77bec51. Weights lấy từ commit 0d1071a (lần phát hành đầu), như trên máy của tác giả dự án này.
Bạn tự chịu trách nhiệm khi dùng; muốn bỏ, đặt `cf_weight=0` và `bass_model="cnnlstm"` trong `FusionParams`.
"""
from __future__ import annotations

import argparse
import hashlib
import os
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


YOURMT3_SPACE = "mimbres/YourMT3"
YOURMT3_COMMIT = "5e66c1ea173a8186e0d20432b841d3180cc015b5"
YOURMT3_CKPT_IN_SPACE = ("amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/"
                         "checkpoints/last.ckpt")
YOURMT3_PYDEPS = ["transformers==4.45.1", "huggingface_hub==0.36.2", "tokenizers==0.20.3", "regex==2026.11.9"]


MUSCRIPTOR_REPO_URL = "https://huggingface.co/MuScriptor/muscriptor-small"
MUSCRIPTOR_STEPS = rf"""   MuScriptor small chỉ cần cho cách dò "Kỹ". Cách dò "Nhanh" (YourMT3+) không cần bước này.
   Muốn bật "Kỹ" (weights CC BY-NC 4.0, chỉ dùng phi thương mại):
     1. Tạo tài khoản Hugging Face, tạo access token loại Read ở https://huggingface.co/settings/tokens
     2. Mở {MUSCRIPTOR_REPO_URL}, đăng nhập và bấm đồng ý điều khoản (Agree and access repository)
     3. Chạy:  .venv\Scripts\hf.exe auth login      (dán token ở bước 1)
     rồi:      .venv\Scripts\python.exe scripts\fetch_models.py --muscriptor
   và khởi động lại server."""


def fetch_yourmt3(check: bool) -> list[str]:
    """Cài YourMT3+ (bắt buộc cho dò giai điệu). Trả về danh sách lỗi (rỗng là đủ)."""
    from chordtrack.melody import yourmt3

    errors: list[str] = []
    print("== YourMT3+ (Apache-2.0, Hugging Face Space mimbres/YourMT3, không cần tài khoản)", flush=True)
    src = yourmt3.SRC_DIR
    if not check:
        try:
            if not (src / "model_helper.py").exists():
                if src.exists() and not (src / ".git").exists():
                    raise RuntimeError(f"{src} có sẵn nhưng không phải repo git. Xóa thư mục này rồi chạy lại.")
                env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")   # checkpoint tải riêng bên dưới, không kéo file LFS
                if not src.exists():
                    subprocess.run(["git", "-c", "core.longpaths=true", "clone", "--no-checkout",
                                    f"https://huggingface.co/spaces/{YOURMT3_SPACE}", str(src)], check=True, env=env)
                # core.longpaths: Space có thư mục `amt/logs/2024/<tên ~70 ký tự>/`, vượt 260 ký tự khi repo nằm sâu.
                subprocess.run(["git", "-c", "core.longpaths=true", "checkout", "--quiet", "--force", YOURMT3_COMMIT],
                               cwd=src, check=True, env=env)
            ckpt = yourmt3.CHECKPOINT
            if not ckpt.exists() or sha256(ckpt) != yourmt3.CHECKPOINT_SHA256:
                download_yourmt3_checkpoint(ckpt)
            if not (yourmt3.PYDEPS_DIR / "transformers").is_dir():
                subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "--target",
                                str(yourmt3.PYDEPS_DIR), *YOURMT3_PYDEPS], check=True)
        except Exception as e:     # mạng, git, pip: vẫn kiểm tra bên dưới để báo thiếu gì
            errors.append(f"lỗi khi tải: {e}")
            if isinstance(e, subprocess.CalledProcessError) and "git" in str(e.cmd):
                errors.append(f"nếu lần clone trước bị ngắt giữa chừng: xóa thư mục {src} rồi chạy lại")
    if not (src / "model_helper.py").exists():
        errors.append(f"thiếu mã YourMT3 ở {src}")
    if not yourmt3.CHECKPOINT.exists():
        errors.append(f"thiếu checkpoint {yourmt3.CHECKPOINT}")
    elif sha256(yourmt3.CHECKPOINT) != yourmt3.CHECKPOINT_SHA256:
        errors.append(f"sai SHA-256: {yourmt3.CHECKPOINT}")
    if not (yourmt3.PYDEPS_DIR / "transformers").is_dir():
        errors.append(f"thiếu {yourmt3.PYDEPS_DIR}")
    print(f"   YourMT3+: {'OK' if not errors else '; '.join(errors)}", flush=True)
    return errors


def download_yourmt3_checkpoint(ckpt: Path) -> None:
    """Tải checkpoint (~540 MB, Space công khai, không cần tài khoản) thẳng về `ckpt` qua URL `resolve` của Hugging
    Face. Không dùng `hf_hub_download`: nó đặt file theo đường dẫn trong Space (`amt/logs/2024/<tên rất dài>/...`),
    dễ vượt giới hạn 260 ký tự của Windows khi thư mục repo nằm sâu, và còn để thêm một bản trong cache."""
    import urllib.request

    from huggingface_hub import hf_hub_url

    from chordtrack.melody import yourmt3

    url = hf_hub_url(YOURMT3_SPACE, YOURMT3_CKPT_IN_SPACE, repo_type="space", revision=YOURMT3_COMMIT)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    part = ckpt.with_name(ckpt.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "chordtrack-fetch-models"})
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, shown = 0, -1
        for chunk in iter(lambda: r.read(1 << 20), b""):
            f.write(chunk)
            done += len(chunk)
            pct = done * 10 // total if total else -1
            if pct != shown:
                shown = pct
                print(f"   checkpoint YourMT3+: {done >> 20} / {total >> 20} MB", flush=True)
    if sha256(part) != yourmt3.CHECKPOINT_SHA256:
        part.unlink(missing_ok=True)
        raise RuntimeError("checkpoint YourMT3+ tải về sai SHA-256")
    os.replace(part, ckpt)


def fetch_roformer(check: bool) -> bool:
    """Tải trước weights BS-Roformer (tách giọng trước khi dò giai điệu; extension luôn dùng khi server có) vào
    `MODELS_DIR/audio-separator`, để lần dò đầu không phải tải. Tùy chọn: thiếu gói audio-separator hoặc tải lỗi thì
    chỉ cảnh báo (server tự dò trên bản mix). Trả về True nếu đã có."""
    import importlib.util

    from chordtrack import separation
    from chordtrack.paths import MODELS_DIR

    print("== BS-Roformer (tách giọng, MIT; tùy chọn)", flush=True)
    d = MODELS_DIR / "audio-separator"
    ckpt = d / separation.ROFORMER_MODEL
    have = lambda: ckpt.is_file() and ckpt.with_suffix(".yaml").is_file()  # noqa: E731
    if importlib.util.find_spec("audio_separator") is None:
        print("   BS-Roformer: chưa cài gói audio-separator (pip install -r requirements/separation.txt "
              "-c requirements/constraints.txt). Dò giai điệu vẫn chạy, không tách giọng.", flush=True)
        return False
    if not check and not have():
        import logging
        import tempfile

        try:
            from audio_separator.separator import Separator

            d.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory() as tmp:
                sep = Separator(log_level=logging.WARNING, model_file_dir=str(d), output_dir=tmp)
                sep.download_model_files(separation.ROFORMER_MODEL)
        except Exception as e:
            print(f"   BS-Roformer: lỗi khi tải ({e}); server sẽ thử tải lại ở lần tách giọng đầu.", flush=True)
    print(f"   BS-Roformer: {'OK' if have() else 'chưa có weights (tải ở lần dò đầu)'}", flush=True)
    return have()


def fetch_muscriptor(check: bool) -> tuple[str, list[str]]:
    """Cài MuScriptor small (tùy chọn, chỉ cách dò "Kỹ" cần). Trả về (trạng thái, chi tiết): "ok", "auth" (chưa
    đăng nhập / chưa đồng ý license), "missing" (lỗi khác hoặc thiếu gói)."""
    import importlib.util

    from chordtrack.melody import muscriptor

    print("== MuScriptor small (tùy chọn; weights CC BY-NC 4.0, repo gated trên Hugging Face)", flush=True)
    problems: list[str] = []
    status = "ok"
    w = muscriptor.weights_dir("small")
    have = lambda: all((w / f).exists() for f in ("model.safetensors", "config.json"))  # noqa: E731
    if not check and not have():
        try:
            muscriptor.ensure_weights("small")
        except muscriptor.MuscriptorAuthError:
            status = "auth"
            problems.append("chưa có quyền tải weights (chưa đăng nhập Hugging Face hoặc chưa đồng ý license)")
        except Exception as e:     # mạng...
            problems.append(f"lỗi khi tải weights: {e}")
    if not have():
        if not problems:
            problems.append(f"chưa có weights ở {w}")
        if status == "ok":
            status = "missing"
    if importlib.util.find_spec("muscriptor") is None:
        problems.append("chưa cài gói muscriptor (pip install --no-deps -r requirements/melody_muscriptor.txt "
                        "-c requirements/constraints.txt)")
        if status == "ok":
            status = "missing"
    print(f"   MuScriptor small: {'OK' if status == 'ok' else '; '.join(problems)}", flush=True)
    return status, problems


def fetch_melody(check: bool, muscriptor_only: bool = False, muscriptor_required: bool = False) -> list[str]:
    """Cài model dò giai điệu. YourMT3+ bắt buộc; MuScriptor small tùy chọn: thiếu thì in cảnh báo kèm cách bật,
    chỉ tính là lỗi khi `muscriptor_required` (cờ `--muscriptor`). Trả về danh sách lỗi (rỗng là được)."""
    errors = [] if muscriptor_only else fetch_yourmt3(check)
    if not muscriptor_only:
        fetch_roformer(check)
    status, problems = fetch_muscriptor(check)
    if status != "ok":
        print("\nCẢNH BÁO: chưa có MuScriptor small, cách dò \"Kỹ\" sẽ bị tắt trong extension "
              "(không ảnh hưởng cách dò \"Nhanh\").", flush=True)
        print(MUSCRIPTOR_STEPS, flush=True)
        if muscriptor_required:
            errors += problems
    return errors


def fetch_beat_this() -> None:
    from chordtrack.beats import _checkpoint_path

    print(f"beat_this: {_checkpoint_path()}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--extra", action="store_true", help="tải thêm consonance-ACE (chỉ dùng cho đánh giá)")
    ap.add_argument("--check", action="store_true", help="chỉ kiểm tra, không tải")
    ap.add_argument("--no-beat-this", action="store_true", help="không tải checkpoint beat_this")
    ap.add_argument("--melody", action="store_true",
                    help="tải thêm model dò giai điệu: YourMT3+ (bắt buộc) và MuScriptor small (tùy chọn, cần "
                         "hf auth login; thiếu thì chỉ cảnh báo)")
    ap.add_argument("--muscriptor", action="store_true",
                    help="chỉ tải MuScriptor small (sau khi hf auth login), báo lỗi nếu không tải được")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.muscriptor:
        if fetch_melody(args.check, muscriptor_only=True, muscriptor_required=True):
            print("Lỗi: chưa có MuScriptor small", file=sys.stderr)
            return 1
        print("Đủ MuScriptor small: cách dò \"Kỹ\" dùng được sau khi khởi động lại server.")
        return 0

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
    if args.melody and fetch_melody(args.check):
        failed.append("dò giai điệu")
    if failed:
        print(f"Lỗi: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("Đủ model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
