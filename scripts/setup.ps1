<#
.SYNOPSIS
  Cài ChordTrack từ đầu trên Windows: venv, PyTorch, thư viện, model, ffmpeg và Deno.

.DESCRIPTION
  Chạy từ thư mục repo vừa clone:
    powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

  Các bước (chạy lại nhiều lần không sao, bước nào xong rồi sẽ bỏ qua):
    1. Tìm Python 3.11 (py -3.11). Không có thì cài bằng winget.
    2. Tạo .venv ở thư mục repo.
    3. Cài PyTorch: bản CUDA 12.6 nếu máy có GPU NVIDIA (nvidia-smi), không thì bản CPU.
    4. Cài thư viện trong requirements\runtime.txt (giữ numpy 1.26).
    5. Tải mã nguồn và weights các model (scripts\fetch_models.py), kiểm tra SHA-256.
    6. Kiểm tra ffmpeg và Deno (yt-dlp cần để tải YouTube), thiếu thì cài bằng winget.
    7. Chạy thử: nạp các model và in thiết bị (GPU/CPU).

.PARAMETER Cpu
  Cài PyTorch bản CPU kể cả khi có GPU NVIDIA.
.PARAMETER Full
  Cài thêm thư viện tách nguồn (--separation) và đánh giá, tải thêm consonance-ACE.
.PARAMETER NoTools
  Không tự cài ffmpeg, Deno bằng winget (chỉ báo thiếu).
#>
param(
    [switch]$Cpu,
    [switch]$Full,
    [switch]$NoTools
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Refresh-Path {
    $m = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $u = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = (@($m, $u, $env:Path) | Where-Object { $_ }) -join ";"
}
function Have($cmd) { return [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }
function Winget-Install($id) {
    if ($NoTools) { return $false }
    if (-not (Have "winget")) { return $false }
    Write-Host "Cài $id bằng winget..."
    & winget install --id $id -e --silent --accept-package-agreements --accept-source-agreements | Out-Host
    Refresh-Path
    return $true
}
function Run($exe, [string[]]$argv) {
    & $exe @argv
    if ($LASTEXITCODE -ne 0) { throw "Lỗi khi chạy: $exe $($argv -join ' ')" }
}

Refresh-Path

# 1. Python 3.11 ------------------------------------------------------------------------------------
Step "Python 3.11"
$py = $null
if (Have "py") {
    & py -3.11 -c "import sys" 2>$null
    if ($LASTEXITCODE -eq 0) { $py = @("py", "-3.11") }
}
if (-not $py) {
    if (Winget-Install "Python.Python.3.11") {
        & py -3.11 -c "import sys" 2>$null
        if ($LASTEXITCODE -eq 0) { $py = @("py", "-3.11") }
    }
}
if (-not $py) { throw "Cần Python 3.11: winget install Python.Python.3.11 (hoặc tải ở python.org), rồi chạy lại." }
Write-Host ("Dùng " + (& $py[0] $py[1] -c "import sys; print(sys.executable)"))

# 2. venv -------------------------------------------------------------------------------------------
Step "Môi trường ảo .venv"
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { Run $py[0] @($py[1], "-m", "venv", ".venv") }
Run $Python @("-m", "pip", "install", "--quiet", "--upgrade", "pip")

# 3. PyTorch ----------------------------------------------------------------------------------------
Step "PyTorch"
$gpu = (-not $Cpu) -and (Have "nvidia-smi")
$index = if ($gpu) { "https://download.pytorch.org/whl/cu126" } else { "https://download.pytorch.org/whl/cpu" }
$tag = if ($gpu) { "+cu126" } else { "+cpu" }
$have = (& $Python -c "import torch; print(torch.__version__)" 2>$null)
if ($LASTEXITCODE -eq 0 -and $have -eq "2.14.1$tag") {
    Write-Host "Đã có torch $have"
} else {
    Write-Host ("Cài torch 2.14.1$tag (" + ($(if ($gpu) { "GPU NVIDIA" } else { "CPU" })) + ", khoảng 2.5 GB với bản CUDA)...")
    Run $Python @("-m", "pip", "install", "torch==2.14.1$tag", "torchaudio==2.11.0$tag", "--index-url", $index)
}

# 4. Thư viện ---------------------------------------------------------------------------------------
Step "Thư viện Python"
$reqs = @("-r", "requirements\runtime.txt")
if ($Full) { $reqs += @("-r", "requirements\separation.txt", "-r", "requirements\eval.txt") }
Run $Python (@("-m", "pip", "install", "--quiet") + $reqs + @("-c", "requirements\constraints.txt"))
Run $Python @("-m", "pip", "install", "--quiet", "--no-deps", "-e", ".")

# 5. Model ------------------------------------------------------------------------------------------
Step "Model hợp âm và bắt nhịp"
if (-not (Have "git")) {
    if (-not (Winget-Install "Git.Git")) { throw "Cần git: winget install Git.Git, rồi chạy lại." }
}
$fetch = @("scripts\fetch_models.py")
if ($Full) { $fetch += "--extra" }
Run $Python $fetch

# 6. ffmpeg, Deno -----------------------------------------------------------------------------------
Step "ffmpeg và Deno"
$missing = @()
if (-not (Have "ffmpeg")) { if (-not ((Winget-Install "Gyan.FFmpeg") -and (Have "ffmpeg"))) { $missing += "ffmpeg (winget install Gyan.FFmpeg)" } }
if (-not (Have "deno")) { if (-not ((Winget-Install "DenoLand.Deno") -and (Have "deno"))) { $missing += "Deno (winget install DenoLand.Deno)" } }
if ($missing) { Write-Warning ("Còn thiếu: " + ($missing -join ", ") + ". Cài xong thì mở PowerShell mới.") }
else { Write-Host "Có ffmpeg và Deno." }

# 7. Chạy thử ---------------------------------------------------------------------------------------
Step "Chạy thử"
Run $Python @("-c", @"
import torch
from chordtrack.chords import chordmini, cnnlstm, chordformer
from chordtrack import beats
chordmini._load_model_cached('btc'); cnnlstm._load_ensemble(*cnnlstm._import_modules()[:2]); chordformer._load_ensemble()
beats._checkpoint_path()
dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (chậm hơn, khoảng 1 phút mỗi bài)'
print('Nạp đủ model. Thiết bị:', dev)
"@)

Write-Host "`nXong. Bước tiếp theo:" -ForegroundColor Green
Write-Host "  1. Chạy server:   powershell -ExecutionPolicy Bypass -File scripts\start_server.ps1 -Background"
Write-Host "  2. Nạp extension: chrome://extensions -> Developer mode -> Load unpacked -> chọn thư mục extension\"
Write-Host "  3. Mở một video YouTube, bấm nút ChordTrack trên thanh điều khiển của player."
