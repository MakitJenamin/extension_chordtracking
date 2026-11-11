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
    5. Tải mã nguồn và weights các model (scripts\fetch_models.py --melody), kiểm tra SHA-256.
       Dò giai điệu cài mặc định: YourMT3+ (cách dò "Nhanh", không cần tài khoản). MuScriptor small (cách dò "Kỹ")
       là tùy chọn: repo gated trên Hugging Face, chưa đăng nhập thì chỉ cảnh báo, setup vẫn chạy tiếp.
    6. Kiểm tra ffmpeg và Deno (yt-dlp cần để tải YouTube), thiếu thì cài bằng winget.
    7. Chạy thử (scripts\smoke_test.py): tạo audio hợp âm C – G, chạy đủ các model, kiểm tra nhận đúng; chạy thử
       YourMT3+ trên một đoạn giai điệu tổng hợp.
  Cuối cùng in tóm tắt: thời gian từng bước, cách dò giai điệu nào dùng được và cách bật phần còn thiếu.

.PARAMETER Cpu
  Cài PyTorch bản CPU kể cả khi có GPU NVIDIA.
.PARAMETER Full
  Cài thêm thư viện đánh giá, tải thêm consonance-ACE (thư viện tách nguồn đã cài cùng dò giai điệu).
.PARAMETER NoMelody
  Không cài dò giai điệu (gồm tách giọng). Mặc định có cài.
.PARAMETER Melody
  Giữ để lệnh cũ vẫn chạy: dò giai điệu nay cài mặc định. Cách dò "Kỹ" cần MuScriptor small (tài khoản Hugging
  Face): tạo token Read, bấm đồng ý license CC BY-NC 4.0 ở https://huggingface.co/MuScriptor/muscriptor-small,
  chạy `.venv\Scripts\hf.exe auth login`, rồi `.venv\Scripts\python.exe scripts\fetch_models.py --muscriptor`.
.PARAMETER NoTools
  Không tự cài ffmpeg, Deno bằng winget (chỉ báo thiếu).
#>
param(
    [switch]$Cpu,
    [switch]$Full,
    [switch]$Melody,
    [switch]$NoMelody,
    [switch]$NoTools
)
$WithMelody = -not $NoMelody

# Không dùng "Stop": trên Windows PowerShell 5.1, chương trình ngoài (pip, python, git) chỉ cần ghi vào stderr
# là bị coi như lỗi và dừng script, kể cả khi chạy thành công. Mọi lệnh ngoài được kiểm tra bằng $LASTEXITCODE.
$ErrorActionPreference = "Continue"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

$Timing = [ordered]@{}
$StepName = $null
$StepWatch = $null
function Step($msg) {
    if ($script:StepName) { $Timing[$script:StepName] = $script:StepWatch.Elapsed }
    $script:StepName = $msg
    $script:StepWatch = [Diagnostics.Stopwatch]::StartNew()
    Write-Host "`n==> $msg" -ForegroundColor Cyan
}
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
# Tách nguồn (audio-separator: BS-Roformer): dò giai điệu luôn tách giọng trước khi dò, nên cài cùng dò giai điệu.
if ($Full -or $WithMelody) { $reqs += @("-r", "requirements\separation.txt") }
if ($Full) { $reqs += @("-r", "requirements\eval.txt") }
Run $Python (@("-m", "pip", "install", "--quiet") + $reqs + @("-c", "requirements\constraints.txt"))
Run $Python @("-m", "pip", "install", "--quiet", "--no-deps", "-e", ".")
if ($WithMelody) {
    Run $Python @("-m", "pip", "install", "--quiet", "-r", "requirements\melody_yourmt3.txt", "-c", "requirements\constraints.txt")
    # Mã MuScriptor (MIT, tải không cần tài khoản; weights gated tải ở bước 5). Lỗi thì chỉ cảnh báo: chỉ cách dò "Kỹ" cần.
    # --no-deps: muscriptor khai báo numpy>=2 nhưng chạy được với numpy 1.26 (không được lên numpy 2)
    & $Python -m pip install --quiet --no-deps -r requirements\melody_muscriptor.txt -c requirements\constraints.txt
    if ($LASTEXITCODE -ne 0) { Write-Warning "Không cài được gói muscriptor: cách dò `"Kỹ`" sẽ tắt, cách dò `"Nhanh`" vẫn dùng được." }
}

# 5. Model ------------------------------------------------------------------------------------------
Step "Model (hợp âm, nhịp, giai điệu)"
if (-not (Have "git")) {
    if (-not (Winget-Install "Git.Git")) { throw "Cần git: winget install Git.Git, rồi chạy lại." }
}
$fetch = @("scripts\fetch_models.py")
if ($Full) { $fetch += "--extra" }
# --melody: YourMT3+ bắt buộc; MuScriptor small thiếu quyền tải (chưa đăng nhập Hugging Face) chỉ cảnh báo, mã thoát 0.
if ($WithMelody) { $fetch += "--melody" }
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
$smoke = @("scripts\smoke_test.py")
if ($WithMelody) { $smoke += "--melody" }
Run $Python $smoke
$Timing[$StepName] = $StepWatch.Elapsed

# Tóm tắt -------------------------------------------------------------------------------------------
Write-Host "`n==> Tóm tắt" -ForegroundColor Cyan
foreach ($k in $Timing.Keys) { Write-Host ("  {0,-34} {1:mm\:ss}" -f $k, $Timing[$k]) }
Write-Host ("  PyTorch: " + (& $Python -c "import torch; print(torch.__version__, '- GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'không có, chạy CPU (chậm hơn)')"))
if ($WithMelody) {
    $ready = @((& $Python -c "from chordtrack.melody import pipeline as p; r = p.variant_ready(p.melody_ready()); print(' '.join(k for k, v in r.items() if v))") -split " ")
    Write-Host "  Dò giai điệu:"
    if ($ready -contains "yourmt3") { Write-Host "    Nhanh (YourMT3+):   có" }
    else { Write-Host "    Nhanh (YourMT3+):   THIẾU, xem lỗi ở bước Model rồi chạy lại setup" -ForegroundColor Red }
    if ($ready -contains "combo") {
        Write-Host "    Kỹ (+ MuScriptor):  có"
    } else {
        Write-Host "    Kỹ (+ MuScriptor):  chưa bật (tùy chọn, cần tài khoản Hugging Face, chỉ dùng phi thương mại). Muốn bật:" -ForegroundColor Yellow
        Write-Host "      1. Tạo access token loại Read ở https://huggingface.co/settings/tokens"
        Write-Host "      2. Mở https://huggingface.co/MuScriptor/muscriptor-small, đăng nhập, bấm đồng ý điều khoản"
        Write-Host "      3. .venv\Scripts\hf.exe auth login        (dán token)"
        Write-Host "         .venv\Scripts\python.exe scripts\fetch_models.py --muscriptor"
        Write-Host "      rồi khởi động lại server."
    }
    if ($ready -contains "yourmt3_vocals") { Write-Host "    Tách giọng:         có" }
    else { Write-Host "    Tách giọng:         chưa cài (gói audio-separator lỗi?), dò trên bản mix. Xem lỗi ở bước Thư viện Python" }
} else {
    Write-Host "  Dò giai điệu: không cài (-NoMelody). Muốn cài: chạy lại setup không có -NoMelody."
}

Write-Host "`nXong. Bước tiếp theo:" -ForegroundColor Green
Write-Host "  1. Chạy server:   powershell -ExecutionPolicy Bypass -File scripts\start_server.ps1 -Background"
Write-Host "  2. Nạp extension: chrome://extensions -> Developer mode -> Load unpacked -> chọn thư mục extension\"
Write-Host "  3. Mở một video YouTube, bấm nút ChordTrack trên thanh điều khiển của player."
