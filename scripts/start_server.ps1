<#
.SYNOPSIS
  Khởi động backend chordtrack (chỉ lắng nghe 127.0.0.1).

.DESCRIPTION
  Nạp lại PATH của máy và user (để thấy ffmpeg, deno vừa cài bằng winget mà không cần mở
  shell mới), kích hoạt venv rồi chạy `python -m chordtrack.server`.

  Venv tìm theo thứ tự: biến môi trường CHORDTRACK_VENV, `.venv` ở thư mục repo, rồi `.venv` của
  checkout chính (khi chạy từ git worktree). Chưa có venv thì chạy scripts\setup.ps1 trước.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\start_server.ps1
  powershell -ExecutionPolicy Bypass -File scripts\start_server.ps1 -Port 8766
  powershell -ExecutionPolicy Bypass -File scripts\start_server.ps1 -Cookies C:\path\cookies.txt
#>
param(
    [int]$Port = 8765,
    # File cookie YouTube (định dạng Netscape), dùng khi YouTube đòi xác minh "not a bot".
    [string]$Cookies = "",
    [string]$LogLevel = "info",
    # Chạy ẩn ở nền, không giữ cửa sổ PowerShell (log ở data\cache\server.log)
    [switch]$Background
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

# PATH mới nhất từ registry: shell mở trước khi cài ffmpeg/deno vẫn thấy chúng.
$machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
$env:Path = (@($machinePath, $userPath, $env:Path) | Where-Object { $_ }) -join ";"

$candidates = @()
if ($env:CHORDTRACK_VENV) { $candidates += $env:CHORDTRACK_VENV }
$candidates += (Join-Path $RepoRoot ".venv")
# Git worktree: venv của checkout chính (thư mục cha của `git rev-parse --git-common-dir`).
try {
    $common = (& git -C $RepoRoot rev-parse --path-format=absolute --git-common-dir 2>$null)
    if ($LASTEXITCODE -eq 0 -and $common) { $candidates += (Join-Path (Split-Path -Parent $common) ".venv") }
} catch { }
$Venv = $candidates | Where-Object { Test-Path (Join-Path $_ "Scripts\python.exe") } | Select-Object -First 1
if (-not $Venv) {
    Write-Error "Không tìm thấy venv. Chạy scripts\setup.ps1 trước, hoặc đặt CHORDTRACK_VENV=<thư mục venv>."
}
$Activate = Join-Path $Venv "Scripts\Activate.ps1"
if (Test-Path $Activate) { . $Activate }
$Python = Join-Path $Venv "Scripts\python.exe"

if ($Cookies) {
    if (-not (Test-Path $Cookies)) { Write-Error "Không thấy file cookie: $Cookies" }
    $env:CHORDTRACK_YT_COOKIES = (Resolve-Path $Cookies).Path
}

foreach ($tool in @("ffmpeg", "deno")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        Write-Warning "Không thấy '$tool' trong PATH. Tải YouTube sẽ lỗi (cài: winget install Gyan.FFmpeg / DenoLand.Deno)."
    }
}

$env:PYTHONIOENCODING = "utf-8"
Set-Location $RepoRoot
$ServerArgs = @("-m", "chordtrack.server", "--port", "$Port", "--log-level", $LogLevel)

if ($Background) {
    # Chạy ẩn, tách khỏi cửa sổ này; log ghi vào data/cache/server.log
    $logDir = Join-Path $RepoRoot "data\cache"
    New-Item -ItemType Directory -Force $logDir | Out-Null
    $p = Start-Process -FilePath $Python -ArgumentList $ServerArgs -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $logDir "server.out.log") -RedirectStandardError (Join-Path $logDir "server.log")
    for ($i = 0; $i -lt 40; $i++) {
        try { Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 2 | Out-Null; break } catch { Start-Sleep -Milliseconds 750 }
    }
    Write-Host "Server chạy nền (PID $($p.Id)): http://127.0.0.1:$Port  — log: $logDir\server.log"
    Write-Host "Tắt: Stop-Process -Id $($p.Id)"
    exit 0
}

Write-Host "Venv: $Venv"
Write-Host "Server: http://127.0.0.1:$Port  (Ctrl+C để dừng)"
# Uvicorn ghi log ra stderr. Với "Stop", PowerShell 5.1 coi mỗi dòng stderr của lệnh native
# (khi output bị chuyển hướng) là lỗi và dừng script, làm server tắt ngay sau dòng log đầu.
$ErrorActionPreference = "Continue"
& $Python @ServerArgs
exit $LASTEXITCODE
