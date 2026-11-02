<#
.SYNOPSIS
  Khởi động backend chordtrack (chỉ lắng nghe 127.0.0.1).

.DESCRIPTION
  Nạp lại PATH của máy và user (để thấy ffmpeg, deno vừa cài bằng winget mà không cần mở
  shell mới), kích hoạt venv rồi chạy `python -m chordtrack.server`.

  Venv tìm theo thứ tự: biến môi trường CHORDTRACK_VENV, `.venv` ở thư mục repo, rồi venv
  của checkout chính (C:\Users\MakitJenamin\Documents\extension_chordtracking\.venv).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\start_server.ps1
  powershell -ExecutionPolicy Bypass -File scripts\start_server.ps1 -Port 8766
  powershell -ExecutionPolicy Bypass -File scripts\start_server.ps1 -Cookies C:\path\cookies.txt
#>
param(
    [int]$Port = 8765,
    # File cookie YouTube (định dạng Netscape), dùng khi YouTube đòi xác minh "not a bot".
    [string]$Cookies = "",
    [string]$LogLevel = "info"
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
$candidates += "C:\Users\MakitJenamin\Documents\extension_chordtracking\.venv"
$Venv = $candidates | Where-Object { Test-Path (Join-Path $_ "Scripts\python.exe") } | Select-Object -First 1
if (-not $Venv) {
    Write-Error "Không tìm thấy venv. Đặt CHORDTRACK_VENV=<thư mục venv> hoặc tạo .venv ở $RepoRoot."
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
Write-Host "Venv: $Venv"
Write-Host "Server: http://127.0.0.1:$Port  (Ctrl+C để dừng)"
& $Python -m chordtrack.server --port $Port --log-level $LogLevel
exit $LASTEXITCODE
