param([string]$UvExecutable = 'uv')
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $ProjectRoot
try {
    # All packages are installed into this project; the global Python is untouched.
    if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
        & $UvExecutable venv --python 3.12.13 .venv
        if ($LASTEXITCODE -ne 0) { throw 'Project environment creation failed.' }
    }
    & $UvExecutable pip sync --python .venv/Scripts/python.exe requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
    if ($LASTEXITCODE -ne 0) { throw 'Pinned dependency installation failed.' }
    & .venv/Scripts/python.exe scripts/smoke_gpu.py
    if ($LASTEXITCODE -ne 0) { throw 'Actual GPU verification failed.' }
} finally {
    Pop-Location
}
