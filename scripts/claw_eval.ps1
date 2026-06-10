param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ClawRoot = Join-Path $ProjectRoot "external\claw-eval"
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$EnvFile = Join-Path $ProjectRoot ".env"

if (-not (Test-Path $ClawRoot)) {
    throw "Claw-Eval checkout not found: $ClawRoot"
}
if (-not (Test-Path $Python)) {
    throw "Project virtualenv python not found: $Python"
}

if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match '^\s*([^#=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
        }
    }
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Push-Location $ClawRoot
try {
    & $Python -m claw_eval.cli @Args
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
