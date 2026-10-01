# Creates this project's virtual environment outside OneDrive and installs requirements.
# Downloads roughly 3 GB (torch cu128 wheel is the bulk). Does not download the model.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $env:USERPROFILE ".venvs\jev-qwen"

if (-not (Test-Path $venv)) {
    py -3.12 -m venv $venv
}
$python = Join-Path $venv "Scripts\python.exe"

& $python -m pip install --upgrade pip
& $python -m pip install -r (Join-Path $root "requirements.txt") --extra-index-url https://download.pytorch.org/whl/cu128
& $python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__)"
Write-Host "Environment ready: $python"
