param(
    [switch]$Dev,
    [switch]$Cpu,
    [string]$Cuda
)

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Get-CudaTag {
    if ($Cpu) { return "cpu" }
    if ($Cuda) { return $Cuda }
    $nvsmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $nvsmi) { return "cpu" }
    $out = & $nvsmi 2>$null
    if ($out -match "CUDA Version:\s+([0-9]+)\.([0-9]+)") {
        $major = [int]$matches[1]
        $minor = [int]$matches[2]
        if ($major -ge 13) { return "cu130" }
        if ($major -eq 12 -and $minor -ge 1) { return "cu121" }
        if ($major -eq 11 -and $minor -ge 8) { return "cu118" }
    }
    return "cpu"
}

$tag = Get-CudaTag
switch ($tag) {
    "cpu" { $indexUrl = "https://download.pytorch.org/whl/cpu" }
    "cu130" { $indexUrl = "https://download.pytorch.org/whl/cu130" }
    "cu121" { $indexUrl = "https://download.pytorch.org/whl/cu121" }
    "cu118" { $indexUrl = "https://download.pytorch.org/whl/cu118" }
    default { $indexUrl = "https://download.pytorch.org/whl/$tag" }
}

Write-Host "Detected CUDA tag: $tag"
Write-Host "Using PyTorch index: $indexUrl"

& python -m pip install -U pip
if ($Dev) {
    & python -m pip install -e ".[dev]"
} else {
    & python -m pip install -e "."
}

& python -m pip install --index-url $indexUrl --extra-index-url https://pypi.org/simple torch torchvision torchaudio
& python -m pip install numpy

Write-Host "Torch install complete. Verify with:"
Write-Host "python -c ""import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"""
