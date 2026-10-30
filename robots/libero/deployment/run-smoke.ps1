param(
    [string]$OutputDir = "",
    [string]$Proxy = "http://127.0.0.1:7890"
)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) -Parent
$previousDirectory = Get-Location
$previousHttp = $env:HTTP_PROXY
$previousHttps = $env:HTTPS_PROXY
$previousBypass = $env:NO_PROXY
try {
    Set-Location -LiteralPath $projectRoot
    if ($Proxy) {
        $env:HTTP_PROXY = $Proxy
        $env:HTTPS_PROXY = $Proxy
    }
    $env:NO_PROXY = "127.0.0.1,localhost"
    if (-not $OutputDir) {
        $OutputDir = "logs/harness-vla-baseline/smoke-" + (Get-Date -Format "yyyyMMdd-HHmmss")
    }
    $experimentArgs = @(
        "--worker-endpoint", "http://127.0.0.1:8870",
        "--memory-dir", "logs/harness-vla-baseline/memory/baseline_gpt55",
        "--memory-provenance", "RLinf/RPent-memory@47f301ef33bf8d0e8c770342f16b3c39c6a0e8c1/libero/GPT_5.5_xhigh",
        "--suite", "libero_object", "--task", "0", "--seed", "0", "--libero-type", "standard",
        "--model", "gpt-6.1-sol", "--reasoning-effort", "xhigh",
        "--max-turns", "100", "--timeout-s", "5000", "--max-env-steps", "10000", "--max-vla-calls", "240",
        "--vla-identity", "RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT@6222623f635769bfc73c9472e29fab9b7fd8e027; sha256:4d9089c941793f170b625c2ed0ac7a3aa09b6f103e52dbbc82e67301529d6683",
        "--vla-endpoint", "http://127.0.0.1:8801", "--sam3-endpoint", "http://127.0.0.1:8802",
        "--cuda-device", "2", "--output-dir", $OutputDir
    )
    & .venv/Scripts/python.exe -m robots.libero.baseline_smoke @experimentArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Baseline smoke did not pass. Keep the output and inspect $OutputDir/baseline-smoke.json."
    }
} finally {
    $env:HTTP_PROXY = $previousHttp
    $env:HTTPS_PROXY = $previousHttps
    $env:NO_PROXY = $previousBypass
    Set-Location -LiteralPath $previousDirectory
}
