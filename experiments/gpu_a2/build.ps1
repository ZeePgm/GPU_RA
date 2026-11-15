# Run from a Visual Studio x64 developer PowerShell (cl and link on PATH).
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$buildDir = Join-Path $repoRoot 'build-a2'
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
Push-Location $buildDir
try {
    & cl /nologo /std:c++17 /O2 /EHsc /MD /utf-8 "/I$repoRoot/third_party/HBF-EventLab/include" "/I$repoRoot/third_party/llama.cpp-integrated/vendor" "$PSScriptRoot/fifo.cpp" "$repoRoot/third_party/HBF-EventLab/src/infrastructure/event_queue.cpp" /Fe:a2_fifo.exe
    if ($LASTEXITCODE -ne 0) { throw 'A2 compilation failed' }
} finally { Pop-Location }
