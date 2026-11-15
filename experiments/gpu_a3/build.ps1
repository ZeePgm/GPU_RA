# Visual Studio x64 developer PowerShell; reuse verified EventLab libraries.
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$buildDir = Join-Path $repoRoot 'build-a3'
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
Push-Location $buildDir
try {
 & cl /nologo /std:c++17 /O2 /EHsc /MD /utf-8 "/I$repoRoot/third_party/HBF-EventLab/include" "/I$repoRoot/third_party/llama.cpp-integrated/vendor" "$PSScriptRoot/scan.cpp" /Fe:a3_scan.exe "$repoRoot/build-portability-model/eventlab/hbf_eventlab_tensor_bridge.lib" "$repoRoot/build-portability-model/eventlab/hbf_eventlab.lib"
 if ($LASTEXITCODE -ne 0) { throw 'A3 compilation failed' }
} finally { Pop-Location }
