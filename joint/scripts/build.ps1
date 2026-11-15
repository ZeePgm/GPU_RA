param([string]$Target='realgpu_llama')
$ErrorActionPreference='Stop'
$jointRoot=Split-Path $PSScriptRoot -Parent
$env:VSLANG='1033'
$jointCuda='D:/HBF-Sim/HBFLab/tools/cuda-13.0.2-m0'
$env:CUDA_PATH=$jointCuda
& 'C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/Tools/Launch-VsDevShell.ps1' -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
$env:PATH="$jointCuda/bin;$jointCuda/bin/x64;$env:PATH"
$jointBuild=Join-Path $jointRoot 'build/model'
cmake -S "$jointRoot/source/adapters/llama_cpp/integrated" -B $jointBuild -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a-real -DHBFLAB_PTX_ARCHITECTURE=compute_75 "-DCUDAToolkit_ROOT=$jointCuda" "-DCMAKE_CUDA_COMPILER=$jointCuda/bin/nvcc.exe" *> "$jointRoot/results/configure.log"
if($LASTEXITCODE -ne 0){Get-Content "$jointRoot/results/configure.log" -Tail 70;throw 'Configure failed'}
cmake --build $jointBuild --target $Target -j 4 *> "$jointRoot/results/build-$Target.log"
if($LASTEXITCODE -ne 0){Get-Content "$jointRoot/results/build-$Target.log" -Tail 70;throw 'Build failed'}
Get-Content "$jointRoot/results/build-$Target.log" -Tail 5
