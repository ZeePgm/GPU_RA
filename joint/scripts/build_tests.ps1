param([switch]$GpuOnly)
$ErrorActionPreference='Stop'
$jointRoot=Split-Path $PSScriptRoot -Parent
$env:VSLANG='1033'
& 'C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/Tools/Launch-VsDevShell.ps1' -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
$env:PATH="D:/HBF-Sim/HBFLab/tools/cuda-13.0.2-m0/bin;D:/HBF-Sim/HBFLab/tools/cuda-13.0.2-m0/bin/x64;$env:PATH"
if(!$GpuOnly){
cmake -S "$jointRoot/source" -B "$jointRoot/build/core" -G Ninja -DCMAKE_BUILD_TYPE=Release -DHBFLAB_ENABLE_CUDA=OFF -DBUILD_TESTING=ON -DHBFLAB_BACKEND_TESTS=ON *> "$jointRoot/results/configure-core.log"
if($LASTEXITCODE -ne 0){Get-Content "$jointRoot/results/configure-core.log" -Tail 40;throw 'Core configure failed'}
cmake --build "$jointRoot/build/core" -j 4 *> "$jointRoot/results/build-core.log"
if($LASTEXITCODE -ne 0){Get-Content "$jointRoot/results/build-core.log" -Tail 40;throw 'Core build failed'}
ctest --test-dir "$jointRoot/build/core" --output-on-failure -j 2 --timeout 120 *> "$jointRoot/results/test-core.log"
if($LASTEXITCODE -ne 0){Get-Content "$jointRoot/results/test-core.log" -Tail 65;throw 'Core tests failed'}
Get-Content "$jointRoot/results/test-core.log" -Tail 5
}
cmake --build "$jointRoot/build/model" --target joint_b2_fixture joint_media_gpu -j 4 *> "$jointRoot/results/build-fixture.log"
if($LASTEXITCODE -ne 0){Get-Content "$jointRoot/results/build-fixture.log" -Tail 40;throw 'Fixture build failed'}
& "$jointRoot/build/model/bin/joint_media_gpu.exe" *> "$jointRoot/results/media-only.log"
if($LASTEXITCODE -ne 0){Get-Content "$jointRoot/results/media-only.log";throw 'Media-only fixture failed'}
& "$jointRoot/build/model/bin/joint_b2_fixture.exe" "$jointRoot/results/b2-fixture.json" normal *> "$jointRoot/results/b2-fixture.log"
if($LASTEXITCODE -ne 0){Get-Content "$jointRoot/results/b2-fixture.log";throw 'GPU fixture failed'}
& "$jointRoot/build/model/bin/joint_b2_fixture.exe" "$jointRoot/results/b2-negative.json" negative *> "$jointRoot/results/b2-negative.log"
if($LASTEXITCODE -ne 23 -or (Test-Path "$jointRoot/results/b2-negative.json")){throw 'GPU negative result failed'}
if(!(Select-String -Path "$jointRoot/results/b2-negative.log" -SimpleMatch 'terminal status 1')){throw 'Unexpected negative failure type'}
Get-Content "$jointRoot/results/b2-negative.log"
