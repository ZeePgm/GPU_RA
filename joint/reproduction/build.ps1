param(
 [Parameter(Mandatory=$true)][string]$CudaRoot,
 [string]$Python='python',
 [string]$BuildRoot=(Join-Path $PSScriptRoot '../build-reproduction'),
 [string]$CudaArchitectures='120a-real',
 [string]$PtxArchitecture='compute_75',
 [int]$Jobs=4,
 [string]$VsRoot,
 [string]$CMake,
 [string]$Ninja,
 [switch]$Reuse,
 [switch]$SkipTests
)
$ErrorActionPreference='Stop'
if($Jobs -lt 1){throw 'Jobs must be positive'}
$taskJoint=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$taskSource=Join-Path $taskJoint 'source'
$taskBuild=[IO.Path]::GetFullPath($BuildRoot)
$taskPython=(Get-Command $Python -ErrorAction Stop).Source
& $taskPython -c "import sys; assert sys.version_info >= (3,12), 'Python 3.12+ required'"
if($LASTEXITCODE -ne 0){throw 'Python check failed'}
if(!(Test-Path -LiteralPath (Join-Path $taskSource 'CMakeLists.txt'))){
 & $taskPython (Join-Path $taskJoint 'restore_source.py')
 if($LASTEXITCODE -ne 0){throw 'Sealed source restore failed'}
}
& $taskPython (Join-Path $PSScriptRoot 'verify_source.py')
if($LASTEXITCODE -ne 0){throw 'Historical V3 source identity mismatch'}
if((Test-Path -LiteralPath $taskBuild) -and !$Reuse){throw 'Use a fresh BuildRoot or explicitly pass -Reuse'}
if(!$VsRoot){
 $taskVswhere=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
 $VsRoot=& $taskVswhere -latest -version '[17.0,18.0)' -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
}
if(!$VsRoot){throw 'Visual Studio 2022 C++ Build Tools are required'}
$env:VSLANG='1033'
& (Join-Path $VsRoot 'Common7/Tools/Launch-VsDevShell.ps1') -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
if(!$CMake){$CMake=Join-Path $VsRoot 'Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'}
if(!$Ninja){$Ninja=Join-Path $VsRoot 'Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe'}
$taskCtest=Join-Path (Split-Path $CMake) 'ctest.exe'
$taskCuda=(Resolve-Path -LiteralPath $CudaRoot).Path
$env:CUDA_PATH=$taskCuda
$env:CUDACXX=Join-Path $taskCuda 'bin/nvcc.exe'
$taskRuntime=@('bin','bin/x64','lib','lib/x64','extras/CUPTI/lib64','extras/CUPTI/lib/x64') | ForEach-Object {Join-Path $taskCuda $_} | Where-Object {Test-Path -LiteralPath $_}
$env:PATH=($taskRuntime -join ';')+';'+(Split-Path $taskPython)+';'+$env:PATH
$taskReport=Join-Path $taskJoint ('reproduction-validation/'+(Get-Date -Format 'yyyyMMdd-HHmmss')+'-'+[guid]::NewGuid().ToString('N').Substring(0,8))
New-Item -ItemType Directory -Path $taskReport -Force | Out-Null
$taskCore=Join-Path $taskBuild 'core'
$taskModel=Join-Path $taskBuild 'model'
if($Reuse){
 foreach($taskPair in @(@($taskCore,$taskSource),@($taskModel,(Join-Path $taskSource 'adapters/llama_cpp/integrated')))){
  $taskCache=Join-Path $taskPair[0] 'CMakeCache.txt'
  if(Test-Path -LiteralPath $taskCache){
   $taskHome=Select-String -LiteralPath $taskCache -Pattern '^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$'
   if(!$taskHome -or ([IO.Path]::GetFullPath($taskHome.Matches[0].Groups[1].Value) -ne [IO.Path]::GetFullPath($taskPair[1]))){throw 'Reuse build source does not match this restored V3 snapshot'}
  }
 }
}
[ordered]@{source=$taskSource;build_root=$taskBuild;cuda_root=$taskCuda;cuda_architectures=$CudaArchitectures;ptx_architecture=$PtxArchitecture;python=$taskPython;vs_root=$VsRoot;cmake=$CMake;ninja=$Ninja;jobs=$Jobs;skip_tests=[bool]$SkipTests;reuse=[bool]$Reuse} | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $taskReport 'build-configuration.json')
& $CMake -S $taskSource -B $taskCore -G Ninja '-DCMAKE_BUILD_TYPE=Release' "-DCMAKE_MAKE_PROGRAM=$Ninja" "-DPython3_EXECUTABLE=$taskPython" '-DHBFLAB_ENABLE_CUDA=OFF' '-DBUILD_TESTING=ON' '-DHBFLAB_BACKEND_TESTS=ON' '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON' *> (Join-Path $taskReport 'configure-core.log')
if($LASTEXITCODE -ne 0){throw "Core configure failed; $taskReport"}
& $CMake --build $taskCore --parallel $Jobs *> (Join-Path $taskReport 'build-core.log')
if($LASTEXITCODE -ne 0){throw "Core build failed; $taskReport"}
if(!$SkipTests){
 & $taskCtest --test-dir $taskCore --output-on-failure --parallel 2 --timeout 120 *> (Join-Path $taskReport 'ctest-core.log')
 if($LASTEXITCODE -ne 0){throw "Core tests failed; $taskReport"}
}
& $CMake -S (Join-Path $taskSource 'adapters/llama_cpp/integrated') -B $taskModel -G Ninja '-DCMAKE_BUILD_TYPE=Release' "-DCMAKE_MAKE_PROGRAM=$Ninja" "-DPython3_EXECUTABLE=$taskPython" "-DCUDAToolkit_ROOT=$taskCuda" "-DCMAKE_CUDA_COMPILER=$env:CUDACXX" "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures" "-DHBFLAB_PTX_ARCHITECTURE=$PtxArchitecture" '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON' *> (Join-Path $taskReport 'configure-model.log')
if($LASTEXITCODE -ne 0){throw "Model configure failed; $taskReport"}
& $CMake --build $taskModel --target realgpu_llama joint_b2_fixture joint_media_gpu --parallel $Jobs *> (Join-Path $taskReport 'build-model.log')
if($LASTEXITCODE -ne 0){throw "Model build failed; $taskReport"}
$env:PATH=(Join-Path $taskModel 'bin')+';'+$env:PATH
if(!$SkipTests){
 & (Join-Path $taskModel 'bin/joint_media_gpu.exe') *> (Join-Path $taskReport 'media-only.log')
 if($LASTEXITCODE -ne 0){throw "Shared-media GPU fixture failed; $taskReport"}
 & (Join-Path $taskModel 'bin/joint_b2_fixture.exe') (Join-Path $taskReport 'b2-fixture.json') normal *> (Join-Path $taskReport 'b2-fixture.log')
 if($LASTEXITCODE -ne 0){throw "B2 GPU fixture failed; $taskReport"}
 & (Join-Path $taskModel 'bin/joint_b2_fixture.exe') (Join-Path $taskReport 'b2-negative.json') negative *> (Join-Path $taskReport 'b2-negative.log')
 if($LASTEXITCODE -ne 23){throw "B2 GPU negative case must reject with exit 23; $taskReport"}
}
$taskAudit=@((Join-Path $PSScriptRoot 'check_build.py'),'--joint-root',$taskJoint,'--build-root',$taskBuild,'--reports',$taskReport)
if($SkipTests){$taskAudit+='--skip-tests'}
& $taskPython @taskAudit
if($LASTEXITCODE -ne 0){throw "Source/build/fixture audit failed; $taskReport"}
Write-Output "PASS: new V3 build and reports $taskReport"
