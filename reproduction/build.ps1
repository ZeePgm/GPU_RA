param(
 [Parameter(Mandatory=$true)][string]$CudaRoot,
 [string]$Python='python',
 [string]$BuildRoot=(Join-Path $PSScriptRoot '../build-reproduction'),
 [string]$CudaArchitectures='120a-real',
 [string]$PtxArchitecture='compute_75',
 [int]$Jobs=4,
 [switch]$ReuseVerifiedModel
)
$ErrorActionPreference='Stop'
$taskRoot=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$taskBuild=[IO.Path]::GetFullPath($BuildRoot)
$taskPython=(Get-Command $Python -ErrorAction Stop).Source
$taskCuda=(Resolve-Path -LiteralPath $CudaRoot).Path
if($Jobs -lt 1){throw 'Jobs must be positive'}
& $taskPython -c 'import sys; assert sys.version_info >= (3, 12), "Python 3.12 or newer is required"'
if($LASTEXITCODE -ne 0){throw 'Python version check failed'}
$env:PATH=(Split-Path -Parent $taskPython)+';'+$env:PATH
if((Test-Path -LiteralPath $taskBuild) -and !$ReuseVerifiedModel){throw 'Use a fresh BuildRoot or explicitly reuse its verified new model build'}
if($ReuseVerifiedModel -and !(Test-Path -LiteralPath (Join-Path $taskBuild 'model/release-identity.json'))){throw 'Reuse requires a sealed model built at this BuildRoot'}
& $taskPython (Join-Path $PSScriptRoot 'verify_source.py')
if($LASTEXITCODE -ne 0){throw 'Historical source identity check failed'}
if(!$ReuseVerifiedModel){
 & (Join-Path $taskRoot 'build.ps1') -Target model -CudaRoot $taskCuda -Python $taskPython -BuildDir (Join-Path $taskBuild 'model') -CudaArchitectures $CudaArchitectures -PtxArchitecture $PtxArchitecture -Jobs $Jobs
 if($LASTEXITCODE -ne 0){throw 'Source model build failed'}
}
$taskBase=Join-Path $taskBuild 'model'
& $taskPython (Join-Path $taskRoot 'tools/integrated_build_identity.py') verify --build-dir $taskBase
if($LASTEXITCODE -ne 0){throw 'New model dependency identity check failed'}
if($ReuseVerifiedModel){
 $taskCache=Get-Content -LiteralPath (Join-Path $taskBase 'CMakeCache.txt')
 function Get-ReproductionCacheValue([string]$Name){
  $taskLine=$taskCache | Where-Object { $_.StartsWith($Name+':') } | Select-Object -First 1
  if(!$taskLine -or !$taskLine.Contains('=')){throw "Missing model cache setting: $Name"}
  return $taskLine.Substring($taskLine.IndexOf('=')+1)
 }
 $taskCachedCuda=(Resolve-Path -LiteralPath (Get-ReproductionCacheValue 'CUDAToolkit_ROOT')).Path
 if($taskCachedCuda -ne $taskCuda){throw 'Reuse requires the model build CUDA toolkit path'}
 if((Get-ReproductionCacheValue 'CMAKE_CUDA_ARCHITECTURES') -ne $CudaArchitectures){throw 'Reuse requires the model build CUDA target architectures'}
 if((Get-ReproductionCacheValue 'HBFLAB_PTX_ARCHITECTURE') -ne $PtxArchitecture){throw 'Reuse requires the model build PTX target architecture'}
}
$taskVswhere=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$taskVs=& $taskVswhere -latest -version '[17.0,18.0)' -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if(!$taskVs){throw 'Visual Studio 2022 C++ build tools not found'}
if($ReuseVerifiedModel){
 $env:VSLANG='1033'
 & (Join-Path $taskVs 'Common7/Tools/Launch-VsDevShell.ps1') -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
 $env:CUDA_PATH=$taskCuda
 $env:PATH=(Join-Path $taskCuda 'bin')+';'+(Join-Path $taskCuda 'bin/x64')+';'+$env:PATH
}
$taskCmake=Join-Path $taskVs 'Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'
$taskCtest=Join-Path $taskVs 'Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/ctest.exe'
$taskNinja=Join-Path $taskVs 'Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe'
$taskFfn=Join-Path $taskBuild 'ffn'
& $taskCmake -S (Join-Path $taskRoot 'experiments/gpu_ffn') -B $taskFfn -G Ninja '-DCMAKE_BUILD_TYPE=Release' "-DBASE_BUILD=$taskBase" "-DCUDAToolkit_ROOT=$taskCuda" "-DCMAKE_MAKE_PROGRAM=$taskNinja"
if($LASTEXITCODE -ne 0){throw 'Source F configure failed'}
& $taskCmake --build $taskFfn --target gpu_ffn ffn_audit_tests --parallel $Jobs
if($LASTEXITCODE -ne 0){throw 'Source F build failed'}
& $taskCtest --test-dir $taskFfn --output-on-failure
if($LASTEXITCODE -ne 0){throw 'F online audit tests failed'}
Copy-Item -Path (Join-Path $taskBase 'bin/*.dll') -Destination (Join-Path $taskFfn 'bin')
& $taskPython -m unittest discover -s (Join-Path $taskRoot 'experiments/gpu_ffn') -p test_run.py -v
if($LASTEXITCODE -ne 0){throw 'F runner checks failed'}
Write-Output "Built from repository sources: $taskBuild"
