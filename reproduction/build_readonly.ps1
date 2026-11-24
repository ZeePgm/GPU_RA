param(
 [Parameter(Mandatory=$true)][string]$CudaRoot,
 [string]$BuildDir,
 [string]$CudaArchitectures='native',
 [string]$PtxArchitecture='compute_75',
 [int]$Jobs=4,
 [string]$VsRoot,
 [string]$CMake,
 [string]$Ninja,
 [switch]$ConfigureOnly
)
$ErrorActionPreference='Stop'
$taskRoot=Split-Path $PSScriptRoot -Parent
if(!$BuildDir){$BuildDir=Join-Path $taskRoot 'build-v2'}
if($Jobs -lt 1){throw 'Jobs must be positive'}
if(!$VsRoot){
 $taskVswhere=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
 $VsRoot=& $taskVswhere -latest -version '[17.0,18.0)' -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
}
if(!$VsRoot){throw 'Visual Studio 2022 C++ Build Tools are required'}
$env:VSLANG='1033'
& (Join-Path $VsRoot 'Common7/Tools/Launch-VsDevShell.ps1') -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
if(!$CMake){$CMake=Join-Path $VsRoot 'Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'}
if(!$Ninja){$Ninja=Join-Path $VsRoot 'Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe'}
$taskCuda=(Resolve-Path -LiteralPath $CudaRoot).Path
$env:CUDA_PATH=$taskCuda
$env:CUDACXX=Join-Path $taskCuda 'bin/nvcc.exe'
$env:PATH=(Join-Path $taskCuda 'bin')+';'+(Join-Path $taskCuda 'bin/x64')+';'+$env:PATH
& $CMake -S $taskRoot -B $BuildDir -G Ninja '-DCMAKE_BUILD_TYPE=Release' '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON' "-DCMAKE_MAKE_PROGRAM=$Ninja" "-DCUDAToolkit_ROOT=$taskCuda" "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures" "-DHBFLAB_PTX_ARCHITECTURE=$PtxArchitecture"
if($LASTEXITCODE -ne 0){throw 'V2 configure failed'}
if(!$ConfigureOnly){
 & $CMake --build $BuildDir --target f_candidate f_clock r3_adaptive r3_policy r3_run --parallel $Jobs
 if($LASTEXITCODE -ne 0){throw 'V2 build failed'}
}
