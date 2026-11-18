param(
 [Parameter(Mandatory=$true)][string]$CudaRoot,
 [Parameter(Mandatory=$true)][string]$BuildRoot,
 [Parameter(Mandatory=$true)][string]$Python,
 [string[]]$Targets=@('five_model','five_weight','r2_down','symbolic_f','realgpu_llama'),
 [int]$Jobs=6
)
$ErrorActionPreference='Stop'
$taskRepo=(Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$taskSource=Join-Path $taskRepo 'joint/source/adapters/llama_cpp/integrated'
$taskVswhere=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$taskVs=& $taskVswhere -latest -version '[17.0,18.0)' -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if(!$taskVs){throw 'VS2022 Build Tools missing'}
$env:VSLANG='1033'
& (Join-Path $taskVs 'Common7/Tools/Launch-VsDevShell.ps1') -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
$taskCmake=Join-Path $taskVs 'Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'
$taskNinja=Join-Path $taskVs 'Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe'
$env:CUDA_PATH=$CudaRoot
$env:PATH=(Join-Path $CudaRoot 'bin')+';'+(Join-Path $CudaRoot 'bin/x64')+';'+$env:PATH
& $taskCmake -S $taskSource -B $BuildRoot -G Ninja '-DCMAKE_BUILD_TYPE=Release' "-DCMAKE_MAKE_PROGRAM=$taskNinja" "-DPython3_EXECUTABLE=$Python" "-DCUDAToolkit_ROOT=$CudaRoot" "-DCMAKE_CUDA_COMPILER=$CudaRoot/bin/nvcc.exe" '-DCMAKE_CUDA_ARCHITECTURES=120a-real' '-DHBFLAB_PTX_ARCHITECTURE=compute_75' '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'
if($LASTEXITCODE -ne 0){throw 'Candidate configure failed'}
& $taskCmake --build $BuildRoot --target @Targets --parallel $Jobs
exit $LASTEXITCODE
