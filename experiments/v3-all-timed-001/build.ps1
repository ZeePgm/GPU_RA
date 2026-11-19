param(
    [Parameter(Mandatory=$true)][string]$CudaRoot,
    [Parameter(Mandatory=$true)][string]$BuildRoot,
    [Parameter(Mandatory=$true)][string]$Python,
    [int]$Jobs=6
)

$ErrorActionPreference='Stop'
$repoRoot=(Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$vswhere=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vs=& $vswhere -latest -version '[17.0,18.0)' -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (!$vs) { throw 'VS2022 Build Tools missing' }
$env:VSLANG='1033'
& (Join-Path $vs 'Common7/Tools/Launch-VsDevShell.ps1') -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
$cmake=Join-Path $vs 'Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'
$ninja=Join-Path $vs 'Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe'
$env:CUDA_PATH=$CudaRoot
$env:PATH=(Join-Path $CudaRoot 'bin')+';'+(Join-Path $CudaRoot 'bin/x64')+';'+$env:PATH
$source=Join-Path $repoRoot 'joint/source/adapters/llama_cpp/integrated'
$launcher=Join-Path $PSScriptRoot 'compile_all.py'
& $cmake -S $source -B $BuildRoot -G Ninja '-DCMAKE_BUILD_TYPE=Release' "-DCMAKE_MAKE_PROGRAM=$ninja" "-DPython3_EXECUTABLE=$Python" "-DCUDAToolkit_ROOT=$CudaRoot" "-DCMAKE_CUDA_COMPILER=$CudaRoot/bin/nvcc.exe" '-DCMAKE_CUDA_ARCHITECTURES=120a-real' '-DHBFLAB_PTX_ARCHITECTURE=compute_75' '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON' "-DHBFLAB_ALL_MEDIA_LAUNCHER=$launcher"
if ($LASTEXITCODE -ne 0) { throw 'Candidate configure failed' }
& $cmake --build $BuildRoot --target five_model --parallel $Jobs
if ($LASTEXITCODE -ne 0) { throw 'Candidate build failed' }
