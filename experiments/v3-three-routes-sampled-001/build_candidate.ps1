param(
 [Parameter(Mandatory=$true)][string]$CudaRoot,
 [Parameter(Mandatory=$true)][string]$BuildRoot,
 [Parameter(Mandatory=$true)][string]$Python,
 [string[]]$Targets=@('sampled_inventory'),
 [int]$Jobs=4,
 [switch]$Stage2,
 [switch]$Stage3
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
if($Stage2){$Targets=@('sampled_inventory','sampled_registration','realgpu_llama','joint_b2_fixture','joint_media_gpu')}
if($Stage3){$Targets=@('sampled_inventory','sampled_registration','sampled_reader_fixture','realgpu_llama','joint_b2_fixture','joint_media_gpu')}
& $taskCmake --build $BuildRoot --target @Targets --parallel $Jobs
if($LASTEXITCODE -ne 0){throw 'Candidate build failed'}
if($Stage2 -or $Stage3){
 $taskCore=Join-Path $BuildRoot 'core'
 & $taskCmake -S (Join-Path $taskRepo 'joint/source') -B $taskCore -G Ninja '-DCMAKE_BUILD_TYPE=Release' "-DCMAKE_MAKE_PROGRAM=$taskNinja" "-DPython3_EXECUTABLE=$Python" '-DHBFLAB_ENABLE_CUDA=OFF' '-DBUILD_TESTING=ON' '-DHBFLAB_BACKEND_TESTS=ON' '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'
 if($LASTEXITCODE -ne 0){throw 'Core configure failed'}
 & $taskCmake --build $taskCore --parallel $Jobs
 if($LASTEXITCODE -ne 0){throw 'Core build failed'}
 $taskCtest=Join-Path (Split-Path $taskCmake) 'ctest.exe'
 & $taskCtest --test-dir $taskCore --output-on-failure --parallel 2 --timeout 120 *> (Join-Path $BuildRoot 'core-tests.log')
 if($LASTEXITCODE -ne 0){throw 'Core regression failed'}
}
& $Python (Join-Path $PSScriptRoot 'stage1.py') --record-build $BuildRoot
if($LASTEXITCODE -ne 0){throw 'Candidate build receipt failed'}
