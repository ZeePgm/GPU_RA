param(
 [Parameter(Mandatory=$true)][string]$CudaRoot,
 [string]$Python='python',
 [string]$BuildRoot=(Join-Path $PSScriptRoot '../build-reproduction'),
 [string]$CudaArchitectures='120a-real',
 [string]$PtxArchitecture='compute_75',
 [int]$Jobs=4
)
$ErrorActionPreference='Stop'
$taskRoot=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$taskBuild=[IO.Path]::GetFullPath($BuildRoot)
$taskPython=(Get-Command $Python -ErrorAction Stop).Source
if(Test-Path -LiteralPath $taskBuild){throw 'Use a fresh BuildRoot; previous results are retained'}
& $taskPython (Join-Path $PSScriptRoot 'verify_source.py')
if($LASTEXITCODE -ne 0){throw 'Historical source identity check failed'}
& (Join-Path $taskRoot 'build.ps1') -Target model -CudaRoot $CudaRoot -Python $taskPython -BuildDir (Join-Path $taskBuild 'model') -CudaArchitectures $CudaArchitectures -PtxArchitecture $PtxArchitecture -Jobs $Jobs
if($LASTEXITCODE -ne 0){throw 'Source model build failed'}
& (Join-Path $taskRoot 'experiments/gpu_ffn/build.ps1') -CudaRoot $CudaRoot -BaseBuild (Join-Path $taskBuild 'model') -BuildDir (Join-Path $taskBuild 'ffn') -Jobs $Jobs
if($LASTEXITCODE -ne 0){throw 'Source F build failed'}
& $taskPython -m unittest discover -s (Join-Path $taskRoot 'experiments/gpu_ffn') -p test_run.py -v
if($LASTEXITCODE -ne 0){throw 'F runner checks failed'}
Write-Output "Built from repository sources: $taskBuild"
