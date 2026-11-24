param(
    [Parameter(Mandatory=$true)][string]$CudaRoot,
    [Parameter(Mandatory=$true)][string]$BuildRoot,
    [Parameter(Mandatory=$true)][string]$Python,
    [int]$Jobs=6
)

$ErrorActionPreference = 'Stop'
$buildScript = Join-Path $PSScriptRoot '../../../experiments/v3-all-timed-001/build.ps1'
& $buildScript @PSBoundParameters
