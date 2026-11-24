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
& (Join-Path $PSScriptRoot '../reproduction/build_readonly.ps1') @PSBoundParameters
