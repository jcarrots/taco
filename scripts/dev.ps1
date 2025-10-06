param(
  [ValidateSet("configure","build","run","rebuild","clean")] [string]$Action = "build",
  [ValidateSet("Debug","Release")] [string]$Config = "Debug",
  [int]$Jobs = 8
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Invoke-CMake([string]$Args) {
  & cmake $Args
  if ($LASTEXITCODE -ne 0) { throw "cmake failed: $Args" }
}

function Ensure-BuildDir {
  if (!(Test-Path build)) { New-Item -ItemType Directory build | Out-Null }
}

function Get-ExePath([string]$Config) {
  $p1 = Join-Path -Path "build" -ChildPath "$Config/tcl2_demo.exe"
  $p2 = Join-Path -Path "build" -ChildPath "tcl2_demo.exe"
  if (Test-Path $p1) { return $p1 }
  if (Test-Path $p2) { return $p2 }
  return $null
}

switch ($Action) {
  'clean' {
    if (Test-Path build) { Remove-Item -Recurse -Force build }
    Write-Host "Cleaned build directory"
    break
  }
  'configure' {
    Ensure-BuildDir
    Invoke-CMake -Args "-S . -B build"
    break
  }
  'build' {
    Ensure-BuildDir
    Invoke-CMake -Args "-S . -B build"
    Invoke-CMake -Args "--build build --config $Config -j $Jobs"
    break
  }
  'run' {
    Ensure-BuildDir
    Invoke-CMake -Args "-S . -B build"
    Invoke-CMake -Args "--build build --config $Config -j $Jobs"
    $exe = Get-ExePath $Config
    if (-not $exe) { throw "Demo executable not found after build" }
    & $exe
    break
  }
  'rebuild' {
    Ensure-BuildDir
    Invoke-CMake -Args "-S . -B build"
    Invoke-CMake -Args "--build build --config $Config -j $Jobs"
    $exe = Get-ExePath $Config
    if (-not $exe) { throw "Demo executable not found after build" }
    & $exe
    break
  }
}

