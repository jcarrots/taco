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

function Build-Project([string]$Config, [switch]$CleanFirst) {
  Ensure-BuildDir
  Invoke-CMake -Args "-S . -B build"
  $cleanArg = if ($CleanFirst) { " --clean-first" } else { "" }
  Invoke-CMake -Args "--build build --config $Config$cleanArg -j $Jobs"
}

function Get-ExePath([string]$Config) {
  $primary1 = Join-Path -Path "build" -ChildPath "$Config/tcl_driver.exe"
  $primary2 = Join-Path -Path "build" -ChildPath "tcl_driver.exe"
  if (Test-Path $primary1) { return $primary1 }
  if (Test-Path $primary2) { return $primary2 }

  # Fallback (doesn't require yaml-cpp)
  $fallback1 = Join-Path -Path "build" -ChildPath "$Config/tcl4_bench.exe"
  $fallback2 = Join-Path -Path "build" -ChildPath "tcl4_bench.exe"
  if (Test-Path $fallback1) { return $fallback1 }
  if (Test-Path $fallback2) { return $fallback2 }
  return $null
}

switch ($Action) {
  'clean' {
    # Prefer `git clean` so we also remove nested git dirs created by FetchContent (_deps/*-src).
    $hasGit = (Get-Command git -ErrorAction SilentlyContinue) -ne $null
    if ($hasGit) {
      & git clean -fdX -ff
      if ($LASTEXITCODE -ne 0) { throw "git clean failed" }
      Write-Host "Cleaned ignored artifacts via git clean"
      break
    }

    $paths = @(
      "build",
      "build-*",
      ".pytest_cache",
      "out",
      "third_party",
      "python/taco/Release",
      "python/taco/__pycache__",
      "python/tests/__pycache__"
    )

    foreach ($p in $paths) {
      Get-ChildItem -Force $p -ErrorAction SilentlyContinue | ForEach-Object {
        Remove-Item -Recurse -Force $_.FullName -ErrorAction SilentlyContinue
      }
    }
    Write-Host "Cleaned build and cache directories"
    break
  }
  'configure' {
    Ensure-BuildDir
    Invoke-CMake -Args "-S . -B build"
    break
  }
  'build' {
    Build-Project -Config $Config
    break
  }
  'run' {
    Build-Project -Config $Config
    $exe = Get-ExePath $Config
    if (-not $exe) { throw "Executable not found after build" }
    & $exe
    break
  }
  'rebuild' {
    Build-Project -Config $Config -CleanFirst
    $exe = Get-ExePath $Config
    if (-not $exe) { throw "Executable not found after build" }
    & $exe
    break
  }
}

