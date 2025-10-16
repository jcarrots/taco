param(
  [ValidateSet("Debug","Release")] [string]$Config = "Release",
  [string]$Out = "",
  [switch]$Build,
  [switch]$Rebuild
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Invoke-CMake([string]$Args) {
  & cmake $Args
  if ($LASTEXITCODE -ne 0) { throw "cmake failed: $Args" }
}

function Ensure-Dir([string]$p) {
  if (-not (Test-Path $p)) { New-Item -ItemType Directory -Force -Path $p | Out-Null }
}

# 1) Configure/build (optional)
if ($Build -or $Rebuild) {
  if (-not (Test-Path build)) { New-Item -ItemType Directory -Force build | Out-Null }
  Invoke-CMake -Args "-S . -B build"
  $clean = ($Rebuild.IsPresent) ? " --clean-first" : ""
  Invoke-CMake -Args "--build build --config $Config$clean --target integrator_tests gamma_tests -j 8"
}

# 2) Locate test executables (multi-config and single-config generators)
$intExe = @(
  Join-Path -Path "build\$Config" -ChildPath "integrator_tests.exe"),
  (Join-Path -Path "build" -ChildPath "integrator_tests.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1

$gamExe = @(
  Join-Path -Path "build\$Config" -ChildPath "gamma_tests.exe"),
  (Join-Path -Path "build" -ChildPath "gamma_tests.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $intExe -and -not $gamExe) {
  Write-Error "No test executables found. Build with -Build first or check your configuration."
}

# 3) Determine output folder
if ([string]::IsNullOrWhiteSpace($Out)) {
  $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
  $Out = Join-Path -Path "out\tests" -ChildPath $stamp
}
Ensure-Dir $Out

Write-Host "Writing test outputs to:" (Resolve-Path $Out)

function Run-TestExe([string]$exePath, [string]$name) {
  if (-not $exePath) { return @{ name=$name; ran=$false; exit=999 } }
  $logFile = Join-Path $Out ("{0}_console.log" -f $name)
  Push-Location $Out
  try {
    Write-Host "Running $name ..."
    & $exePath *>&1 | Tee-Object -FilePath $logFile | Out-Null
    $code = $LASTEXITCODE
  } finally {
    Pop-Location
  }
  # Try to gather per-test result text files if the test produced them in CWD or elsewhere
  $patterns = @("integrator_test_results.txt","gamma_test_results.txt")
  foreach ($p in $patterns) {
    $candidates = @(
      (Join-Path $Out $p),
      (Join-Path (Split-Path $exePath -Parent) $p),
      (Join-Path (Get-Location) $p)
    )
    foreach ($c in $candidates) {
      if (Test-Path $c) {
        $dest = Join-Path $Out $p
        Copy-Item -Path $c -Destination $dest -Force
      }
    }
  }
  return @{ name=$name; ran=$true; exit=$code; log=$logFile }
}

$results = @()
if ($intExe) { $results += Run-TestExe -exePath $intExe -name "integrator" }
if ($gamExe) { $results += Run-TestExe -exePath $gamExe -name "gamma" }

Write-Host "\nSummary:"
foreach ($r in $results) {
  if (-not $r.ran) { Write-Host ("{0}: SKIPPED (not found)" -f $r.name); continue }
  $status = ($r.exit -eq 0) ? "PASS" : "FAIL($($r.exit))"
  Write-Host ("{0}: {1}  log={2}" -f $r.name, $status, (Split-Path $r.log -Leaf))
}

Write-Host "\nArtifacts in:" (Resolve-Path $Out)
