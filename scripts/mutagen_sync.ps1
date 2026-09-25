param(
    [Parameter(Position = 0)]
    [ValidateSet("start", "status", "flush", "pause", "resume", "monitor", "stop")]
    [string]$Action = "status"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ProjectFile = Join-Path $ProjectRoot "mutagen.yml"
$SessionNames = @("grid-src", "grid-configs", "grid-scripts", "grid-root-code")

if (-not (Test-Path -LiteralPath $ProjectFile -PathType Leaf)) {
    throw "Mutagen project file not found: $ProjectFile"
}

$requiredDirectories = @("src", "configs", "scripts")
foreach ($directory in $requiredDirectories) {
    $path = Join-Path $ProjectRoot $directory
    if (-not (Test-Path -LiteralPath $path -PathType Container)) {
        throw "Required local directory not found: $path"
    }
}

if (-not (Get-Command mutagen -ErrorAction SilentlyContinue)) {
    throw "Mutagen is not available on PATH."
}

function Invoke-Mutagen {
    param([string[]]$Arguments)

    & mutagen @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Mutagen command failed with exit code $LASTEXITCODE."
    }
}

Push-Location $ProjectRoot
try {
    switch ($Action) {
        "start" {
            Invoke-Mutagen @("project", "start", "--paused", "--no-global-configuration")
        }
        "status" {
            Invoke-Mutagen (@("sync", "list") + $SessionNames)
        }
        "flush" {
            Invoke-Mutagen @("project", "flush")
        }
        "pause" {
            Invoke-Mutagen @("project", "pause")
        }
        "resume" {
            Invoke-Mutagen @("project", "resume")
            Invoke-Mutagen @("project", "flush")
        }
        "monitor" {
            Invoke-Mutagen (@("sync", "monitor") + $SessionNames)
        }
        "stop" {
            Invoke-Mutagen @("project", "terminate")
        }
    }
}
finally {
    Pop-Location
}
