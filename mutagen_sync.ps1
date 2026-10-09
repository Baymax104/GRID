param(
    [Parameter(Position = 0)]
    [ValidateSet("start", "status", "flush", "pause", "resume", "monitor", "stop")]
    [string]$Action = "status"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = $PSScriptRoot
$ProjectFile = Join-Path $ProjectRoot "mutagen.yml"
$SessionNames = @("grid-src", "grid-configs", "grid-root-code")

if (-not (Test-Path -LiteralPath $ProjectFile -PathType Leaf)) {
    throw "Mutagen project file not found: $ProjectFile"
}

$requiredDirectories = @("src", "configs")
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
    if ($Action -in @("start", "resume", "flush")) {
        # 本地工作树是 Git 来源的唯一权威，记录通过 src session 单向同步。
        & uv run --no-sync python -c "from src.utils.source_snapshot import write_source_origin; write_source_origin('.')"
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to capture local source origin; synchronization was not started."
        }
    }
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
