$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
git config core.hooksPath .githooks
Write-Host '[OK] Git hooks enabled for this checkout.'
