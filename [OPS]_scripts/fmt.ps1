#Requires -Version 5.0
$ErrorActionPreference = "Stop"
Write-Host "▶ FORMAT: Black (auto-formato)"
poetry run black .
