#Requires -Version 5.0
$ErrorActionPreference = "Stop"
Write-Host "▶ LINT: Ruff + Black (check)"
poetry run ruff check .
poetry run black --check .
