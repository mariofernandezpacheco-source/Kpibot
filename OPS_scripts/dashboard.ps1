#Requires -Version 5.0
$ErrorActionPreference = "Stop"

# Load .env into this PowerShell session (NAME=VALUE per line, ignore comments and blanks)
if (Test-Path ".env") {
  Get-Content .env | Where-Object { $_ -notmatch '^\s*#' -and $_ -match '=' } | ForEach-Object {
    $name, $value = $_ -split '=', 2
    $name = $name.Trim()
    $value = $value.Trim()
    if ($name) { Set-Item -Path "Env:$name" -Value $value }
  }
}


$port = if ($env:PORT) { $env:PORT } else { "8501" }
Write-Host "▶ DASHBOARD: http://localhost:$port"
poetry run streamlit run 06_dashboard.py --server.port $port
