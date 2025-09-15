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


Write-Host "▶ PREP: descargando/preparando datos…"
$timeframe = if ($env:TIMEFRAME) { $env:TIMEFRAME } else { "10min" }
$tickers   = if ($env:TICKERS)   { $env:TICKERS }   else { "AAPL" }
$datadir   = if ($env:DATA_DIR)  { $env:DATA_DIR }  else { "./data" }
poetry run python 01_data_download.py --timeframe "$timeframe" --tickers "$tickers" --data-dir "$datadir"
Write-Host "✅ PREP completado"
