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


$timeframe = if ($env:TIMEFRAME)  { $env:TIMEFRAME }  else { "10min" }
$tickers   = if ($env:TICKERS)    { $env:TICKERS }    else { "AAPL" }
$tp        = if ($env:TP_PCT)     { $env:TP_PCT }     else { "0.005" }
$sl        = if ($env:SL_PCT)     { $env:SL_PCT }     else { "0.005" }
$extra = ""
if ($env:EOD_FLATTEN -eq "1") { $extra = "--eod_flatten" }

Write-Host "▶ BACKTEST: ejecutando…"
poetry run python engine/run_backtest.py --timeframe "$timeframe" --tickers "$tickers" --tp_pct "$tp" --sl_pct "$sl" $extra
Write-Host "✅ BACKTEST completado"
