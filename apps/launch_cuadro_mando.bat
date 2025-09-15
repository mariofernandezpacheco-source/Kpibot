@echo off
setlocal
set "HERE=%~dp0"
pushd "%HERE%\.."

set PYTHONUTF8=1
set "PYTHONPATH=."

echo [poetry] Ejecutando con Poetry...
poetry run python -c "import sys; print('[poetry] python =', sys.executable)"
poetry run python -m streamlit run apps\APP_cuadro_mando.py %*

popd
endlocal