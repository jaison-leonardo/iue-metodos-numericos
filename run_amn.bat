@echo off
cd /d "%~dp0"

set "PYEXE=python-3.12.0-embed-amd64\python.exe"

if not exist "%PYEXE%" (
    echo ERROR: no encuentro "%PYEXE%".
    echo Este script espera el Python portable dentro de la carpeta del repo.
    pause
    exit /b 1
)

echo === AMN - Asistente Automatizado de Metodos Numericos ===
echo Usando Python portable del proyecto: %PYEXE%
echo.
"%PYEXE%" amn_cli.py %*
echo.
pause
