@echo off
cd /d "%~dp0"

set "PYEXE=python-3.12.0-embed-amd64\python.exe"

if not exist "%PYEXE%" (
    echo ERROR: no encuentro "%PYEXE%".
    echo Este script espera el Python portable dentro de la carpeta del repo.
    pause
    exit /b 1
)

echo Usando Python portable del proyecto:
"%PYEXE%" -c "import sys; print(sys.executable)"
echo.

echo === Verificando pip en el Python portable ===
"%PYEXE%" -m pip --version >nul 2>&1
if errorlevel 1 (
    echo pip no esta disponible todavia, instalandolo con get-pip.py...
    "%PYEXE%" "python-3.12.0-embed-amd64\get-pip.py" --disable-pip-version-check -q
)

echo === Instalando dependencias del proyecto (aisladas: solo dentro de esta carpeta portable) ===
"%PYEXE%" -m pip install --disable-pip-version-check -q --upgrade pip
"%PYEXE%" -m pip install --disable-pip-version-check -q -r requirements.txt
echo.
echo === Ejecutando prueba de conexion LLM real ===
"%PYEXE%" assistant\examples\test_llm_connection.py
echo.
echo === FIN (esta ventana se cerrara sola en 45 segundos) ===
timeout /t 45
