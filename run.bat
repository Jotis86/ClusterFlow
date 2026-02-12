@echo off
REM Script para ejecutar ClusterFlow
REM Se puede ejecutar haciendo doble clic o desde la terminal

REM Ir al directorio del script
cd /d "%~dp0"

echo.
echo ========================================
echo    ClusterFlow - Clustering APP
echo ========================================
echo.
echo Directorio: %CD%
echo.
echo Verificando instalacion de Streamlit...

REM Verificar si Streamlit está instalado
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo.
    echo [ERROR] Streamlit no esta instalado
    echo.
    echo Instalando dependencias...
    pip install -r requirements.txt
    echo.
)

echo.
echo Iniciando aplicacion Streamlit...
echo La aplicacion se abrira en: http://localhost:8501
echo.
echo Presiona Ctrl+C para detener el servidor
echo.

REM Cambiar al directorio app y ejecutar
cd app
streamlit run main.py

pause
