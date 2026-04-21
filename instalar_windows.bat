@echo off
echo === Reel Editor - Instalador Windows ===

REM Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python no esta instalado.
    echo Descargalo de https://www.python.org/downloads/
    echo Asegurate de marcar "Add Python to PATH" al instalar
    pause
    exit /b 1
)

REM Verificar ffmpeg
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo ERROR: ffmpeg no esta instalado.
    echo Descargalo de https://www.gyan.dev/ffmpeg/builds/
    echo Descarga "ffmpeg-release-essentials.zip", extraelo y agrega la carpeta bin al PATH
    echo.
    echo Instrucciones PATH: https://www.architectryan.com/2018/03/17/add-to-the-path-on-windows-10/
    pause
    exit /b 1
)

echo Instalando dependencias Python...
pip install flask openai-whisper librosa numpy yt-dlp

echo Descargando modelo Whisper small (460MB, solo una vez)...
python -c "import whisper; whisper.load_model('small')"

if not exist uploads mkdir uploads
if not exist outputs mkdir outputs

echo.
echo Instalacion lista!
echo Ahora ejecuta: iniciar_windows.bat
pause
