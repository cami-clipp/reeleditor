@echo off
echo Iniciando Reel Editor...
start /b python app.py
timeout /t 3 >nul
echo Servidor corriendo en http://localhost:5050
start http://localhost:5050
echo Para detener: cierra esta ventana o ejecuta taskkill /f /im python.exe
pause
