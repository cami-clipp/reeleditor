#!/bin/bash
# Instala todo lo necesario para Reel Editor
echo "🎬 Instalando Reel Editor..."

# ffmpeg
if ! command -v ffmpeg &> /dev/null; then
  echo "📦 Instalando ffmpeg..."
  sudo apt-get install -y ffmpeg
fi

# Python packages
echo "🐍 Instalando dependencias Python..."
pip install -r requirements.txt --break-system-packages -q

# Fuente Montserrat
echo "🔤 Instalando fuente Montserrat..."
mkdir -p ~/.local/share/fonts/Montserrat
curl -sL "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-Bold.ttf" \
     -o ~/.local/share/fonts/Montserrat/Montserrat-Bold.ttf
curl -sL "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-Regular.ttf" \
     -o ~/.local/share/fonts/Montserrat/Montserrat-Regular.ttf
fc-cache -f ~/.local/share/fonts

# Pre-descargar modelo Whisper small (~460MB, solo la primera vez)
echo "🤖 Descargando modelo Whisper small (460MB, solo una vez)..."
python3 -c "import whisper; whisper.load_model('small')"

mkdir -p uploads outputs

echo ""
echo "✅ ¡Listo! Ejecutá: bash iniciar.sh"
echo "   Abrí el navegador en: http://localhost:5050"
