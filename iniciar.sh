#!/bin/bash
cd "$(dirname "$0")"
nohup python3 app.py > /tmp/reel-editor.log 2>&1 &
echo "✅ Reel Editor corriendo en http://localhost:5050"
echo "   Para ver logs: tail -f /tmp/reel-editor.log"
echo "   Para detener:  pkill -f 'python3 app.py'"
