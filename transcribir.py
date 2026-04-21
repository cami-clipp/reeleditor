#!/usr/bin/env python3
"""Corre Whisper en proceso aislado y escribe resultado a stdout JSON."""
import sys, json, whisper

video_path = sys.argv[1]
import os
try: os.nice(15)
except: pass

model = whisper.load_model("small")
result = model.transcribe(video_path, language=None, word_timestamps=True, fp16=False)
print(json.dumps(result.get('segments', [])))
