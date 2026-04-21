#!/usr/bin/env python3
from flask import Flask, request, jsonify, Response, send_from_directory, stream_with_context, send_file
import os, sys, json, subprocess, threading, uuid, time, io, zipfile
import whisper
import librosa
import numpy as np
import yt_dlp

app = Flask(__name__, static_folder='static')
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

jobs = {}

FONT_PATH = os.path.expanduser('~/.local/share/fonts/Montserrat/Montserrat-Bold.ttf')
FONT_DIR  = os.path.dirname(FONT_PATH)

# ─── helpers ──────────────────────────────────────────────────────────────────

def log(job_id, msg, progress=None):
    jobs[job_id]['messages'].append(msg)
    if progress is not None:
        jobs[job_id]['progress'] = progress
    print(f"[{job_id[:6]}] {msg}")

def get_duration(path):
    r = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', path],
        capture_output=True, text=True
    )
    return float(json.loads(r.stdout)['format']['duration'])

def get_video_size(path):
    r = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams',
         '-select_streams', 'v:0', path],
        capture_output=True, text=True
    )
    info = json.loads(r.stdout)
    s = info['streams'][0]
    return s.get('width', 1920), s.get('height', 1080)

def detect_beats(music_path):
    y, sr = librosa.load(music_path, sr=22050, mono=True)
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    return librosa.frames_to_time(beat_frames, sr=sr).tolist()

def detect_silences(audio_path, min_dur=0.4, threshold=-35):
    """Retorna lista de (start, end) de silencios."""
    r = subprocess.run(
        ['ffmpeg', '-i', audio_path, '-af',
         f'silencedetect=noise={threshold}dB:d={min_dur}',
         '-f', 'null', '-'],
        capture_output=True, text=True
    )
    silences = []
    start = None
    for line in r.stderr.splitlines():
        if 'silence_start' in line:
            try: start = float(line.split('silence_start:')[1].strip().split()[0])
            except: pass
        if 'silence_end' in line and start is not None:
            try:
                end = float(line.split('silence_end:')[1].strip().split('|')[0].strip())
                silences.append((start, end))
                start = None
            except: pass
    return silences

def find_natural_end(silences, clip_start, min_dur=50, max_dur=180):
    """Encuentra el fin natural más cercano al mínimo, sin exceder el máximo."""
    best = clip_start + min_dur
    for s_start, s_end in silences:
        t = s_start - clip_start
        if t >= min_dur:
            return s_start  # corta al inicio del silencio
        if t >= max_dur:
            break
    return clip_start + min(max_dur, min_dur + 30)

def detect_smart_clips(video_path, n_clips, min_dur=50, max_dur=180):
    """Detecta clips de duración variable usando energía + silencios naturales."""
    audio_tmp = video_path + '_audio_det.wav'
    subprocess.run(
        ['ffmpeg', '-y', '-i', video_path, '-ar', '22050', '-ac', '1', audio_tmp],
        capture_output=True
    )
    y, sr = librosa.load(audio_tmp, sr=22050, mono=True)
    hop = int(sr * 3)
    frame_len = int(sr * min_dur)
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0]
    times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop)
    total_dur = get_duration(video_path)
    silences = detect_silences(audio_tmp)
    os.unlink(audio_tmp)

    clips = []  # list of (start, end)
    for idx in np.argsort(rms)[::-1]:
        t = float(times[idx])
        if t + min_dur > total_dur:
            continue
        # no superponer con clips ya elegidos
        overlap = any(not (t + max_dur < cs or t > ce) for cs, ce in clips)
        if overlap:
            continue
        end_t = find_natural_end(silences, t, min_dur, max_dur)
        end_t = min(end_t, total_dur - 0.5)
        clips.append((t, end_t))
        if len(clips) >= n_clips:
            break
    return sorted(clips, key=lambda x: x[0])

# ─── corte de silencios ───────────────────────────────────────────────────────

def cut_silences(raw_in, raw_out, words, clip_start, clip_end, max_gap=0.4):
    """Re-corta raw_in eliminando pausas >max_gap usando concat de segmentos de habla."""
    # construir segmentos de habla a partir de palabras
    rel_words = [w for w in words
                 if w.get('start', 0) >= clip_start and w.get('start', 0) < clip_end]
    if len(rel_words) < 2:
        # sin datos suficientes, copiar sin cambios
        import shutil; shutil.copy2(raw_in, raw_out)
        return get_duration(raw_out)

    # agrupar palabras en segmentos continuos (gap < max_gap → mismo segmento)
    segments = []
    seg_s = rel_words[0]['start'] - clip_start
    seg_e = rel_words[0].get('end', rel_words[0]['start'] + 0.3) - clip_start
    for w in rel_words[1:]:
        ws = w['start'] - clip_start
        we = w.get('end', w['start'] + 0.3) - clip_start
        if ws - seg_e <= max_gap:
            seg_e = we
        else:
            segments.append((max(0, seg_s - 0.05), seg_e + 0.08))
            seg_s = ws
            seg_e = we
    segments.append((max(0, seg_s - 0.05), seg_e + 0.08))

    if len(segments) <= 1:
        import shutil; shutil.copy2(raw_in, raw_out)
        return get_duration(raw_out)

    # construir filter_complex de concat
    clip_dur = clip_end - clip_start
    inputs = []
    filter_parts = []
    for k, (ss, se) in enumerate(segments):
        se = min(se, clip_dur)
        inputs += ['-ss', f'{ss:.3f}', '-to', f'{se:.3f}', '-i', raw_in]
        filter_parts.append(f'[{k}:v][{k}:a]')

    n = len(segments)
    concat_f = ''.join(filter_parts) + f'concat=n={n}:v=1:a=1[vout][aout]'
    cmd = ['ffmpeg', '-y'] + inputs + [
        '-filter_complex', concat_f,
        '-map', '[vout]', '-map', '[aout]',
        '-c:v', 'libx264', '-preset', 'ultrafast', '-c:a', 'aac', raw_out
    ]
    subprocess.run(cmd, capture_output=True)
    return get_duration(raw_out)

# ─── subtítulos ───────────────────────────────────────────────────────────────

def _ass_ts(t):
    t = max(0, t)
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    cs = int((s % 1) * 100)
    return f"{int(h)}:{int(m):02d}:{int(s):02d}.{cs:02d}"

def segments_to_srt(segments, srt_path):
    """Genera ASS word-by-word con estilo grande y contorno (se guarda con extensión .srt pero es ASS)."""
    font_name = "Montserrat" if os.path.exists(FONT_PATH) else "Arial"
    header = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font_name},88,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
        "1,0,0,0,100,100,0,0,1,0,0,2,20,20,120,1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    dialogues = []
    for seg in segments:
        words = seg.get('words', [])
        if not words:
            parts = seg['text'].strip().split()
            if not parts: continue
            dur = (seg['end'] - seg['start']) / len(parts)
            words = [{'word': w, 'start': seg['start']+i*dur, 'end': seg['start']+(i+1)*dur}
                     for i, w in enumerate(parts)]
        for w in words:
            text = w.get('word', '').strip().upper()
            if not text: continue
            s = _ass_ts(w.get('start', seg['start']))
            e = _ass_ts(w.get('end', seg['end']))
            dialogues.append(f"Dialogue: 0,{s},{e},Default,,0,0,0,,{text}")
    with open(srt_path, 'w', encoding='utf-8') as f:
        f.write(header + '\n'.join(dialogues))

def subtitle_filter(srt_path):
    import platform
    srt_path = os.path.abspath(srt_path)
    srt_esc = srt_path.replace('\\', '/').replace(':', r'\:').replace("'", r"\'")
    font_dir_esc = FONT_DIR.replace('\\', '/').replace(':', r'\:').replace("'", r"\'")
    if os.path.exists(FONT_PATH):
        return f"subtitles='{srt_esc}':fontsdir='{font_dir_esc}'"
    return f"subtitles='{srt_esc}'"

def _ffmpeg_env():
    """Env vars para ffmpeg — en Windows crea un fontconfig mínimo si no existe."""
    import platform, shutil
    env = os.environ.copy()
    if platform.system() == 'Windows':
        fc_dir = os.path.join(os.path.dirname(__file__), '.fontconfig')
        fc_file = os.path.join(fc_dir, 'fonts.conf')
        if not os.path.exists(fc_file):
            os.makedirs(fc_dir, exist_ok=True)
            font_path = FONT_DIR.replace('\\', '/')
            with open(fc_file, 'w') as f:
                f.write(f'''<?xml version="1.0"?>
<!DOCTYPE fontconfig SYSTEM "fonts.dtd">
<fontconfig>
  <dir>{font_path}</dir>
  <dir>C:/Windows/Fonts</dir>
</fontconfig>''')
        env['FONTCONFIG_FILE'] = fc_file
    return env

# ─── video filters ────────────────────────────────────────────────────────────

def vertical_filter(w, h):
    """
    Escala el video a 1080x1920 (vertical Instagram).
    Si no llena el frame, usa el video blureado de fondo.
    """
    target_w, target_h = 1080, 1920
    # escala para cubrir alto
    scale_h = f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h}"
    # versión borrosa de fondo
    bg = f"[0:v]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h},boxblur=20:5[bg]"
    # video principal centrado
    fg = f"[0:v]scale={target_w}:-2:force_original_aspect_ratio=decrease[fg_scaled];[fg_scaled]pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black@0[fg]"
    # overlay
    overlay = f"[bg][fg]overlay=(W-w)/2:(H-h)/2[vbase]"
    return bg, fg, overlay

# ─── render ───────────────────────────────────────────────────────────────────

def _vertical_filters(srt_f, cta_filter):
    return (
        f"[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
        f"crop=1080:1920,boxblur=25:8[bg];"
        f"[0:v]scale=1080:-2:force_original_aspect_ratio=decrease[fg_s];"
        f"[fg_s]pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black@0[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2[vbase];"
        f"[vbase]{srt_f}{cta_filter}[vout]"
    )

def render_youtube_clip(raw_clip, music_path, srt_path, clip_idx, job_id, clip_dur, add_cta=True):
    """Renderiza clip YouTube: 1080x1920, blur de fondo, subtítulos, música opcional, CTA."""
    import shutil, platform
    out = os.path.join(OUTPUT_DIR, f'clip_{job_id}_{clip_idx:03d}.mp4')

    srt_f = subtitle_filter(srt_path)
    cta_filter = ""

    vf = _vertical_filters(srt_f, cta_filter)

    if music_path:
        fc = vf + f";[0:a]volume=1.0[va];[1:a]volume=0.4[vm];[va][vm]amix=inputs=2:duration=first[aout]"
        cmd = ['ffmpeg', '-y', '-i', raw_clip, '-i', music_path,
               '-filter_complex', fc,
               '-map', '[vout]', '-map', '[aout]']
    else:
        cmd = ['ffmpeg', '-y', '-i', raw_clip,
               '-filter_complex', vf,
               '-map', '[vout]', '-map', '0:a']

    cmd += ['-t', str(clip_dur), '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k', out]
    env = _ffmpeg_env()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        print("ffmpeg stderr:", proc.stderr[-800:])
        import shutil
        # fallback: solo escalar a vertical sin subtítulos
        cmd2 = ['ffmpeg', '-y', '-i', raw_clip,
                '-vf', 'scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black',
                '-t', str(clip_dur), '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-c:a', 'aac', '-b:a', '128k', out]
        proc2 = subprocess.run(cmd2, capture_output=True, text=True, env=env)
        if proc2.returncode != 0:
            shutil.copy2(raw_clip, out)
    return out

def render_reel_clip(raw_clip, music_path, srt_path, clip_idx, job_id, clip_dur):
    """Renderiza clip reel normal: mantiene resolución original."""
    out = os.path.join(OUTPUT_DIR, f'clip_{job_id}_{clip_idx:03d}.mp4')
    srt_f = subtitle_filter(srt_path)
    if music_path:
        fc = (f"[0:v]{srt_f}[vout];"
              f"[0:a]volume=1.0[va];[1:a]volume=0.4[vm];[va][vm]amix=inputs=2:duration=first[aout]")
        cmd = ['ffmpeg', '-y', '-i', raw_clip, '-i', music_path,
               '-filter_complex', fc, '-map', '[vout]', '-map', '[aout]']
    else:
        fc = f"[0:v]{srt_f}[vout]"
        cmd = ['ffmpeg', '-y', '-i', raw_clip,
               '-filter_complex', fc, '-map', '[vout]', '-map', '0:a']
    cmd += ['-t', str(clip_dur), '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k', out]
    env = _ffmpeg_env()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        print("ffmpeg stderr:", proc.stderr[-800:])
        if not os.path.exists(out):
            import shutil; shutil.copy2(raw_clip, out)
    return out

# ─── YouTube download ─────────────────────────────────────────────────────────

def download_youtube(url, job_id):
    out_tmpl = os.path.join(UPLOAD_DIR, f'yt_{job_id}.%(ext)s')
    # busca node en nvm o en PATH para que yt-dlp pueda extraer YouTube
    import shutil
    node_path = (
        shutil.which('node') or
        '/home/camila/.nvm/versions/node/v24.14.1/bin/node'
    )
    ydl_opts = {
        'format': 'bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[height<=720]/best',
        'outtmpl': out_tmpl,
        'quiet': False,
        'no_warnings': False,
        'merge_output_format': 'mp4',
        'noplaylist': True,
        'retries': 15,
        'fragment_retries': 15,
        'extractor_args': {'youtube': {'player_client': ['android_vr', 'web']}},
        'js_runtimes': {'node': {'path': node_path}},
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        },
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get('title', 'video')
        ext = info.get('ext', 'mp4')
    out_path = os.path.join(UPLOAD_DIR, f'yt_{job_id}.mp4')
    if not os.path.exists(out_path):
        # try with detected ext
        alt = os.path.join(UPLOAD_DIR, f'yt_{job_id}.{ext}')
        if os.path.exists(alt):
            os.rename(alt, out_path)
    return out_path, title

# ─── job runner ───────────────────────────────────────────────────────────────

def run_job(job_id, mode, files, music_path, n_clips_req, extra_prompt, yt_url=None):
    is_youtube = mode == 'youtube'
    try:
        jobs[job_id]['status'] = 'running'
        clips_dir = os.path.join(OUTPUT_DIR, f'clips_{job_id}')
        os.makedirs(clips_dir, exist_ok=True)

        # ── YouTube: descargar ──
        if is_youtube:
            log(job_id, "⬇️ Descargando video de YouTube...", 3)
            video_path, title = download_youtube(yt_url, job_id)
            log(job_id, f"✅ Descargado: {title}", 10)
            files = [video_path]

        # ── Video largo / YouTube: detectar clips inteligentes ──
        if mode in ('long', 'youtube'):
            video_path = files[0]
            total_dur = get_duration(video_path)
            n_clips = n_clips_req if n_clips_req > 0 else max(3, int(total_dur // 90))

            log(job_id, f"🔍 Analizando audio para encontrar {n_clips} momentos clave...", 12)
            clip_ranges = detect_smart_clips(video_path, n_clips, min_dur=45, max_dur=120)
            log(job_id, f"📍 {len(clip_ranges)} clips detectados: " +
                ", ".join(f"{s:.0f}s-{e:.0f}s ({e-s:.0f}s)" for s, e in clip_ranges), 20)

            log(job_id, "🎙️ Transcribiendo con Whisper tiny...", 25)
            _script = os.path.join(os.path.dirname(__file__), 'transcribir.py')
            _r = subprocess.run([sys.executable, _script, video_path], capture_output=True, text=True)
            all_whisper_segs = json.loads(_r.stdout) if _r.stdout.strip() else []

            if music_path:
                log(job_id, "🎵 Detectando beats de la música...", 40)
                detect_beats(music_path)

            log(job_id, "🎬 Renderizando clips...", 45)
            output_clips = []

            # recolectar todas las palabras para corte de silencios
            all_words = []
            for seg in all_whisper_segs:
                all_words.extend(seg.get('words', []))

            for i, (start_t, end_t) in enumerate(clip_ranges):
                dur = end_t - start_t
                raw_tmp = os.path.join(clips_dir, f'raw_tmp_{i:03d}.mp4')
                raw = os.path.join(clips_dir, f'raw_{i:03d}.mp4')

                # extraer segmento crudo
                subprocess.run(
                    ['ffmpeg', '-y', '-ss', str(start_t), '-i', video_path,
                     '-t', str(dur), '-c:v', 'libx264', '-preset', 'ultrafast',
                     '-c:a', 'aac', raw_tmp],
                    capture_output=True
                )

                # cortar silencios usando timestamps de Whisper
                clip_words = [w for w in all_words
                              if w.get('start', 0) >= start_t and w.get('start', 0) < end_t]
                dur = cut_silences(raw_tmp, raw, clip_words, start_t, end_t, max_gap=0.4)
                try: os.unlink(raw_tmp)
                except: pass

                # subtítulos relativos al clip (post corte de silencios, usar timestamps pre-corte)
                segs = [s for s in all_whisper_segs
                        if s['start'] >= start_t and s['start'] < end_t]
                clip_segs = []
                for seg in segs:
                    ns = dict(seg, start=seg['start']-start_t, end=seg['end']-start_t)
                    if 'words' in seg:
                        ns['words'] = [dict(w, start=max(0, w['start']-start_t),
                                               end=max(0, w['end']-start_t))
                                       for w in seg['words']]
                    clip_segs.append(ns)
                srt_i = os.path.join(OUTPUT_DIR, f'srt_{job_id}_{i}.srt')
                segments_to_srt(clip_segs, srt_i)

                # render
                if is_youtube:
                    rendered = render_youtube_clip(raw, music_path, srt_i, i, job_id, dur, add_cta=True)
                else:
                    rendered = render_reel_clip(raw, music_path, srt_i, i, job_id, dur)

                output_clips.append(f'/outputs/clip_{job_id}_{i:03d}.mp4')
                pct = 45 + int((i+1)/len(clip_ranges)*52)
                log(job_id, f"   ✅ Clip {i+1}/{len(clip_ranges)} listo ({dur:.0f}s)", pct)

            jobs[job_id]['outputs'] = output_clips
            jobs[job_id]['output'] = output_clips[0]
            jobs[job_id]['status'] = 'done'
            log(job_id, f"✅ ¡{len(output_clips)} clips listos!", 100)
            return

        # ── Clips ya listos → un solo reel ──
        files_to_join = files
        log(job_id, "🎙️ Transcribiendo clips (Whisper tiny)...", 20)
        _script = os.path.join(os.path.dirname(__file__), 'transcribir.py')
        all_segs = []
        offset = 0.0
        for clip in files_to_join:
            _r = subprocess.run([sys.executable, _script, clip], capture_output=True, text=True)
            res_segs = json.loads(_r.stdout) if _r.stdout.strip() else []
            dur = get_duration(clip)
            for seg in res_segs:
                ns = dict(seg, start=seg['start']+offset, end=seg['end']+offset)
                if 'words' in seg:
                    ns['words'] = [dict(w, start=w['start']+offset, end=w['end']+offset)
                                   for w in seg['words']]
                all_segs.append(ns)
            offset += dur

        srt_path = os.path.join(OUTPUT_DIR, f'reel_{job_id}.srt')
        segments_to_srt(all_segs, srt_path)

        if music_path:
            log(job_id, "🎵 Detectando beats...", 55)
            detect_beats(music_path)

        log(job_id, "🔗 Uniendo clips...", 65)
        list_path = os.path.join(clips_dir, 'list.txt')
        with open(list_path, 'w') as f:
            for c in files_to_join:
                f.write(f"file '{os.path.abspath(c)}'\n")
        joined = os.path.join(clips_dir, 'joined.mp4')
        subprocess.run(
            ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_path, '-c', 'copy', joined],
            capture_output=True
        )

        log(job_id, "🎬 Renderizando reel final...", 75)
        output_path = os.path.join(OUTPUT_DIR, f'reel_{job_id}.mp4')
        total_dur = get_duration(joined)

        srt_f = subtitle_filter(srt_path)
        if music_path:
            fc = (f"[0:v]{srt_f}[vout];"
                  f"[0:a]volume=1.0[va];[1:a]volume=0.4[vm];[va][vm]amix=inputs=2:duration=first[aout]")
            cmd = ['ffmpeg', '-y', '-i', joined, '-i', music_path,
                   '-filter_complex', fc, '-map', '[vout]', '-map', '[aout]']
        else:
            fc = f"[0:v]{srt_f}[vout]"
            cmd = ['ffmpeg', '-y', '-i', joined,
                   '-filter_complex', fc, '-map', '[vout]', '-map', '0:a']
        cmd += ['-t', str(total_dur), '-c:v', 'libx264', '-preset', 'fast',
                '-crf', '23', '-c:a', 'aac', '-b:a', '128k', output_path]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise Exception(f"ffmpeg error: {proc.stderr[-600:]}")

        jobs[job_id]['output'] = f'/outputs/reel_{job_id}.mp4'
        jobs[job_id]['outputs'] = [f'/outputs/reel_{job_id}.mp4']
        jobs[job_id]['status'] = 'done'
        log(job_id, "✅ ¡Reel listo!", 100)

    except Exception as e:
        jobs[job_id]['status'] = 'error'
        log(job_id, f"❌ Error: {str(e)}", None)
        import traceback; traceback.print_exc()

# ─── routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/upload', methods=['POST'])
def upload():
    saved = []
    for key in request.files:
        f = request.files[key]
        if f.filename:
            ext = os.path.splitext(f.filename)[1]
            name = f"{uuid.uuid4().hex}{ext}"
            path = os.path.join(UPLOAD_DIR, name)
            f.save(path)
            saved.append({'original': f.filename, 'path': path})
    return jsonify(saved)

@app.route('/start', methods=['POST'])
def start():
    data = request.json
    mode    = data.get('mode', 'clips')
    files   = data.get('files', [])
    music   = data.get('music')
    n_clips = int(data.get('n_clips', 0))   # 0 = auto
    prompt  = data.get('prompt', '')
    yt_url  = data.get('yt_url', '')

    if mode == 'youtube' and not yt_url:
        return jsonify({'error': 'Falta la URL de YouTube'}), 400
    if mode != 'youtube' and not files:
        return jsonify({'error': 'Faltan archivos de video'}), 400
    # música opcional

    job_id = uuid.uuid4().hex
    jobs[job_id] = {'status': 'pending', 'progress': 0, 'messages': [], 'output': None, 'outputs': []}

    t = threading.Thread(target=run_job,
                         args=(job_id, mode, files, music, n_clips, prompt, yt_url))
    t.daemon = True
    t.start()
    return jsonify({'job_id': job_id})

@app.route('/progress/<job_id>')
def progress(job_id):
    def stream():
        last = 0
        while True:
            job = jobs.get(job_id)
            if not job:
                yield f"data: {json.dumps({'error':'not found'})}\n\n"
                break
            msgs = job['messages'][last:]
            last = len(job['messages'])
            yield f"data: {json.dumps({'status':job['status'],'progress':job['progress'],'new_messages':msgs,'output':job.get('output'),'outputs':job.get('outputs',[])})}\n\n"
            if job['status'] in ('done', 'error'):
                break
            time.sleep(0.5)
    return Response(stream_with_context(stream()), mimetype='text/event-stream',
                    headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

@app.route('/outputs/<filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)

@app.route('/download-all/<job_id>')
def download_all(job_id):
    job = jobs.get(job_id, {})
    outputs = job.get('outputs', [])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for i, url_path in enumerate(outputs):
            full = os.path.join(OUTPUT_DIR, os.path.basename(url_path))
            if os.path.exists(full):
                zf.write(full, f'clip_{i+1:03d}.mp4')
    buf.seek(0)
    return send_file(buf, mimetype='application/zip',
                     as_attachment=True, download_name='clips.zip')

if __name__ == '__main__':
    print("🎬 Reel Editor → http://localhost:5050")
    app.run(host='0.0.0.0', port=5050, debug=False, threaded=True)
