#!/usr/bin/env python3
from flask import Flask, request, jsonify, Response, send_from_directory, stream_with_context, send_file
import os, sys, json, subprocess, threading, uuid, time, io, zipfile
import librosa
import numpy as np
import yt_dlp

def transcribe_video(video_path):
    """Transcribe con Whisper small en subprocess. Retorna lista de segmentos."""
    _script = os.path.join(os.path.dirname(__file__), 'transcribir.py')
    _r = subprocess.run([sys.executable, _script, video_path], capture_output=True, text=True, timeout=600)
    if _r.stdout.strip():
        return json.loads(_r.stdout)
    return []

def detect_coherent_clips(segments, n_clips, min_dur=45, max_dur=90):
    """
    Elige clips que empiezan y terminan en frases completas.
    Busca los n_clips bloques de segmentos más 'densos' (más palabras por segundo).
    """
    if not segments:
        return []

    total_dur = segments[-1]['end'] if segments else 0

    # Puntuar cada segmento por densidad de palabras
    scored = []
    for seg in segments:
        dur = seg['end'] - seg['start']
        words = len(seg.get('text', '').split())
        if dur > 0:
            scored.append((words / dur, seg['start'], seg['end']))

    scored.sort(reverse=True)

    clips = []
    for _, anchor_start, anchor_end in scored:
        if len(clips) >= n_clips:
            break

        # Expandir desde este segmento hasta llenar min_dur sin pasar max_dur
        clip_start = anchor_start
        clip_end = anchor_end

        # Extender hacia adelante sumando segmentos completos
        for seg in segments:
            if seg['start'] < clip_end:
                continue
            if seg['end'] - clip_start > max_dur:
                break
            clip_end = seg['end']
            if clip_end - clip_start >= min_dur:
                break

        # Si quedó muy corto, extender hacia atrás
        if clip_end - clip_start < min_dur:
            for seg in reversed(segments):
                if seg['end'] > clip_start:
                    continue
                if clip_end - seg['start'] > max_dur:
                    break
                clip_start = seg['start']
                if clip_end - clip_start >= min_dur:
                    break

        dur = clip_end - clip_start
        if dur < 20:
            continue

        # No superponer con clips ya elegidos
        overlap = any(not (clip_end < cs or clip_start > ce) for cs, ce in clips)
        if overlap:
            continue

        clips.append((clip_start, min(clip_end, total_dur - 0.5)))

    return sorted(clips, key=lambda x: x[0])

app = Flask(__name__, static_folder='static')
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

jobs = {}

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

def detect_beats(music_path):
    y, sr = librosa.load(music_path, sr=22050, mono=True)
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    return librosa.frames_to_time(beat_frames, sr=sr).tolist()

def detect_silences(audio_path, min_dur=0.4, threshold=-35):
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

def find_natural_end(silences, clip_start, min_dur=50, max_dur=90):
    for s_start, s_end in silences:
        t = s_start - clip_start
        if t >= min_dur:
            return s_start
        if t >= max_dur:
            break
    return clip_start + min(max_dur, min_dur + 30)

def detect_smart_clips(video_path, n_clips, min_dur=50, max_dur=90):
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

    clips = []
    for idx in np.argsort(rms)[::-1]:
        t = float(times[idx])
        if t + min_dur > total_dur:
            continue
        overlap = any(not (t + max_dur < cs or t > ce) for cs, ce in clips)
        if overlap:
            continue
        end_t = find_natural_end(silences, t, min_dur, max_dur)
        end_t = min(end_t, total_dur - 0.5)
        clips.append((t, end_t))
        if len(clips) >= n_clips:
            break
    return sorted(clips, key=lambda x: x[0])

# ─── render ───────────────────────────────────────────────────────────────────

def render_youtube_clip(raw_clip, music_path, clip_idx, job_id, clip_dur):
    """Renderiza clip YouTube: 1080x1920, blur de fondo, sin subtítulos."""
    out = os.path.join(OUTPUT_DIR, f'clip_{job_id}_{clip_idx:03d}.mp4')

    vf = (
        "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920[vout]"
    )

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
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("ffmpeg stderr:", proc.stderr[-800:])
        cmd2 = ['ffmpeg', '-y', '-i', raw_clip,
                '-vf', 'scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920',
                '-t', str(clip_dur), '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-c:a', 'aac', '-b:a', '128k', out]
        proc2 = subprocess.run(cmd2, capture_output=True, text=True)
        if proc2.returncode != 0:
            import shutil; shutil.copy2(raw_clip, out)
    return out

def render_reel_clip(raw_clip, music_path, clip_idx, job_id, clip_dur):
    """Renderiza clip reel normal: mantiene resolución original."""
    out = os.path.join(OUTPUT_DIR, f'clip_{job_id}_{clip_idx:03d}.mp4')
    if music_path:
        fc = ("[0:a]volume=1.0[va];[1:a]volume=0.4[vm];[va][vm]amix=inputs=2:duration=first[aout]")
        cmd = ['ffmpeg', '-y', '-i', raw_clip, '-i', music_path,
               '-filter_complex', fc, '-map', '0:v', '-map', '[aout]']
    else:
        cmd = ['ffmpeg', '-y', '-i', raw_clip, '-c', 'copy']
    cmd += ['-t', str(clip_dur), '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k', out]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("ffmpeg stderr:", proc.stderr[-800:])
        import shutil; shutil.copy2(raw_clip, out)
    return out

# ─── YouTube download ─────────────────────────────────────────────────────────

def download_youtube(url, job_id):
    out_tmpl = os.path.join(UPLOAD_DIR, f'yt_{job_id}.%(ext)s')
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
            log(job_id, f"✅ Descargado: {title}", 15)
            files = [video_path]

        # ── Video largo / YouTube: detectar clips inteligentes ──
        if mode in ('long', 'youtube'):
            video_path = files[0]
            total_dur = get_duration(video_path)
            n_clips = n_clips_req if n_clips_req > 0 else max(3, int(total_dur // 90))

            log(job_id, "🎙️ Transcribiendo para detectar frases completas (Whisper small)...", 20)
            segments = transcribe_video(video_path)
            if segments:
                clip_ranges = detect_coherent_clips(segments, n_clips, min_dur=45, max_dur=90)
                log(job_id, f"📍 {len(clip_ranges)} clips con coherencia narrativa: " +
                    ", ".join(f"{s:.0f}s-{e:.0f}s ({e-s:.0f}s)" for s, e in clip_ranges), 35)
            else:
                log(job_id, "⚠️ Sin transcripción, usando análisis de audio...", 28)
                clip_ranges = detect_smart_clips(video_path, n_clips, min_dur=45, max_dur=90)
                log(job_id, f"📍 {len(clip_ranges)} clips detectados: " +
                    ", ".join(f"{s:.0f}s-{e:.0f}s ({e-s:.0f}s)" for s, e in clip_ranges), 35)

            if music_path:
                log(job_id, "🎵 Detectando beats de la música...", 40)
                detect_beats(music_path)

            log(job_id, "🎬 Renderizando clips...", 45)
            output_clips = []

            for i, (start_t, end_t) in enumerate(clip_ranges):
                dur = end_t - start_t
                raw = os.path.join(clips_dir, f'raw_{i:03d}.mp4')

                subprocess.run(
                    ['ffmpeg', '-y', '-ss', str(start_t), '-i', video_path,
                     '-t', str(dur), '-c:v', 'libx264', '-preset', 'ultrafast',
                     '-c:a', 'aac', raw],
                    capture_output=True
                )

                if is_youtube:
                    rendered = render_youtube_clip(raw, music_path, i, job_id, dur)
                else:
                    rendered = render_reel_clip(raw, music_path, i, job_id, dur)

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

        if music_path:
            log(job_id, "🎵 Detectando beats...", 30)
            detect_beats(music_path)

        log(job_id, "🔗 Uniendo clips...", 50)
        list_path = os.path.join(clips_dir, 'list.txt')
        with open(list_path, 'w') as f:
            for c in files_to_join:
                f.write(f"file '{os.path.abspath(c)}'\n")
        joined = os.path.join(clips_dir, 'joined.mp4')
        subprocess.run(
            ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_path, '-c', 'copy', joined],
            capture_output=True
        )

        log(job_id, "🎬 Renderizando reel final...", 70)
        output_path = os.path.join(OUTPUT_DIR, f'reel_{job_id}.mp4')
        total_dur = get_duration(joined)

        if music_path:
            fc = ("[0:a]volume=1.0[va];[1:a]volume=0.4[vm];[va][vm]amix=inputs=2:duration=first[aout]")
            cmd = ['ffmpeg', '-y', '-i', joined, '-i', music_path,
                   '-filter_complex', fc, '-map', '0:v', '-map', '[aout]']
        else:
            cmd = ['ffmpeg', '-y', '-i', joined, '-c', 'copy']
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
    n_clips = int(data.get('n_clips', 0))
    prompt  = data.get('prompt', '')
    yt_url  = data.get('yt_url', '')

    if mode == 'youtube' and not yt_url:
        return jsonify({'error': 'Falta la URL de YouTube'}), 400
    if mode != 'youtube' and not files:
        return jsonify({'error': 'Faltan archivos de video'}), 400

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
