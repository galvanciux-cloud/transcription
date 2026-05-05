import json
import os
import re
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from flask import Flask, Response, request, send_file
from faster_whisper import WhisperModel

# ─── Configuración ───────────────────────────────────────────────
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 5000))
DEFAULT_MODEL = os.environ.get("WHISPER_MODEL", "small")
DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE", "int8")
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".m4v", ".mpeg", ".mpg", ".3gp"}
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a", ".wma", ".opus"}
ALL_EXTS = VIDEO_EXTS | AUDIO_EXTS

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

HAS_FFMPEG = check_ffmpeg()

# ─── Caché de modelos ────────────────────────────────────────────
loaded_models = {}
def get_model(size):
    if size not in loaded_models:
        print(f"Cargando modelo Whisper '{size}' en {DEVICE} ({COMPUTE_TYPE})...")
        loaded_models[size] = WhisperModel(size, device=DEVICE, compute_type=COMPUTE_TYPE)
        print(f"Modelo '{size}' listo.")
    return loaded_models[size]

print(f"Cargando modelo por defecto ({DEFAULT_MODEL})...")
get_model(DEFAULT_MODEL)

# ─── Tareas en memoria ───────────────────────────────────────────
tasks = {}
def cleanup_tasks():
    while True:
        time.sleep(3600)
        now = time.time()
        to_remove = [tid for tid, t in tasks.items() if now - t.get("created_at", 0) > 7200]
        for tid in to_remove:
            t = tasks.pop(tid, {})
            txt = t.get("txt_path")
            if txt and os.path.exists(txt):
                try: os.unlink(txt)
                except OSError: pass

threading.Thread(target=cleanup_tasks, daemon=True).start()

# ─── Clean Verbatim ──────────────────────────────────────────────
FILLERS = {
    "es": [r"\beh\b", r"\behm\b", r"\bem\b", r"\bum\b", r"\bah\b", r"\bo\s*sea\b", r"\bo\s*sea\s*que\b", r"\bpues\b", r"\bverás\b", r"\beste\b", r"\bést[ea]\b", r"\ba\s*ver\b", r"\bcomo\s*que\b", r"\bdigo\b", r"\bva\s*a\s*ser\b", r"\bo\s*sea\s*y\s*que\b", r"\bno\s*\?\s*", r"\b¿\s*no\s*\?\s*", r"\bcorrecto\b", r"\b¿\s*eh\s*\?\s*"],
    "en": [r"\bum+\b", r"\bah+\b", r"\ber\b", r"\buh\b", r"\blike\b", r"\byou\s*know\b", r"\bi\s*mean\b", r"\bsort\s*of\b", r"\bkind\s*of\b", r"\bbasically\b", r"\bliterally\b", r"\bright\b(?!\s*[.!?,;:])", r"\bso\s+(?=like|um|uh)\b", r"\bwell\b(?=\s*[,.]|\s*$)"],
    "fr": [r"\beuh\b", r"\bah\b", r"\bhein\b", r"\bdu\s*coup\b", r"\ben\s*fait\b", r"\bquoi\b(?=\s*[,.!?]|\s*$)", r"\bvoil[àa]\b(?=\s*[,.]|\s*$)", r"\bbref\b(?=\s*[,.]|\s*$)"],
    "pt": [r"\béh\b", r"\bum\b", r"\bah\b", r"\btipo\b", r"\bent[ãa]o\b", r"\bpoxa\b(?=\s*[,.]|\s*$)", r"\bn[ée]\b(?=\s*[,.]|\s*$)"],
    "de": [r"\bäh\b", r"\bum\b", r"\böh\b", r"\balso\b(?=\s*[,.]|\s*$)", r"\bsagen\s*wir\s*mal\b"],
    "it": [r"\beh\b", r"\bumh\b", r"\bah\b", r"\bcio[èe]\b", r"\bpraticamente\b", r"\bdiciamo\b(?=\s*[,.]|\s*$)"],
}

def clean_verbatim(text, lang="es"):
    if not text or not text.strip():
        return ""
    t = text.strip()
    t = re.sub(r'\*[^*]+\*', '', t)
    non_verbal = r'\((?:tose|tos|risa|risas|suspira|bostez|carraspea|respira|aplauso|aplausos|llora|llanto|toser|carcajada|silbido|toser|estornuda)[^)]*\)'
    t = re.sub(non_verbal, '', t, flags=re.IGNORECASE)
    lang_key = lang[:2] if lang else "es"
    fillers = FILLERS.get(lang_key, FILLERS["es"])
    generic_fillers = [r"\beh\b", r"\behm\b", r"\bum+\b", r"\bah+\b"]
    all_fillers = list(set(fillers + generic_fillers))
    for filler in all_fillers:
        t = re.sub(filler, '', t, flags=re.IGNORECASE)
    t = re.sub(r'\b(\S+?)(?:\s+\1)+\b', r'\1', t, flags=re.IGNORECASE)
    t = re.sub(r'(\w)(?:\s*[-–—]\s*\1)+(?:\s*[-–—]\s*)(\w+)', r'\2', t)
    t = re.sub(r'(\w)(?:\s*[-–—]\s*\1)+\b', r'\1', t)
    t = re.sub(r'^\S{1,3}?\s*[-–—]+\s*', '', t)
    t = re.sub(r'^(?:\S+\s+){0,2}\S*?[-–—]+\s*', '', t)
    t = re.sub(r'^[\s.]+', '', t)
    t = re.sub(r'^(?:\S+\s*\.{2,}\s*)+', '', t)
    t = re.sub(r'(?:^|\s)[¡!]?[aouh]{1,3}[¡!]?(?:\s|[.,;!?]|$)', ' ', t, flags=re.IGNORECASE)
    t = re.sub(r'[()[\]{}]', '', t)
    t = re.sub(r'\s{2,}', ' ', t)
    t = re.sub(r'\s+([.,;:!?])', r'\1', t)
    t = re.sub(r'([.,;:!?])(?![\s]|$)', r'\1 ', t)
    t = re.sub(r'\.{2,}', '.', t)
    t = re.sub(r'[,]{2,}', ',', t)
    t = re.sub(r'^\s*[,\s;:]+', '', t)
    t = re.sub(r'[,\s;:]+$', '', t)
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    if t and t[-1] not in ".!?":
        t += "."
    return t.strip()

def clean_and_merge_segments(segments, lang="es"):
    cleaned = []
    for seg in segments:
        text = clean_verbatim(seg["text"], lang)
        if text:
            cleaned.append({"start": seg["start"], "end": seg["end"], "text": text})
        elif cleaned:
            cleaned[-1]["end"] = seg["end"]
    if len(cleaned) > 1:
        merged = []
        i = 0
        while i < len(cleaned):
            seg = cleaned[i]
            word_count = len(seg["text"].split())
            if word_count <= 2 and i + 1 < len(cleaned):
                next_seg = cleaned[i + 1]
                merged.append({"start": seg["start"], "end": next_seg["end"], "text": seg["text"] + " " + next_seg["text"]})
                i += 2
            else:
                merged.append(seg)
                i += 1
        cleaned = merged
    return cleaned

# ─── Helpers ─────────────────────────────────────────────────────
def fmt_time(s):
    m, sec = divmod(int(s), 60)
    return f"{m:02d}:{sec:02d}"

def save_txt(segments, task_id, filename, full_text):
    """Guarda la transcripción limpia en un archivo TXT con formato profesional."""
    txt_path = str(DATA_DIR / f"{task_id}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"TRANSCRIPCIÓN: {filename}\n")
        f.write("="*60 + "\n\n")
        for seg in segments:
            ts = f"[{fmt_time(seg['start'])} - {fmt_time(seg['end'])}]"
            f.write(f"{ts}  {seg['text'].strip()}\n\n")
        f.write("\n" + "="*60 + "\n")
        f.write("TEXTO COMPLETO (CLEAN VERBATIM)\n")
        f.write("="*60 + "\n\n")
        f.write(full_text + "\n")
    return txt_path

# ─── Procesamiento (hilo separado) ───────────────────────────────
def process_file(task_id, file_path, filename, language, model_size):
    audio_path = file_path
    try:
        tasks[task_id]["status"] = "extracting"
        tasks[task_id]["message"] = "Extrayendo audio..."
        ext = os.path.splitext(file_path)[1].lower()
        if ext in VIDEO_EXTS:
            if not HAS_FFMPEG:
                raise Exception("FFmpeg no está instalado. Necesario para procesar video.")
            audio_path = file_path + ".wav"
            r = subprocess.run(["ffmpeg", "-i", file_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", audio_path], capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                raise Exception(f"FFmpeg error: {r.stderr[:300]}")
                
        tasks[task_id]["status"] = "loading_model"
        tasks[task_id]["message"] = f"Cargando modelo {model_size}..."
        whisper = get_model(model_size)
        
        tasks[task_id]["status"] = "transcribing"
        tasks[task_id]["message"] = "Transcribiendo audio con Whisper..."
        tasks[task_id]["progress"] = 0
        kwargs = {"beam_size": 5, "vad_filter": True}
        if language and language != "auto":
            kwargs["language"] = language
            
        # progress_callback eliminado para evitar bloqueos
        seg_iter, info = whisper.transcribe(audio_path, **kwargs)
        raw_segments = []
        for seg in seg_iter:
            raw_segments.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": seg.text})
            
        tasks[task_id]["progress"] = 85
        tasks[task_id]["status"] = "cleaning"
        tasks[task_id]["message"] = "Aplicando formato Clean Verbatim..."
        detected_lang = info.language if info.language else "es"
        segments = clean_and_merge_segments(raw_segments, detected_lang)
        full_text = " ".join(s["text"].strip() for s in segments)
        
        tasks[task_id]["progress"] = 92
        tasks[task_id]["status"] = "saving"
        tasks[task_id]["message"] = "Guardando archivo TXT..."
        txt_path = save_txt(segments, task_id, filename, full_text)
        
        tasks[task_id].update({
            "status": "completed", "progress": 100, "message": "Completado",
            "result": {"full_text": full_text, "segments": segments, "language": detected_lang, "duration": round(info.duration, 2), "filename": filename, "format": "clean_verbatim"},
            "txt_path": txt_path,
        })
    except Exception as e:
        tasks[task_id].update({"status": "error", "error": str(e), "message": str(e)})
    finally:
        for p in set([file_path, file_path + ".wav", audio_path]):
            if p and os.path.exists(p):
                try: os.unlink(p)
                except OSError: pass

# ─── Flask ───────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

@app.route("/")
def index():
    return HTML_TEMPLATE

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return json.dumps({"error": "No se envió ningún archivo"}), 400, {"Content-Type": "application/json"}
    f = request.files["file"]
    if not f.filename:
        return json.dumps({"error": "Nombre de archivo vacío"}), 400, {"Content-Type": "application/json"}
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALL_EXTS:
        return json.dumps({"error": f"Formato no soportado: {ext}"}), 400, {"Content-Type": "application/json"}
        
    tid = str(uuid.uuid4())
    lang = request.form.get("language", "auto")
    model_size = request.form.get("model_size", DEFAULT_MODEL)
    suffix = ext if ext else ".bin"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.save(tmp.name)
    tmp.close()
    
    tasks[tid] = {"status": "starting", "progress": 0, "message": "Iniciando...", "result": None, "txt_path": None, "error": None, "created_at": time.time()}
    threading.Thread(target=process_file, args=(tid, tmp.name, f.filename, lang, model_size), daemon=True).start()
    return json.dumps({"task_id": tid}), 200, {"Content-Type": "application/json"}

@app.route("/progress/<tid>")
def progress(tid):
    def stream():
        while True:
            t = tasks.get(tid)
            if not t:
                yield f"data: {json.dumps({'error': 'Tarea no encontrada'})}\n\n"
                break
            d = {"status": t["status"], "progress": t["progress"], "message": t.get("message", "")}
            if t["status"] == "completed":
                d["result"] = t["result"]
                yield f"data: {json.dumps(d)}\n\n"
                break
            elif t["status"] == "error":
                d["error"] = t.get("error", "Error desconocido")
                yield f"data: {json.dumps(d)}\n\n"
                break
            else:
                yield f"data: {json.dumps(d)}\n\n"
            time.sleep(0.4)
    return Response(stream(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.route("/download/<tid>")
def download(tid):
    t = tasks.get(tid)
    if not t or not t.get("txt_path") or not os.path.exists(t["txt_path"]):
        return json.dumps({"error": "Archivo no encontrado"}), 404, {"Content-Type": "application/json"}
    name = os.path.splitext(t["result"]["filename"])[0] + "_transcripcion.txt"
    return send_file(t["txt_path"], as_attachment=True, download_name=name, mimetype='text/plain')

# ─── HTML ────────────────────────────────────────────────────────
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TranscribeAI — Clean Verbatim</title>
<script src="https://cdn.tailwindcss.com"></script>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,700;1,9..40,400&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
<style>
:root { --bg:#08090d;--bg-card:rgba(14,16,24,0.85);--accent:#10b981;--accent-light:#34d399;--accent-glow:rgba(16,185,129,0.12);--text:#e5e7eb;--text-muted:#6b7280;--border:rgba(255,255,255,0.07);--error:#ef4444; }
*,*::before,*::after{box-sizing:border-box}
body{font-family:'DM Sans',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden}
h1,h2,h3,h4{font-family:'Space Grotesk',sans-serif}
.bg-glow{position:fixed;inset:0;pointer-events:none;z-index:0;overflow:hidden}
.bg-glow::before{content:'';position:absolute;top:-20%;right:-10%;width:700px;height:700px;border-radius:50%;background:radial-gradient(circle,rgba(16,185,129,0.08) 0%,transparent 70%);filter:blur(40px);animation:f1 20s ease-in-out infinite alternate}
.bg-glow::after{content:'';position:absolute;bottom:-15%;left:-5%;width:500px;height:500px;border-radius:50%;background:radial-gradient(circle,rgba(16,185,129,0.05) 0%,transparent 70%);filter:blur(60px);animation:f2 25s ease-in-out infinite alternate}
@keyframes f1{from{transform:translate(0,0)}to{transform:translate(-60px,40px)}}
@keyframes f2{from{transform:translate(0,0)}to{transform:translate(40px,-30px)}}
.noise{position:fixed;inset:0;pointer-events:none;z-index:1;opacity:0.03;background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E")}
.drop-zone{border:2px dashed var(--border);border-radius:16px;transition:all .3s ease;cursor:pointer;position:relative;overflow:hidden}
.drop-zone::before{content:'';position:absolute;inset:0;background:var(--accent-glow);opacity:0;transition:opacity .3s}
.drop-zone:hover,.drop-zone.drag-over{border-color:var(--accent);box-shadow:0 0 30px rgba(16,185,129,0.1),inset 0 0 30px rgba(16,185,129,0.03)}
.drop-zone:hover::before,.drop-zone.drag-over::before{opacity:1}
.wave-icon{display:flex;align-items:center;justify-content:center;gap:4px;height:48px}
.wave-icon span{display:block;width:3px;border-radius:2px;background:var(--accent);opacity:.5;animation:wave 1.2s ease-in-out infinite alternate}
.drop-zone:hover .wave-icon span{opacity:.9}
@keyframes wave{0%{height:8px}100%{height:36px}}
.equalizer{display:flex;align-items:flex-end;justify-content:center;gap:3px;height:32px}
.equalizer span{display:block;width:5px;border-radius:3px;background:var(--accent);animation:eq .5s ease-in-out infinite alternate}
@keyframes eq{0%{height:4px}100%{height:30px}}
.progress-track{width:100%;height:8px;border-radius:4px;background:rgba(255,255,255,0.06);overflow:hidden}
.progress-fill{height:100%;border-radius:4px;background:linear-gradient(90deg,var(--accent),var(--accent-light));transition:width .4s ease;box-shadow:0 0 12px rgba(16,185,129,0.4)}
.card{background:var(--bg-card);border:1px solid var(--border);border-radius:16px;backdrop-filter:blur(12px)}
.seg-row{padding:10px 14px;border-radius:8px;transition:background .15s}
.seg-row:hover{background:rgba(255,255,255,0.03)}
.seg-ts{font-family:'Space Grotesk',monospace;font-size:12px;font-weight:600;color:var(--accent);white-space:nowrap}
.seg-text{font-size:15px;line-height:1.7;color:#d1d5db}
.btn-primary{display:inline-flex;align-items:center;gap:8px;padding:12px 28px;border-radius:10px;border:none;background:linear-gradient(135deg,var(--accent),#059669);color:#fff;font-weight:600;font-size:15px;cursor:pointer;transition:all .2s;font-family:'Space Grotesk',sans-serif;box-shadow:0 4px 20px rgba(16,185,129,0.25)}
.btn-primary:hover{transform:translateY(-1px);box-shadow:0 6px 28px rgba(16,185,129,0.35)}
.btn-primary:active{transform:translateY(0)}
.btn-ghost{display:inline-flex;align-items:center;gap:8px;padding:12px 24px;border-radius:10px;border:1px solid var(--border);background:transparent;color:var(--text);font-weight:500;font-size:15px;cursor:pointer;transition:all .2s;font-family:'Space Grotesk',sans-serif}
.btn-ghost:hover{border-color:rgba(255,255,255,0.15);background:rgba(255,255,255,0.04)}
.sel{appearance:none;padding:10px 36px 10px 14px;border-radius:10px;border:1px solid var(--border);background:rgba(255,255,255,0.04);color:var(--text);font-size:14px;font-family:'DM Sans',sans-serif;cursor:pointer;transition:border-color .2s;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' fill='%236b7280' viewBox='0 0 16 16'%3E%3Cpath d='M8 11L3 6h10z'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 12px center}
.sel:hover{border-color:rgba(255,255,255,0.15)}
.sel:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 2px rgba(16,185,129,0.15)}
.toast{position:fixed;bottom:24px;right:24px;z-index:999;padding:14px 22px;border-radius:12px;background:rgba(14,16,24,0.95);border:1px solid var(--border);backdrop-filter:blur(12px);font-size:14px;color:var(--text);box-shadow:0 8px 32px rgba(0,0,0,0.4);animation:ti .3s ease,to .3s ease 3.5s forwards}
.toast.error{border-color:rgba(239,68,68,0.3);color:#fca5a5}
.toast.success{border-color:rgba(16,185,129,0.3);color:var(--accent-light)}
@keyframes ti{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
@keyframes to{to{opacity:0;transform:translateY(16px)}}
.check-svg{width:56px;height:56px}
.check-circle{stroke-dasharray:166;stroke-dashoffset:166;animation:cs .6s cubic-bezier(.65,0,.45,1) .2s forwards}
.check-mark{stroke-dasharray:48;stroke-dashoffset:48;animation:cs .4s cubic-bezier(.65,0,.45,1) .6s forwards}
@keyframes cs{to{stroke-dashoffset:0}}
.custom-scroll::-webkit-scrollbar{width:6px}
.custom-scroll::-webkit-scrollbar-track{background:transparent}
.custom-scroll::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.1);border-radius:3px}
.custom-scroll::-webkit-scrollbar-thumb:hover{background:rgba(255,255,255,0.18)}
.hidden{display:none!important}
.fade-in{animation:fi .5s ease}
@keyframes fi{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
.format-badge{display:inline-flex;align-items:center;gap:6px;padding:6px 14px;border-radius:20px;font-size:12px;font-weight:600;letter-spacing:.03em;background:rgba(16,185,129,0.1);color:var(--accent-light);border:1px solid rgba(16,185,129,0.2)}
.clean-item{display:flex;align-items:flex-start;gap:8px;font-size:12px;color:var(--text-muted);line-height:1.5}
.clean-item i{color:var(--accent);margin-top:3px;font-size:10px}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{animation-duration:.01ms!important;transition-duration:.01ms!important}}
</style>
</head>
<body>
<div class="bg-glow"></div>
<div class="noise"></div>
<main class="relative z-10 min-h-screen flex flex-col items-center px-4 py-8 sm:py-12">
<header class="text-center mb-10 fade-in">
<div class="flex items-center justify-center gap-3 mb-3">
<div class="wave-icon" style="height:32px"><span style="animation-delay:0s;height:12px"></span><span style="animation-delay:.15s;height:20px"></span><span style="animation-delay:.3s;height:28px"></span><span style="animation-delay:.1s;height:16px"></span><span style="animation-delay:.25s;height:24px"></span><span style="animation-delay:.05s;height:14px"></span><span style="animation-delay:.2s;height:22px"></span></div>
<h1 class="text-3xl sm:text-4xl font-bold tracking-tight">Transcribe<span style="color:var(--accent)">AI</span></h1>
</div>
<p class="text-sm sm:text-base mb-3" style="color:var(--text-muted);max-width:480px">Transcripción Clean Verbatim con marcas de tiempo. Modelo gratuito, privado, sin límites.</p>
<div class="format-badge"><i class="fa-solid fa-wand-magic-sparkles"></i> CLEAN VERBATIM</div>
</header>
<div class="w-full max-w-xl mb-8 fade-in">
<div class="card p-5">
<div class="flex items-center gap-2 mb-3"><i class="fa-solid fa-broom text-sm" style="color:var(--accent)"></i><h3 class="text-sm font-bold" style="font-family:'Space Grotesk',sans-serif">Formato de salida: Clean Verbatim</h3></div>
<div class="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-2">
<div class="clean-item"><i class="fa-solid fa-xmark"></i><span>Tartamudeos y repeticiones</span></div>
<div class="clean-item"><i class="fa-solid fa-xmark"></i><span>Muletillas (eh, o sea, pues...)</span></div>
<div class="clean-item"><i class="fa-solid fa-xmark"></i><span>Palabras de relleno</span></div>
<div class="clean-item"><i class="fa-solid fa-xmark"></i><span>Falsos comienzos</span></div>
<div class="clean-item"><i class="fa-solid fa-xmark"></i><span>Sonidos no verbales (toses, risas)</span></div>
<div class="clean-item"><i class="fa-solid fa-xmark"></i><span>Interjecciones sin significado</span></div>
<div class="clean-item"><i class="fa-solid fa-check"></i><span style="color:var(--accent-light)">Texto fluido y profesional</span></div>
<div class="clean-item"><i class="fa-solid fa-check"></i><span style="color:var(--accent-light)">Marcas de tiempo [mm:ss – mm:ss]</span></div>
</div>
</div>
</div>
<section id="sec-upload" class="w-full max-w-xl fade-in">
<div id="drop-zone" class="drop-zone p-10 sm:p-14 text-center">
<input type="file" id="file-input" accept="audio/*,video/*" class="hidden">
<div class="relative z-10">
<div class="wave-icon mx-auto mb-5"><span style="animation-delay:0s"></span><span style="animation-delay:.15s"></span><span style="animation-delay:.3s"></span><span style="animation-delay:.1s"></span><span style="animation-delay:.25s"></span><span style="animation-delay:.05s"></span><span style="animation-delay:.2s"></span></div>
<p class="text-lg font-semibold mb-2" style="font-family:'Space Grotesk',sans-serif">Arrastra tu archivo aqui</p>
<p class="text-sm mb-5" style="color:var(--text-muted)">o haz clic para seleccionar</p>
<span class="inline-block px-4 py-2 rounded-lg text-xs font-medium" style="background:rgba(255,255,255,0.05);color:var(--text-muted)">MP3 · WAV · FLAC · MP4 · MKV · MOV · OGG · hasta 500 MB</span>
</div>
</div>
<div class="flex flex-col sm:flex-row gap-3 mt-5">
<div class="flex-1"><label class="block text-xs font-medium mb-1.5" style="color:var(--text-muted)">Idioma</label><select id="sel-lang" class="sel w-full"><option value="auto">Detectar automáticamente</option><option value="es">Español</option><option value="en">Inglés</option><option value="fr">Francés</option><option value="de">Alemán</option><option value="it">Italiano</option><option value="pt">Portugués</option><option value="ja">Japonés</option><option value="ko">Coreano</option><option value="zh">Chino</option><option value="ru">Ruso</option><option value="ar">Árabe</option><option value="hi">Hindi</option></select></div>
<div class="flex-1"><label class="block text-xs font-medium mb-1.5" style="color:var(--text-muted)">Modelo</label><select id="sel-model" class="sel w-full"><option value="tiny">Tiny — Ultra rápido</option><option value="base">Base — Rápido</option><option value="small" selected>Small — Recomendado</option><option value="medium">Medium — Preciso</option><option value="large-v3">Large v3 — Máxima precisión</option></select></div>
</div>
</section>
<section id="sec-processing" class="w-full max-w-xl hidden">
<div class="card p-8 fade-in">
<div class="flex items-center gap-3 mb-2"><i class="fa-solid fa-file-audio text-lg" style="color:var(--accent)"></i><p id="proc-filename" class="text-sm font-semibold truncate"></p></div>
<div class="mt-6 mb-3"><div class="flex justify-between items-center mb-2"><span id="proc-status" class="text-xs" style="color:var(--text-muted)">Preparando...</span><span id="proc-pct" class="text-xs font-bold" style="color:var(--accent)">0%</span></div><div class="progress-track"><div id="proc-bar" class="progress-fill" style="width:0%"></div></div></div>
<div class="flex items-center justify-center gap-6 mt-8"><div class="equalizer"><span style="animation-delay:0s"></span><span style="animation-delay:.08s"></span><span style="animation-delay:.16s"></span><span style="animation-delay:.04s"></span><span style="animation-delay:.12s"></span><span style="animation-delay:.2s"></span><span style="animation-delay:.06s"></span></div><p id="proc-msg" class="text-sm" style="color:var(--text-muted)">Iniciando...</p></div>
<div class="mt-8 pt-5" style="border-top:1px solid var(--border)"><div id="step-1" class="flex items-center gap-3 text-xs mb-2 opacity-40"><i class="fa-solid fa-circle text-[6px]"></i><span>Extraer audio</span></div><div id="step-2" class="flex items-center gap-3 text-xs mb-2 opacity-40"><i class="fa-solid fa-circle text-[6px]"></i><span>Transcribir con Whisper</span></div><div id="step-3" class="flex items-center gap-3 text-xs mb-2 opacity-40"><i class="fa-solid fa-circle text-[6px]"></i><span>Limpiar (Clean Verbatim)</span></div><div id="step-4" class="flex items-center gap-3 text-xs opacity-40"><i class="fa-solid fa-circle text-[6px]"></i><span>Guardar TXT</span></div></div>
</div>
</section>
<section id="sec-result" class="w-full max-w-3xl hidden">
<div class="card p-6 sm:p-8 fade-in">
<div class="flex flex-col sm:flex-row sm:items-center gap-4 mb-6">
<div class="flex items-center gap-3"><svg class="check-svg" viewBox="0 0 52 52"><circle class="check-circle" cx="26" cy="26" r="25" fill="none" stroke="var(--accent)" stroke-width="2"/><path class="check-mark" fill="none" stroke="var(--accent)" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" d="M14.1 27.2l7.1 7.2 16.7-16.8"/></svg><div><h2 class="text-xl font-bold">Transcripción lista</h2><p id="res-filename" class="text-xs" style="color:var(--text-muted)"></p></div></div>
<div class="sm:ml-auto flex flex-wrap gap-3 text-xs" style="color:var(--text-muted)"><span class="format-badge" style="font-size:11px;padding:4px 10px"><i class="fa-solid fa-wand-magic-sparkles" style="font-size:9px"></i> CLEAN VERBATIM</span><span class="flex items-center gap-1.5"><i class="fa-regular fa-clock"></i><span id="res-duration"></span></span><span class="flex items-center gap-1.5"><i class="fa-solid fa-globe"></i><span id="res-lang"></span></span><span class="flex items-center gap-1.5"><i class="fa-solid fa-bars-staggered"></i><span id="res-segs"></span> segmentos</span></div>
</div>
<div class="flex gap-1 mb-4 p-1 rounded-lg" style="background:rgba(255,255,255,0.04)">
<button id="tab-timestamps" onclick="switchTab('timestamps')" class="flex-1 py-2 px-3 rounded-md text-sm font-medium transition-all" style="background:var(--accent);color:#fff">Con timestamps</button>
<button id="tab-text" onclick="switchTab('text')" class="flex-1 py-2 px-3 rounded-md text-sm font-medium transition-all" style="color:var(--text-muted)">Texto completo</button>
</div>
<div id="view-timestamps" class="custom-scroll" style="max-height:460px;overflow-y:auto"><div id="res-segments"></div></div>
<div id="view-text" class="hidden custom-scroll" style="max-height:460px;overflow-y:auto"><p id="res-fulltext" class="text-sm leading-relaxed" style="color:#d1d5db;white-space:pre-wrap"></p></div>
<div class="flex flex-wrap gap-3 mt-6 pt-6" style="border-top:1px solid var(--border)">
<button onclick="downloadTXT()" class="btn-primary"><i class="fa-solid fa-file-lines"></i> Descargar TXT</button>
<button onclick="copyText()" class="btn-ghost"><i class="fa-regular fa-copy"></i> Copiar texto</button>
<button onclick="resetApp()" class="btn-ghost ml-auto"><i class="fa-solid fa-arrow-rotate-left"></i> Nueva transcripción</button>
</div>
</div>
</section>
<section id="sec-error" class="w-full max-w-xl hidden">
<div class="card p-8 text-center fade-in" style="border-color:rgba(239,68,68,0.2)"><i class="fa-solid fa-circle-exclamation text-4xl mb-4" style="color:var(--error)"></i><h2 class="text-xl font-bold mb-2">Error en la transcripción</h2><p id="err-msg" class="text-sm mb-6" style="color:var(--text-muted)"></p><button onclick="resetApp()" class="btn-ghost"><i class="fa-solid fa-arrow-rotate-left"></i> Intentar de nuevo</button></div>
</section>
<footer class="mt-auto pt-12 pb-4 text-center text-xs" style="color:var(--text-muted)"><p>Impulsado por <strong style="color:var(--accent)">Whisper</strong> vía HuggingFace · 100% local · Sin datos enviados a terceros</p></footer>
</main>
<script>
let currentTaskId=null,currentResult=null,eventSource=null;
const $=id=>document.getElementById(id);
const secUpload=$('sec-upload'),secProc=$('sec-processing'),secResult=$('sec-result'),secError=$('sec-error');
const dropZone=$('drop-zone'),fileInput=$('file-input');
['dragenter','dragover'].forEach(ev=>dropZone.addEventListener(ev,e=>{e.preventDefault();e.stopPropagation();dropZone.classList.add('drag-over')}));
['dragleave','drop'].forEach(ev=>dropZone.addEventListener(ev,e=>{e.preventDefault();e.stopPropagation();dropZone.classList.remove('drag-over')}));
dropZone.addEventListener('drop',e=>{if(e.dataTransfer.files.length)handleFile(e.dataTransfer.files[0])});
dropZone.addEventListener('click',()=>fileInput.click());
fileInput.addEventListener('change',()=>{if(fileInput.files.length)handleFile(fileInput.files[0])});
function handleFile(file){
const ext='.'+file.name.split('.').pop().toLowerCase();
const v=['mp3','wav','flac','ogg','aac','m4a','wma','opus','mp4','avi','mkv','mov','wmv','flv','webm','m4v','mpeg','mpg','3gp'];
if(!v.includes(ext.replace('.',''))){showToast('Formato no soportado: '+ext,'error');return}
if(file.size>500*1024*1024){showToast('El archivo supera los 500 MB','error');return}
startTranscription(file)
}
function setStep(n){
for(let i=1;i<=4;i++){const s=$('step-'+i);s.style.opacity='0.4';s.style.color='var(--text-muted)';s.querySelector('i').style.color='var(--text-muted)'}
if(n>0){const s=$('step-'+n);s.style.opacity='1';s.style.color='var(--accent-light)';s.querySelector('i').style.color='var(--accent)'}
for(let i=1;i<n;i++){const s=$('step-'+i);s.style.opacity='0.7';s.querySelector('i').className='fa-solid fa-check text-[8px]';s.querySelector('i').style.color='var(--accent)'}
}
function startTranscription(file){
showSection('processing');setStep(0);$('proc-filename').textContent=file.name;$('proc-status').textContent='Subiendo...';$('proc-msg').textContent='Enviando archivo al servidor...';$('proc-pct').textContent='0%';$('proc-bar').style.width='0%';
const fd=new FormData();fd.append('file',file);fd.append('language',$('sel-lang').value);fd.append('model_size',$('sel-model').value);
fetch('/transcribe',{method:'POST',body:fd}).then(r=>r.json()).then(d=>{if(d.error)throw new Error(d.error);currentTaskId=d.task_id;connectProgress()}).catch(e=>showError(e.message))
}
function connectProgress(){
if(eventSource)eventSource.close();eventSource=new EventSource('/progress/'+currentTaskId);
const sm={starting:[0,'Iniciando...'],extracting:[10,'Extrayendo audio del video...'],loading_model:[20,'Cargando modelo de IA...'],transcribing:[30,'Transcribiendo con Whisper...'],cleaning:[85,'Aplicando formato Clean Verbatim...'],saving:[92,'Guardando archivo TXT...']};
eventSource.onmessage=function(e){
const d=JSON.parse(e.data);
if(d.error){eventSource.close();showError(d.error);return}
if(d.status==='completed'){eventSource.close();currentResult=d.result;showResult(d.result);return}
const m=sm[d.status]||[d.progress,d.message];const pct=Math.max(m[0],d.progress);
$('proc-bar').style.width=pct+'%';$('proc-pct').textContent=pct+'%';$('proc-status').textContent=m[1];$('proc-msg').textContent=d.message||m[1];
const stepMap={extracting:1,transcribing:2,cleaning:3,saving:4};if(stepMap[d.status])setStep(stepMap[d.status])
};
eventSource.onerror=function(){eventSource.close();showError('Conexión perdida con el servidor.')}
}
function showResult(r){
showSection('result');$('res-filename').textContent=r.filename;const mm=Math.floor(r.duration/60),ss=Math.floor(r.duration%60);$('res-duration').textContent=String(mm).padStart(2,'0')+':'+String(ss).padStart(2,'0');$('res-lang').textContent=r.language.toUpperCase();$('res-segs').textContent=r.segments.length;$('res-fulltext').textContent=r.full_text;
const c=$('res-segments');c.innerHTML='';const ft=s=>{const m=Math.floor(s/60);return String(m).padStart(2,'0')+':'+String(Math.floor(s%60)).padStart(2,'0')};
r.segments.forEach((seg,i)=>{const row=document.createElement('div');row.className='seg-row flex gap-3 sm:gap-5';row.innerHTML='<span class="seg-ts pt-0.5">['+ft(seg.start)+' – '+ft(seg.end)+']</span><span class="seg-text">'+esc(seg.text.trim())+'</span>';c.appendChild(row)});
switchTab('timestamps')
}
function switchTab(t){
const a=$('tab-timestamps'),b=$('tab-text'),va=$('view-timestamps'),vb=$('view-text');
if(t==='timestamps'){a.style.background='var(--accent)';a.style.color='#fff';b.style.background='transparent';b.style.color='var(--text-muted)';va.classList.remove('hidden');vb.classList.add('hidden')}
else{b.style.background='var(--accent)';b.style.color='#fff';a.style.background='transparent';a.style.color='var(--text-muted)';vb.classList.remove('hidden');va.classList.add('hidden')}
}
function downloadTXT(){if(!currentTaskId)return;window.location.href='/download/'+currentTaskId;showToast('TXT descargado','success')}
function copyText(){if(!currentResult)return;navigator.clipboard.writeText(currentResult.full_text).then(()=>showToast('Texto copiado al portapapeles','success')).catch(()=>{const ta=document.createElement('textarea');ta.value=currentResult.full_text;document.body.appendChild(ta);ta.select();document.execCommand('copy');ta.remove();showToast('Texto copiado al portapapeles','success')})}
function resetApp(){if(eventSource)eventSource.close();currentTaskId=null;currentResult=null;fileInput.value='';showSection('upload')}
function showSection(n){secUpload.classList.add('hidden');secProc.classList.add('hidden');secResult.classList.add('hidden');secError.classList.add('hidden');if(n==='upload')secUpload.classList.remove('hidden');if(n==='processing')secProc.classList.remove('hidden');if(n==='result')secResult.classList.remove('hidden');if(n==='error')secError.classList.remove('hidden')}
function showError(m){$('err-msg').textContent=m;showSection('error')}
function showToast(m,t){const el=document.createElement('div');el.className='toast '+(t||'');el.innerHTML='<i class="fa-solid '+(t==='error'?'fa-circle-xmark':'fa-circle-check')+' mr-2"></i>'+esc(m);document.body.appendChild(el);setTimeout(()=>el.remove(),4000)}
function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML}
</script>
</body>
</html>"""

# ─── Inicio ──────────────────────────────────────────────────────
if __name__ == "__main__":
    if not HAS_FFMPEG:
        print("\n⚠️ AVISO: FFmpeg no instalado. Solo audio (no video).")
        print("  Instalar: sudo apt install ffmpeg\n")
    print(f"\n{'═'*55}")
    print(f"  TranscribeAI — Clean Verbatim")
    print(f"  Modelo: {DEFAULT_MODEL}  |  Dispositivo: {DEVICE}")
    print(f"  FFmpeg: {'Sí' if HAS_FFMPEG else 'No'}")
    print(f"  URL:    http://localhost:{PORT}")
    print(f"{'═'*55}\n")
    app.run(host=HOST, port=PORT, debug=False, threaded=True)