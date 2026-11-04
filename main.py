import os
import re
import gc
import json
import math
import shutil
import subprocess
import site
import traceback
import asyncio

# Add ffmpeg to PATH if FFMPEG_DIR points to its bin directory
ffmpeg_dir = os.environ.get("FFMPEG_DIR")
if ffmpeg_dir and os.path.exists(ffmpeg_dir) and ffmpeg_dir not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + ffmpeg_dir

# Add NVIDIA CUDA libraries to PATH and DLL directories to fix cublas64_12.dll error
try:
    for sp in site.getsitepackages():
        nvidia_bin_paths = [
            os.path.join(sp, "nvidia", "cublas", "bin"),
            os.path.join(sp, "nvidia", "cudnn", "bin"),
            os.path.join(sp, "nvidia", "cuda_nvrtc", "bin")
        ]
        for p in nvidia_bin_paths:
            if os.path.exists(p):
                if p not in os.environ["PATH"]:
                    os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(p)
except Exception as e:
    print(f"Failed to add NVIDIA DLLs: {e}")

from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import yt_dlp
import cv2
import torch
import requests
from faster_whisper import WhisperModel

# Настройки
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5"
OLLAMA_NUM_CTX = 32768
TEMP_DIR = "temp_processing"
OUTPUT_DIR = "output_shorts"
HOST = os.environ.get("HOST", "127.0.0.1")
BUSY_MESSAGE = "Задача уже выполняется. Дождитесь её завершения."

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def cleanup_temp_files():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)

def free_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def download_youtube_video(url, output_path):
    ydl_opts = {
        'outtmpl': output_path,
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'merge_output_format': 'mp4'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def run_ffmpeg(command):
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"FFmpeg завершился с кодом {result.returncode}: {stderr[-500:]}")

def extract_audio(video_path, audio_path):
    command = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path
    ]
    run_ffmpeg(command)

def transcribe_audio(audio_path):
    model = WhisperModel("medium", device="cuda", compute_type="float16")
    segments_gen, _ = model.transcribe(audio_path, beam_size=5, word_timestamps=False, language="ru")
    
    segments = []
    transcript_lines = []
    for s in segments_gen:
        segments.append({"start": s.start, "end": s.end, "text": s.text.strip()})
        transcript_lines.append(f"[{s.start:.2f} - {s.end:.2f}] {s.text.strip()}")
        
    full_text = "\n".join(transcript_lines)
    
    del model
    free_vram()
    return full_text, segments

class OllamaError(Exception):
    pass

def validate_shorts(raw_shorts):
    if not isinstance(raw_shorts, list):
        return []

    valid_shorts = []
    for i, short in enumerate(raw_shorts):
        try:
            start = float(short["start"])
            end = float(short["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            continue

        title = short.get("title")
        if not isinstance(title, str) or not title.strip():
            title = f"short_{i+1}"
        valid_shorts.append({"start": start, "end": end, "title": title})

    dropped = len(raw_shorts) - len(valid_shorts)
    if dropped:
        print(f"Отброшено некорректных клипов от модели: {dropped}")
    return valid_shorts

def analyze_with_ollama(transcript):
    prompt = f"""
    Ты — гениальный ИИ-режиссер монтажа. Твоя главная цель — КАЧЕСТВО, а не количество. Найди самые яркие, целостные и виральные ситуации из этого транскрипта для YouTube Shorts/Reels/TikTok.
    
    КРИТИЧЕСКИЕ ПРАВИЛА (СМЫСЛ И АДАПТИВНОСТЬ):
    1. ДИНАМИЧНОЕ КОЛИЧЕСТВО: Не пытайся вырезать как можно больше! Если исходник длится 5 минут — найди 1-3 действительно крутых момента. Если видео идет час и там много экшена — можешь сделать 10-15 шортсов. Если ничего интересного нет, лучше сделай 1 отличный шортс, чем 5 скучных. Пропускай тишину, скучный гринд и бессмысленные разговоры.
    2. Полноценные сцены: Фрагмент должен обязательно включать предысторию (завязку ситуации), само событие и логический конец (реакцию, смех, последствия или вывод). Зритель должен понимать весь контекст происходящего.
    3. Жесткий лимит времени: Длительность каждого фрагмента ОБЯЗАТЕЛЬНО должна быть от 15 до 120 секунд. Делай упор на продолжительные, осмысленные сцены.
    4. Контекст: Расширяй таймкоды (start и end) назад и вперед так, чтобы захватить атмосферу вокруг момента.
    
    Транскрипт имеет формат [старт - конец] текст.
    
    Выведи ТОЛЬКО валидный JSON в таком формате, без markdown разметки и без лишнего текста:
    {{
        "shorts": [
            {{"start": 12.5, "end": 45.0, "title": "Название первого шортса"}},
            {{"start": 120.0, "end": 225.5, "title": "Название второго шортса"}}
        ]
    }}
    
    Транскрипт:
    {transcript}
    """
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "keep_alive": 0,
        "options": {
            "num_ctx": OLLAMA_NUM_CTX
        }
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
    except requests.RequestException as e:
        raise OllamaError(f"Не удалось обратиться к Ollama ({OLLAMA_API_URL}): {e}") from e

    if response.status_code != 200:
        raise OllamaError(f"Ollama вернула код {response.status_code}: {response.text[:300]}")

    try:
        data = response.json()
        response_text = data.get("response", "{}")
        response_text = re.sub(r"```(?:json)?\n?|```", "", response_text).strip()
        shorts_data = json.loads(response_text)
        raw_shorts = shorts_data.get("shorts", [])
    except (ValueError, AttributeError, TypeError) as e:
        raise OllamaError(f"Ollama вернула некорректный JSON: {e}") from e

    # Ollama обрезает длинный промпт без ошибки, поэтому о потере текста судим по счётчику токенов.
    truncated = data.get("prompt_eval_count", 0) >= OLLAMA_NUM_CTX * 0.95
    return validate_shorts(raw_shorts), truncated

def format_time_srt(seconds):
    ms = int((seconds % 1) * 1000)
    s = int(seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def generate_srt(segments, srt_path):
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments):
            start_str = format_time_srt(seg['start'])
            end_str = format_time_srt(seg['end'])
            f.write(f"{i+1}\n{start_str} --> {end_str}\n{seg['text']}\n\n")

def prepare_render_data(short_start, short_end, all_segments, do_silence, do_subs, temp_dir, short_idx):
    valid_segments = []
    for seg in all_segments:
        if seg["end"] > short_start and seg["start"] < short_end:
            c_start = max(short_start, seg["start"])
            c_end = min(short_end, seg["end"])
            if c_end - c_start > 0.5:
                valid_segments.append({"start": c_start, "end": c_end, "text": seg["text"]})
                
    if not valid_segments:
        valid_segments = [{"start": short_start, "end": short_end, "text": ""}]
        do_silence = False
        
    concat_path = os.path.join(temp_dir, f"concat_{short_idx}.txt")
    srt_path = os.path.join(temp_dir, f"subs_{short_idx}.srt")
    
    if do_silence:
        current_time = 0.0
        adjusted_segments = []
        with open(concat_path, 'w', encoding='utf-8') as f:
            for seg in valid_segments:
                f.write(f"file 'input_video.mp4'\n")
                f.write(f"inpoint {seg['start']}\n")
                f.write(f"outpoint {seg['end']}\n")
                
                dur = seg['end'] - seg['start']
                adjusted_segments.append({
                    "start": current_time,
                    "end": current_time + dur,
                    "text": seg["text"]
                })
                current_time += dur
        
        if do_subs:
            generate_srt(adjusted_segments, srt_path)
            
        return concat_path, srt_path
    else:
        adjusted_segments = []
        for seg in valid_segments:
            adjusted_segments.append({
                "start": seg["start"] - short_start,
                "end": seg["end"] - short_start,
                "text": seg["text"]
            })
        
        if do_subs:
            generate_srt(adjusted_segments, srt_path)
            
        return None, srt_path

def process_short_video(input_video, short_start, short_end, output_path, format_type, concat_path, srt_path, do_subs):
    cap = cv2.VideoCapture(input_video)
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    target_w = 1080
    target_h = 1920
    if orig_h > 0:
        target_w = orig_h
        target_h = int(orig_h * 16 / 9)
        target_w -= target_w % 2
        target_h -= target_h % 2
        
    filters = []
    if format_type == "Шортс (9:16 с рамками)":
        filters.append(f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease")
        filters.append(f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=black")
        
    if do_subs and srt_path and os.path.exists(srt_path):
        srt_path_fwd = srt_path.replace('\\', '/')
        style = "FontName=Arial,FontSize=18,PrimaryColour=&H0000FFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=0,Alignment=2,MarginV=40"
        filters.append(f"subtitles='{srt_path_fwd}':force_style='{style}'")
        
    vf_string = ",".join(filters) if filters else None
    
    out_cmd = ["ffmpeg", "-y"]
    
    if concat_path:
        out_cmd.extend(["-f", "concat", "-safe", "0", "-i", concat_path])
    else:
        duration = short_end - short_start
        out_cmd.extend(["-ss", str(short_start), "-i", input_video, "-t", str(duration)])
        
    if vf_string:
        out_cmd.extend(["-vf", vf_string])
        
    out_cmd.extend([
        "-c:v", "h264_nvenc", "-preset", "p6", "-b:v", "5M",
        "-c:a", "aac", "-b:a", "192k",
        output_path
    ])
    
    run_ffmpeg(out_cmd)

def unique_output_filename(title, index):
    safe_title = "".join(c for c in title if c.isalnum() or c in " _-").strip()
    if not safe_title:
        safe_title = f"short_{index+1}"

    filename = f"{safe_title}.mp4"
    suffix = 2
    while os.path.exists(os.path.join(OUTPUT_DIR, filename)):
        filename = f"{safe_title}_{suffix}.mp4"
        suffix += 1
    return filename


# --- FastAPI Application ---
app = FastAPI()
app.mount("/output_shorts", StaticFiles(directory="output_shorts"), name="output_shorts")
templates = Jinja2Templates(directory="templates")

# Глобальное состояние (для простоты однопользовательского режима локально)
app_state = {
    "segments": [],
    "shorts_metadata": [],
    "video_url": None,
    "file_path": None,
    "render_payload": {}
}
task_lock = asyncio.Lock()

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/upload")
async def upload_video(
    video_url: str = Form(None),
    video_file: UploadFile = File(None)
):
    if task_lock.locked():
        raise HTTPException(status_code=409, detail=BUSY_MESSAGE)

    await asyncio.to_thread(cleanup_temp_files)
    input_video = os.path.join(TEMP_DIR, "input_video.mp4")

    app_state["video_url"] = video_url
    if video_file and video_file.filename:
        with open(input_video, "wb") as buffer:
            await asyncio.to_thread(shutil.copyfileobj, video_file.file, buffer)
        app_state["file_path"] = input_video
    else:
        app_state["file_path"] = None
        
    return {"status": "ok"}

@app.get("/api/analyze_stream")
async def analyze_stream():
    async def event_generator():
        if task_lock.locked():
            yield f"data: {json.dumps({'status': f'❌ {BUSY_MESSAGE}', 'error': True})}\n\n"
            return

        async with task_lock:
            try:
                input_video = os.path.join(TEMP_DIR, "input_video.mp4")
                audio_path = os.path.join(TEMP_DIR, "audio.wav")

                if app_state["video_url"]:
                    yield f"data: {json.dumps({'status': '⏳ Скачивание видео с YouTube...', 'progress': 10})}\n\n"
                    await asyncio.to_thread(download_youtube_video, app_state["video_url"], input_video)
                elif app_state["file_path"]:
                    yield f"data: {json.dumps({'status': '⏳ Видео загружено...', 'progress': 10})}\n\n"
                else:
                    yield f"data: {json.dumps({'status': '❌ Ошибка: Нет видео', 'error': True})}\n\n"
                    return

                yield f"data: {json.dumps({'status': '⏳ Извлечение аудиодорожки...', 'progress': 30})}\n\n"
                await asyncio.to_thread(extract_audio, input_video, audio_path)

                yield f"data: {json.dumps({'status': '⏳ Транскрибация (Whisper, GPU)...', 'progress': 50})}\n\n"
                transcript, segments = await asyncio.to_thread(transcribe_audio, audio_path)
                app_state["segments"] = segments

                yield f"data: {json.dumps({'status': '⏳ Анализ нейросетью (Ollama)...', 'progress': 80})}\n\n"
                shorts_metadata, truncated = await asyncio.to_thread(analyze_with_ollama, transcript)
                app_state["shorts_metadata"] = shorts_metadata

                if not shorts_metadata:
                    yield f"data: {json.dumps({'status': '❌ Ошибка: Ollama не нашла моментов', 'error': True})}\n\n"
                    return

                status = '✅ Анализ завершен!'
                if truncated:
                    status = '⚠️ Анализ завершен, но транскрипт не поместился в контекст модели: часть видео могла не учитываться'
                yield f"data: {json.dumps({'status': status, 'progress': 100, 'done': True, 'shorts': shorts_metadata})}\n\n"
            except Exception as e:
                traceback.print_exc()
                yield f"data: {json.dumps({'status': f'❌ Ошибка: {str(e)}', 'error': True})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

class RenderPayload(BaseModel):
    selected_indices: list[int]
    format_type: str
    do_subs: bool
    do_silence: bool

@app.post("/api/render")
async def render_start(payload: RenderPayload):
    app_state["render_payload"] = payload.model_dump()
    return {"status": "ok"}

@app.get("/api/render_stream")
async def render_stream():
    async def event_generator():
        if task_lock.locked():
            yield f"data: {json.dumps({'status': f'❌ {BUSY_MESSAGE}', 'error': True})}\n\n"
            return

        async with task_lock:
            try:
                payload = app_state.get("render_payload", {})
                selected_indices = payload.get("selected_indices", [])
                format_type = payload.get("format_type", "Шортс (9:16 с рамками)")
                do_subs = payload.get("do_subs", False)
                do_silence = payload.get("do_silence", False)

                all_segments = app_state.get("segments", [])
                shorts_metadata = app_state.get("shorts_metadata", [])
                input_video = os.path.join(TEMP_DIR, "input_video.mp4")

                selected_shorts = [shorts_metadata[i] for i in selected_indices if 0 <= i < len(shorts_metadata)]

                output_files = []
                errors = []
                last_error = ""

                for i, short in enumerate(selected_shorts):
                    title = short.get("title", f"short_{i+1}")
                    start = float(short.get("start", 0))
                    end = float(short.get("end", 0))

                    yield f"data: {json.dumps({'status': f'⏳ Рендер {i+1}/{len(selected_shorts)}: {title}...', 'progress': int(((i) / len(selected_shorts))*100)})}\n\n"

                    concat_path, srt_path = prepare_render_data(start, end, all_segments, do_silence, do_subs, TEMP_DIR, i)

                    filename = unique_output_filename(title, i)
                    out_file = os.path.join(OUTPUT_DIR, filename)

                    try:
                        await asyncio.to_thread(process_short_video, input_video, start, end, out_file, format_type, concat_path, srt_path, do_subs)
                        output_files.append(f"/output_shorts/{filename}")
                    except Exception as e:
                        traceback.print_exc()
                        errors.append(title)
                        last_error = str(e)

                if errors and not output_files:
                    yield f"data: {json.dumps({'status': f'❌ Ошибка рендера: {last_error}', 'error': True, 'failed': errors})}\n\n"
                else:
                    status = '✅ Успешно отрендерено!'
                    if errors:
                        status = f'⚠️ Отрендерено {len(output_files)} из {len(selected_shorts)}'
                    yield f"data: {json.dumps({'status': status, 'progress': 100, 'done': True, 'files': output_files, 'failed': errors})}\n\n"
            except Exception as e:
                traceback.print_exc()
                yield f"data: {json.dumps({'status': f'❌ Ошибка: {str(e)}', 'error': True})}\n\n"
            finally:
                free_vram()

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    if not shutil.which("ffmpeg"):
        print("❌ ОШИБКА: FFmpeg не найден! Убедитесь, что он установлен и добавлен в PATH.")
        exit(1)

    print(f"Запуск FastAPI сервера на http://{HOST}:7860 ...")
    uvicorn.run(app, host=HOST, port=7860, log_level="warning")