# AI Shorts Maker Pro

[Русский](#русский) | [English](#english)

## Русский

AI Shorts Maker Pro - локальное FastAPI-приложение для поиска и рендера
коротких клипов из длинных видео. Приложение принимает ссылку на YouTube или
локальный видеофайл, извлекает аудио через FFmpeg, транскрибирует речь через
Faster Whisper, отправляет транскрипт в локальную модель Ollama и рендерит
выбранные фрагменты в MP4.

Текущая реализация рассчитана на один локальный компьютер с NVIDIA GPU. Это не
многопользовательский веб-сервис.

## Возможности

- Загрузка исходника по YouTube-ссылке или из локального файла.
- Извлечение аудио через FFmpeg.
- Транскрибация русской речи через Faster Whisper.
- Поиск кандидатов для шортсов через локальную модель Ollama.
- Рендер выбранных фрагментов в двух режимах:
  - вертикальный 9:16 с черными полями;
  - исходное разрешение видео.
- Опциональное вшивание субтитров в итоговый ролик.
- Опциональное удаление пауз между сегментами транскрипта внутри клипа.
- Потоковый вывод прогресса анализа и рендера в браузере.

## Требования

Перед запуском установите:

- Python 3.10 или новее.
- FFmpeg, доступный из `PATH`.
- Ollama, запущенную локально.
- Модель Ollama `qwen2.5`.
- NVIDIA GPU с поддержкой CUDA для Faster Whisper.
- Сборку FFmpeg с поддержкой энкодера `h264_nvenc`.

В коде сейчас жестко заданы следующие параметры:

- `WhisperModel("medium", device="cuda", compute_type="float16")`
- Ollama API: `http://localhost:11434/api/generate`
- модель Ollama: `qwen2.5`
- видеоэнкодер FFmpeg: `h264_nvenc`
- порт приложения: `7860`
- язык транскрибации: `ru`

Если ваша среда отличается, измените соответствующие значения в `main.py`.

## Установка

### Windows

Запустите установочный скрипт:

```bat
install.bat
```

Скрипт проверяет Python, устанавливает зависимости из `requirements.txt`,
проверяет Ollama, скачивает модель `qwen2.5`, если Ollama доступна, и проверяет
наличие FFmpeg.

### Ручная установка

Если вы не используете `install.bat`, выполните команды вручную:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull qwen2.5
```

Перед анализом видео убедитесь, что сервис Ollama запущен.

## Запуск

### Windows

Запустите:

```bat
run.bat
```

Скрипт открывает `http://localhost:7860` в браузере и запускает
`python main.py`.

### Ручной запуск

Запустите приложение:

```bash
python main.py
```

Затем откройте в браузере:

```text
http://localhost:7860
```

## Использование

1. Откройте веб-интерфейс на `http://localhost:7860`.
2. Укажите YouTube-ссылку или выберите локальный видеофайл.
3. Выберите формат вывода:
   - `Шортс (9:16 с рамками)` - вертикальный ролик с черными полями;
   - `Исходное разрешение` - клип в разрешении исходника.
4. Включите субтитры, если нужны вшитые `.srt`-субтитры.
5. Включите удаление тишины, если нужно склеивать только сегменты с речью.
6. Запустите анализ.
7. Просмотрите найденные фрагменты и отметьте нужные.
8. Запустите рендер.
9. Заберите готовые файлы из `output_shorts` или из блока результатов в
   браузере.

## Структура проекта

```text
.
├── main.py
├── requirements.txt
├── install.bat
├── run.bat
└── templates
    └── index.html
```

### `main.py`

Основной backend-файл. В нем находятся:

- настройка путей для FFmpeg и NVIDIA DLL;
- создание временной и выходной директорий;
- скачивание YouTube-видео через `yt-dlp`;
- извлечение аудио через FFmpeg;
- транскрибация через Faster Whisper;
- запрос к Ollama и разбор JSON-ответа;
- генерация `.srt`-субтитров;
- сборка команд FFmpeg для рендера;
- FastAPI-эндпоинты и запуск Uvicorn.

### `templates/index.html`

Единственный HTML-шаблон приложения. В нем находятся:

- поля выбора источника;
- настройки формата и обработки;
- индикатор прогресса;
- клиентская логика для Server-Sent Events;
- список найденных клипов;
- предпросмотр и ссылки на скачивание готовых роликов.

## API

Интерфейс использует следующие эндпоинты:

| Метод | Путь | Назначение |
| --- | --- | --- |
| `GET` | `/` | Возвращает HTML-интерфейс. |
| `POST` | `/api/upload` | Сохраняет YouTube-ссылку или загруженный файл. |
| `GET` | `/api/analyze_stream` | Стримит прогресс анализа и возвращает найденные клипы. |
| `POST` | `/api/render` | Сохраняет выбранные клипы и параметры рендера. |
| `GET` | `/api/render_stream` | Стримит прогресс рендера и возвращает ссылки на MP4. |

Готовые ролики раздаются из статического маршрута `/output_shorts`.

## Pipeline обработки

1. При новой загрузке приложение очищает `temp_processing`.
2. Исходное видео сохраняется как `temp_processing/input_video.mp4`.
3. Для YouTube-источника `yt-dlp` скачивает и объединяет видео в MP4.
4. FFmpeg извлекает mono WAV 16 kHz в `temp_processing/audio.wav`.
5. Faster Whisper возвращает сегменты транскрипта с таймкодами.
6. Ollama получает транскрипт и возвращает JSON такого формата:

```json
{
  "shorts": [
    {
      "start": 12.5,
      "end": 45.0,
      "title": "Название клипа"
    }
  ]
}
```

7. Пользователь выбирает клипы для рендера.
8. Приложение генерирует `.srt`-файлы и, при включенном удалении тишины,
   concat-файлы для FFmpeg.
9. FFmpeg записывает итоговые MP4-файлы в `output_shorts`.

## Рабочие директории

- `temp_processing` хранит текущий исходник, извлеченное аудио, временные
  субтитры и concat-файлы.
- `output_shorts` хранит готовые ролики и раздается FastAPI как статическая
  директория.

`temp_processing` очищается при каждой новой загрузке. `output_shorts`
автоматически не очищается.

## Ограничения текущей реализации

- Состояние хранится в глобальном словаре `app_state`. Параллельные пользователи
  или несколько одновременных задач будут конфликтовать.
- Тяжелые операции анализа и рендера выполняются внутри request handler'ов
  FastAPI. Длинная задача может блокировать worker.
- Транскрибация жестко настроена на русский язык: `language="ru"`.
- Whisper жестко настроен на CUDA и `float16`.
- Рендер жестко использует `h264_nvenc`.
- Endpoint и модель Ollama заданы константами в `main.py`.
- Имена выходных файлов строятся из названий, которые вернула модель. Повторные
  названия могут перезаписать существующие файлы.
- `extract_audio()` скрывает вывод FFmpeg и не проверяет код возврата через
  `check=True`, поэтому ошибка извлечения аудио может проявиться позже.
- UI частично англоязычный, а backend-статусы русскоязычные.

## Решение частых проблем

### `FFmpeg не найден`

Установите FFmpeg и проверьте, что команда доступна из терминала:

```bash
ffmpeg -version
```

### Ollama не возвращает клипы

Проверьте, что Ollama запущена и модель установлена:

```bash
ollama list
ollama pull qwen2.5
```

### Ошибка CUDA или Faster Whisper

Текущий код требует CUDA-capable GPU. Для CPU fallback измените
`transcribe_audio()` в `main.py`: укажите CPU-устройство и совместимый
`compute_type`.

### Ошибка рендера FFmpeg

Проверьте, что FFmpeg поддерживает NVENC:

```bash
ffmpeg -encoders
```

В списке должен быть `h264_nvenc`. Если его нет, установите FFmpeg со сборкой
NVIDIA encoder support или замените энкодер в `process_short_video()`.

## English

AI Shorts Maker Pro is a local FastAPI application for finding and rendering
short clips from long videos. The application accepts a YouTube URL or a local
video file, extracts audio with FFmpeg, transcribes speech with Faster Whisper,
sends the transcript to a local Ollama model, and renders the selected segments
as MP4 files.

The current implementation is designed for one local workstation with an NVIDIA
GPU. It is not a multi-user web service.

## Features

- Load a source video from a YouTube URL or a local file.
- Extract audio with FFmpeg.
- Transcribe Russian speech with Faster Whisper.
- Find short-form clip candidates with a local Ollama model.
- Render selected segments in two modes:
  - vertical 9:16 video with black padding;
  - original source resolution.
- Optionally burn subtitles into the output video.
- Optionally remove pauses between transcript segments inside a clip.
- Stream analysis and rendering progress to the browser.

## Requirements

Install the following tools before running the project:

- Python 3.10 or newer.
- FFmpeg available from `PATH`.
- Ollama running locally.
- Ollama model `qwen2.5`.
- NVIDIA GPU with CUDA support for Faster Whisper.
- FFmpeg build with `h264_nvenc` encoder support.

The code currently hard-codes these settings:

- `WhisperModel("medium", device="cuda", compute_type="float16")`
- Ollama API: `http://localhost:11434/api/generate`
- Ollama model: `qwen2.5`
- FFmpeg video encoder: `h264_nvenc`
- application port: `7860`
- transcription language: `ru`

If your environment differs, update the corresponding values in `main.py`.

## Installation

### Windows

Run the installer script:

```bat
install.bat
```

The script checks Python, installs dependencies from `requirements.txt`, checks
Ollama, pulls `qwen2.5` when Ollama is available, and checks FFmpeg.

### Manual installation

If you do not use `install.bat`, run the commands manually:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull qwen2.5
```

Make sure the Ollama service is running before analyzing a video.

## Running

### Windows

Run:

```bat
run.bat
```

The script opens `http://localhost:7860` in your browser and starts
`python main.py`.

### Manual run

Start the application:

```bash
python main.py
```

Then open in your browser:

```text
http://localhost:7860
```

## Usage

1. Open the web interface at `http://localhost:7860`.
2. Provide a YouTube URL or choose a local video file.
3. Choose the output format:
   - `Шортс (9:16 с рамками)` - vertical video with black padding;
   - `Исходное разрешение` - clip in the source resolution.
4. Enable subtitles if you need burned-in `.srt` captions.
5. Enable silence trimming if you want to concatenate only speech segments.
6. Start analysis.
7. Review the detected segments and select the ones you want.
8. Start rendering.
9. Take the rendered files from `output_shorts` or from the browser results
   section.

## Project structure

```text
.
├── main.py
├── requirements.txt
├── install.bat
├── run.bat
└── templates
    └── index.html
```

### `main.py`

The main backend file. It contains:

- FFmpeg and NVIDIA DLL path setup;
- temporary and output directory creation;
- YouTube video download through `yt-dlp`;
- audio extraction through FFmpeg;
- transcription through Faster Whisper;
- Ollama request and JSON response parsing;
- `.srt` subtitle generation;
- FFmpeg command construction for rendering;
- FastAPI endpoints and Uvicorn startup.

### `templates/index.html`

The only HTML template in the application. It contains:

- source input fields;
- format and processing options;
- progress indicator;
- client-side logic for Server-Sent Events;
- detected clip list;
- preview and download links for rendered videos.

## API

The interface uses these endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Returns the HTML interface. |
| `POST` | `/api/upload` | Stores the YouTube URL or uploaded file. |
| `GET` | `/api/analyze_stream` | Streams analysis progress and returns detected clips. |
| `POST` | `/api/render` | Stores selected clips and render options. |
| `GET` | `/api/render_stream` | Streams render progress and returns MP4 links. |

Rendered videos are served from the `/output_shorts` static route.

## Processing pipeline

1. On every new upload, the application clears `temp_processing`.
2. The source video is stored as `temp_processing/input_video.mp4`.
3. For a YouTube source, `yt-dlp` downloads and merges the video into MP4.
4. FFmpeg extracts mono 16 kHz WAV audio to `temp_processing/audio.wav`.
5. Faster Whisper returns timestamped transcript segments.
6. Ollama receives the transcript and returns JSON in this format:

```json
{
  "shorts": [
    {
      "start": 12.5,
      "end": 45.0,
      "title": "Clip title"
    }
  ]
}
```

7. The user selects clips for rendering.
8. The application generates `.srt` files and, when silence trimming is enabled,
   FFmpeg concat files.
9. FFmpeg writes the final MP4 files to `output_shorts`.

## Runtime directories

- `temp_processing` stores the current source file, extracted audio, temporary
  subtitles, and concat files.
- `output_shorts` stores rendered videos and is served by FastAPI as a static
  directory.

`temp_processing` is cleared on every new upload. `output_shorts` is not cleared
automatically.

## Current limitations

- State is stored in the global `app_state` dictionary. Parallel users or
  multiple simultaneous jobs will conflict.
- Heavy analysis and rendering work runs inside FastAPI request handlers. A long
  job can block the worker.
- Transcription is hard-coded to Russian: `language="ru"`.
- Whisper is hard-coded to CUDA and `float16`.
- Rendering is hard-coded to `h264_nvenc`.
- The Ollama endpoint and model are constants in `main.py`.
- Output filenames are built from model-generated titles. Duplicate titles can
  overwrite existing files.
- `extract_audio()` hides FFmpeg output and does not check the return code with
  `check=True`, so an audio extraction error can appear later.
- The UI is partly English, while backend status messages are in Russian.

## Troubleshooting

### `FFmpeg не найден`

Install FFmpeg and check that the command is available from your terminal:

```bash
ffmpeg -version
```

### Ollama does not return clips

Check that Ollama is running and the model is installed:

```bash
ollama list
ollama pull qwen2.5
```

### CUDA or Faster Whisper error

The current code requires a CUDA-capable GPU. For CPU fallback, update
`transcribe_audio()` in `main.py`: set a CPU device and a compatible
`compute_type`.

### FFmpeg render error

Check that FFmpeg supports NVENC:

```bash
ffmpeg -encoders
```

The list must contain `h264_nvenc`. If it is missing, install an FFmpeg build
with NVIDIA encoder support or replace the encoder in `process_short_video()`.
