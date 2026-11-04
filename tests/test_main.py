import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests
from fastapi.testclient import TestClient

# Тяжёлые модули требуют GPU и в этих тестах не вызываются.
for module_name in ("yt_dlp", "cv2", "torch", "faster_whisper"):
    sys.modules[module_name] = MagicMock()

# main.py создаёт рабочие каталоги относительно текущей директории при импорте.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(tempfile.mkdtemp())

import main


def ollama_response(text=None, status_code=200, prompt_eval_count=100, raw_body=None):
    response = MagicMock()
    response.status_code = status_code
    response.text = raw_body or ""
    if raw_body is not None:
        response.json.side_effect = ValueError("not json")
    else:
        response.json.return_value = {"response": text, "prompt_eval_count": prompt_eval_count}
    return response


def read_events(response):
    return [
        json.loads(line[len("data: "):])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def busy(monkeypatch):
    lock = MagicMock()
    lock.locked.return_value = True
    monkeypatch.setattr(main, "task_lock", lock)


def test_validate_shorts_keeps_valid_and_converts_numbers():
    shorts = main.validate_shorts([{"start": "1.5", "end": 20, "title": "Клип"}])
    assert shorts == [{"start": 1.5, "end": 20.0, "title": "Клип"}]


@pytest.mark.parametrize("bad", [
    {"start": 10, "end": 5, "title": "конец раньше начала"},
    {"start": 5, "end": 5, "title": "нулевая длина"},
    {"start": -1, "end": 5, "title": "отрицательное начало"},
    {"start": "abc", "end": 5, "title": "не число"},
    {"start": float("nan"), "end": 5, "title": "nan"},
    {"start": 0, "end": float("inf"), "title": "inf"},
    {"end": 5, "title": "нет начала"},
    "не словарь",
])
def test_validate_shorts_drops_invalid(bad):
    assert main.validate_shorts([bad]) == []


@pytest.mark.parametrize("title", [None, "", "   ", 42])
def test_validate_shorts_replaces_bad_title(title):
    shorts = main.validate_shorts([{"start": 0, "end": 5, "title": title}])
    assert shorts[0]["title"] == "short_1"


def test_validate_shorts_rejects_non_list():
    assert main.validate_shorts({"start": 0}) == []


def test_unique_output_filename_adds_suffix(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "OUTPUT_DIR", str(tmp_path))
    assert main.unique_output_filename("Клип", 0) == "Клип.mp4"
    (tmp_path / "Клип.mp4").touch()
    assert main.unique_output_filename("Клип", 0) == "Клип_2.mp4"
    (tmp_path / "Клип_2.mp4").touch()
    assert main.unique_output_filename("Клип", 0) == "Клип_3.mp4"


def test_unique_output_filename_handles_title_without_safe_chars(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "OUTPUT_DIR", str(tmp_path))
    assert main.unique_output_filename("?!/", 2) == "short_3.mp4"


def test_format_time_srt():
    assert main.format_time_srt(3661.5) == "01:01:01,500"


def test_prepare_render_data_shifts_subtitles_to_clip_start(tmp_path):
    segments = [
        {"start": 8.0, "end": 12.0, "text": "первый"},
        {"start": 15.0, "end": 18.0, "text": "второй"},
        {"start": 40.0, "end": 45.0, "text": "вне клипа"},
    ]
    concat_path, srt_path = main.prepare_render_data(10.0, 20.0, segments, False, True, str(tmp_path), 0)
    assert concat_path is None
    srt = Path(srt_path).read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:02,000" in srt
    assert "00:00:05,000 --> 00:00:08,000" in srt
    assert "вне клипа" not in srt


def test_prepare_render_data_silence_removal_joins_segments(tmp_path):
    segments = [
        {"start": 10.0, "end": 12.0, "text": "первый"},
        {"start": 15.0, "end": 18.0, "text": "второй"},
    ]
    concat_path, srt_path = main.prepare_render_data(10.0, 20.0, segments, True, True, str(tmp_path), 0)
    concat = Path(concat_path).read_text(encoding="utf-8")
    assert "inpoint 15.0" in concat and "outpoint 18.0" in concat
    assert "00:00:02,000 --> 00:00:05,000" in Path(srt_path).read_text(encoding="utf-8")


def test_ollama_returns_validated_shorts(monkeypatch):
    body = json.dumps({"shorts": [{"start": 1, "end": 20, "title": "ok"}, {"start": 9, "end": 2}]})
    monkeypatch.setattr(main.requests, "post", lambda *a, **k: ollama_response(body))
    shorts, truncated = main.analyze_with_ollama("текст")
    assert [s["title"] for s in shorts] == ["ok"]
    assert truncated is False


def test_ollama_reports_possible_truncation(monkeypatch):
    response = ollama_response('{"shorts": []}', prompt_eval_count=main.OLLAMA_NUM_CTX)
    monkeypatch.setattr(main.requests, "post", lambda *a, **k: response)
    assert main.analyze_with_ollama("текст") == ([], True)


def test_ollama_connection_error_raises(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.ConnectionError("refused")
    monkeypatch.setattr(main.requests, "post", fail)
    with pytest.raises(main.OllamaError, match="Не удалось обратиться"):
        main.analyze_with_ollama("текст")


def test_ollama_bad_status_raises(monkeypatch):
    response = ollama_response(status_code=500)
    monkeypatch.setattr(main.requests, "post", lambda *a, **k: response)
    with pytest.raises(main.OllamaError, match="код 500"):
        main.analyze_with_ollama("текст")


@pytest.mark.parametrize("response", [
    ollama_response("это не json"),
    ollama_response('["список"]'),
    ollama_response(raw_body="<html>"),
])
def test_ollama_bad_json_raises(monkeypatch, response):
    monkeypatch.setattr(main.requests, "post", lambda *a, **k: response)
    with pytest.raises(main.OllamaError, match="некорректный JSON"):
        main.analyze_with_ollama("текст")


def test_extract_audio_raises_with_ffmpeg_stderr(monkeypatch):
    result = subprocess.CompletedProcess([], 1, stderr="нет такого файла".encode("utf-8"))
    monkeypatch.setattr(main.subprocess, "run", lambda *a, **k: result)
    with pytest.raises(RuntimeError, match="нет такого файла"):
        main.extract_audio("in.mp4", "out.wav")


def test_analyze_stream_reports_ollama_failure(client, monkeypatch):
    def fail(transcript):
        raise main.OllamaError("Ollama вернула код 500")
    monkeypatch.setitem(main.app_state, "video_url", None)
    monkeypatch.setitem(main.app_state, "file_path", "input.mp4")
    monkeypatch.setattr(main, "extract_audio", lambda *a: None)
    monkeypatch.setattr(main, "transcribe_audio", lambda *a: ("текст", []))
    monkeypatch.setattr(main, "analyze_with_ollama", fail)

    final = read_events(client.get("/api/analyze_stream"))[-1]
    assert final["error"] is True
    assert "код 500" in final["status"]


def test_analyze_stream_rejected_when_busy(client, busy):
    events = read_events(client.get("/api/analyze_stream"))
    assert len(events) == 1
    assert events[0]["error"] is True
    assert main.BUSY_MESSAGE in events[0]["status"]


def test_render_stream_rejected_when_busy(client, busy):
    events = read_events(client.get("/api/render_stream"))
    assert len(events) == 1
    assert main.BUSY_MESSAGE in events[0]["status"]


def test_upload_rejected_when_busy(client, busy):
    response = client.post("/api/upload", data={"video_url": "https://example.com/v"})
    assert response.status_code == 409


def prepare_render(monkeypatch, tmp_path, titles):
    monkeypatch.setattr(main, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setitem(main.app_state, "segments", [])
    monkeypatch.setitem(main.app_state, "shorts_metadata", [
        {"start": 0.0, "end": 5.0, "title": title} for title in titles
    ])
    monkeypatch.setitem(main.app_state, "render_payload", {
        "selected_indices": list(range(len(titles))),
        "format_type": "Исходное разрешение",
        "do_subs": False,
        "do_silence": False,
    })


def test_render_stream_reports_partial_failure(client, monkeypatch, tmp_path):
    def render(input_video, start, end, output_path, *args):
        if "плохой" in output_path:
            raise RuntimeError("FFmpeg завершился с кодом 1")
    prepare_render(monkeypatch, tmp_path, ["хороший", "плохой"])
    monkeypatch.setattr(main, "process_short_video", render)

    final = read_events(client.get("/api/render_stream"))[-1]
    assert final["done"] is True
    assert final["files"] == ["/output_shorts/хороший.mp4"]
    assert final["failed"] == ["плохой"]
    assert "1 из 2" in final["status"]


def test_render_stream_reports_error_when_all_fail(client, monkeypatch, tmp_path):
    def render(*args):
        raise RuntimeError("FFmpeg завершился с кодом 1")
    prepare_render(monkeypatch, tmp_path, ["плохой"])
    monkeypatch.setattr(main, "process_short_video", render)

    final = read_events(client.get("/api/render_stream"))[-1]
    assert final["error"] is True
    assert "кодом 1" in final["status"]
    assert final["failed"] == ["плохой"]


def test_lock_released_after_stream(client, monkeypatch, tmp_path):
    prepare_render(monkeypatch, tmp_path, [])
    client.get("/api/render_stream")
    assert not main.task_lock.locked()
