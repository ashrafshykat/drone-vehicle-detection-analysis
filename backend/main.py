from __future__ import annotations

import asyncio
import aiofiles
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from video_processor import CountingLine, VideoProcessor


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR.parent / "models"
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
REPORTS_DIR = DATA_DIR / "reports"
OUTPUTS_DIR = DATA_DIR / "outputs"

for directory in (UPLOADS_DIR, REPORTS_DIR, OUTPUTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / ".env")

MODEL_PATH = os.getenv("MODEL_PATH", str((BASE_DIR.parent / "models" / "model.pt").resolve()))
DB_DSN = os.getenv("DATABASE_URL", "postgresql://postgres:1521@localhost:5432/ants_traffic")
COUNT_LINE_ORIENTATION = os.getenv("COUNT_LINE_ORIENTATION", "horizontal")
COUNT_LINE_VALUE = int(os.getenv("COUNT_LINE_VALUE", "350"))
COUNT_LINE_DEAD_ZONE = int(os.getenv("COUNT_LINE_DEAD_ZONE", "6"))


app = FastAPI(title="Smart Drone Traffic Analyzer")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AppState:
    def __init__(self) -> None:
        self.is_processing = False
        self.job_id: str | None = None
        self.last_status: dict[str, Any] = {
            "progress_percent": 0.0,
            "counts": {},
            "total_unique_count": 0,
            "frame_index": 0,
            "done": False,
            "error": None,
        }
        self.report_path: Path | None = None
        self.output_video_path: Path | None = None
        self.clients: set[WebSocket] = set()


state = AppState()
main_loop: asyncio.AbstractEventLoop | None = None


@app.on_event("startup")
async def startup_event() -> None:
    global main_loop
    main_loop = asyncio.get_running_loop()


async def _broadcast_status(payload: dict[str, Any]) -> None:
    disconnected: list[WebSocket] = []
    for ws in list(state.clients):
        try:
            await ws.send_json(payload)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        state.clients.discard(ws)


def _status_callback(payload: dict[str, Any]) -> None:
    state.last_status.update(payload)
    if main_loop is not None:
        asyncio.run_coroutine_threadsafe(_broadcast_status(dict(state.last_status)), main_loop)


def _available_models() -> list[Path]:
    if not MODELS_DIR.exists():
        return []
    return sorted([p for p in MODELS_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".pt"])


def _resolve_model_path(model_name: str | None) -> Path:
    if model_name:
        candidate = (MODELS_DIR / model_name).resolve()
        if candidate.parent != MODELS_DIR.resolve() or not candidate.exists() or candidate.suffix.lower() != ".pt":
            raise HTTPException(status_code=400, detail=f"Invalid model selected: {model_name}")
        return candidate

    fallback = Path(MODEL_PATH)
    if fallback.exists():
        return fallback

    models = _available_models()
    if models:
        return models[0]
    raise HTTPException(status_code=500, detail="No model found. Add at least one .pt file in models/.")


def _process_job(video_path: Path, report_path: Path, output_video_path: Path, model_path: Path) -> None:
    try:
        line = CountingLine(
            orientation=COUNT_LINE_ORIENTATION,
            value=COUNT_LINE_VALUE,
            dead_zone_px=COUNT_LINE_DEAD_ZONE,
        )
        processor = VideoProcessor(
            model_path=model_path,
            db_dsn=DB_DSN,
            counting_line=line,
            status_callback=_status_callback,
        )
        result = processor.process_video(
            video_path=video_path,
            report_path=report_path,
            output_video_path=output_video_path,
        )
        state.report_path = Path(result["report_path"])
        out_path = result.get("output_video_path")
        state.output_video_path = Path(out_path) if out_path else None
        state.last_status.update({"done": True, "error": None, **result, "is_processing": False})
    except Exception as exc:
        state.last_status.update({"done": True, "error": str(exc), "is_processing": False})
    finally:
        state.is_processing = False
        if main_loop is not None:
            asyncio.run_coroutine_threadsafe(_broadcast_status(dict(state.last_status)), main_loop)


@app.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_name: str | None = Form(default=None),
) -> JSONResponse:
    if not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only .mp4 uploads are supported.")

    if state.is_processing:
        raise HTTPException(status_code=409, detail="A processing job is already running.")

    selected_model_path = _resolve_model_path(model_name)
    job_id = str(uuid4())
    upload_path = UPLOADS_DIR / f"{job_id}_{file.filename}"
    report_path = REPORTS_DIR / f"{job_id}_traffic_report.xlsx"
    output_video_path = OUTPUTS_DIR / f"{job_id}_annotated.mp4"

    try:
        async with aiofiles.open(upload_path, "wb") as out_f:
            while chunk := await file.read(1024 * 1024): 
                await out_f.write(chunk)
    except Exception as e:
        # Reset state if upload fails
        state.is_processing = False
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
    state.is_processing = True
    state.job_id = job_id
    state.report_path = report_path
    state.output_video_path = output_video_path
    state.last_status = {
        "progress_percent": 0.0,
        "counts": {},
        "total_unique_count": 0,
        "frame_index": 0,
        "done": False,
        "error": None,
        "is_processing": True,
        "job_id": job_id,
        "model_name": selected_model_path.name,
    }
    await _broadcast_status(dict(state.last_status))
    background_tasks.add_task(_process_job, upload_path, report_path, output_video_path, selected_model_path)
    return JSONResponse(
        {
            "message": "Upload accepted. Processing started.",
            "job_id": job_id,
            "model_name": selected_model_path.name,
        }
    )


@app.get("/models")
async def list_models() -> JSONResponse:
    models = _available_models()
    model_names = [p.name for p in models]
    default_name = Path(MODEL_PATH).name if Path(MODEL_PATH).exists() else (model_names[0] if model_names else None)
    return JSONResponse({"models": model_names, "default": default_name})


@app.websocket("/status")
async def status_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    state.clients.add(websocket)
    await websocket.send_json(dict(state.last_status))
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        state.clients.discard(websocket)


@app.get("/download-report")
async def download_report() -> FileResponse:
    if state.is_processing:
        raise HTTPException(status_code=409, detail="Processing still running.")
    if state.report_path is None or not state.report_path.exists():
        raise HTTPException(status_code=404, detail="Report not available.")
    return FileResponse(
        path=state.report_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=state.report_path.name,
    )


@app.get("/processed-video")
async def processed_video() -> FileResponse:
    if state.is_processing:
        raise HTTPException(status_code=409, detail="Processing still running.")
    if state.output_video_path is None or not state.output_video_path.exists():
        raise HTTPException(status_code=404, detail="Processed video not available.")
    return FileResponse(path=state.output_video_path, media_type="video/mp4", filename=state.output_video_path.name)