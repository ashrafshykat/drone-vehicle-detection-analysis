# Smart Drone Traffic Analyzer

Full-stack proof-of-concept for drone traffic analysis using YOLO + ByteTrack with strict anti-double-counting logic.

## Tech Stack

- Backend: FastAPI
- Frontend: React + Tailwind (Vite)
- CV/AI: Ultralytics YOLO with ByteTrack (`persist=True`, `tracker="bytetrack.yaml"`)
- Database: PostgreSQL
- Reporting: Pandas + Openpyxl (`.xlsx`)

## Project Structure

```text
backend/
  main.py
  video_processor.py
  requirements.txt
  Dockerfile
  .env.example
  data/
frontend/
  src/
  package.json
  Dockerfile
models/
  *.pt
docker-compose.yml
README.md
```

## Setup

### 1) Place models

Put one or more YOLO models in:

`models/`

Examples:

- `models/model.pt`
- `models/yolo_model_v1.pt`
- `models/yolo_model_v2.pt`

The UI fetches available models from backend and lets you choose one before processing.

### 2) PostgreSQL

Create DB (example):

- Host: `localhost`
- Port: `5432`
- DB: `ants_traffic`
- User: `postgres`
- Password: `postgres`

### 3) Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Configuration uses a `.env` file in `backend/` (loaded automatically on startup). Copy the template and edit:

```bash
copy .env.example .env
```

Example `backend/.env`:

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ants_traffic
MODEL_PATH=E:\ANTS\models\model.pt
COUNT_LINE_ORIENTATION=horizontal
COUNT_LINE_VALUE=350
COUNT_LINE_DEAD_ZONE=6
```

`MODEL_PATH` is a fallback/default model. During upload, user-selected `model_name` from `models/` takes priority.

Run backend:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4) Frontend

```bash
cd frontend
npm install
```

Optional API base URL:

```powershell
$env:VITE_API_BASE="http://localhost:8000"
```

Run frontend:

```bash
npm run dev
```

## API Endpoints

- `GET /models` - lists all available `.pt` files in `models/`.
- `POST /upload` - uploads `.mp4` and starts background processing.
- `WS /status` - pushes live progress, per-class counts, total unique count, and completion/error state.
- `GET /download-report` - downloads generated `.xlsx` report.
- `GET /processed-video` - returns annotated output video with boxes, IDs, line, and overlay counters.

`/upload` accepts:

- `file`: video `.mp4`
- `model_name`: selected model filename (from `/models`)

## Docker (Reproducible Run)

Run all services (PostgreSQL + FastAPI + React static frontend):

```bash
docker compose up --build
```

Then open:

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000`

Notes:

- Place your model files in local `models/` before starting compose.
- Backend container reads models from mounted `/models`.
- If `MODEL_PATH` is unset/invalid, backend uses first available `.pt` in `models/`.

## Tracking & Counting Methodology

### ByteTrack + Persistent IDs

Each frame is processed with:

`model.track(source=frame, persist=True, tracker="bytetrack.yaml")`

`persist=True` keeps tracker state (Kalman-filter based association) across frames for temporary occlusions and short disappearances.

### Counting Line Geometry

`CountingLine` defines:

- Orientation: `horizontal` (`y = value`) or `vertical` (`x = value`)
- `dead_zone_px`: small tolerance around line to suppress jitter

For each tracked centroid:

1. Compute side of line: `-1`, `0`, `+1`
2. Ignore side `0` (dead-zone)
3. Count only when side transitions `-1 -> +1` or `+1 -> -1`

### No Double Counting

- `GlobalCountedSet` stores counted `track_id`s
- If a `track_id` already exists in the set, it is never counted again
- Handles edge cases where vehicles slow down, stop on the line, or oscillate near line

## Report Contents (`.xlsx`)

Each counted crossing includes:

- `track_id`
- `vehicle_class`
- `crossed_at_utc`
- `direction`
- `frame_index`
- `video_seconds`

## Engineering Assumptions

- One processing job at a time (single-worker PoC behavior).
- Database stores only successful crossing events (not every raw detection).
- Counting line is configured through `backend/.env` and is global per run.
- Input format is `.mp4`.
