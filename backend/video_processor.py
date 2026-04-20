from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import subprocess
from typing import Any, Callable

import cv2
import pandas as pd
import psycopg2
from ultralytics import YOLO


@dataclass
class CountingLine:
    """
    Defines a single counting line.

    orientation:
      - "horizontal" => y = value
      - "vertical" => x = value
    """

    orientation: str
    value: int
    dead_zone_px: int = 6

    def side_of_point(self, x: float, y: float) -> int:
        """
        Returns:
          -1 for one side of line
           0 for dead zone around line
          +1 for opposite side of line
        """
        if self.orientation == "vertical":
            distance = x - float(self.value)
        elif self.orientation == "horizontal":
            distance = y - float(self.value)
        else:
            raise ValueError("orientation must be 'horizontal' or 'vertical'")

        if abs(distance) <= self.dead_zone_px:
            return 0
        return 1 if distance > 0 else -1


@dataclass
class TrackState:
    last_nonzero_side: int = 0
    last_seen_frame: int = 0


class VideoProcessor:
    """
    Smart Drone Traffic Analyzer core processor.

    Responsibilities:
      1) Run YOLO + ByteTrack per frame.
      2) Maintain track state for robust line-crossing decisions.
      3) Enforce single-count policy with GlobalCountedSet.
      4) Log counted crossings to PostgreSQL.
      5) Export report rows to XLSX.
    """

    def __init__(
        self,
        model_path: str | Path,
        db_dsn: str,
        counting_line: CountingLine,
        tracker_config: str = "bytetrack.yaml",
        status_callback: Callable[[dict[str, Any]], None] | None = None,
        max_inactive_frames: int = 90,
    ) -> None:
        self.model = YOLO(str(model_path))
        self.db_dsn = db_dsn
        self.counting_line = counting_line
        self.tracker_config = tracker_config
        self.status_callback = status_callback
        self.max_inactive_frames = max_inactive_frames

        # GlobalCountedSet: once a track_id is counted, it is never counted again.
        self.global_counted_set: set[int] = set()
        self.track_states: dict[int, TrackState] = {}
        self.class_counts: dict[str, int] = {}
        self.report_rows: list[dict[str, Any]] = []

    def process_video(
        self,
        video_path: str | Path,
        report_path: str | Path,
        output_video_path: str | Path | None = None,
    ) -> dict[str, Any]:
        video_path = Path(video_path)
        report_path = Path(report_path)
        output_video = Path(output_video_path) if output_video_path else None

        conn = psycopg2.connect(self.db_dsn)
        conn.autocommit = True
        self._ensure_table(conn)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            conn.close()
            raise RuntimeError(f"Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        frame_idx = 0
        writer: cv2.VideoWriter | None = None

        if output_video:
            output_video.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(output_video),
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (frame_w, frame_h),
            )

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_idx += 1
                annotated_frame = self._process_frame(frame=frame, frame_idx=frame_idx, fps=fps, conn=conn)

                progress = (frame_idx / total_frames * 100.0) if total_frames > 0 else 0.0
                if self.status_callback:
                    self.status_callback(
                        {
                            "progress_percent": round(progress, 2),
                            "counts": dict(self.class_counts),
                            "total_unique_count": len(self.global_counted_set),
                            "frame_index": frame_idx,
                            "done": False,
                        }
                    )

                if writer is not None and annotated_frame is not None:
                    writer.write(annotated_frame)

                self._cleanup_stale_tracks(frame_idx)
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            conn.close()

        self._write_report(report_path)
        if output_video and output_video.exists():
            self._make_video_web_playable(output_video)
        final_payload = {
            "counts": dict(self.class_counts),
            "total_unique_count": len(self.global_counted_set),
            "report_path": str(report_path),
            "output_video_path": str(output_video) if output_video else None,
        }
        if self.status_callback:
            self.status_callback(
                {
                    "progress_percent": 100.0,
                    "counts": dict(self.class_counts),
                    "total_unique_count": len(self.global_counted_set),
                    "frame_index": frame_idx,
                    "done": True,
                }
            )
        return final_payload

    def _make_video_web_playable(self, output_video: Path) -> None:
        """
        Convert OpenCV output to browser-friendly H.264/AAC-compatible MP4.
        If ffmpeg is unavailable, keep the original file.
        """
        temp_output = output_video.with_name(f"{output_video.stem}_web.mp4")
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(output_video),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temp_output),
        ]
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            temp_output.replace(output_video)
        except (FileNotFoundError, subprocess.CalledProcessError):
            if temp_output.exists():
                temp_output.unlink(missing_ok=True)

    def _process_frame(self, frame: Any, frame_idx: int, fps: float, conn: Any) -> Any:
        """
        Runs detector + tracker on a single frame and applies crossing logic.

        NOTE: `persist=True` is essential for keeping ByteTrack state frame-to-frame.
        """
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            verbose=False,
        )

        if not results:
            return frame

        result = results[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            self._draw_counting_line(frame)
            return frame

        track_ids = boxes.id.int().cpu().tolist()
        classes = boxes.cls.int().cpu().tolist() if boxes.cls is not None else []
        xyxy_list = boxes.xyxy.cpu().tolist() if boxes.xyxy is not None else []
        names = result.names if isinstance(result.names, dict) else {}

        for idx, track_id in enumerate(track_ids):
            if idx >= len(xyxy_list):
                continue

            x1, y1, x2, y2 = xyxy_list[idx]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            crossed, direction = self._did_cross_counting_line(track_id=track_id, cx=cx, cy=cy, frame_idx=frame_idx)
            cls_id = classes[idx] if idx < len(classes) else -1
            cls_name = names.get(cls_id, f"class_{cls_id}")
            self._draw_detection(
                frame,
                track_id=track_id,
                cls_name=cls_name,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                counted_now=False,
            )
            if not crossed:
                continue

            crossed_at = datetime.now(timezone.utc)
            video_seconds = frame_idx / fps if fps > 0 else 0.0

            self.class_counts[cls_name] = self.class_counts.get(cls_name, 0) + 1
            self.global_counted_set.add(track_id)

            self._log_crossing_to_db(
                conn=conn,
                track_id=track_id,
                vehicle_class=cls_name,
                crossed_at=crossed_at,
                direction=direction,
                frame_index=frame_idx,
                video_seconds=video_seconds,
            )
            self.report_rows.append(
                {
                    "track_id": track_id,
                    "vehicle_class": cls_name,
                    "crossed_at_utc": crossed_at.isoformat(),
                    "direction": direction,
                    "frame_index": frame_idx,
                    "video_seconds": round(video_seconds, 3),
                }
            )
            self._draw_detection(frame, track_id, cls_name, x1, y1, x2, y2, counted_now=True)

        self._draw_counting_line(frame)
        self._draw_overlay_counts(frame, frame_idx)
        return frame

    def _draw_counting_line(self, frame: Any) -> None:
        h, w = frame.shape[:2]
        color = (0, 255, 255)
        if self.counting_line.orientation == "vertical":
            x = int(self.counting_line.value)
            cv2.line(frame, (x, 0), (x, h), color, 2)
        else:
            y = int(self.counting_line.value)
            cv2.line(frame, (0, y), (w, y), color, 2)

    def _draw_detection(
        self,
        frame: Any,
        track_id: int,
        cls_name: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        counted_now: bool = False,
    ) -> None:
        box_color = (0, 255, 0) if counted_now else (255, 255, 0)
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, box_color, 2)
        label = f"{cls_name} | ID:{track_id}"
        cv2.putText(frame, label, (p1[0], max(18, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

    def _draw_overlay_counts(self, frame: Any, frame_idx: int) -> None:
        cv2.putText(
            frame,
            f"Frame: {frame_idx}  Unique: {len(self.global_counted_set)}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        y = 56
        for cls_name, value in sorted(self.class_counts.items()):
            cv2.putText(frame, f"{cls_name}: {value}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += 24

    def _did_cross_counting_line(self, track_id: int, cx: float, cy: float, frame_idx: int) -> tuple[bool, str]:
        """
        Crossing logic with anti-double-count guarantees:
          - Uses side transition across the counting line: (-1 -> +1) or (+1 -> -1)
          - Ignores centroid in dead-zone to reduce jitter around the line
          - Uses GlobalCountedSet so each track_id is counted once globally
        """
        if track_id in self.global_counted_set:
            return False, ""

        current_side = self.counting_line.side_of_point(cx, cy)
        state = self.track_states.get(track_id, TrackState(last_nonzero_side=0, last_seen_frame=frame_idx))
        state.last_seen_frame = frame_idx

        if current_side == 0:
            # Inside dead-zone: do not update side to avoid oscillation false positives.
            self.track_states[track_id] = state
            return False, ""

        if state.last_nonzero_side == 0:
            # First confident side observation for this track.
            state.last_nonzero_side = current_side
            self.track_states[track_id] = state
            return False, ""

        if current_side != state.last_nonzero_side:
            direction = self._direction_label(from_side=state.last_nonzero_side, to_side=current_side)
            # Update state before returning; GlobalCountedSet update happens in caller.
            state.last_nonzero_side = current_side
            self.track_states[track_id] = state
            return True, direction

        state.last_nonzero_side = current_side
        self.track_states[track_id] = state
        return False, ""

    def _direction_label(self, from_side: int, to_side: int) -> str:
        if self.counting_line.orientation == "vertical":
            if from_side < to_side:
                return "left_to_right"
            return "right_to_left"
        if from_side < to_side:
            return "top_to_bottom"
        return "bottom_to_top"

    def _cleanup_stale_tracks(self, current_frame_idx: int) -> None:
        stale_ids = [
            tid
            for tid, state in self.track_states.items()
            if current_frame_idx - state.last_seen_frame > self.max_inactive_frames
        ]
        for tid in stale_ids:
            self.track_states.pop(tid, None)

    def _ensure_table(self, conn: Any) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS vehicle_crossings (
                    id BIGSERIAL PRIMARY KEY,
                    track_id BIGINT NOT NULL,
                    vehicle_class TEXT NOT NULL,
                    crossed_at_utc TIMESTAMPTZ NOT NULL,
                    direction TEXT NOT NULL,
                    frame_index INTEGER NOT NULL,
                    video_seconds DOUBLE PRECISION NOT NULL
                );
                """
            )

    def _log_crossing_to_db(
        self,
        conn: Any,
        track_id: int,
        vehicle_class: str,
        crossed_at: datetime,
        direction: str,
        frame_index: int,
        video_seconds: float,
    ) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO vehicle_crossings
                    (track_id, vehicle_class, crossed_at_utc, direction, frame_index, video_seconds)
                VALUES (%s, %s, %s, %s, %s, %s);
                """,
                (track_id, vehicle_class, crossed_at, direction, frame_index, video_seconds),
            )

    def _write_report(self, report_path: Path) -> None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            self.report_rows,
            columns=[
                "track_id",
                "vehicle_class",
                "crossed_at_utc",
                "direction",
                "frame_index",
                "video_seconds",
            ],
        )
        df.to_excel(report_path, index=False, engine="openpyxl")
