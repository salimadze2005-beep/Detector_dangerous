from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class FrameSnapshot:
    camera_id: str
    source: str
    captured_at: float
    frame: np.ndarray


@dataclass(frozen=True, slots=True)
class BufferedFrameSnapshot:
    """JPEG-compressed frame kept only for short pre/post-event evidence clips."""

    camera_id: str
    source: str
    captured_at: float
    jpeg: bytes

    def decode(self) -> np.ndarray | None:
        encoded = np.frombuffer(self.jpeg, dtype=np.uint8)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


class FrameStore:
    """Thread-safe latest-frame registry with an optional compressed history ring."""

    def __init__(self) -> None:
        self._frames: dict[str, FrameSnapshot] = {}
        self._history: dict[str, deque[BufferedFrameSnapshot]] = {}
        self._last_history_at: dict[str, float] = {}
        self._audio_history: dict[str, deque[tuple[float, float, int, bytes]]] = {}
        self._history_sec = 0.0
        self._history_fps = 0.0
        self._history_jpeg_quality = 70
        self._history_max_width = 960
        self._history_max_height = 540
        self._lock = threading.Lock()

    def configure_history(
        self,
        history_sec: float,
        history_fps: float = 5.0,
        jpeg_quality: int = 70,
        max_width: int = 960,
        max_height: int = 540,
    ) -> None:
        """Configure a sampled JPEG history used for incident clips."""
        history_sec = max(0.0, float(history_sec))
        history_fps = max(0.0, float(history_fps))
        with self._lock:
            self._history_sec = history_sec
            self._history_fps = history_fps
            self._history_jpeg_quality = int(min(max(jpeg_quality, 30), 95))
            self._history_max_width = max(1, int(max_width))
            self._history_max_height = max(1, int(max_height))
            self._history.clear()
            self._last_history_at.clear()
            self._audio_history.clear()

    def update(
        self,
        camera_id: str,
        source: str,
        frame: np.ndarray,
        captured_at: float | None = None,
    ) -> None:
        captured = time.monotonic() if captured_at is None else float(captured_at)
        array = np.asarray(frame)
        snapshot = FrameSnapshot(camera_id, source, captured, array.copy())

        should_buffer = False
        quality = 70
        max_width = 960
        max_height = 540
        with self._lock:
            self._frames[camera_id] = snapshot
            if self._history_sec > 0 and self._history_fps > 0:
                interval = 1.0 / self._history_fps
                previous = self._last_history_at.get(camera_id)
                if previous is None or captured - previous >= interval:
                    self._last_history_at[camera_id] = captured
                    should_buffer = True
                    quality = self._history_jpeg_quality
                    max_width = self._history_max_width
                    max_height = self._history_max_height

        if not should_buffer:
            return

        history_frame = self._resize_for_history(array, max_width, max_height)
        success, encoded = cv2.imencode(
            ".jpg", history_frame, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        if not success:
            return
        buffered = BufferedFrameSnapshot(camera_id, source, captured, encoded.tobytes())
        with self._lock:
            if self._history_sec <= 0:
                return
            history = self._history.setdefault(camera_id, deque())
            history.append(buffered)
            cutoff = captured - self._history_sec
            while history and history[0].captured_at < cutoff:
                history.popleft()

    @staticmethod
    def _resize_for_history(frame: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
        height, width = frame.shape[:2]
        scale = min(1.0, max_width / width, max_height / height)
        if scale >= 1.0:
            return frame
        return cv2.resize(
            frame,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    def fresh(self, max_age_sec: float, camera_id: str | None = None) -> list[FrameSnapshot]:
        now = time.monotonic()
        with self._lock:
            snapshots = list(self._frames.values())
        result = []
        for snapshot in snapshots:
            if camera_id is not None and snapshot.camera_id != camera_id:
                continue
            if now - snapshot.captured_at <= max_age_sec:
                result.append(
                    FrameSnapshot(
                        snapshot.camera_id,
                        snapshot.source,
                        snapshot.captured_at,
                        snapshot.frame.copy(),
                    )
                )
        return sorted(result, key=lambda item: item.camera_id)

    def history_between(
        self,
        camera_id: str,
        start_at: float,
        end_at: float,
    ) -> list[BufferedFrameSnapshot]:
        """Return compressed frames captured inside a monotonic-time interval."""
        with self._lock:
            history = list(self._history.get(camera_id, ()))
        return [
            BufferedFrameSnapshot(
                item.camera_id,
                item.source,
                item.captured_at,
                bytes(item.jpeg),
            )
            for item in history
            if start_at <= item.captured_at <= end_at
        ]

    def update_audio(
        self,
        camera_ids: tuple[str, ...],
        audio: np.ndarray,
        sample_rate: int,
        captured_at: float | None = None,
    ) -> None:
        """Keep short PCM audio history on the same monotonic clock as frames."""
        rate = max(1, int(sample_rate))
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not samples.size:
            return
        captured = time.monotonic() if captured_at is None else float(captured_at)
        started = captured - (len(samples) / rate)
        pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        targets = tuple(str(item) for item in camera_ids if str(item)) or ("__default__",)
        with self._lock:
            if self._history_sec <= 0:
                return
            cutoff = captured - self._history_sec
            for camera_id in targets:
                history = self._audio_history.setdefault(camera_id, deque())
                history.append((started, captured, rate, pcm))
                while history and history[0][1] < cutoff:
                    history.popleft()

    def audio_between(
        self, camera_id: str, start_at: float, end_at: float
    ) -> list[tuple[float, float, int, bytes]]:
        """Return audio chunks intersecting the requested video time range."""
        with self._lock:
            chunks = list(self._audio_history.get(camera_id, ()))
            if not chunks:
                chunks = list(self._audio_history.get("__default__", ()))
        return [
            (float(started), float(ended), int(rate), bytes(pcm))
            for started, ended, rate, pcm in chunks
            if ended > start_at and started < end_at
        ]

    def camera_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._frames)

    def history_memory_bytes(self) -> int:
        with self._lock:
            return (
                sum(len(item.jpeg) for history in self._history.values() for item in history)
                + sum(len(item[3]) for history in self._audio_history.values() for item in history)
            )

    def remove(self, camera_id: str) -> None:
        with self._lock:
            self._frames.pop(camera_id, None)
            self._history.pop(camera_id, None)
            self._audio_history.pop(camera_id, None)
            self._last_history_at.pop(camera_id, None)


def build_mosaic(snapshots: list[FrameSnapshot]) -> np.ndarray | None:
    """Build one API image from all fresh cameras without distorting frames."""
    if not snapshots:
        return None
    if len(snapshots) == 1:
        return snapshots[0].frame.copy()

    columns = math.ceil(math.sqrt(len(snapshots)))
    rows = math.ceil(len(snapshots) / columns)
    cell_width = max(frame.frame.shape[1] for frame in snapshots)
    cell_height = max(frame.frame.shape[0] for frame in snapshots)
    canvas = np.zeros((rows * cell_height, columns * cell_width, 3), dtype=np.uint8)

    for index, snapshot in enumerate(snapshots):
        frame = snapshot.frame
        scale = min(cell_width / frame.shape[1], cell_height / frame.shape[0])
        width = max(1, int(frame.shape[1] * scale))
        height = max(1, int(frame.shape[0] * scale))
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        row, column = divmod(index, columns)
        x = column * cell_width + (cell_width - width) // 2
        y = row * cell_height + (cell_height - height) // 2
        canvas[y:y + height, x:x + width] = resized
        cv2.putText(
            canvas,
            snapshot.camera_id,
            (column * cell_width + 12, row * cell_height + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas
