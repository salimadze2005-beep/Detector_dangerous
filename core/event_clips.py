from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import wave
import time
from datetime import timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from core.event_bus import Event, EventType
from core.frames import BufferedFrameSnapshot, FrameStore

logger = logging.getLogger(__name__)


class EventClipRecorder:
    """Save short annotated MP4 evidence clips around alert events.

    Gunshot/keyword clips use a symmetric pre/post window. Fall clips are
    special: they include context before lying started, the whole confirmation
    period, and a short period after the alarm is confirmed.
    """

    SCHEMA_VERSION = 1
    EVENT_FOLDERS = {
        EventType.FALL_DETECTED: "fall",
        EventType.GUNSHOT_DETECTED: "gunshot",
        EventType.KEYWORD_DETECTED: "keyword",
    }

    def __init__(
        self,
        frame_store: FrameStore,
        output_dir: str,
        pre_sec: float = 5.0,
        post_sec: float = 5.0,
        fall_pre_sec: float = 2.0,
        fall_post_sec: float = 2.0,
        fall_confirmation_sec: float = 5.0,
        fps: float = 5.0,
        jpeg_quality: int = 70,
        max_width: int = 960,
        max_height: int = 540,
        enabled: bool = True,
        on_clip_ready: Callable[[Event, str | None], None] | None = None,
    ) -> None:
        self.frame_store = frame_store
        self.output_dir = Path(output_dir).resolve()
        self.pre_sec = max(0.0, float(pre_sec))
        self.post_sec = max(0.0, float(post_sec))
        self.fall_pre_sec = max(0.0, float(fall_pre_sec))
        self.fall_post_sec = max(0.0, float(fall_post_sec))
        self.fall_confirmation_sec = max(0.0, float(fall_confirmation_sec))
        self.fps = max(1.0, float(fps))
        self.enabled = bool(enabled)
        self.on_clip_ready = on_clip_ready
        self._timers: set[threading.Timer] = set()
        self._lock = threading.Lock()
        self._closed = False

        # Keep enough history for the longest supported clip. Fall alarms are
        # emitted only after their confirmation delay, so their pre-event side
        # must retain: 2 s before lying + the whole confirmation period.
        regular_window = self.pre_sec + self.post_sec
        fall_window = self.fall_pre_sec + self.fall_confirmation_sec + self.fall_post_sec
        history_sec = max(regular_window, fall_window) + max(1.0, 2.0 / self.fps)
        self.frame_store.configure_history(
            history_sec=history_sec if self.enabled else 0.0,
            history_fps=self.fps,
            jpeg_quality=jpeg_quality,
            max_width=max_width,
            max_height=max_height,
        )
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def capture_event(self, event: Event) -> None:
        if not self.enabled or event.type not in self.EVENT_FOLDERS:
            return
        with self._lock:
            if self._closed:
                return

        event_monotonic = time.monotonic()
        camera_ids = self._camera_ids_for_event(event)
        if not camera_ids:
            logger.warning("Клип тревоги %s не записан: нет активной камеры", event.id)
            self._notify_clip_ready(event, None)
            return

        post_delay = self.fall_post_sec if event.type == EventType.FALL_DETECTED else self.post_sec
        timer = threading.Timer(
            post_delay,
            self._finalize_from_timer,
            args=(event, event_monotonic, camera_ids),
        )
        timer.daemon = True
        with self._lock:
            if self._closed:
                return
            self._timers.add(timer)
        timer.start()

    def _camera_ids_for_event(self, event: Event) -> list[str]:
        if event.type == EventType.FALL_DETECTED:
            camera_id = str(event.metadata.get("camera_id", "")).strip()
            if camera_id:
                return [camera_id]
        associated = [
            str(value).strip()
            for value in event.metadata.get("camera_ids", [])
            if str(value).strip()
        ]
        return sorted(set(associated)) if associated else self.frame_store.camera_ids()

    def _event_window(self, event: Event, event_monotonic: float) -> tuple[float, float, dict]:
        if event.type == EventType.FALL_DETECTED:
            # duration_sec is measured by FallDetector from fall_since until the
            # alarm is emitted. Prefer it so the clip includes the actual full
            # lying interval even if frame cadence makes confirmation 5.1 s.
            try:
                lying_sec = float(event.metadata.get("duration_sec", self.fall_confirmation_sec))
            except (TypeError, ValueError):
                lying_sec = self.fall_confirmation_sec
            lying_sec = max(self.fall_confirmation_sec, lying_sec)
            start_at = event_monotonic - lying_sec - self.fall_pre_sec
            end_at = event_monotonic + self.fall_post_sec
            timing = {
                "mode": "fall_confirmation",
                "pre_fall_sec": self.fall_pre_sec,
                "lying_before_confirmation_sec": round(lying_sec, 3),
                "post_confirmation_sec": self.fall_post_sec,
            }
            return start_at, end_at, timing

        return (
            event_monotonic - self.pre_sec,
            event_monotonic + self.post_sec,
            {
                "mode": "event_centered",
                "pre_event_sec": self.pre_sec,
                "post_event_sec": self.post_sec,
            },
        )

    def _finalize_from_timer(
        self,
        event: Event,
        event_monotonic: float,
        camera_ids: list[str],
    ) -> None:
        current = threading.current_thread()
        try:
            self._finalize(event, event_monotonic, camera_ids)
        except Exception:
            logger.exception("Не удалось сохранить клип тревоги %s", event.id)
            self._notify_clip_ready(event, None)
        finally:
            if isinstance(current, threading.Timer):
                with self._lock:
                    self._timers.discard(current)

    def _finalize(
        self,
        event: Event,
        event_monotonic: float,
        camera_ids: list[str],
    ) -> None:
        start_at, end_at, timing = self._event_window(event, event_monotonic)
        event_time = event.timestamp.astimezone(timezone.utc)
        category = self.EVENT_FOLDERS.get(event.type, "other")
        directory = (
            self.output_dir
            / category
            / f"{event_time:%Y}"
            / f"{event_time:%m}"
            / event.id
        )
        directory.mkdir(parents=True, exist_ok=True)

        clips = []
        clip_paths: list[Path] = []
        for camera_id in camera_ids:
            frames = self.frame_store.history_between(camera_id, start_at, end_at)
            if not frames:
                continue
            target = directory / f"{self._safe_name(camera_id)}.mp4"
            written = self._write_mp4(target, frames)
            if written <= 0:
                continue
            clip_start = frames[0].captured_at
            audio_attached = self._attach_audio(
                target,
                self.frame_store.audio_between(
                    camera_id, clip_start, clip_start + written / self.fps
                ),
                clip_start,
                written / self.fps,
            )
            clips.append(
                {
                    "camera_id": camera_id,
                    "source": frames[0].source,
                    "file": target.name,
                    "relative_path": str(target.relative_to(self.output_dir)).replace("\\", "/"),
                    "media_type": "video/mp4",
                    "codec": "mp4v",
                    "audio": audio_attached,
                    "fps": self.fps,
                    "frames": written,
                    "duration_sec": round(written / self.fps, 3),
                    "first_offset_sec": round(frames[0].captured_at - event_monotonic, 3),
                    "last_offset_sec": round(frames[-1].captured_at - event_monotonic, 3),
                    "bytes": target.stat().st_size,
                }
            )
            clip_paths.append(target)

        manifest = {
            "schema_version": self.SCHEMA_VERSION,
            "category": category,
            "event": event.to_dict(),
            "evidence": {
                "format": "mp4",
                "annotated_video": True,
                "timing": timing,
                "target_fps": self.fps,
                "clips": clips,
            },
        }
        manifest_path = directory / "event.json"
        temporary = directory / ".event.json.tmp"
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, manifest_path)
        logger.info(
            "Сохранено видео тревоги %s (%s): %d клип(ов), %s",
            event.id,
            category,
            len(clips),
            directory,
        )
        self._notify_clip_ready(
            event,
            str(clip_paths[0]) if clip_paths else None,
        )

    def _notify_clip_ready(self, event: Event, clip_path: str | None) -> None:
        callback = self.on_clip_ready
        if callback is None:
            return
        try:
            callback(event, clip_path)
        except Exception:
            logger.exception("Не удалось передать клип тревоги %s в REST-outbox", event.id)

    def _attach_audio(
        self,
        target: Path,
        chunks: list[tuple[float, float, int, bytes]],
        clip_start: float,
        duration_sec: float,
    ) -> bool:
        """Mux mono PCM captured over the exact displayed video interval."""
        if not chunks or duration_sec <= 0:
            return False
        sample_rate = int(chunks[0][2])
        if sample_rate <= 0 or any(int(chunk[2]) != sample_rate for chunk in chunks):
            logger.warning("Audio for incident clip %s has incompatible sample rates", target)
            return False
        total_samples = max(1, round(duration_sec * sample_rate))
        mix = np.zeros(total_samples, dtype=np.int16)
        clip_end = clip_start + duration_sec
        for started, ended, rate, pcm in chunks:
            samples = np.frombuffer(pcm, dtype="<i2")
            if not samples.size or ended <= clip_start or started >= clip_end:
                continue
            first = max(started, clip_start)
            last = min(ended, clip_end)
            source_start = max(0, round((first - started) * rate))
            target_start = max(0, round((first - clip_start) * sample_rate))
            count = min(
                len(samples) - source_start,
                total_samples - target_start,
                max(0, round((last - first) * sample_rate)),
            )
            if count > 0:
                mix[target_start:target_start + count] = samples[source_start:source_start + count]
        # Archive clips must be audible in an ordinary media player.  This only
        # changes the saved evidence copy, never the detector input.
        peak = int(np.max(np.abs(mix.astype(np.int32)))) if mix.size else 0
        if peak > 0:
            gain = min(64.0, 12_000.0 / peak)
            mix = np.clip(mix.astype(np.float32) * gain, -32767, 32767).astype(np.int16)
        wav_path = target.with_name(f".{target.stem}.audio.tmp.wav")
        muxed_path = target.with_name(f".{target.stem}.mux.tmp.mp4")
        try:
            with wave.open(str(wav_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(mix.astype("<i2").tobytes())
            from imageio_ffmpeg import get_ffmpeg_exe

            completed = subprocess.run(
                [
                    get_ffmpeg_exe(), "-y", "-loglevel", "error",
                    "-i", str(target), "-i", str(wav_path),
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "96k",
                    "-shortest", str(muxed_path),
                ],
                capture_output=True, text=True, timeout=30, check=False,
            )
            if completed.returncode != 0 or not muxed_path.is_file():
                logger.warning(
                    "Could not add synchronized audio to %s: %s",
                    target, completed.stderr.strip()[:500],
                )
                return False
            os.replace(muxed_path, target)
            return True
        except Exception:
            logger.exception("Could not add synchronized audio to incident clip %s", target)
            return False
        finally:
            wav_path.unlink(missing_ok=True)
            muxed_path.unlink(missing_ok=True)

    def _write_mp4(self, target: Path, frames: list[BufferedFrameSnapshot]) -> int:
        decoded = []
        for item in frames:
            frame = item.decode()
            if frame is not None:
                decoded.append(frame)
        if not decoded:
            return 0

        height, width = decoded[0].shape[:2]
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.stem}.tmp.mp4")
        writer = cv2.VideoWriter(
            str(temporary),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (width, height),
        )
        if not writer.isOpened():
            logger.error("OpenCV не смог открыть MP4 writer для %s", target)
            return 0
        written = 0
        try:
            for frame in decoded:
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(frame)
                written += 1
        finally:
            writer.release()
        if written:
            os.replace(temporary, target)
        else:
            temporary.unlink(missing_ok=True)
        return written

    @staticmethod
    def _safe_name(value: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)
        return safe[:80] or "camera"

    def close(self, timeout: float = 5.0) -> None:
        with self._lock:
            self._closed = True
            timers = list(self._timers)
        deadline = time.monotonic() + max(0.0, timeout)
        for timer in timers:
            remaining = max(0.0, deadline - time.monotonic())
            timer.join(remaining)
        with self._lock:
            self._timers = {timer for timer in self._timers if timer.is_alive()}
