from __future__ import annotations

import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit, urlunsplit

import cv2
import requests

from core.config import AlertsConfig
from core.event_bus import Event, EventType
from core.frames import FrameStore, build_mosaic
from core.rest_video import encode_delivery_video
from core.storage import EventStore, OutboxItem

logger = logging.getLogger(__name__)

try:
    import winsound
except ImportError:  # pragma: no cover - Windows is the supported desktop target
    winsound = None


ALERT_EVENT_TYPES = {
    EventType.GUNSHOT_DETECTED,
    EventType.KEYWORD_DETECTED,
    EventType.FALL_DETECTED,
}


def describe_event(event: Event, image_available: bool = True) -> str:
    confidence = f"Уверенность: {event.confidence:.0%}."
    if event.type == EventType.GUNSHOT_DETECTED:
        microphone = str(event.metadata.get("microphone", "")).strip()
        details = f" Микрофон: {microphone}." if microphone else ""
        description = f"Обнаружен возможный выстрел. {confidence}{details}"
    elif event.type == EventType.KEYWORD_DETECTED:
        keyword = str(event.metadata.get("keyword", "")).strip()
        text = str(event.metadata.get("text", "")).strip()
        details = f" Ключевое слово: «{keyword}»." if keyword else ""
        if text:
            details += f" Распознанная фраза: «{text[:500]}»."
        description = f"Обнаружено тревожное ключевое слово. {confidence}{details}"
    elif event.type == EventType.FALL_DETECTED:
        camera_id = str(event.metadata.get("camera_id", "")).strip()
        duration = event.metadata.get("duration_sec")
        details = f" Камера: {camera_id}." if camera_id else ""
        if duration is not None:
            details += f" Горизонтальная поза: {float(duration):.1f} с."
        description = f"Обнаружено возможное падение человека. {confidence}{details}"
    else:
        description = f"Обнаружено событие {event.type.value}. {confidence}"
    if not image_available:
        description += " Снимок камеры недоступен."
    return description


class AlertService:
    """Persists incidents, plays local sounds and delivers a durable HTTP outbox."""

    def __init__(
        self,
        config: AlertsConfig,
        event_store: EventStore | None = None,
        frame_store: FrameStore | None = None,
        screenshots_dir: str | None = None,
        config_persistor: Callable[[], None] | None = None,
    ):
        self.config = config
        self.event_store = event_store
        self.frame_store = frame_store or FrameStore()
        self.screenshots_dir = Path(screenshots_dir or "data/screenshots").resolve()
        self.rest_videos_dir = self.screenshots_dir.parent / "rest_videos"
        self._config_persistor = config_persistor
        self._delivery_thread: threading.Thread | None = None
        self._running = threading.Event()
        self._wake_delivery = threading.Event()

    def start(self) -> None:
        if self.event_store is None:
            return
        if self._delivery_thread and self._delivery_thread.is_alive():
            return
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        self.rest_videos_dir.mkdir(parents=True, exist_ok=True)
        self._running.set()
        self._delivery_thread = threading.Thread(
            target=self._delivery_loop, name="alert-delivery", daemon=True
        )
        self._delivery_thread.start()

    def handle_event(self, event: Event) -> None:
        if event.type not in ALERT_EVENT_TYPES:
            return
        if self.event_store is None:
            self._play_local(event)
            if self.config.webhook_url:
                self._send_webhook(event)
            return
        if not self._running.is_set():
            self.start()
        image_path = self._save_snapshot(event)
        description = describe_event(event, image_available=image_path is not None)
        awaiting_video = bool(self.config.video_delivery_enabled)
        if self.event_store.enqueue_alert(
            event,
            description,
            image_path,
            await_video=awaiting_video,
        ) and not awaiting_video:
            self._wake_delivery.set()

    def prepare_video_delivery(self, event: Event, clip_path: str | None) -> None:
        """Convert the recorder output and release the durable outbox item."""
        if self.event_store is None or not self.config.video_delivery_enabled:
            return
        current = self.event_store.get_alert(event.id)
        if current is None or current.status != "preparing":
            return
        if not clip_path:
            message = "REST-video не подготовлено: в буфере камеры нет кадров"
            self.event_store.fail_video_preparation(event.id, message)
            logger.error("%s (%s)", message, event.id)
            return
        try:
            prepared = encode_delivery_video(
                clip_path,
                self.rest_videos_dir,
                fps=self.config.delivery_video_fps,
                max_bytes=self.config.delivery_video_max_bytes,
                initial_crf=self.config.delivery_video_initial_crf,
            )
        except Exception as exc:
            message = f"REST-video не подготовлено: {exc}"
            self.event_store.fail_video_preparation(event.id, message)
            logger.exception("Не удалось подготовить REST-video тревоги %s", event.id)
            return
        if self.event_store.attach_delivery_video(event.id, prepared.path):
            logger.info(
                "REST-video тревоги %s готово к доставке: %s",
                event.id,
                Path(prepared.path).name,
            )
            self._wake_delivery.set()

    def _send_webhook(self, event: Event) -> None:
        """Compatibility path for integrations without a durable outbox."""
        headers = {}
        if self.config.webhook_token:
            headers["Authorization"] = f"Bearer {self.config.webhook_token}"
        last_error: str | None = None
        for attempt in range(max(1, self.config.webhook_retries)):
            try:
                response = requests.post(
                    self.config.webhook_url,
                    json=event.to_dict(),
                    headers=headers,
                    timeout=self.config.webhook_timeout_sec,
                )
                if 200 <= response.status_code < 300:
                    return
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                if response.status_code < 500:
                    logger.error(
                        "Webhook отклонил событие %s без повтора: %s",
                        event.id,
                        last_error,
                    )
                    return
            except requests.RequestException as exc:
                last_error = str(exc)
            if attempt + 1 < max(1, self.config.webhook_retries):
                time.sleep(min(2 ** attempt, 4))
        logger.error("Webhook не доставил событие %s: %s", event.id, last_error)

    def stop(self, timeout: float = 5.0) -> bool:
        if not self._delivery_thread:
            return True
        self._running.clear()
        self._wake_delivery.set()
        self._delivery_thread.join(timeout)
        stopped = not self._delivery_thread.is_alive()
        if stopped:
            self._delivery_thread = None
        return stopped

    def _snapshot_frame(self, event: Event):
        camera_id = None
        if event.type == EventType.FALL_DETECTED:
            camera_id = str(event.metadata.get("camera_id", "")).strip() or None
        if camera_id is not None:
            snapshots = self.frame_store.fresh(self.config.snapshot_max_age_sec, camera_id)
        else:
            associated = [str(value) for value in event.metadata.get("camera_ids", [])]
            if associated:
                snapshots = []
                for associated_id in associated:
                    snapshots.extend(
                        self.frame_store.fresh(self.config.snapshot_max_age_sec, associated_id)
                    )
            else:
                snapshots = self.frame_store.fresh(self.config.snapshot_max_age_sec)
        return build_mosaic(snapshots)

    def _resize_snapshot(self, frame):
        height, width = frame.shape[:2]
        scale = min(
            1.0,
            self.config.image_max_width / width,
            self.config.image_max_height / height,
        )
        if scale >= 1.0:
            return frame
        return cv2.resize(
            frame,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    def _save_snapshot(self, event: Event) -> str | None:
        frame = self._snapshot_frame(event)
        if frame is None:
            return None
        frame = self._resize_snapshot(frame)
        success, encoded = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
        )
        if not success:
            logger.error("Не удалось закодировать снимок события %s", event.id)
            return None
        event_time = event.timestamp.astimezone(timezone.utc)
        directory = self.screenshots_dir / f"{event_time:%Y}" / f"{event_time:%m}"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{event.id}.jpg"
        temporary = directory / f".{event.id}.tmp"
        try:
            temporary.write_bytes(encoded.tobytes())
            os.replace(temporary, target)
            return str(target)
        except OSError:
            logger.exception("Не удалось сохранить снимок события %s", event.id)
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            return None

    def _delivery_loop(self) -> None:
        try:
            self._cleanup_expired_media()
        except Exception:
            logger.exception("Не удалось выполнить начальную очистку старых снимков")
        last_cleanup = time.monotonic()
        while self._running.is_set():
            if not self.config.webhook_url:
                self._wake_delivery.wait(1.0)
                self._wake_delivery.clear()
                continue
            items = self.event_store.due_alerts(limit=20)
            if not items:
                self._wake_delivery.wait(1.0)
                self._wake_delivery.clear()
            for item in items:
                if not self._running.is_set():
                    return
                try:
                    self._deliver(item)
                except Exception:
                    logger.exception("Необработанная ошибка доставки тревоги %s", item.event_id)
            if time.monotonic() - last_cleanup >= 3600:
                try:
                    self._cleanup_expired_media()
                except Exception:
                    logger.exception("Не удалось выполнить очистку старых снимков")
                last_cleanup = time.monotonic()

    def _deliver(self, item: OutboxItem) -> None:
        if self.config.video_delivery_enabled:
            self._deliver_video(item)
            return
        headers = {"Idempotency-Key": item.event_id}
        if self.config.webhook_token:
            headers["Authorization"] = f"Bearer {self.config.webhook_token}"
        modes = self._payload_modes()
        for mode in modes:
            current = self.event_store.get_alert(item.event_id)
            if current is None or current.status != "pending":
                logger.info("REST-доставка тревоги %s отменена оператором", item.event_id)
                return
            logger.info("REST отправка тревоги %s: режим=%s", item.event_id, mode)
            response = self._post_payload(item, mode, headers)
            if 200 <= response.status_code < 300:
                if self.config.webhook_payload_mode == "auto":
                    self.config.webhook_payload_mode = mode
                    try:
                        if self._config_persistor is not None:
                            self._config_persistor()
                    except (OSError, ValueError):
                        logger.debug("Не удалось сохранить рабочий REST-формат", exc_info=True)
                    logger.info("REST-формат тревог сохранён: %s", mode)
                if self.event_store.mark_delivered(item.event_id):
                    logger.info(
                        "REST тревога %s доставлена: HTTP %s, режим=%s",
                        item.event_id,
                        response.status_code,
                        mode,
                    )
                return
            message = f"HTTP {response.status_code}: {response.text[:500]}"
            if response.status_code == 599 or response.status_code >= 500:
                self._retry(item, message)
            else:
                if self.event_store.mark_permanent_failure(item.event_id, message):
                    logger.error("REST отклонил тревогу %s: %s", item.event_id, message)
                else:
                    logger.info(
                        "Ответ REST для отменённой тревоги %s проигнорирован: %s",
                        item.event_id,
                        message,
                    )
            return

    def _deliver_video(self, item: OutboxItem) -> None:
        current = self.event_store.get_alert(item.event_id)
        if current is None or current.status != "pending":
            logger.info("REST-доставка тревоги %s отменена оператором", item.event_id)
            return
        video_path = Path(current.video_path or "")
        if not video_path.is_file():
            self._fail_delivery(current, None, "REST-video отсутствует на диске")
            return
        if video_path.stat().st_size >= self.config.delivery_video_max_bytes:
            self._fail_delivery(
                current,
                None,
                (
                    f"REST-video имеет размер {video_path.stat().st_size} байт; "
                    f"требуется меньше {self.config.delivery_video_max_bytes}"
                ),
            )
            return
        if not re.fullmatch(r"[0-9a-f]{64}\.mp4", video_path.name):
            self._fail_delivery(current, None, "Имя REST-video не соответствует SHA-256")
            return

        headers = {"Idempotency-Key": current.event_id}
        if self.config.webhook_token:
            headers["Authorization"] = f"Bearer {self.config.webhook_token}"

        if current.delivery_stage == "trigger":
            logger.info("REST POST /Trigger для тревоги %s", current.event_id)
            response = self._post_trigger(current, video_path.name, headers)
            if response.status_code != 204:
                self._handle_video_failure(current, response, "Trigger")
                return
            if not self.event_store.advance_to_upload(current.event_id):
                logger.info("REST-доставка тревоги %s отменена после /Trigger", current.event_id)
                return
            current = self.event_store.get_alert(current.event_id)
            if current is None or current.status != "pending":
                return
            logger.info("REST /Trigger принял тревогу %s: HTTP 204", current.event_id)
        elif current.delivery_stage != "upload":
            self._fail_delivery(
                current,
                None,
                f"Неизвестная стадия REST-доставки: {current.delivery_stage}",
            )
            return

        logger.info("REST POST /UploadVideo для тревоги %s", current.event_id)
        response = self._post_video(current, video_path, headers)
        if response.status_code == 204:
            if self.event_store.mark_delivered(current.event_id):
                logger.info(
                    "REST-video тревоги %s загружено: HTTP 204, файл=%s",
                    current.event_id,
                    video_path.name,
                )
            return
        self._handle_video_failure(current, response, "UploadVideo")

    def _post_trigger(
        self,
        item: OutboxItem,
        video_name: str,
        headers: dict[str, str],
    ):
        try:
            return requests.post(
                self.config.webhook_url,
                json={"Description": item.description, "Video": video_name},
                headers=headers,
                timeout=self.config.webhook_timeout_sec,
            )
        except requests.RequestException as exc:
            return type("FailedResponse", (), {"status_code": 599, "text": str(exc)})()

    def _post_video(
        self,
        item: OutboxItem,
        video_path: Path,
        headers: dict[str, str],
    ):
        del item
        try:
            with video_path.open("rb") as handle:
                return requests.post(
                    self._video_upload_url(),
                    files={"video": (video_path.name, handle, "video/mp4")},
                    headers=headers,
                    timeout=self.config.webhook_timeout_sec,
                )
        except (requests.RequestException, OSError) as exc:
            return type("FailedResponse", (), {"status_code": 599, "text": str(exc)})()

    def _video_upload_url(self) -> str:
        configured = self.config.video_upload_url.strip()
        if configured:
            return configured
        parsed = urlsplit(self.config.webhook_url)
        parent = parsed.path.rsplit("/", 1)[0]
        path = f"{parent}/UploadVideo" if parent else "/UploadVideo"
        return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))

    @staticmethod
    def _problem_detail(response) -> str:
        try:
            payload = response.json()
        except Exception:
            payload = None
        if isinstance(payload, dict):
            for key in ("detail", "details", "title"):
                value = payload.get(key)
                if value:
                    return str(value)[:1_000]
        return str(getattr(response, "text", ""))[:1_000]

    def _handle_video_failure(self, item: OutboxItem, response, stage: str) -> None:
        detail = self._problem_detail(response)
        message = f"{stage}: HTTP {response.status_code}"
        if detail:
            message += f": {detail}"
        if response.status_code == 599 or response.status_code >= 500:
            self._retry(item, message)
        else:
            self._fail_delivery(item, response, message)

    def _fail_delivery(self, item: OutboxItem, _response, message: str) -> None:
        if self.event_store.mark_permanent_failure(item.event_id, message):
            logger.error("REST отклонил тревогу %s: %s", item.event_id, message)
        else:
            logger.info("REST-ответ отменённой тревоги %s проигнорирован", item.event_id)

    def _payload_modes(self) -> list[str]:
        mode = self.config.webhook_payload_mode
        if mode == "auto":
            return ["description_json", "event_json", "multipart"]
        if mode in {"description_json", "event_json", "multipart"}:
            return [mode]
        return ["description_json"]

    def _post_payload(self, item: OutboxItem, mode: str, headers: dict[str, str]):
        try:
            if mode == "description_json":
                return requests.post(
                    self.config.webhook_url,
                    json={"description": item.description},
                    headers={**headers, "Content-Type": "application/json"},
                    timeout=self.config.webhook_timeout_sec,
                )
            if mode == "event_json":
                return requests.post(
                    self.config.webhook_url,
                    json={"id": item.event_id, "type": item.event_type, "description": item.description},
                    headers={**headers, "Content-Type": "application/json"},
                    timeout=self.config.webhook_timeout_sec,
                )
            image_handle = None
            files = {"description": (None, item.description.encode("utf-8"), "text/plain; charset=utf-8")}
            try:
                if item.image_path and Path(item.image_path).is_file():
                    image_handle = Path(item.image_path).open("rb")
                    files["image"] = (Path(item.image_path).name, image_handle, "image/jpeg")
                return requests.post(
                    self.config.webhook_url, files=files, headers=headers,
                    timeout=self.config.webhook_timeout_sec,
                )
            finally:
                if image_handle is not None:
                    image_handle.close()
        except (requests.RequestException, OSError) as exc:
            return type("FailedResponse", (), {"status_code": 599, "text": str(exc)})()

    def _retry(self, item: OutboxItem, message: str) -> None:
        delay = min(
            self.config.retry_initial_sec * (2 ** min(item.attempts, 20)),
            self.config.retry_max_sec,
        )
        if self.event_store.schedule_retry(item.event_id, time.time() + delay, message):
            logger.warning(
                "REST-тревога %s будет повторена через %.1f с: %s",
                item.event_id,
                delay,
                message,
            )
        else:
            logger.info("Повтор REST-тревоги %s отменён оператором", item.event_id)

    def _cleanup_expired_media(self) -> None:
        cutoff = time.time() - self.config.snapshot_retention_days * 86400
        paths = self.event_store.detach_expired_images(cutoff)
        paths.extend(self.event_store.detach_expired_videos(cutoff))
        for raw_path in paths:
            path = Path(raw_path)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.exception("Не удалось удалить устаревший снимок %s", path)

    def _play_local(self, event: Event) -> None:
        if not self.config.local_sound_enabled:
            return
        sound_by_type = {
            EventType.GUNSHOT_DETECTED: self.config.gunshot_sound,
            EventType.KEYWORD_DETECTED: self.config.keyword_sound,
            EventType.FALL_DETECTED: self.config.fall_sound,
        }
        path = sound_by_type.get(event.type)
        if winsound is None:
            logger.warning("Локальные звуковые оповещения недоступны на этой ОС")
            return
        try:
            if path and Path(path).is_file():
                winsound.PlaySound(
                    path,
                    winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP,
                )
            else:
                logger.warning("Файл тревоги не найден: %s", path)
                winsound.Beep(1000, 400)
        except Exception:
            logger.exception("Ошибка воспроизведения тревоги %s", path)

    def replay_local_alarm(self, event: Event) -> None:
        """Replay the current incident siren when the operator advances the alert queue."""
        self._play_local(event)

    def acknowledge_local_alarm(self) -> None:
        if winsound is None:
            return
        try:
            winsound.PlaySound(None, 0)
        except Exception:
            logger.exception("Не удалось остановить сирену")


def play_sound(path: str) -> None:
    """Compatibility helper for small standalone scripts."""
    if winsound is None or not path or not os.path.isfile(path):
        return
    winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
