import tempfile
import threading
import time
import unittest
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import cv2
import numpy as np

from core.alerts import AlertService, describe_event
from core.config import AlertsConfig
from core.event_bus import Event, EventType
from core.frames import FrameStore, build_mosaic
from core.storage import EventStore


def wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class FrameStoreTests(unittest.TestCase):
    def test_returns_copies_and_filters_stale_frames(self):
        store = FrameStore()
        frame = np.full((20, 30, 3), 10, dtype=np.uint8)
        store.update("camera-a", "0", frame)
        current = store.fresh(5.0, "camera-a")
        current[0].frame[:] = 99
        self.assertEqual(int(store.fresh(5.0)[0].frame[0, 0, 0]), 10)

        store.update("camera-old", "1", frame, captured_at=time.monotonic() - 10)
        self.assertEqual([item.camera_id for item in store.fresh(5.0)], ["camera-a"])

    def test_mosaic_contains_all_fresh_cameras(self):
        store = FrameStore()
        store.update("camera-a", "0", np.full((20, 30, 3), (0, 0, 255), dtype=np.uint8))
        store.update("camera-b", "1", np.full((20, 30, 3), (0, 255, 0), dtype=np.uint8))
        mosaic = build_mosaic(store.fresh(5.0))
        self.assertIsNotNone(mosaic)
        self.assertGreaterEqual(mosaic.shape[1], 60)


class MultipartAlertTests(unittest.TestCase):
    def setUp(self):
        self.received = []
        self.response_codes = []
        received = self.received
        response_codes = self.response_codes

        class Handler(BaseHTTPRequestHandler):
            def do_POST(handler_self):
                body = handler_self.rfile.read(int(handler_self.headers["Content-Length"]))
                message = BytesParser(policy=default).parsebytes(
                    b"Content-Type: "
                    + handler_self.headers["Content-Type"].encode("ascii")
                    + b"\r\nMIME-Version: 1.0\r\n\r\n"
                    + body
                )
                parts = {part.get_param("name", header="content-disposition"): part for part in message.iter_parts()}
                received.append(
                    {
                        "description": parts["description"].get_content(),
                        "image": parts.get("image").get_payload(decode=True) if "image" in parts else None,
                        "image_type": parts.get("image").get_content_type() if "image" in parts else None,
                        "event_id": handler_self.headers.get("Idempotency-Key"),
                    }
                )
                code = response_codes.pop(0) if response_codes else 204
                handler_self.send_response(code)
                handler_self.end_headers()

            def log_message(self, *_):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.store = EventStore(str(root / "events.sqlite3"))
        self.frames = FrameStore()
        self.config = AlertsConfig(
            local_sound_enabled=False,
            webhook_url=f"http://127.0.0.1:{self.server.server_port}/trigger",
            retry_initial_sec=0.02,
            retry_max_sec=0.05,
            video_delivery_enabled=False,
        )
        self.service = AlertService(self.config, self.store, self.frames, str(root / "screenshots"))

    def tearDown(self):
        self.service.stop()
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join(2)
        self.temp_dir.cleanup()

    def test_posts_description_and_jpeg_for_all_alert_types(self):
        red = np.full((40, 60, 3), (0, 0, 255), dtype=np.uint8)
        green = np.full((40, 60, 3), (0, 255, 0), dtype=np.uint8)
        self.frames.update("camera-a", "0", red)
        self.frames.update("camera-b", "1", green)
        events = [
            Event(EventType.GUNSHOT_DETECTED, confidence=0.8),
            Event(EventType.KEYWORD_DETECTED, confidence=0.7, metadata={"keyword": "помоги", "text": "помоги мне"}),
            Event(EventType.FALL_DETECTED, confidence=0.9, metadata={"camera_id": "camera-b", "duration_sec": 5.2}),
        ]
        self.service.start()
        for event in events:
            self.service.handle_event(event)
        self.assertTrue(wait_until(lambda: len(self.received) == 3))
        for request, event in zip(self.received, events):
            self.assertEqual(request["event_id"], event.id)
            self.assertEqual(request["image_type"], "image/jpeg")
            self.assertEqual(cv2.imdecode(np.frombuffer(request["image"], np.uint8), cv2.IMREAD_COLOR).ndim, 3)
        self.assertIn("выстрел", self.received[0]["description"])
        self.assertIn("помоги", self.received[1]["description"])
        fall_image = cv2.imdecode(np.frombuffer(self.received[2]["image"], np.uint8), cv2.IMREAD_COLOR)
        self.assertGreater(float(fall_image[:, :, 1].mean()), float(fall_image[:, :, 2].mean()))

    def test_audio_alert_uses_only_cameras_bound_to_its_microphone(self):
        self.frames.update("camera-a", "0", np.full((30, 40, 3), (0, 0, 255), dtype=np.uint8))
        self.frames.update("camera-b", "1", np.full((30, 40, 3), (0, 255, 0), dtype=np.uint8))
        event = Event(
            EventType.GUNSHOT_DETECTED,
            metadata={"microphone": "4", "camera_ids": ["camera-b"]},
        )
        selected = self.service._snapshot_frame(event)
        self.assertGreater(float(selected[:, :, 1].mean()), float(selected[:, :, 2].mean()))

    def test_sends_description_without_image_when_no_frame_exists(self):
        event = Event(EventType.GUNSHOT_DETECTED)
        self.service.start()
        self.service.handle_event(event)
        self.assertTrue(wait_until(lambda: len(self.received) == 1))
        self.assertIsNone(self.received[0]["image"])
        self.assertIn("Снимок камеры недоступен", self.received[0]["description"])

    def test_retries_500_and_marks_delivery(self):
        self.response_codes[:] = [500, 204]
        event = Event(EventType.GUNSHOT_DETECTED)
        self.service.start()
        self.service.handle_event(event)
        self.assertTrue(
            wait_until(
                lambda: len(self.received) == 2
                and self.store.get_alert(event.id).status == "delivered"
            )
        )
        self.assertEqual(self.store.get_alert(event.id).status, "delivered")
        self.assertEqual(self.store.get_alert(event.id).attempts, 2)

    def test_204_marks_delivery_without_retry(self):
        self.response_codes[:] = [204]
        event = Event(EventType.GUNSHOT_DETECTED)
        self.service.start()
        self.service.handle_event(event)
        self.assertTrue(wait_until(lambda: self.store.get_alert(event.id).status == "delivered"))
        self.assertEqual(len(self.received), 1)
        self.assertEqual(self.store.get_alert(event.id).attempts, 1)

    def test_marks_non_retryable_400_as_failed(self):
        self.response_codes[:] = [400]
        event = Event(EventType.GUNSHOT_DETECTED)
        self.service.start()
        self.service.handle_event(event)
        self.assertTrue(wait_until(lambda: self.store.get_alert(event.id).status == "failed"))
        time.sleep(0.12)
        self.assertEqual(len(self.received), 1)


class OutboxRetentionTests(unittest.TestCase):
    def test_only_delivered_images_are_detached(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = EventStore(str(Path(temp_dir) / "events.sqlite3"))
            delivered = Event(EventType.GUNSHOT_DETECTED)
            pending = Event(EventType.KEYWORD_DETECTED)
            store.enqueue_alert(delivered, "done", "done.jpg", now=1.0)
            store.enqueue_alert(pending, "waiting", "waiting.jpg", now=1.0)
            store.mark_delivered(delivered.id, delivered_at=2.0)
            self.assertEqual(store.detach_expired_images(3.0), ["done.jpg"])
            self.assertIsNone(store.get_alert(delivered.id).image_path)
            self.assertEqual(store.get_alert(pending.id).image_path, "waiting.jpg")

    def test_shared_video_is_retained_while_another_alert_references_it(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = EventStore(str(Path(temp_dir) / "events.sqlite3"))
            video_path = str(Path(temp_dir) / ("b" * 64 + ".mp4"))
            delivered = Event(EventType.GUNSHOT_DETECTED)
            pending = Event(EventType.KEYWORD_DETECTED)
            for event in (delivered, pending):
                store.enqueue_alert(event, "video", None, now=1.0, await_video=True)
                self.assertTrue(store.attach_delivery_video(event.id, video_path, now=1.5))
            store.mark_delivered(delivered.id, delivered_at=2.0)

            self.assertEqual(store.detach_expired_videos(3.0), [])
            self.assertIsNone(store.get_alert(delivered.id).video_path)
            self.assertEqual(store.get_alert(pending.id).video_path, video_path)

            store.mark_delivered(pending.id, delivered_at=2.5)
            self.assertEqual(store.detach_expired_videos(3.0), [video_path])

    def test_pending_outbox_is_available_after_store_restart(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = str(Path(temp_dir) / "events.sqlite3")
            event = Event(EventType.FALL_DETECTED)
            EventStore(path).enqueue_alert(event, "pending incident", None, now=1.0)
            reopened = EventStore(path)
            due = reopened.due_alerts(now=2.0)
            self.assertEqual([item.event_id for item in due], [event.id])

    def test_video_upload_stage_is_available_after_store_restart(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = str(Path(temp_dir) / "events.sqlite3")
            video_path = str(Path(temp_dir) / ("a" * 64 + ".mp4"))
            event = Event(EventType.FALL_DETECTED)
            store = EventStore(path)
            store.enqueue_alert(event, "pending video", None, now=1.0, await_video=True)
            self.assertTrue(store.attach_delivery_video(event.id, video_path, now=2.0))
            self.assertTrue(store.advance_to_upload(event.id, now=3.0))

            item = EventStore(path).get_alert(event.id)
            self.assertEqual(item.status, "pending")
            self.assertEqual(item.delivery_stage, "upload")
            self.assertEqual(item.video_path, video_path)

    def test_duplicate_event_id_is_enqueued_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = EventStore(str(Path(temp_dir) / "events.sqlite3"))
            event = Event(EventType.GUNSHOT_DETECTED)
            self.assertTrue(store.enqueue_alert(event, "first", None))
            self.assertFalse(store.enqueue_alert(event, "second", None))
            self.assertEqual(store.get_alert(event.id).description, "first")


class DescriptionTests(unittest.TestCase):
    def test_describes_each_danger_type(self):
        self.assertIn("выстрел", describe_event(Event(EventType.GUNSHOT_DETECTED)))
        self.assertIn("ключевое слово", describe_event(Event(EventType.KEYWORD_DETECTED)))
        self.assertIn("падение", describe_event(Event(EventType.FALL_DETECTED)))


if __name__ == "__main__":
    unittest.main()

