import hashlib
import json
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

from core.alerts import AlertService
from core.config import AlertsConfig
from core.event_bus import Event, EventType
from core.rest_video import encode_delivery_video
from core.storage import EventStore


def wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class VideoDeliveryContractTests(unittest.TestCase):
    def setUp(self):
        self.trigger_requests = []
        self.upload_requests = []
        self.trigger_codes = []
        self.upload_codes = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(handler):
                length = int(handler.headers.get("Content-Length", "0"))
                body = handler.rfile.read(length)
                if handler.path == "/Trigger":
                    outer.trigger_requests.append(
                        {
                            "json": json.loads(body.decode("utf-8")),
                            "idempotency": handler.headers.get("Idempotency-Key"),
                            "content_type": handler.headers.get("Content-Type"),
                        }
                    )
                    code = outer.trigger_codes.pop(0) if outer.trigger_codes else 204
                elif handler.path == "/UploadVideo":
                    message = BytesParser(policy=default).parsebytes(
                        b"Content-Type: "
                        + handler.headers["Content-Type"].encode("ascii")
                        + b"\r\nMIME-Version: 1.0\r\n\r\n"
                        + body
                    )
                    parts = {
                        part.get_param("name", header="content-disposition"): part
                        for part in message.iter_parts()
                    }
                    video = parts["video"]
                    outer.upload_requests.append(
                        {
                            "filename": video.get_filename(),
                            "content_type": video.get_content_type(),
                            "body": video.get_payload(decode=True),
                            "idempotency": handler.headers.get("Idempotency-Key"),
                        }
                    )
                    code = outer.upload_codes.pop(0) if outer.upload_codes else 204
                else:
                    code = 404

                payload = b""
                if code == 400:
                    payload = json.dumps(
                        {"title": "Bad Request", "detail": "Некорректное имя видео"},
                        ensure_ascii=False,
                    ).encode("utf-8")
                    handler.send_response(code)
                    handler.send_header("Content-Type", "application/problem+json")
                    handler.send_header("Content-Length", str(len(payload)))
                    handler.end_headers()
                    handler.wfile.write(payload)
                    return
                handler.send_response(code)
                handler.end_headers()

            def log_message(self, *_):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.store = EventStore(str(root / "events.sqlite3"))
        self.config = AlertsConfig(
            local_sound_enabled=False,
            webhook_url=f"http://127.0.0.1:{self.server.server_port}/Trigger",
            video_delivery_enabled=True,
            retry_initial_sec=0.02,
            retry_max_sec=0.05,
        )
        self.service = AlertService(
            self.config,
            self.store,
            screenshots_dir=str(root / "screenshots"),
        )
        self.video_bytes = b"\x00\x00\x00\x18ftypmp42test-video"
        digest = hashlib.sha256(self.video_bytes).hexdigest()
        self.video_path = root / f"{digest}.mp4"
        self.video_path.write_bytes(self.video_bytes)

    def tearDown(self):
        self.service.stop()
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join(2)
        self.temp_dir.cleanup()

    def enqueue(self):
        event = Event(EventType.GUNSHOT_DETECTED, confidence=0.8)
        self.store.enqueue_alert(event, "Обнаружен возможный выстрел", None, await_video=True)
        self.assertTrue(self.store.attach_delivery_video(event.id, str(self.video_path)))
        return event

    def test_trigger_204_then_uploads_hashed_mp4(self):
        event = self.enqueue()
        self.service.start()
        self.assertTrue(wait_until(lambda: self.store.get_alert(event.id).status == "delivered"))

        self.assertEqual(len(self.trigger_requests), 1)
        self.assertEqual(len(self.upload_requests), 1)
        trigger = self.trigger_requests[0]
        self.assertEqual(
            trigger["json"],
            {
                "Description": "Обнаружен возможный выстрел",
                "Video": self.video_path.name,
            },
        )
        self.assertTrue(trigger["content_type"].startswith("application/json"))
        upload = self.upload_requests[0]
        self.assertEqual(upload["filename"], self.video_path.name)
        self.assertEqual(upload["content_type"], "video/mp4")
        self.assertEqual(upload["body"], self.video_bytes)
        self.assertEqual(trigger["idempotency"], event.id)
        self.assertEqual(upload["idempotency"], event.id)

    def test_trigger_400_is_not_retried_and_problem_detail_is_saved(self):
        self.trigger_codes[:] = [400]
        event = self.enqueue()
        self.service.start()
        self.assertTrue(wait_until(lambda: self.store.get_alert(event.id).status == "failed"))
        time.sleep(0.12)

        item = self.store.get_alert(event.id)
        self.assertEqual(len(self.trigger_requests), 1)
        self.assertEqual(self.upload_requests, [])
        self.assertIn("Некорректное имя видео", item.last_error)

    def test_upload_500_retries_only_upload_stage(self):
        self.upload_codes[:] = [500, 204]
        event = self.enqueue()
        self.service.start()
        self.assertTrue(wait_until(lambda: self.store.get_alert(event.id).status == "delivered"))

        self.assertEqual(len(self.trigger_requests), 1)
        self.assertEqual(len(self.upload_requests), 2)
        self.assertEqual(self.store.get_alert(event.id).delivery_stage, "upload")

    def test_upload_400_is_not_retried_and_problem_detail_is_saved(self):
        self.upload_codes[:] = [400]
        event = self.enqueue()
        self.service.start()
        self.assertTrue(wait_until(lambda: self.store.get_alert(event.id).status == "failed"))
        time.sleep(0.12)

        item = self.store.get_alert(event.id)
        self.assertEqual(len(self.trigger_requests), 1)
        self.assertEqual(len(self.upload_requests), 1)
        self.assertIn("Некорректное имя видео", item.last_error)


class DeliveryVideoEncodingTests(unittest.TestCase):
    def test_avc_file_is_below_limit_and_named_by_sha256(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.mp4"
            writer = cv2.VideoWriter(
                str(source),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10.0,
                (480, 270),
            )
            self.assertTrue(writer.isOpened())
            rng = np.random.default_rng(42)
            for _ in range(100):
                writer.write(rng.integers(0, 255, (270, 480, 3), dtype=np.uint8))
            writer.release()

            result = encode_delivery_video(
                source,
                root / "delivery",
                fps=10.0,
                max_bytes=2_000_000,
                initial_crf=28,
            )
            target = Path(result.path)
            self.assertLess(target.stat().st_size, 2_000_000)
            self.assertEqual(target.stem, hashlib.sha256(target.read_bytes()).hexdigest())
            self.assertRegex(target.name, r"^[0-9a-f]{64}\.mp4$")
            self.assertIn(b"avc1", target.read_bytes())
            capture = cv2.VideoCapture(str(target))
            self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 10.0, delta=0.2)
            self.assertAlmostEqual(
                capture.get(cv2.CAP_PROP_FRAME_COUNT) / capture.get(cv2.CAP_PROP_FPS),
                10.0,
                delta=0.3,
            )
            capture.release()


if __name__ == "__main__":
    unittest.main()
