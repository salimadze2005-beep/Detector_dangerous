import json
import os
import re
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

from core.config import load_config, save_operator_settings
from core.alerts import AlertService
from core.config import AlertsConfig
from core.event_bus import Event, EventBus, EventType, Severity
from core.storage import EventStore
from ui.workers import LatestFrameSlot


class LatestFrameSlotTests(unittest.TestCase):
    def test_replaces_unconsumed_frame_instead_of_queueing_it(self):
        slot = LatestFrameSlot()
        first = slot.publish("old", captured_at=1.0)
        newest = slot.publish("new", captured_at=2.0)

        received = slot.take_after(first.sequence, timeout_sec=0.01)

        self.assertIsNotNone(received)
        self.assertEqual(received.frame, "new")
        self.assertEqual(received.sequence, newest.sequence)
        self.assertEqual(slot.dropped, 1)


class EventBusTests(unittest.TestCase):
    def test_dispatch_is_ordered_and_isolates_broken_handler(self):
        bus = EventBus()
        received = []

        def broken(_):
            raise RuntimeError("expected")

        bus.subscribe_all(broken)
        bus.subscribe(EventType.GUNSHOT_DETECTED, lambda event: received.append(event.id))
        first = Event(EventType.GUNSHOT_DETECTED)
        second = Event(EventType.GUNSHOT_DETECTED)
        with self.assertLogs("core.event_bus", level="ERROR") as captured:
            bus.publish(first)
            bus.publish(second)
            self.assertTrue(bus.flush())
        self.assertEqual(len(captured.records), 2)
        self.assertEqual(received, [first.id, second.id])
        self.assertTrue(bus.stop())

    def test_event_normalizes_confidence_and_copies_metadata(self):
        metadata = {"value": 1}
        event = Event("FALL_DETECTED", confidence=4.0, severity=Severity.CRITICAL, metadata=metadata)
        metadata["value"] = 2
        self.assertEqual(event.confidence, 1.0)
        self.assertEqual(event.metadata["value"], 1)
        self.assertEqual(event.type, EventType.FALL_DETECTED)


class EventStoreTests(unittest.TestCase):
    def test_persists_and_counts_events(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = EventStore(str(Path(temp_dir) / "events.sqlite3"))
            event = Event(
                EventType.KEYWORD_DETECTED,
                source="test",
                metadata={"text": "помогите"},
            )
            store.record(event)
            self.assertEqual(store.counts()[EventType.KEYWORD_DETECTED.value], 1)
            self.assertEqual(store.recent(1)[0]["metadata"]["text"], "помогите")


class ConfigTests(unittest.TestCase):
    def test_operator_settings_are_persisted_without_losing_advanced_options(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "detector_config.json"
            path.write_text('{"audio":{"ema_alpha":0.4}}', encoding="utf-8")
            loaded = load_config(path)
            loaded.audio.trigger_dbfs = -24
            loaded.audio.microphone_profiles = {
                "3": {"preset": "noisy", "trigger_dbfs": -7.0}
            }
            loaded.video.sources = ["rtsp://camera/stream"]
            loaded.video.microphone_by_source = {"rtsp://camera/stream": 3}
            save_operator_settings(loaded)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["audio"]["ema_alpha"], 0.4)
            self.assertEqual(saved["audio"]["trigger_dbfs"], -24)
            self.assertEqual(saved["audio"]["microphone_profiles"]["3"]["preset"], "noisy")
            self.assertEqual(saved["video"]["microphone_by_source"], {"rtsp://camera/stream": 3})

    def test_json_and_environment_overrides(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.json"
            path.write_text(
                json.dumps({"audio": {"peak_min": 0.42}, "video": {"sources": ["2"]}}),
                encoding="utf-8",
            )
            data_dir = str(Path(temp_dir) / "runtime")
            with patch.dict(os.environ, {"DETECTOR_DATA_DIR": data_dir}, clear=False):
                loaded = load_config(path)
            self.assertEqual(loaded.audio.peak_min, 0.42)
            self.assertEqual(loaded.video.sources, ["2"])
            self.assertEqual(loaded.paths.event_db, str(Path(data_dir).resolve() / "events.sqlite3"))

    def test_unknown_option_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.json"
            path.write_text('{"audio": {"typo_threshold": 1}}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)


class AlertServiceTests(unittest.TestCase):
    def test_webhook_posts_serialized_event(self):
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                received.append(json.loads(self.rfile.read(length)))
                self.send_response(204)
                self.end_headers()

            def log_message(self, *_):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            service = AlertService(
                AlertsConfig(
                    local_sound_enabled=False,
                    webhook_url=f"http://127.0.0.1:{server.server_port}/events",
                )
            )
            event = Event(EventType.FALL_DETECTED, metadata={"track_id": 4})
            service._send_webhook(event)
            self.assertEqual(received[0]["id"], event.id)
            self.assertEqual(received[0]["metadata"]["track_id"], 4)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)

    def test_compatibility_webhook_does_not_retry_400(self):
        service = AlertService(
            AlertsConfig(
                local_sound_enabled=False,
                webhook_url="http://example.invalid/trigger",
                webhook_retries=3,
            )
        )
        response = type("Response", (), {"status_code": 400, "text": "bad request"})()
        with patch("core.alerts.requests.post", return_value=response) as post:
            service._send_webhook(Event(EventType.GUNSHOT_DETECTED))
        post.assert_called_once()

    def test_compatibility_webhook_retries_5xx_until_204(self):
        service = AlertService(
            AlertsConfig(
                local_sound_enabled=False,
                webhook_url="http://example.invalid/trigger",
                webhook_retries=3,
            )
        )
        failed = type("Response", (), {"status_code": 500, "text": "server error"})()
        accepted = type("Response", (), {"status_code": 204, "text": ""})()
        with (
            patch("core.alerts.requests.post", side_effect=[failed, accepted]) as post,
            patch("core.alerts.time.sleep"),
        ):
            service._send_webhook(Event(EventType.GUNSHOT_DETECTED))
        self.assertEqual(post.call_count, 2)


if __name__ == "__main__":
    unittest.main()
