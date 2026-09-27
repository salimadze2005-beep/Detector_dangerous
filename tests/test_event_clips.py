import json
import tempfile
import time
import unittest
from datetime import timezone
from pathlib import Path

import numpy as np

from core.event_bus import Event, EventType
from core.event_clips import EventClipRecorder
from core.frames import FrameStore


class FrameHistoryTests(unittest.TestCase):
    def test_history_is_sampled_compressed_and_bounded(self):
        store = FrameStore()
        store.configure_history(
            history_sec=1.0,
            history_fps=2.0,
            jpeg_quality=60,
            max_width=100,
            max_height=80,
        )
        frame = np.full((240, 320, 3), 100, dtype=np.uint8)
        base = 1000.0
        for index in range(16):
            store.update("camera-a", "0", frame, captured_at=base + index * 0.1)

        history = store.history_between("camera-a", base, base + 2.0)
        self.assertGreaterEqual(len(history), 2)
        self.assertLessEqual(len(history), 3)
        self.assertGreater(store.history_memory_bytes(), 0)
        decoded = history[-1].decode()
        self.assertIsNotNone(decoded)
        self.assertLessEqual(decoded.shape[1], 100)
        self.assertLessEqual(decoded.shape[0], 80)


class EventClipRecorderTests(unittest.TestCase):
    def test_camera_selection_prefers_event_binding(self):
        store = FrameStore()
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = EventClipRecorder(store, temp_dir, post_sec=0.0)
            fall = Event(EventType.FALL_DETECTED, metadata={"camera_id": "camera-b"})
            gunshot = Event(
                EventType.GUNSHOT_DETECTED,
                metadata={"camera_ids": ["camera-a", "camera-b"]},
            )
            self.assertEqual(recorder._camera_ids_for_event(fall), ["camera-b"])
            self.assertEqual(
                recorder._camera_ids_for_event(gunshot),
                ["camera-a", "camera-b"],
            )
            recorder.close()

    def test_event_types_have_separate_archive_categories(self):
        self.assertEqual(EventClipRecorder.EVENT_FOLDERS[EventType.FALL_DETECTED], "fall")
        self.assertEqual(EventClipRecorder.EVENT_FOLDERS[EventType.GUNSHOT_DETECTED], "gunshot")
        self.assertEqual(EventClipRecorder.EVENT_FOLDERS[EventType.KEYWORD_DETECTED], "keyword")

    def test_fall_window_is_two_before_five_confirm_two_after(self):
        store = FrameStore()
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = EventClipRecorder(
                store,
                temp_dir,
                fall_pre_sec=2.0,
                fall_post_sec=2.0,
                fall_confirmation_sec=5.0,
            )
            event_monotonic = 100.0
            fall = Event(
                EventType.FALL_DETECTED,
                metadata={"camera_id": "camera-a", "duration_sec": 5.0},
            )
            start_at, end_at, timing = recorder._event_window(fall, event_monotonic)
            self.assertEqual(start_at, 93.0)
            self.assertEqual(end_at, 102.0)
            self.assertEqual(timing["mode"], "fall_confirmation")
            self.assertEqual(timing["pre_fall_sec"], 2.0)
            self.assertEqual(timing["lying_before_confirmation_sec"], 5.0)
            self.assertEqual(timing["post_confirmation_sec"], 2.0)
            recorder.close()

    def test_fall_window_keeps_actual_longer_confirmation(self):
        store = FrameStore()
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = EventClipRecorder(
                store,
                temp_dir,
                fall_pre_sec=2.0,
                fall_post_sec=2.0,
                fall_confirmation_sec=5.0,
            )
            fall = Event(EventType.FALL_DETECTED, metadata={"duration_sec": 5.4})
            start_at, end_at, timing = recorder._event_window(fall, 100.0)
            self.assertAlmostEqual(start_at, 92.6)
            self.assertEqual(end_at, 102.0)
            self.assertEqual(timing["lying_before_confirmation_sec"], 5.4)
            recorder.close()

    def test_manifest_links_event_and_saved_clip(self):
        store = FrameStore()
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = EventClipRecorder(
                store,
                temp_dir,
                pre_sec=1.0,
                post_sec=0.0,
                fps=5.0,
                jpeg_quality=70,
                max_width=160,
                max_height=120,
            )
            event_monotonic = time.monotonic()
            for index, offset in enumerate((-0.8, -0.6, -0.4, -0.2, 0.0)):
                frame = np.full((120, 160, 3), 20 + index * 30, dtype=np.uint8)
                store.update(
                    "camera-a",
                    "0",
                    frame,
                    captured_at=event_monotonic + offset,
                )

            event = Event(
                EventType.GUNSHOT_DETECTED,
                confidence=0.8,
                metadata={"camera_ids": ["camera-a"]},
            )
            recorder._finalize(event, event_monotonic, ["camera-a"])

            event_time = event.timestamp.astimezone(timezone.utc)
            directory = (
                Path(temp_dir)
                / "gunshot"
                / f"{event_time:%Y}"
                / f"{event_time:%m}"
                / event.id
            )
            manifest_path = directory / "event.json"
            self.assertTrue(manifest_path.is_file())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["category"], "gunshot")
            self.assertEqual(manifest["event"]["id"], event.id)
            self.assertEqual(manifest["event"]["type"], "GUNSHOT_DETECTED")
            self.assertEqual(manifest["evidence"]["timing"]["pre_event_sec"], 1.0)
            self.assertEqual(manifest["evidence"]["timing"]["post_event_sec"], 0.0)

            clips = manifest["evidence"]["clips"]
            if clips:
                clip_path = Path(temp_dir) / clips[0]["relative_path"]
                self.assertTrue(clip_path.is_file())
                self.assertGreater(clip_path.stat().st_size, 0)
                self.assertEqual(clips[0]["media_type"], "video/mp4")
                self.assertEqual(clips[0]["camera_id"], "camera-a")
            recorder.close()

    def test_saved_clip_is_forwarded_to_delivery_callback(self):
        store = FrameStore()
        received = []
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = EventClipRecorder(
                store,
                temp_dir,
                pre_sec=1.0,
                post_sec=0.0,
                fps=10.0,
                max_width=160,
                max_height=120,
                on_clip_ready=lambda event, path: received.append((event.id, path)),
            )
            event_monotonic = time.monotonic()
            for index in range(10):
                store.update(
                    "camera-a",
                    "0",
                    np.full((120, 160, 3), index * 20, dtype=np.uint8),
                    captured_at=event_monotonic - 0.9 + index * 0.1,
                )
            event = Event(
                EventType.GUNSHOT_DETECTED,
                metadata={"camera_ids": ["camera-a"]},
            )
            recorder._finalize(event, event_monotonic, ["camera-a"])
            self.assertEqual(received[0][0], event.id)
            self.assertTrue(Path(received[0][1]).is_file())
            recorder.close()


if __name__ == "__main__":
    unittest.main()
