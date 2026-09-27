import tempfile
import unittest
from pathlib import Path

from core.event_bus import Event, EventType
from core.incidents import IncidentStore
from core.storage import EventStore


class IncidentHistoryStoreTests(unittest.TestCase):
    def test_false_alarm_cancels_only_pending_delivery_and_saves_note(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "events.sqlite3")
            event_store = EventStore(db_path)
            incident_store = IncidentStore(db_path, str(Path(temp_dir) / "event_clips"))
            event = Event(EventType.FALL_DETECTED, confidence=0.8, metadata={"camera_id": "camera-a"})
            event_store.record(event)
            event_store.enqueue_alert(event, "Падение", None, now=100.0)

            incident_store.review(event.id, "false_alarm", "Человек лёг на пол намеренно")
            record = incident_store.get(event.id)

            self.assertIsNotNone(record)
            self.assertEqual(record.decision, "false_alarm")
            self.assertEqual(record.note, "Человек лёг на пол намеренно")
            self.assertEqual(record.delivery_status, "cancelled")

            # A delivery worker may still hold a stale copy of this outbox item.
            # Its late result must never resurrect or overwrite the cancellation.
            self.assertFalse(event_store.schedule_retry(event.id, 200.0, "HTTP 500"))
            self.assertFalse(event_store.mark_delivered(event.id, delivered_at=201.0))
            self.assertFalse(event_store.mark_permanent_failure(event.id, "HTTP 400"))
            self.assertEqual(event_store.get_alert(event.id).status, "cancelled")
            self.assertEqual(event_store.get_alert(event.id).attempts, 0)

    def test_false_alarm_cancels_video_that_is_still_being_prepared(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "events.sqlite3")
            event_store = EventStore(db_path)
            incident_store = IncidentStore(db_path, str(Path(temp_dir) / "event_clips"))
            event = Event(EventType.FALL_DETECTED)
            event_store.record(event)
            event_store.enqueue_alert(
                event,
                "Падение",
                None,
                now=100.0,
                await_video=True,
            )

            incident_store.review(event.id, "false_alarm")
            self.assertEqual(event_store.get_alert(event.id).status, "cancelled")
            self.assertFalse(
                event_store.attach_delivery_video(event.id, str(Path(temp_dir) / "late.mp4"))
            )

    def test_review_does_not_rewrite_already_delivered_alert(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "events.sqlite3")
            event_store = EventStore(db_path)
            incident_store = IncidentStore(db_path, str(Path(temp_dir) / "event_clips"))
            event = Event(EventType.GUNSHOT_DETECTED, confidence=0.9)
            event_store.record(event)
            event_store.enqueue_alert(event, "Выстрел", None, now=100.0)
            event_store.mark_delivered(event.id, delivered_at=101.0)

            incident_store.review(event.id, "false_alarm", "Хлопок двери")
            record = incident_store.get(event.id)

            self.assertEqual(record.decision, "false_alarm")
            self.assertEqual(record.delivery_status, "delivered")

    def test_history_resolves_snapshot_and_event_clip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db_path = str(root / "events.sqlite3")
            clips_root = root / "event_clips"
            event_store = EventStore(db_path)
            incident_store = IncidentStore(db_path, str(clips_root))
            event = Event(EventType.KEYWORD_DETECTED, confidence=0.75)
            snapshot = root / "shot.jpg"
            snapshot.write_bytes(b"jpg")
            event_store.record(event)
            event_store.enqueue_alert(event, "Ключевое слово", str(snapshot), now=100.0)

            dt = event.timestamp
            clip_dir = clips_root / "keyword" / f"{dt:%Y}" / f"{dt:%m}" / event.id
            clip_dir.mkdir(parents=True)
            clip = clip_dir / "camera-a.mp4"
            clip.write_bytes(b"mp4")

            record = incident_store.get(event.id)
            self.assertEqual(record.image_path, str(snapshot))
            self.assertEqual(record.clip_dir, str(clip_dir.resolve()))
            self.assertEqual(record.clip_files, (str(clip.resolve()),))

    def test_note_can_be_saved_without_changing_review_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "events.sqlite3")
            event_store = EventStore(db_path)
            incident_store = IncidentStore(db_path, str(Path(temp_dir) / "event_clips"))
            event = Event(EventType.FALL_DETECTED)
            event_store.record(event)
            event_store.enqueue_alert(event, "Падение", None, now=100.0)

            incident_store.update_note(event.id, "Проверить камеру 2")
            record = incident_store.get(event.id)

            self.assertEqual(record.decision, "unreviewed")
            self.assertEqual(record.note, "Проверить камеру 2")

    def test_editing_note_preserves_existing_confirmed_decision(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "events.sqlite3")
            event_store = EventStore(db_path)
            incident_store = IncidentStore(db_path, str(Path(temp_dir) / "event_clips"))
            event = Event(EventType.FALL_DETECTED)
            event_store.record(event)
            event_store.enqueue_alert(event, "Падение", None, now=100.0)

            incident_store.review(event.id, "confirmed", "Первичная проверка")
            incident_store.update_note(event.id, "Уточнённая заметка")
            record = incident_store.get(event.id)

            self.assertEqual(record.decision, "confirmed")
            self.assertEqual(record.note, "Уточнённая заметка")


if __name__ == "__main__":
    unittest.main()
