import os
import tempfile
from datetime import datetime, timezone
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import QApplication, QLabel, QScrollArea, QTextEdit

from audio.calibration import AmbientCalibration
from core.config import AppConfig
from core.event_bus import Event, EventType
from ui.dashboard import SecurityDashboard
from ui.audio_calibration_dialog import AudioCalibrationDialog
from ui.incident_history import _format_incident_timestamp
from ui.settings_dialog import SettingsDialog
from ui.settings_dialog import discover_local_cameras


class OperatorUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.config = AppConfig()
        self.config.paths.data_dir = str(root)
        self.config.paths.event_db = str(root / "events.sqlite3")
        self.config.paths.screenshots_dir = str(root / "screenshots")
        self.config.paths.log_file = str(root / "detector.log")
        self.config.alerts.local_sound_enabled = False
        self.config.alerts.webhook_url = ""
        self.config.video.sources = ["0", "1", "2"]
        self.window = SecurityDashboard(self.config)
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        self.app.processEvents()
        self.temp_dir.cleanup()

    def test_main_screen_contains_only_operator_controls_and_camera_grid(self):
        self.assertEqual(len(self.window.camera_tiles), 3)
        self.assertEqual(self.window.control_button.text(), "ЗАПУСТИТЬ")
        self.assertTrue(self.window.settings_button.isVisible())
        self.assertTrue(self.window.logo.pixmap() and not self.window.logo.pixmap().isNull())
        self.assertEqual(self.window.logo.pixmap().toImage().pixelColor(0, 0).alpha(), 0)
        self.assertEqual(self.window.findChildren(QTextEdit), [])
        labels = " ".join(label.text() for label in self.window.findChildren(QLabel))
        self.assertNotIn("Выстрелов:", labels)
        self.assertNotIn("Падений:", labels)

    def test_alert_overlay_queues_incidents_until_acknowledged(self):
        first = Event(EventType.GUNSHOT_DETECTED)
        second = Event(EventType.KEYWORD_DETECTED, metadata={"keyword": "помоги"})
        self.window.event_store.enqueue_alert(first, "Первое описание", None)
        self.window.event_store.enqueue_alert(second, "Второе описание", None)
        self.window._handle_event(first)
        self.window._handle_event(second)
        self.app.processEvents()
        self.assertTrue(self.window.alert_overlay.isVisible())
        self.assertEqual(self.window.current_alert.id, first.id)
        self.assertEqual(self.window.alert_overlay.description.text(), "Первое описание")
        self.window.acknowledge_alert()
        self.assertEqual(self.window.current_alert.id, second.id)
        self.window.acknowledge_alert()
        self.assertIsNone(self.window.current_alert)
        self.assertFalse(self.window.alert_overlay.isVisible())

    def test_settings_dialog_updates_runtime_configuration(self):
        with patch("ui.settings_dialog.discover_local_cameras", return_value=[0, 1, 2]):
            dialog = SettingsDialog(self.config)
        threshold_help = dialog.findChild(QLabel, "gunshotThresholdHelp")
        self.assertIsNotNone(threshold_help)
        self.assertTrue(threshold_help.toolTip())
        first = dialog.camera_rows[0]
        first.name_edit.setText("Главный вход")
        first.source_combo.setEditText("rtsp://camera/stream")
        if first.microphone_combo.count() > 1:
            first.microphone_combo.setCurrentIndex(1)
        dialog.add_camera_row("Склад", "3", None)
        dialog.db_slider.setValue(-24)
        dialog.keyword_edit.setText("тревога")
        with patch.object(dialog, "_keyword_exists", return_value=True):
            dialog._add_keyword()
        dialog.webhook_edit.setText("http://127.0.0.1:8080/trigger")
        dialog.local_sound_check.setChecked(True)
        dialog._apply_and_accept()
        self.assertEqual(self.config.video.sources, ["rtsp://camera/stream", "1", "2", "3"])
        self.assertEqual(self.config.video.names_by_source["rtsp://camera/stream"], "Главный вход")
        self.assertEqual(self.config.audio.trigger_dbfs, -24.0)
        self.assertIn("тревога", self.config.speech.keywords)
        self.assertEqual(self.config.alerts.webhook_url, "http://127.0.0.1:8080/trigger")
        self.assertTrue(self.config.alerts.local_sound_enabled)

    def test_settings_store_top_down_view_mode_per_camera(self):
        dialog = SettingsDialog(self.config)
        row = dialog.camera_rows[0]
        row.view_mode_combo.setCurrentIndex(row.view_mode_combo.findData("top_down"))
        dialog._apply_and_accept()
        self.assertEqual(self.config.video.view_mode_by_source["0"], "top_down")
    def test_settings_accept_rtsp_microphone_for_gunshot_and_keyword(self):
        dialog = SettingsDialog(self.config)
        source = "rtsp://user:password@microphone.local/live"
        dialog.camera_rows[0].microphone_combo.setEditText(source)
        dialog.speech_microphone_combo.setEditText(source)
        dialog._apply_and_accept()
        first_camera = self.config.video.sources[0]
        self.assertEqual(self.config.video.microphone_by_source[first_camera], source)
        self.assertEqual(self.config.speech.microphone_device, source)

    def test_new_camera_row_starts_with_all_fields_empty(self):
        dialog = SettingsDialog(self.config)
        row = dialog.add_camera_row(empty=True)
        self.assertEqual(row.name_edit.text(), "")
        self.assertEqual(row.source_combo.currentIndex(), -1)
        self.assertEqual(row.source_combo.currentText(), "")
        self.assertEqual(row.microphone_combo.currentIndex(), -1)
        self.assertEqual(row.microphone_combo.currentText(), "")

    def test_rest_delivery_log_is_visible_in_alert_settings(self):
        Path(self.config.paths.log_file).write_text(
            "2026-08-18 12:00:00 INFO Система запущена\n"
            "2026-08-18 12:00:01 INFO REST отправлен: HTTP 200\n"
            "2026-08-18 12:00:02 ERROR Webhook: HTTP 415\n",
            encoding="utf-8",
        )
        dialog = SettingsDialog(self.config)
        log_text = dialog.rest_log_view.toPlainText()
        self.assertIn("REST отправлен: HTTP 200", log_text)
        self.assertIn("Webhook: HTTP 415", log_text)
        self.assertNotIn("Система запущена", log_text)

    def test_audio_calibration_starts_on_the_active_microphone(self):
        dialog = AudioCalibrationDialog(
            self.config.audio,
            [(3, "Entrance"), (4, "Workshop")],
            initial_device=4,
        )
        self.assertEqual(dialog.microphone_combo.currentData(), 4)
        self.assertEqual(dialog._current_key, "4")

    def test_manual_threshold_change_marks_profile_custom(self):
        dialog = AudioCalibrationDialog(
            self.config.audio,
            [(17, "Active microphone")],
            initial_device=17,
        )
        dialog.preset_combo.setCurrentIndex(dialog.preset_combo.findData("balanced"))
        dialog.snr_spin.setValue(12.5)
        self.assertEqual(dialog.preset_combo.currentData(), "custom")
        dialog._save_and_accept()
        self.assertEqual(dialog.profiles["17"]["preset"], "custom")
        self.assertEqual(dialog.profiles["17"]["min_snr_db"], 12.5)

    def test_audio_calibration_dialog_stores_separate_microphone_profiles(self):
        dialog = AudioCalibrationDialog(self.config.audio, [(3, "Entrance"), (4, "Workshop")])
        dialog.microphone_combo.setCurrentIndex(dialog.microphone_combo.findData(3))
        dialog.preset_combo.setCurrentIndex(dialog.preset_combo.findData("noisy"))
        result = AmbientCalibration(
            preset="noisy",
            windows=40,
            ambient_rms=0.02,
            ambient_peak_dbfs=-19.0,
            ambient_crest_p95=3.1,
            clipping_ratio=0.0,
            trigger_dbfs=-5.0,
            rms_min=0.10,
            min_snr_db=14.0,
            min_crest_factor=3.5,
        )
        dialog._calibration_completed(result)
        dialog.microphone_combo.setCurrentIndex(dialog.microphone_combo.findData(4))
        dialog.preset_combo.setCurrentIndex(dialog.preset_combo.findData("sensitive"))
        dialog._save_and_accept()
        self.assertEqual(dialog.profiles["3"]["preset"], "noisy")
        self.assertEqual(dialog.profiles["3"]["trigger_dbfs"], -5.0)
        self.assertEqual(dialog.profiles["4"]["preset"], "sensitive")

    def test_incident_history_converts_utc_to_local_time(self):
        stored = "2026-08-25T10:00:00+00:00"
        expected = datetime.fromisoformat(stored).astimezone().strftime("%d.%m.%Y %H:%M:%S")
        self.assertEqual(_format_incident_timestamp(stored), expected)

    def test_audio_calibration_content_is_scrollable(self):
        dialog = AudioCalibrationDialog(self.config.audio, [(3, "Entrance")])
        scrolls = dialog.findChildren(QScrollArea)
        self.assertEqual(len(scrolls), 1)
        self.assertIsNotNone(scrolls[0].widget())
    def test_calibration_wheel_only_scrolls_and_does_not_change_controls(self):
        dialog = AudioCalibrationDialog(self.config.audio, [(3, "Entrance")])
        dialog.show()
        self.app.processEvents()
        scroll = dialog.findChildren(QScrollArea)[0]
        self.assertGreater(scroll.verticalScrollBar().maximum(), 0)
        before = dialog.cnn_threshold.value()
        scroll.verticalScrollBar().setValue(0)
        wheel = QWheelEvent(
            QPointF(2, 2), QPointF(2, 2), QPoint(0, -120), QPoint(0, -120),
            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.ScrollUpdate, False,
        )
        QApplication.sendEvent(dialog.cnn_threshold, wheel)
        self.assertEqual(dialog.cnn_threshold.value(), before)
        self.assertGreater(scroll.verticalScrollBar().value(), 0)
        dialog.close()

    def test_manual_distance_calibration_accepts_three_separate_takes(self):
        dialog = AudioCalibrationDialog(self.config.audio, [(3, "Entrance")])
        self.assertEqual(set(dialog.manual_distance_buttons), {"10–20 см", "50 см", "1 м", "2 м", "5 м"})
        generator = np.random.default_rng(42)
        dialog._manual_ambient_chunks = [
            generator.normal(0.0, 0.002, 4_000).astype(np.float32)
            for _ in range(12)
        ]
        for distance, amplitude in (("10–20 см", 0.70), ("50 см", 0.45), ("2 м", 0.25)):
            take = generator.normal(0.0, 0.002, 48_000).astype(np.float32)
            take[24_000:24_320] += amplitude * np.hanning(320).astype(np.float32)
            dialog._manual_reference_takes[distance] = [take]
        dialog._apply_manual_reference_calibration()
        profile = dialog.profiles["__default__"]
        self.assertEqual(profile["reference_count"], 3)
        self.assertEqual(profile["calibration_mode"], "ambient_plus_reference")
    def test_audio_calibration_has_visible_help_for_every_setting(self):
        dialog = AudioCalibrationDialog(self.config.audio, [(3, "Entrance")])
        self.assertGreaterEqual(len(dialog.help_badges), 9)
        self.assertTrue(all(badge.text() == "?" for badge in dialog.help_badges))
        self.assertTrue(
            all(len(badge.toolTip().strip()) > 40 for badge in dialog.help_badges)
        )
        dialog.close()
    def test_cancelled_audio_calibration_does_not_change_app_config(self):
        original = {"3": {"preset": "balanced", "trigger_dbfs": -12.0}}
        self.config.audio.microphone_profiles = original
        with patch("ui.settings_dialog.discover_local_cameras", return_value=[]):
            dialog = SettingsDialog(self.config)
        with patch("ui.settings_dialog_base.AudioCalibrationDialog") as calibration:
            calibration.return_value.exec.return_value = 0
            dialog._open_audio_calibration()
        self.assertIs(self.config.audio.microphone_profiles, original)

    def test_unknown_keyword_is_rejected_with_notification(self):
        with patch("ui.settings_dialog.discover_local_cameras", return_value=[]):
            dialog = SettingsDialog(self.config)
        initial_count = dialog.keywords_list.count()
        dialog.keyword_edit.setText("несуществующееслово")
        with (
            patch.object(dialog, "_keyword_exists", return_value=False),
            patch("ui.settings_dialog.QMessageBox.warning") as warning,
        ):
            dialog._add_keyword()
        self.assertEqual(dialog.keywords_list.count(), initial_count)
        warning.assert_called_once()

    def test_keyword_dictionary_requires_every_word_in_phrase(self):
        with patch("ui.settings_dialog.discover_local_cameras", return_value=[]):
            dialog = SettingsDialog(self.config)
        model = Mock()
        model.vosk_model_find_word.side_effect = lambda word: {
            "помоги": 10,
            "срочно": 11,
            "чужоеслово": -1,
        }[word]
        dialog._keyword_model = model
        self.assertTrue(dialog._keyword_exists("помоги срочно"))
        self.assertFalse(dialog._keyword_exists("помоги чужоеслово"))

    def test_single_camera_expands_after_grid_rebuild(self):
        self.config.video.sources = ["0"]
        self.window._prepare_camera_tiles()
        self.app.processEvents()
        tile = next(iter(self.window.camera_tiles.values()))
        self.assertGreater(tile.width(), self.window.video_scroll.viewport().width() * 0.8)

    def test_one_worker_per_microphone_combines_gunshot_and_speech_roles(self):
        self.config.video.sources = ["cam-a", "cam-b", "cam-c"]
        self.config.video.microphone_by_source = {"cam-a": 3, "cam-b": 3, "cam-c": 4}
        self.config.speech.enabled = True
        self.config.speech.microphone_device = 3
        with patch("ui.dashboard.AudioSystemWorker") as worker_class:
            worker_class.return_value.start.return_value = None
            self.window._start_audio_workers()
        self.assertEqual(worker_class.call_count, 2)
        roles = {
            call.kwargs["mic_device"]: (
                call.kwargs["enable_gunshot"], call.kwargs["enable_speech"]
            )
            for call in worker_class.call_args_list
        }
        self.assertEqual(roles, {3: (True, True), 4: (True, False)})
        self.window.audio_workers.clear()

    def test_fullscreen_toggle_and_escape(self):
        self.window.toggle_fullscreen()
        self.app.processEvents()
        self.assertTrue(self.window.isFullScreen())
        self.window.toggle_fullscreen()
        self.app.processEvents()
        self.assertFalse(self.window.isFullScreen())

    def test_camera_viewer_opens_first_camera_and_navigates_with_keyboard(self):
        camera_ids = list(self.window.camera_tiles)
        frames = [np.full((40, 60, 3), value, dtype=np.uint8) for value in (20, 80, 140)]
        for camera_id, frame in zip(camera_ids, frames):
            self.window.update_video(camera_id, frame, frame)

        self.window.layout_view_button.click()
        self.app.processEvents()
        self.assertTrue(self.window.camera_viewer.isVisible())
        self.assertEqual(self.window.viewer_camera_index, 0)
        self.assertEqual(self.window.camera_viewer.title.text(), "Камера 1")
        self.assertEqual(self.window.camera_viewer.previous_button.y(), 0)
        self.assertEqual(
            self.window.camera_viewer.previous_button.height(),
            self.window.camera_viewer.height(),
        )
        self.assertLessEqual(self.window.camera_viewer.previous_button.width(), 100)

        QTest.keyClick(self.window.camera_viewer, Qt.Key.Key_Right)
        self.assertEqual(self.window.viewer_camera_index, 1)
        self.assertEqual(self.window.camera_viewer.title.text(), "Камера 2")
        QTest.keyClick(self.window.camera_viewer, Qt.Key.Key_Left)
        self.assertEqual(self.window.viewer_camera_index, 0)
        QTest.keyClick(self.window.camera_viewer, Qt.Key.Key_Left)
        self.assertEqual(self.window.viewer_camera_index, 2)

        QTest.keyClick(self.window.camera_viewer, Qt.Key.Key_Escape)
        self.assertFalse(self.window.camera_viewer.isVisible())

    def test_global_camera_mode_switch_updates_all_tiles_without_resizing(self):
        sizes_before = [tile.mode_button.size() for tile in self.window.camera_tiles.values()]
        self.window.all_mode_toggle.setChecked(False)
        self.app.processEvents()
        self.assertTrue(all(tile.display_mode == "raw" for tile in self.window.camera_tiles.values()))
        self.assertEqual(
            [tile.mode_button.size() for tile in self.window.camera_tiles.values()],
            sizes_before,
        )
        self.window.all_mode_toggle.setChecked(True)
        self.assertTrue(
            all(tile.display_mode == "annotated" for tile in self.window.camera_tiles.values())
        )

    def test_stopping_monitoring_clears_last_frames(self):
        frame = np.full((40, 60, 3), 120, dtype=np.uint8)
        for tile in self.window.camera_tiles.values():
            tile.set_frames(frame, frame)
        self.window.monitoring_active = True

        with patch.object(self.window, "stop_all", return_value=True):
            self.window.stop_monitoring()

        for tile in self.window.camera_tiles.values():
            self.assertIsNone(tile.raw_frame)
            self.assertIsNone(tile.annotated_frame)
            self.assertTrue(tile.video.pixmap().isNull())
            self.assertEqual(tile.video.text(), "Мониторинг остановлен")

    def test_camera_dropdown_discovers_only_readable_local_ids(self):
        captures = []
        for opened, readable in ((True, True), (True, False), (False, False)):
            capture = Mock()
            capture.open.return_value = opened
            capture.read.return_value = (readable, object() if readable else None)
            captures.append(capture)
        with patch("ui.settings_dialog.cv2.VideoCapture", side_effect=captures):
            self.assertEqual(discover_local_cameras(3), [0])
        for capture in captures:
            capture.open.assert_called_once()
            capture.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
