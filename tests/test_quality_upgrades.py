import queue
import re
import threading
import unittest
from unittest.mock import Mock
import numpy as np
from audio.gunshot_detector_runtime import (
    EnergyGate,
    GunshotDetector,
    PANNsYAMNetFusion,
)
from audio.keyword_detector import KeywordDetector
from core.event_bus import EventType
from video.fall_detector import FallDetector, TrackState
class FallGeometryUpgradeTests(unittest.TestCase):
    def setUp(self):
        self.detector = FallDetector.__new__(FallDetector)
        self.detector.standing_angle_deg = 30.0
        self.detector.lying_angle_deg = 60.0
        self.detector.lying_aspect_ratio = 1.0
        self.detector.keypoint_confidence = 0.5
        self.detector.seated_knee_drop_ratio = 0.12
        self.detector._states = {}
    @staticmethod
    def keypoints(points):
        result = np.zeros((17, 3), dtype=np.float32)
        for index, (x, y) in points.items():
            result[index] = (x, y, 0.95)
        return result

    def test_full_body_axis_recovers_side_lying_missed_by_torso_only(self):
        points = self.keypoints(
            {
                5: (10, 20),
                6: (10, 30),
                11: (35, 35),
                12: (35, 45),
                13: (60, 42),
                14: (62, 48),
                15: (82, 43),
                16: (84, 49),
            }
        )
        old = self.detector.classify_posture(59.0, 0.85, False)
        posture, metrics = self.detector.classify_side_posture(points, 85, 100)
        self.assertEqual(old, "\u041f\u0430\u0434\u0430\u0435\u0442")
        self.assertEqual(posture, "\u041b\u0435\u0436\u0438\u0442")
        self.assertGreater(metrics["body_axis_angle"], 60.0)

    def test_existing_side_lying_decision_is_never_demoted(self):
        points = self.keypoints(
            {
                5: (10, 20),
                6: (10, 30),
                11: (70, 20),
                12: (70, 30),
            }
        )
        posture, _ = self.detector.classify_side_posture(points, 120, 80)
        self.assertEqual(posture, "\u041b\u0435\u0436\u0438\u0442")

    def test_top_down_footprint_growth_promotes_ambiguous_pose(self):
        self.detector._states[7] = TrackState(safe_bbox_area_ema=1000.0)
        metrics = {
            "lying_score": 0.60,
            "visible_points": 7.0,
        }
        posture = self.detector._refine_top_down_posture(
            7,
            "\u041f\u0430\u0434\u0430\u0435\u0442",
            metrics,
            1500.0,
        )
        self.assertEqual(posture, "\u041b\u0435\u0436\u0438\u0442")
        self.assertAlmostEqual(metrics["footprint_growth"], 1.5)

    def test_top_down_without_growth_does_not_force_lying(self):
        self.detector._states[7] = TrackState(safe_bbox_area_ema=1000.0)
        metrics = {
            "lying_score": 0.60,
            "visible_points": 7.0,
        }
        posture = self.detector._refine_top_down_posture(
            7,
            "\u041f\u0430\u0434\u0430\u0435\u0442",
            metrics,
            1100.0,
        )
        self.assertEqual(posture, "\u041f\u0430\u0434\u0430\u0435\u0442")
class GunshotUpgradeTests(unittest.TestCase):
    @staticmethod
    def detector():
        detector = GunshotDetector.__new__(GunshotDetector)
        detector.fusion = PANNsYAMNetFusion(0.30, 0.15, 0.30)
        return detector

    def test_strict_short_rescue_requires_both_models(self):
        detector = self.detector()
        accepted = detector._strict_short_rescue_decision(
            0.55, 0.04, 0.10, 0.62, 0.05, 0.12, False
        )
        weak_panns = detector._strict_short_rescue_decision(
            0.20, 0.01, 0.02, 0.80, 0.05, 0.10, False
        )
        nuisance = detector._strict_short_rescue_decision(
            0.55, 0.04, 0.10, 0.70, 0.05, 0.80, True
        )
        self.assertTrue(accepted.accepted)
        self.assertFalse(weak_panns.accepted)
        self.assertFalse(nuisance.accepted)
    @staticmethod
    def runtime_detector():
        detector = GunshotDetector.__new__(GunshotDetector)
        detector.sample_rate = 16_000
        detector.yamnet_sample_rate = 16_000
        detector.gate = Mock(spec=EnergyGate)
        detector.gate.peak_min = 0.2
        detector.gate.last_metrics = {"dynamic_peak_min": 0.2}
        detector.rescue_gate = Mock(spec=EnergyGate)
        detector.rescue_gate.peak_min = 0.2
        detector.fusion = PANNsYAMNetFusion(0.30, 0.15, 0.30)
        detector.use_veto = True
        detector.panns = Mock()
        detector.yamnet = Mock()
        detector.multiscale_rescue = True
        detector.rescue_window_sec = 1.0
        detector.window_sec = 3.0
        detector.last = -1e9
        detector.cooldown = 2.0
        detector.event_bus = Mock()
        detector.source = "test"
        detector.camera_ids = ()
        detector.pref = 0.10
        return detector

    def test_baseline_accept_is_authoritative_and_skips_rescue(self):
        detector = self.runtime_detector()
        detector.gate.check.return_value = (True, 0.01, 0.5)
        detector.panns.check.return_value = (0.59, 0.01, 0.04, "Gunshot", 0.59)
        detector.yamnet.check.return_value = (0.50, 0.10, 0.08, "Gunshot", 0.50)
        detector._analyze(10.0, np.zeros(48_000, dtype=np.float32))
        event = detector.event_bus.publish.call_args.args[0]
        self.assertEqual(event.type, EventType.GUNSHOT_DETECTED)
        self.assertEqual(event.metadata["detection_path"], "baseline_3s")
        detector.rescue_gate.check.assert_not_called()
        detector.panns.check.assert_called_once()
        detector.yamnet.check.assert_called_once()

    def test_strict_short_path_adds_detection_after_baseline_reject(self):
        detector = self.runtime_detector()
        detector.gate.check.return_value = (True, 0.01, 0.5)
        detector.rescue_gate.check.return_value = (True, 0.02, 0.5)
        detector.panns.check.side_effect = [
            (0.20, 0.01, 0.10, "Noise", 0.20),
            (0.55, 0.04, 0.10, "Gunshot", 0.55),
        ]
        detector.yamnet.check.side_effect = [
            (0.30, 0.05, 0.12, "Noise", 0.30),
            (0.62, 0.05, 0.12, "Gunshot", 0.62),
        ]
        detector._analyze(10.0, np.zeros(48_000, dtype=np.float32))
        event = detector.event_bus.publish.call_args.args[0]
        self.assertEqual(event.metadata["detection_path"], "strict_short_rescue")
        self.assertEqual(detector.panns.check.call_count, 2)
        self.assertEqual(detector.yamnet.check.call_count, 2)
class SpeechUpgradeTests(unittest.TestCase):
    def detector(self):
        detector = KeywordDetector.__new__(KeywordDetector)
        detector.event_bus = Mock()
        detector.keywords = ["\u043f\u043e\u043c\u043e\u0449\u044c"]
        detector._patterns = {
            word: re.compile(
                rf"(?<![\w-]){re.escape(word)}(?![\w-])",
                re.IGNORECASE,
            )
            for word in detector.keywords
        }
        detector.secondary_min_confidence = 0.30
        detector.cooldown_sec = 3.0
        detector.last_event_time = -float("inf")
        detector.microphone_source = "test"
        detector.camera_ids = ()
        return detector

    def test_secondary_exact_keyword_can_rescue_vosk_miss(self):
        detector = self.detector()
        detector._publish_secondary("\u043f\u043e\u043c\u043e\u0449\u044c \u043d\u0443\u0436\u043d\u0430", 0.84)
        events = [call.args[0] for call in detector.event_bus.publish.call_args_list]
        self.assertEqual(
            [event.type for event in events],
            [EventType.SPEECH_RECOGNIZED, EventType.KEYWORD_DETECTED],
        )
        self.assertEqual(
            events[-1].metadata["recognizer"], "faster_whisper_rescue"
        )

    def test_secondary_sound_imitation_does_not_use_fuzzy_alarm(self):
        detector = self.detector()
        detector._publish_secondary("\u0431\u0434\u044b\u0449", 0.99)
        events = [call.args[0] for call in detector.event_bus.publish.call_args_list]
        self.assertEqual([event.type for event in events], [EventType.SPEECH_RECOGNIZED])

    def test_secondary_disagreement_vetoes_vosk_false_alarm(self):
        detector = self.detector()
        detector._publish_secondary(
            "\u0431\u0434\u044b\u0449",
            0.99,
            (("\u043f\u043e\u043c\u043e\u0449\u044c", 0.95),),
        )
        events = [
            call.args[0] for call in detector.event_bus.publish.call_args_list
        ]
        self.assertEqual(
            [event.type for event in events],
            [EventType.SPEECH_RECOGNIZED],
        )

    def test_secondary_same_word_confirms_vosk_candidate(self):
        detector = self.detector()
        detector._publish_secondary(
            "\u043f\u043e\u043c\u043e\u0449\u044c",
            0.57,
            (("\u043f\u043e\u043c\u043e\u0449\u044c", 0.91),),
        )
        events = [
            call.args[0] for call in detector.event_bus.publish.call_args_list
        ]
        self.assertEqual(events[-1].type, EventType.KEYWORD_DETECTED)
        self.assertEqual(
            events[-1].metadata["recognizer"], "vosk+faster_whisper"
        )

    def test_secondary_queue_skips_non_alarm_speech(self):
        detector = self.detector()
        detector._secondary_queue = Mock()
        detector._secondary_ready = threading.Event()
        detector._secondary_ready.set()
        detector.sample_rate = 16_000
        self.assertTrue(detector._queue_secondary(np.ones(16_000), ()))
        detector._secondary_queue.put_nowait.assert_not_called()

    def test_secondary_queue_replaces_stale_candidate_with_current_one(self):
        detector = self.detector()
        detector._secondary_queue = queue.Queue(maxsize=1)
        detector._secondary_queue.put_nowait((np.ones(16_000), (("старое", 0.9),)))
        detector._secondary_ready = threading.Event()
        detector._secondary_ready.set()
        detector.sample_rate = 16_000
        current = (("помощь", 0.95),)
        self.assertTrue(detector._queue_secondary(np.zeros(16_000), current))
        _, candidates = detector._secondary_queue.get_nowait()
        self.assertEqual(candidates, current)


if __name__ == "__main__":
    unittest.main()
