import re
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import cv2

from audio.gunshot_detector import EnergyGate, GunshotFusion, TemporalEvidenceTracker
from audio.keyword_detector import KeywordDetector, normalize_speech_text
from audio.microphone import (
    DualRateAudioResampler,
    MicrophoneStream,
    StreamingAudioResampler,
    dbfs_to_amplitude,
)
from video.fall_detector import FallDetector, TrackState


class KeywordMatchingTests(unittest.TestCase):
    def setUp(self):
        self.detector = KeywordDetector.__new__(KeywordDetector)
        self.detector._patterns = {
            word: re.compile(rf"(?<![\w-]){re.escape(word)}(?![\w-])", re.IGNORECASE)
            for word in ("помоги", "помощь")
        }
        self.detector.keywords = ["помоги", "помощь"]
        self.detector.fuzzy_match = True

    def test_short_sound_imitation_is_not_an_alarm_word(self):
        self.assertEqual(self.detector.find_keywords("бдыщ"), [])

    def test_recognizer_is_not_restricted_to_alarm_vocabulary(self):
        fake_vosk = SimpleNamespace(Model=Mock(), KaldiRecognizer=Mock())
        with patch.dict(sys.modules, {"vosk": fake_vosk}):
            detector = KeywordDetector(
                event_bus=Mock(),
                model_path="unused",
                keyword=["помощь"],
                fuzzy_match=False,
            )
        fake_vosk.KaldiRecognizer.assert_called_once_with(
            fake_vosk.Model.return_value,
            detector.sample_rate,
        )
    def test_matches_whole_words_only(self):
        self.assertEqual(self.detector.find_keywords("Помоги, пожалуйста!"), [("помоги", 1)])
        self.assertEqual(self.detector.find_keywords("вспомоги мне"), [])

    def test_counts_repeated_words(self):
        self.assertEqual(self.detector.find_keywords("помощь! помощь!"), [("помощь", 2)])

    def test_normalizes_yo_and_accepts_one_typo_in_long_alarm_word(self):
        self.assertEqual(normalize_speech_text("  ПОМОГИ, ёж! "), "помоги еж")
        self.assertEqual(self.detector.find_keywords("помогиь"), [("помоги", 1)])
        self.assertEqual(self.detector.find_keywords("помидор"), [])

    def test_uses_confidence_of_matched_word(self):
        result = {
            "text": "пожалуйста помоги",
            "result": [
                {"word": "пожалуйста", "conf": 0.2},
                {"word": "помоги", "conf": 0.91},
            ],
        }
        self.assertAlmostEqual(
            self.detector._keyword_confidence(result, [("помоги", 1)]), 0.91
        )


class GunshotLogicTests(unittest.TestCase):
    def test_energy_gate_rejects_tonal_background_but_accepts_impulse(self):
        gate = EnergyGate(rms_min=0.005, peak_min=0.2, min_crest_factor=2.5)
        t = np.linspace(0, 1, 16_000, endpoint=False)
        tonal = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        self.assertFalse(gate.check(tonal)[0])

        impulse = np.zeros(16_000, dtype=np.float32)
        impulse[8_000:8_100] = np.hanning(100).astype(np.float32)
        self.assertTrue(gate.check(impulse)[0])
        self.assertGreater(gate.last_metrics["crest_factor"], 2.5)

    def test_strict_fusion_allows_strong_cnn_override(self):
        fusion = GunshotFusion(
            "CNN + YAMNet (Строгий)",
            yamnet_threshold=0.2,
            veto_threshold=0.3,
            veto_margin=0.12,
            cnn_override_threshold=0.92,
        )
        result = fusion.combine(cnn=0.97, gun=0.05, veto=0.5)
        self.assertTrue(result.yamnet_confirmed)
        self.assertFalse(result.semantic_veto)

    def test_speech_veto_needs_a_margin_over_gun_score(self):
        fusion = GunshotFusion("CNN + YAMNet", 0.2, 0.3, veto_margin=0.12)
        self.assertFalse(fusion.combine(0.8, 0.35, 0.4).semantic_veto)
        self.assertTrue(fusion.combine(0.8, 0.1, 0.4).semantic_veto)

    def test_temporal_evidence_pairs_model_peaks_from_adjacent_windows(self):
        evidence = TemporalEvidenceTracker(
            window_sec=2.0,
            cnn_threshold=0.82,
            yamnet_threshold=0.15,
        )
        self.assertFalse(evidence.update(9.5, cnn=0.257, gun=0.675).matched)
        result = evidence.update(10.5, cnn=0.95, gun=0.042)
        self.assertTrue(result.matched)
        self.assertAlmostEqual(result.cnn, 0.95)
        self.assertAlmostEqual(result.gun, 0.675)

    def test_temporal_evidence_expires_and_respects_veto(self):
        evidence = TemporalEvidenceTracker(2.0, 0.82, 0.15)
        evidence.update(1.0, cnn=0.2, gun=0.7)
        self.assertFalse(evidence.update(3.1, cnn=0.95, gun=0.01).matched)

        evidence.reset()
        evidence.update(5.0, cnn=0.2, gun=0.7)
        evidence.update(5.5, cnn=0.1, gun=0.01, semantic_veto=True)
        result = evidence.update(6.0, cnn=0.95, gun=0.01)
        self.assertTrue(result.matched)
        self.assertTrue(result.vetoed)


class PostureTests(unittest.TestCase):
    def setUp(self):
        self.detector = FallDetector.__new__(FallDetector)
        self.detector.standing_angle_deg = 30.0
        self.detector.lying_angle_deg = 60.0
        self.detector.lying_aspect_ratio = 0.8
        self.detector.keypoint_confidence = 0.5
        self.detector.seated_knee_drop_ratio = 0.12

    def test_angle_is_measured_from_vertical(self):
        self.assertAlmostEqual(self.detector.calculate_angle((0, 0), (0, 10)), 0.0)
        self.assertAlmostEqual(self.detector.calculate_angle((0, 0), (10, 0)), 90.0)

    def test_horizontal_torso_requires_wide_bbox_to_be_lying(self):
        self.assertEqual(self.detector.classify_posture(75.0, 1.2), "Лежит")
        self.assertEqual(self.detector.classify_posture(75.0, 0.4), "Падает")

    def test_squat_like_compact_bbox_is_not_lying(self):
        self.detector.lying_aspect_ratio = 1.0
        self.assertEqual(self.detector.classify_posture(65.0, 0.85), "Падает")

    def test_horizontal_torso_with_knees_below_hips_is_seated(self):
        hip_center = (50.0, 40.0)
        knees = [(42.0, 68.0, 0.9), (58.0, 70.0, 0.9)]
        seated = self.detector.has_seated_leg_geometry(hip_center, knees, 100.0)
        self.assertTrue(seated)
        self.assertEqual(self.detector.classify_posture(75.0, 1.2, seated), "Сидит")

    def test_horizontal_lying_legs_do_not_look_seated(self):
        hip_center = (50.0, 40.0)
        knees = [(70.0, 45.0, 0.9), (80.0, 44.0, 0.9)]
        seated = self.detector.has_seated_leg_geometry(hip_center, knees, 100.0)
        self.assertFalse(seated)
        self.assertEqual(self.detector.classify_posture(75.0, 1.2, seated), "Лежит")

    @staticmethod
    def _top_down_keypoints(points: dict[int, tuple[float, float]]) -> np.ndarray:
        keypoints = np.zeros((17, 3), dtype=np.float32)
        for index, (x, y) in points.items():
            keypoints[index] = (x, y, 0.95)
        return keypoints

    def test_top_down_mode_detects_a_long_straight_body_independent_of_rotation(self):
        keypoints = self._top_down_keypoints({
            5: (25, 45), 6: (25, 55), 11: (45, 45), 12: (45, 55),
            13: (72, 45), 14: (72, 55), 15: (102, 45), 16: (102, 55),
        })
        posture, metrics = self.detector.classify_top_down_posture(keypoints, 120, 90)
        self.assertEqual(posture, "Лежит")
        self.assertGreater(metrics["elongation"], 2.0)
        self.assertGreater(metrics["body_extent"], 0.5)

    def test_top_down_mode_keeps_a_compact_person_out_of_lying_state(self):
        keypoints = self._top_down_keypoints({
            5: (47, 48), 6: (47, 52), 11: (50, 48), 12: (50, 52),
            13: (53, 48), 14: (53, 52), 15: (55, 48), 16: (55, 52),
        })
        posture, metrics = self.detector.classify_top_down_posture(keypoints, 100, 100)
        self.assertEqual(posture, "Стоит")
        self.assertLess(metrics["body_extent"], 0.5)

    def test_top_down_mode_recognizes_bent_legs_as_seated(self):
        keypoints = self._top_down_keypoints({
            5: (42, 25), 6: (58, 25), 11: (45, 42), 12: (55, 42),
            13: (30, 55), 14: (70, 55), 15: (34, 75), 16: (66, 75),
        })
        posture, _ = self.detector.classify_top_down_posture(keypoints, 110, 110)
        self.assertEqual(posture, "Сидит")
    def test_sitting_clears_pending_false_fall_after_grace(self):
        self.detector._states = {}
        self.detector.current_time = 0.0
        self.detector.reset_grace_sec = 2.0
        self.detector.posture_window = 1
        self.detector.posture_hits = 1
        self.detector.rapid_descent_heights_per_sec = 0.75
        state = self.detector._update_state(9, "Лежит")
        self.assertIsNotNone(state.fall_since)
        self.detector.current_time = 0.5
        self.detector._update_state(9, "Сидит")
        self.detector.current_time = 2.6
        state = self.detector._update_state(9, "Сидит")
        self.assertIsNone(state.fall_since)
        self.assertFalse(state.seen_upright)

    def test_rapid_descent_never_bypasses_no_recovery_delay(self):
        self.detector.fall_duration_sec = 5.0
        self.detector.motion_confirm_sec = 1.5
        self.detector.current_time = 10.0
        state = TrackState(rapid_descent_at=9.5)
        self.assertEqual(self.detector._required_confirmation_sec(state), 5.0)

    def test_alerted_missing_track_is_kept_active(self):
        self.detector._states = {7: TrackState(alerted=True, fall_since=1.0)}
        self.detector.current_time = 10.0
        self.detector.missing_grace_sec = 2.0
        self.detector._mark_missing(set())
        self.assertIn(7, self.detector._states)

    def test_static_lying_starts_the_five_second_confirmation(self):
        self.detector._states = {}
        self.detector.current_time = 0.0
        self.detector.require_upright_transition = True
        self.detector.require_fall_motion = True
        self.detector.transition_memory_sec = 2.0
        self.detector.posture_window = 3
        self.detector.posture_hits = 2
        self.detector.reset_grace_sec = 2.0
        self.detector.rapid_descent_heights_per_sec = 0.75
        state = None
        for _ in range(3):
            self.detector.current_time += 0.1
            state = self.detector._update_state(8, "Лежит", center_y=100, height=80)
        self.assertEqual(state.fall_since, 0.1)
        self.assertFalse(state.seen_upright)
        self.assertEqual(state.sequence_state, "LYING_CONFIRMATION")

    def test_track_resets_only_after_standing_grace(self):
        self.detector._states = {}
        self.detector.current_time = 0.0
        self.detector.require_upright_transition = True
        self.detector.require_fall_motion = False
        self.detector.reset_grace_sec = 2.0
        self.detector.posture_window = 1
        self.detector.posture_hits = 1
        self.detector.rapid_descent_heights_per_sec = 0.75
        self.detector._update_state(7, "Стоит")
        self.detector.current_time = 1.0
        state = self.detector._update_state(7, "Лежит")
        self.assertEqual(state.fall_since, 1.0)
        self.detector.current_time = 2.0
        self.detector._update_state(7, "Стоит")
        self.detector.current_time = 4.1
        state = self.detector._update_state(7, "Стоит")
        self.assertIsNone(state.fall_since)

    def test_ambiguous_still_pose_does_not_start_fall_transition(self):
        self.detector._states = {}
        self.detector.current_time = 0.0
        self.detector.posture_window = 1
        self.detector.posture_hits = 1
        self.detector.rapid_descent_heights_per_sec = 0.75
        self.detector._update_state(5, "Падает", center_y=50, height=100)
        self.detector.current_time = 0.5
        state = self.detector._update_state(
            5, "Падает", center_y=50, height=100
        )
        self.assertEqual(state.vertical_speed, 0.0)
        self.assertIsNone(state.transition_since)

    def test_rapid_descent_keeps_full_confirmation_delay(self):
        self.detector._states = {}
        self.detector.current_time = 0.0
        self.detector.require_upright_transition = True
        self.detector.reset_grace_sec = 2.0
        self.detector.fall_duration_sec = 5.0
        self.detector.motion_confirm_sec = 1.5
        self.detector.transition_memory_sec = 3.0
        self.detector.rapid_descent_heights_per_sec = 0.75
        self.detector.posture_window = 3
        self.detector.posture_hits = 2

        self.detector._update_state(3, "Стоит", center_y=10, height=100)
        self.detector.current_time = 0.1
        self.detector._update_state(3, "Стоит", center_y=10, height=100)
        self.detector.current_time = 0.6
        state = self.detector._update_state(3, "Падает", center_y=60, height=100)
        self.assertIsNone(state.fall_since)
        self.detector.current_time = 0.9
        self.detector._update_state(3, "Лежит", center_y=70, height=100)
        self.detector.current_time = 1.1
        state = self.detector._update_state(3, "Лежит", center_y=72, height=100)
        self.assertEqual(state.fall_since, 0.9)
        self.assertEqual(self.detector._detection_basis(state), "rapid_descent")
        self.assertEqual(self.detector._required_confirmation_sec(state), 5.0)

    def test_static_lying_uses_long_confirmation(self):
        self.detector._states = {}
        self.detector.current_time = 0.0
        self.detector.require_upright_transition = False
        self.detector.require_fall_motion = False
        self.detector.fall_duration_sec = 5.0
        self.detector.motion_confirm_sec = 1.5
        self.detector.transition_memory_sec = 2.0
        self.detector.rapid_descent_heights_per_sec = 0.75
        self.detector.posture_window = 1
        self.detector.posture_hits = 1
        state = self.detector._update_state(4, "Лежит", center_y=50, height=100)
        self.assertEqual(self.detector._required_confirmation_sec(state), 5.0)

    def test_fall_event_stays_bound_to_its_camera_and_frame(self):
        class Tensor:
            def __init__(self, value):
                self.value = np.asarray(value)

            def cpu(self):
                return self

            def numpy(self):
                return self.value

            def int(self):
                return Tensor(self.value.astype(int))

            def tolist(self):
                return self.value.tolist()

        keypoints = np.zeros((1, 17, 3), dtype=np.float32)
        keypoints[0, 5] = (10, 10, 0.9)
        keypoints[0, 6] = (20, 10, 0.9)
        keypoints[0, 11] = (50, 10, 0.9)
        keypoints[0, 12] = (60, 10, 0.9)
        annotated = np.full((50, 100, 3), 123, dtype=np.uint8)
        result = Mock()
        result.keypoints.data = Tensor(keypoints)
        result.boxes.xyxy = Tensor([[0, 0, 100, 50]])
        result.boxes.cls = Tensor([0])
        result.boxes.id = Tensor([7])
        result.plot.return_value = annotated
        model = Mock()
        model.track.return_value = [result]

        fake_ultralytics = SimpleNamespace(YOLO=Mock(return_value=model))
        with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
            detector = FallDetector(
                camera_id="camera-test",
                model_path="unused.pt",
                fall_duration_sec=1.0,
                fps=1.0,
                use_wall_clock=False,
                require_upright_transition=False,
                require_fall_motion=False,
                posture_window=1,
                posture_hits=1,
            )
        detector.use_half_precision = True
        first = detector.process_frame(np.zeros_like(annotated))
        second = detector.process_frame(np.zeros_like(annotated))
        self.assertEqual(first.events, ())
        self.assertEqual(second.events[0].metadata["camera_id"], "camera-test")
        self.assertIs(second.annotated_frame, annotated)
        self.assertEqual(model.track.call_args.kwargs["quantize"], 16)
        self.assertNotIn("half", model.track.call_args.kwargs)
        result.plot.assert_called_with(kpt_line=True, kpt_radius=4)


class MicrophoneBufferTests(unittest.TestCase):
    def test_windows_microphone_list_prefers_wasapi(self):
        fake_sounddevice = SimpleNamespace(
            query_devices=Mock(return_value=[
                {"name": "Mic A", "hostapi": 0, "max_input_channels": 1},
                {"name": "Mic A", "hostapi": 1, "max_input_channels": 1},
                {"name": "Mic B", "hostapi": 1, "max_input_channels": 2},
            ]),
            query_hostapis=Mock(return_value=[
                {"name": "MME"},
                {"name": "Windows WASAPI"},
            ]),
        )
        with (
            patch.dict(sys.modules, {"sounddevice": fake_sounddevice}),
            patch("audio.microphone.os.name", "nt"),
        ):
            self.assertEqual(MicrophoneStream.list_input_devices(), [(1, "Mic A"), (2, "Mic B")])

    def test_dbfs_threshold_is_converted_to_linear_amplitude(self):
        self.assertAlmostEqual(dbfs_to_amplitude(-20), 0.1)
        self.assertAlmostEqual(dbfs_to_amplitude(0), 1.0)

    def test_explicit_microphone_does_not_fallback(self):
        stream = MicrophoneStream(device=7, allow_fallback=False)
        with patch.object(stream, "_open_stream", side_effect=RuntimeError("missing")) as opened:
            with self.assertRaises(RuntimeError):
                stream.start()
        opened.assert_called_once_with(7)

    def test_callback_produces_fixed_size_chunks(self):
        stream = MicrophoneStream(sample_rate=4, chunk_duration_sec=0.5, max_buffer_sec=1.0)
        stream.device_sample_rate = 4
        stream._stopped = False
        stream._callback(np.array([[0.1], [0.2], [0.3], [0.4]], dtype=np.float32), 4, None, None)
        np.testing.assert_allclose(stream.get_chunk(), [0.1, 0.2])
        np.testing.assert_allclose(stream.get_chunk(), [0.3, 0.4])

    def test_native_mode_keeps_device_rate_and_chunk_duration(self):
        stream = MicrophoneStream(
            sample_rate=4,
            chunk_duration_sec=0.5,
            max_buffer_sec=1.0,
            preserve_native=True,
        )
        stream._configure_device_rate(8)
        self.assertEqual(stream.sample_rate, 8)
        self.assertEqual(stream.chunk_size, 4)
        stream._stopped = False
        values = np.arange(8, dtype=np.float32).reshape(-1, 1)
        stream._callback(values, 8, None, None)
        np.testing.assert_allclose(stream.get_chunk(), [0, 1, 2, 3])
        np.testing.assert_allclose(stream.get_chunk(), [4, 5, 6, 7])

    def test_stateful_resampler_preserves_panns_high_band_only_at_32khz(self):
        source_rate = 48_000
        timeline = np.arange(source_rate, dtype=np.float32) / source_rate
        source = np.sin(2 * np.pi * 12_000 * timeline).astype(np.float32)
        to_panns = StreamingAudioResampler(source_rate, 32_000)
        to_semantic = StreamingAudioResampler(source_rate, 16_000)

        chunks = (source[:17_000], source[17_000:33_000], source[33_000:])
        panns = np.concatenate([
            to_panns.process(chunks[0]),
            to_panns.process(chunks[1]),
            to_panns.process(chunks[2], last=True),
        ])
        semantic = np.concatenate([
            to_semantic.process(chunks[0]),
            to_semantic.process(chunks[1]),
            to_semantic.process(chunks[2], last=True),
        ])

        self.assertEqual(len(panns), 32_000)
        self.assertEqual(len(semantic), 16_000)
        self.assertGreater(float(np.sqrt(np.mean(panns * panns))), 0.60)
        self.assertLess(float(np.sqrt(np.mean(semantic * semantic))), 0.01)

    def test_dual_rate_resampler_keeps_equal_time_and_full_audio(self):
        source_rate = 48_000
        timeline = np.arange(source_rate, dtype=np.float32) / source_rate
        source = np.sin(2 * np.pi * 2_000 * timeline).astype(np.float32)
        resampler = DualRateAudioResampler(source_rate, 32_000, 16_000)
        chunks = (source[:17_000], source[17_000:33_000], source[33_000:])
        pairs = [
            resampler.process(chunks[0]),
            resampler.process(chunks[1]),
            resampler.process(chunks[2], last=True),
        ]
        panns = [panns for panns, _ in pairs if panns.size]
        semantic = [semantic for _, semantic in pairs if semantic.size]
        self.assertTrue(panns)
        self.assertTrue(semantic)
        self.assertTrue(all(len(p) == 2 * len(s) for p, s in pairs))
        self.assertEqual(sum(map(len, panns)), 32_000)
        self.assertEqual(sum(map(len, semantic)), 16_000)

    def test_dual_rate_resampler_keeps_pending_branch_until_partner_arrives(self):
        resampler = DualRateAudioResampler(48_000, 32_000, 16_000)
        resampler._panns_resampler.process = Mock(side_effect=[
            np.array([1.0, 2.0], dtype=np.float32),
            np.empty(0, dtype=np.float32),
        ])
        resampler._semantic_resampler.process = Mock(side_effect=[
            np.empty(0, dtype=np.float32),
            np.array([3.0], dtype=np.float32),
        ])
        panns, semantic = resampler.process(np.ones(8, dtype=np.float32))
        self.assertFalse(panns.size)
        self.assertFalse(semantic.size)
        panns, semantic = resampler.process(np.ones(8, dtype=np.float32))
        np.testing.assert_array_equal(panns, [1.0, 2.0])
        np.testing.assert_array_equal(semantic, [3.0])

    def test_rtsp_audio_uses_ffmpeg_audio_stream_and_normalizes_pcm(self):
        capture = Mock()
        capture.open.return_value = True
        capture.grab.return_value = True
        capture.retrieve.return_value = (
            True,
            np.array([[0, 16384, -16384, 32767]], dtype=np.int16),
        )

        def capture_property(prop):
            return {
                cv2.CAP_PROP_AUDIO_TOTAL_STREAMS: 1,
                cv2.CAP_PROP_AUDIO_SAMPLES_PER_SECOND: 8,
                cv2.CAP_PROP_AUDIO_BASE_INDEX: 0,
            }.get(prop, 0)

        capture.get.side_effect = capture_property
        stream = MicrophoneStream(
            sample_rate=8,
            chunk_duration_sec=0.5,
            device="rtsp://camera/audio",
            allow_fallback=False,
        )
        with (
            patch("cv2.VideoCapture", return_value=capture),
            patch("audio.microphone.threading.Thread") as thread,
        ):
            stream.start()
            self.assertTrue(stream._read_network_audio())
            chunk = stream.get_chunk()
            stream.stop()

        params = capture.open.call_args.args[2]
        self.assertIn(cv2.CAP_PROP_AUDIO_STREAM, params)
        self.assertIn(cv2.CAP_PROP_VIDEO_STREAM, params)
        np.testing.assert_allclose(chunk, [0.0, 0.5, -0.5, 32767 / 32768], rtol=1e-4)
        thread.return_value.start.assert_called_once()
        capture.release.assert_called()


if __name__ == "__main__":
    unittest.main()
