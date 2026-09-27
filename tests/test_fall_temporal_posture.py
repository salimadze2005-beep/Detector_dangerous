import unittest

import numpy as np

from video.fall_detector import FallDetector


class TemporalPostureTests(unittest.TestCase):
    def make_detector(self) -> FallDetector:
        detector = FallDetector.__new__(FallDetector)
        detector._states = {}
        detector.current_time = 0.0
        detector.standing_angle_deg = 30.0
        detector.lying_angle_deg = 60.0
        detector.lying_aspect_ratio = 1.0
        detector.keypoint_confidence = 0.5
        detector.seated_knee_drop_ratio = 0.12
        detector.temporal_window_sec = 0.75
        detector.posture_window = 5
        detector.rapid_descent_heights_per_sec = 0.75
        detector.transition_memory_sec = 2.0
        detector.reset_grace_sec = 2.0
        detector.fall_duration_sec = 5.0
        detector.require_upright_transition = True
        detector.require_fall_motion = True
        detector.view_mode = "side"
        detector.lying_stable_motion_heights_per_sec = 0.12
        return detector

    @staticmethod
    def keypoints(points: dict[int, tuple[float, float]], confidence=0.95) -> np.ndarray:
        result = np.zeros((17, 3), dtype=np.float32)
        for index, (x, y) in points.items():
            result[index] = (x, y, confidence)
        return result

    def test_standing_hysteresis_ignores_small_threshold_jitter(self):
        detector = self.make_detector()
        metrics = {"torso_angle": 34.0, "aspect_ratio": 0.65, "lying_score": 0.10, "sitting_score": 0.10}
        self.assertEqual(detector._posture(metrics, "Стоит"), "Стоит")
        metrics["torso_angle"] = 45.0
        self.assertEqual(detector._posture(metrics, "Стоит"), "Падает")

    def test_lying_hysteresis_does_not_flip_on_one_borderline_frame(self):
        detector = self.make_detector()
        metrics = {"torso_angle": 53.0, "aspect_ratio": 0.90, "lying_score": 0.58, "sitting_score": 0.10}
        self.assertEqual(detector._posture(metrics, "Лежит"), "Лежит")

    def test_temporal_window_is_time_based_not_frame_count_based(self):
        detector = self.make_detector()
        standing = self.keypoints({5:(45,10),6:(55,10),11:(45,40),12:(55,40),13:(45,65),14:(55,65),15:(45,90),16:(55,90)})
        for timestamp in (0.00, 0.03, 0.06, 0.09, 0.40, 0.80):
            detector.current_time = timestamp
            detector._temporal_side_posture(1, standing, 60, 100)
        samples = list(detector._states[1].pose_samples)
        self.assertTrue(samples)
        self.assertGreaterEqual(samples[0].t, 0.05)
        self.assertLessEqual(samples[-1].t - samples[0].t, 0.75)

    def test_bent_dropped_legs_strengthen_sitting_classification(self):
        detector = self.make_detector()
        seated = self.keypoints({5:(45,10),6:(55,10),11:(45,40),12:(55,40),13:(35,65),14:(65,65),15:(35,90),16:(65,90)})
        posture, metrics = detector.classify_side_posture(seated, 60, 100)
        self.assertEqual(posture, "Сидит")
        self.assertGreaterEqual(metrics["sitting_score"], 0.68)
        self.assertGreater(metrics["knee_drop_ratio"], 0.20)

    def test_multifeature_score_recovers_horizontal_full_body(self):
        detector = self.make_detector()
        points = self.keypoints({5:(10,20),6:(10,30),11:(35,35),12:(35,45),13:(60,42),14:(62,48),15:(82,43),16:(84,49)})
        posture, metrics = detector.classify_side_posture(points, 85, 100)
        self.assertEqual(posture, "Лежит")
        self.assertGreater(metrics["body_axis_angle"], 60.0)
        self.assertGreater(metrics["lying_score"], 0.60)

    def test_end_on_lying_feet_toward_camera_is_not_sitting(self):
        detector = self.make_detector()
        prone = self.keypoints({
            5:(46,20),6:(54,20),11:(47,31),12:(53,31),
            13:(46,56),14:(54,56),15:(45,90),16:(55,90),
        })
        posture, metrics = detector.classify_side_posture(prone, 60, 100)
        self.assertEqual(posture, "Лежит")
        self.assertGreaterEqual(metrics["end_on_lying_score"], 0.50)
        self.assertLess(metrics["sitting_score"], 0.68)

    def test_end_on_lying_head_toward_camera_is_not_sitting(self):
        detector = self.make_detector()
        prone = self.keypoints({
            5:(40,10),6:(60,10),11:(44,40),12:(56,40),
            13:(47,48),14:(53,48),15:(48,52),16:(52,52),
        })
        posture, metrics = detector.classify_side_posture(prone, 55, 100)
        self.assertEqual(posture, "Лежит")
        self.assertGreater(metrics["segment_imbalance"], 2.0)
        self.assertGreater(metrics["end_on_lying_score"], 0.35)

    def test_normal_standing_does_not_trigger_end_on_expert(self):
        detector = self.make_detector()
        standing = self.keypoints({
            5:(45,10),6:(55,10),11:(45,40),12:(55,40),
            13:(45,65),14:(55,65),15:(45,90),16:(55,90),
        })
        posture, metrics = detector.classify_side_posture(standing, 60, 100)
        self.assertEqual(posture, "Стоит")
        self.assertLess(metrics["end_on_lying_score"], 0.50)

    def test_lying_hysteresis_survives_end_on_rotation(self):
        detector = self.make_detector()
        metrics = {
            "torso_angle": 12.0,
            "aspect_ratio": 0.62,
            "lying_score": 0.57,
            "sitting_score": 0.40,
            "end_on_lying_score": 0.57,
            "leg_bend_score": 0.1,
        }
        self.assertEqual(detector._posture(metrics, "Лежит"), "Лежит")

    def test_partial_horizontal_skeleton_from_nano_is_still_lying(self):
        detector = self.make_detector()
        detector.partial_pose_confidence = 0.25
        detector.partial_pose_min_keypoints = 3
        detector.partial_lie_min_score = 0.72
        prone = self.keypoints(
            {5: (10, 30), 11: (40, 35), 15: (80, 38), 16: (85, 42)},
            confidence=0.35,
        )
        posture, metrics = detector.classify_side_posture(prone, 100, 70)
        self.assertEqual(posture, "Лежит")
        self.assertEqual(metrics["full_pose"], 0.0)
        self.assertEqual(metrics["partial_pose"], 1.0)
        self.assertTrue(detector._side_pose_is_usable(metrics))

    def test_partial_vertical_skeleton_is_not_lying(self):
        detector = self.make_detector()
        detector.partial_pose_confidence = 0.25
        detector.partial_pose_min_keypoints = 3
        detector.partial_lie_min_score = 0.72
        upright = self.keypoints(
            {5: (50, 10), 11: (51, 40), 15: (52, 80)}, confidence=0.35
        )
        posture, metrics = detector.classify_side_posture(upright, 70, 100)
        self.assertNotEqual(posture, "Лежит")
        self.assertFalse(detector._side_pose_is_usable(metrics))

    def test_timer_posture_matches_visible_lying_label(self):
        detector = self.make_detector()
        self.assertEqual(
            detector._timer_posture("Лежит", {"lying_score": 0.51}),
            "Лежит",
        )

    def test_person_already_lying_at_start_starts_alarm_timer(self):
        detector = self.make_detector()
        detector.current_time = 1.25
        state = detector._update_state(2, "Лежит", center_y=50, height=100)
        self.assertEqual(state.fall_since, 1.25)
        self.assertEqual(state.lying_since, 1.25)
        self.assertFalse(state.seen_upright)
        self.assertEqual(state.sequence_state, "LYING_CONFIRMATION")

    def test_static_falling_label_does_not_start_timer(self):
        detector = self.make_detector()
        detector._update_state(3, "Стоит", center_y=50, height=100)
        detector.current_time = 0.5
        state = detector._update_state(3, "Падает", center_y=50, height=100)
        self.assertEqual(state.posture, "Падает")
        self.assertIsNone(state.fall_since)

    def test_rapid_descent_starts_timer_when_lying_is_first_seen(self):
        detector = self.make_detector()
        detector._update_state(4, "Стоит", center_y=10, height=100)
        detector.current_time = 0.1
        detector._update_state(4, "Стоит", center_y=10, height=100)
        detector.current_time = 0.6
        falling = detector._update_state(4, "Падает", center_y=60, height=100)
        self.assertIsNotNone(falling.rapid_descent_at)
        detector.current_time = 0.9
        lying = detector._update_state(4, "Лежит", center_y=70, height=100)
        self.assertEqual(lying.fall_since, 0.9)
        self.assertEqual(lying.sequence_state, "LYING_CONFIRMATION")
        self.assertEqual(detector._detection_basis(lying), "rapid_descent")

    def test_timer_keeps_original_start_while_person_remains_lying(self):
        detector = self.make_detector()
        detector.current_time = 2.0
        first = detector._update_state(5, "Лежит", center_y=50, height=100)
        detector.current_time = 3.0
        second = detector._update_state(5, "Лежит", center_y=50, height=100)
        self.assertEqual(first.fall_since, 2.0)
        self.assertEqual(second.fall_since, 2.0)
        self.assertEqual(second.lying_since, 2.0)

    def test_top_down_accepts_lower_confidence_keypoints(self):
        detector = self.make_detector()
        detector.view_mode = "top_down"
        lying = self.keypoints({5:(20,45),6:(20,55),11:(45,45),12:(45,55),13:(75,45),14:(75,55),15:(105,45),16:(105,55)}, confidence=0.40)
        posture, metrics = detector.classify_top_down_posture(lying, 120, 90)
        self.assertEqual(posture, "Лежит")
        self.assertGreaterEqual(metrics["visible_points"], 5)
        self.assertGreater(metrics["lying_score"], 0.55)

    def test_top_down_static_lying_starts_alarm_timer(self):
        detector = self.make_detector()
        detector.view_mode = "top_down"
        detector.current_time = 4.0
        state = detector._update_state(11, "Лежит", center_x=50, bbox_area=5000)
        self.assertEqual(state.fall_since, 4.0)
        self.assertEqual(state.lying_since, 4.0)
        self.assertEqual(state.sequence_state, "LYING_CONFIRMATION")


if __name__ == "__main__":
    unittest.main()
