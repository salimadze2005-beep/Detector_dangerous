import unittest

import numpy as np

from video.fall_detector import FallDetector


class TopDownPostureTests(unittest.TestCase):
    def make_detector(self) -> FallDetector:
        detector = FallDetector.__new__(FallDetector)
        detector._states = {}
        detector.current_time = 0.0
        detector.keypoint_confidence = 0.50
        detector.standing_angle_deg = 30.0
        detector.lying_angle_deg = 60.0
        detector.lying_aspect_ratio = 1.0
        detector.seated_knee_drop_ratio = 0.12
        detector.temporal_window_sec = 0.75
        detector.posture_window = 5
        detector.rapid_descent_heights_per_sec = 0.75
        detector.transition_memory_sec = 2.0
        detector.reset_grace_sec = 2.0
        detector.fall_duration_sec = 5.0
        detector.require_upright_transition = True
        detector.require_fall_motion = True
        detector.view_mode = "top_down"
        return detector

    @staticmethod
    def keypoints(points, confidence=0.95):
        result = np.zeros((17, 3), dtype=np.float32)
        for index, (x, y) in points.items():
            result[index] = (x, y, confidence)
        return result

    def test_top_down_straight_lying_is_rotation_independent(self):
        detector = self.make_detector()
        lying = self.keypoints(
            {
                5: (20, 45), 6: (20, 55),
                11: (45, 45), 12: (45, 55),
                13: (75, 45), 14: (75, 55),
                15: (105, 45), 16: (105, 55),
            },
            confidence=0.40,
        )
        posture, metrics = detector.classify_top_down_posture(lying, 120, 100)
        self.assertEqual(posture, "Лежит")
        self.assertGreater(metrics["axial_ratio"], 4.0)
        self.assertGreater(metrics["lying_score"], 0.60)

    def test_top_down_compact_standing_remains_standing(self):
        detector = self.make_detector()
        standing = self.keypoints(
            {
                5: (42, 40), 6: (58, 40),
                11: (45, 50), 12: (55, 50),
                13: (47, 58), 14: (53, 58),
                15: (48, 66), 16: (52, 66),
            }
        )
        posture, metrics = detector.classify_top_down_posture(standing, 120, 100)
        self.assertEqual(posture, "Стоит")
        self.assertGreater(metrics["standing_score"], metrics["lying_score"])
        self.assertLess(metrics["torso_ratio"], 1.25)

    def test_top_down_real_sitting_requires_knee_bend(self):
        detector = self.make_detector()
        sitting = self.keypoints(
            {
                5: (42, 25), 6: (58, 25),
                11: (45, 42), 12: (55, 42),
                13: (30, 55), 14: (70, 55),
                15: (34, 75), 16: (66, 75),
            }
        )
        posture, metrics = detector.classify_top_down_posture(sitting, 120, 100)
        self.assertEqual(posture, "Сидит")
        self.assertGreaterEqual(metrics["leg_bend_score"], 0.35)
        self.assertGreater(metrics["sitting_score"], metrics["lying_score"])

    def test_top_down_curled_diagonal_lying_not_misread_as_sitting(self):
        detector = self.make_detector()
        curled = self.keypoints(
            {
                5: (20, 30), 6: (30, 22),
                11: (48, 38), 12: (54, 45),
                13: (70, 60), 14: (61, 66),
                15: (52, 78), 16: (43, 72),
            }
        )
        posture, metrics = detector.classify_top_down_posture(curled, 120, 100)
        self.assertEqual(posture, "Лежит")
        self.assertGreater(metrics["torso_ratio"], 2.0)
        self.assertGreater(metrics["lying_score"], metrics["sitting_score"])

    def test_top_down_hysteresis_keeps_lying_across_borderline_frame(self):
        detector = self.make_detector()
        state = detector._state(8)
        state.stable_posture = "Лежит"
        state.posture = "Лежит"
        metrics = {
            "lying_score": 0.50,
            "sitting_score": 0.42,
            "standing_score": 0.20,
            "footprint_growth": 1.0,
        }
        refined = detector._refine_top_down_posture(
            8, "Падает", metrics, bbox_area=5000
        )
        self.assertEqual(refined, "Лежит")


if __name__ == "__main__":
    unittest.main()
