import unittest

import numpy as np

from video.fall_detector import FallDetector, TrackState


class LyingClassifierFusionTests(unittest.TestCase):
    def make_detector(self):
        detector = FallDetector.__new__(FallDetector)
        detector._states = {}
        detector.current_time = 0.0
        detector.camera_id = "camera-test"
        detector.view_mode = "side"
        detector.temporal_window_sec = 0.75
        detector.posture_window = 1
        detector.reset_grace_sec = 2.0
        detector.transition_memory_sec = 2.0
        detector.rapid_descent_heights_per_sec = 0.75
        detector.require_upright_transition = False
        detector.require_fall_motion = False
        detector.fall_duration_sec = 5.0
        detector.lying_classifier_threshold = 0.75
        detector.lying_alert_score_threshold = 0.55
        detector.lying_classifier_pose_quality_min = 0.45
        detector.lying_classifier_rgb_weight = 0.65
        detector.lying_stable_duration_sec = 1.0
        detector.lying_stable_motion_heights_per_sec = 0.12
        return detector

    def test_rgb_fallback_accepts_lying_when_pose_is_unusable(self):
        detector = self.make_detector()
        metrics = {"lying_score": 0.0, "pose_quality": 0.0, "full_pose": 0.0}

        posture = detector._fuse_side_lying_scores("Неизвестно", metrics, 0.91)

        self.assertEqual(posture, "Лежит")
        self.assertAlmostEqual(metrics["lying_score"], 0.91)
        self.assertEqual(metrics["pose_good"], 0.0)

    def test_pose_and_rgb_scores_are_fused_when_pose_is_good(self):
        detector = self.make_detector()
        metrics = {"lying_score": 0.50, "pose_quality": 0.90, "full_pose": 1.0}

        posture = detector._fuse_side_lying_scores("Падает", metrics, 0.90)

        self.assertEqual(posture, "Лежит")
        self.assertAlmostEqual(metrics["lying_score"], 0.76, places=2)
        self.assertEqual(metrics["pose_good"], 1.0)

    def test_non_lying_rgb_does_not_promote_broken_pose(self):
        detector = self.make_detector()
        metrics = {"lying_score": 0.0, "pose_quality": 0.0, "full_pose": 0.0}

        posture = detector._fuse_side_lying_scores("Неизвестно", metrics, 0.10)

        self.assertEqual(posture, "Неизвестно")

    def test_stability_requires_one_second_without_bbox_motion(self):
        detector = self.make_detector()
        state = detector._update_state(7, "Лежит", center_x=50, center_y=50, height=100)
        self.assertEqual(state.lying_since, 0.0)
        self.assertFalse(detector._stable_for_fall(state))

        detector.current_time = 1.0
        state = detector._update_state(7, "Лежит", center_x=50, center_y=50, height=100)
        self.assertTrue(detector._stable_for_fall(state))

        detector.current_time = 1.1
        state = detector._update_state(7, "Лежит", center_x=100, center_y=50, height=100)
        self.assertIsNone(state.stable_since)

    def test_visible_lying_label_is_the_timer_source_of_truth(self):
        detector = self.make_detector()
        confident = {"lying_score": 0.55}
        weak = {"lying_score": 0.54}

        self.assertEqual(detector._timer_posture("Лежит", confident), "Лежит")
        self.assertEqual(detector._timer_posture("Лежит", weak), "Лежит")

    def test_qualified_lying_sends_event_after_five_seconds(self):
        detector = self.make_detector()
        detector.current_time = 5.0
        detector.fall_min_confidence = 0.55
        detector.keypoint_confidence = 0.35
        detector.lying_stable_duration_sec = 0.0
        detector.event_bus = None
        state = TrackState(fall_since=0.0, lying_since=0.0, posture="Лежит")
        keypoints = np.zeros((17, 3), dtype=np.float32)
        for index in (5, 6, 11, 12):
            keypoints[index, 2] = 0.9
        metrics = {
            "lying_score": 0.55,
            "pose_lying_score": 0.55,
            "rgb_lying_score": None,
            "partial_pose": 0.0,
            "sitting_score": 0.0,
            "torso_angle": 70.0,
        }
        events = []

        detector._side_event_and_label(
            keypoints, state, 7, "Лежит", metrics, 120.0, 80.0, events
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].metadata["track_id"], 7)
        self.assertTrue(state.alerted)


if __name__ == "__main__":
    unittest.main()
