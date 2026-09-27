import unittest

from video.fall_detector import FallDetector, TrackState, dismiss_fall_track


class FallAlarmLifecycleTests(unittest.TestCase):
    def make_detector(self):
        detector = FallDetector.__new__(FallDetector)
        detector.camera_id = "camera-test"
        detector._states = {}
        detector.current_time = 0.0
        detector.posture_window = 1
        detector.posture_hits = 1
        detector.reset_grace_sec = 2.0
        detector.rapid_descent_heights_per_sec = 0.75
        detector.transition_memory_sec = 2.0
        detector.fall_duration_sec = 5.0
        detector.motion_confirm_sec = 1.5
        detector.temporal_window_sec = 0.75
        detector.view_mode = "side"
        return detector

    def test_missing_tracker_ids_get_fallback_ids(self):
        self.assertEqual(FallDetector._resolve_track_ids(None, 3), [-1, -2, -3])

    def test_alerted_fall_marks_recovery_after_standing_grace(self):
        detector = self.make_detector()
        detector._states[7] = TrackState(alerted=True, fall_since=0.0, posture="Лежит")

        detector.current_time = 10.0
        state = detector._update_state(7, "Стоит", center_y=50, height=100)
        self.assertTrue(state.alerted)
        self.assertFalse(state.recovered_pending)

        detector.current_time = 12.1
        state = detector._update_state(7, "Стоит", center_y=50, height=100)
        self.assertFalse(state.alerted)
        self.assertIsNone(state.fall_since)
        self.assertTrue(state.recovered_pending)

    def test_side_alarm_returns_to_observing_after_recovery(self):
        detector = self.make_detector()
        detector._states[4] = TrackState(alerted=True, fall_since=1.0, posture="Лежит")

        detector.current_time = 5.0
        first = detector._update_state(4, "Стоит", center_x=20, center_y=20, height=100)
        self.assertTrue(first.alerted)

        detector.current_time = 7.1
        recovered = detector._update_state(4, "Стоит", center_x=20, center_y=20, height=100)
        self.assertFalse(recovered.alerted)
        self.assertIsNone(recovered.fall_since)
        self.assertTrue(recovered.recovered_pending)
        self.assertFalse(any(state.alerted for state in detector._states.values()))

    def test_false_alarm_is_suppressed_until_person_stands(self):
        detector = self.make_detector()
        detector._states[9] = TrackState(alerted=True, fall_since=1.0, posture="Лежит")
        dismiss_fall_track("camera-test", 9)

        detector.current_time = 10.0
        state = detector._update_state(9, "Лежит", center_y=50, height=100)
        self.assertFalse(state.alerted)
        self.assertIsNone(state.fall_since)
        self.assertTrue(state.suppressed_until_standing)

        detector.current_time = 15.0
        state = detector._update_state(9, "Лежит", center_y=50, height=100)
        self.assertIsNone(state.fall_since)
        self.assertTrue(state.suppressed_until_standing)

        detector.current_time = 16.0
        detector._update_state(9, "Стоит", center_y=20, height=100)
        detector.current_time = 18.1
        state = detector._update_state(9, "Стоит", center_y=20, height=100)
        self.assertFalse(state.suppressed_until_standing)

        detector.current_time = 19.0
        state = detector._update_state(9, "Лежит", center_y=50, height=100)
        self.assertEqual(state.fall_since, 19.0)

    def test_false_alarm_suppression_survives_single_person_track_id_change(self):
        detector = self.make_detector()
        detector._states[9] = TrackState(alerted=True, fall_since=1.0, posture="Лежит")
        dismiss_fall_track("camera-test", 9)

        detector._apply_pending_dismissals({21})
        self.assertNotIn(9, detector._states)
        self.assertIn(21, detector._states)

        detector.current_time = 10.0
        state = detector._update_state(21, "Лежит", center_y=50, height=100)
        self.assertTrue(state.suppressed_until_standing)
        self.assertIsNone(state.fall_since)

    def test_suppressed_fall_survives_temporary_tracking_loss(self):
        detector = self.make_detector()
        detector.missing_grace_sec = 2.0
        detector._states[9] = TrackState(
            posture="Лежит",
            suppressed_until_standing=True,
            missing_since=9.0,
        )

        detector.current_time = 10.0
        detector._mark_missing(set())
        self.assertIn(9, detector._states)

        detector._apply_pending_dismissals({21})
        detector.current_time = 10.5
        state = detector._update_state(21, "Лежит", center_y=50, height=100)
        self.assertTrue(state.suppressed_until_standing)
        self.assertIsNone(state.fall_since)

        detector._apply_pending_dismissals({35})
        detector.current_time = 11.0
        state = detector._update_state(35, "Лежит", center_y=50, height=100)
        self.assertTrue(state.suppressed_until_standing)
        self.assertIsNone(state.fall_since)

    def test_suppression_expires_after_track_is_missing(self):
        detector = self.make_detector()
        detector.missing_grace_sec = 2.0
        detector._states[9] = TrackState(
            posture="Лежит",
            suppressed_until_standing=True,
            missing_since=1.0,
        )

        detector.current_time = 2.5
        detector._apply_pending_dismissals({})
        self.assertTrue(detector._states[9].suppressed_until_standing)

        detector.current_time = 3.1
        detector._apply_pending_dismissals({})
        self.assertFalse(detector._states[9].suppressed_until_standing)

        detector.current_time = 4.0
        state = detector._update_state(9, "Лежит", center_y=50, height=100)
        self.assertEqual(state.fall_since, 4.0)

    def test_multiple_people_keep_their_own_state_after_tracker_id_changes(self):
        detector = self.make_detector()
        detector._states[1] = TrackState(
            alerted=True, fall_since=1.0, last_center_x=100.0,
            last_center_y=120.0, last_height=90.0,
        )
        detector._states[2] = TrackState(
            fall_since=2.0, last_center_x=420.0,
            last_center_y=150.0, last_height=110.0,
        )

        detector._apply_pending_dismissals({
            11: (108.0, 122.0, 92.0),
            12: (414.0, 151.0, 108.0),
        })

        self.assertNotIn(1, detector._states)
        self.assertNotIn(2, detector._states)
        self.assertTrue(detector._states[11].alerted)
        self.assertEqual(detector._states[12].fall_since, 2.0)

        detector.current_time = 10.0
        detector._update_state(11, "Стоит", center_x=108.0, center_y=122.0, height=92.0)
        detector.current_time = 12.1
        recovered = detector._update_state(
            11, "Стоит", center_x=108.0, center_y=122.0, height=92.0
        )
        self.assertFalse(recovered.alerted)
        self.assertTrue(recovered.recovered_pending)


if __name__ == "__main__":
    unittest.main()
