import unittest

from audio.gunshot_detector import ASTResult, AST_GROUPS, PANNsASTFusion


def ast_result(**overrides):
    scores = {
        "gunshot": 0.05,
        "firecracker": 0.02,
        "clap": 0.02,
        "click": 0.02,
        "metal_impact": 0.02,
        "door_slam": 0.02,
        "other": 0.10,
    }
    scores.update(overrides)
    top = max(scores, key=scores.get)
    return ASTResult(scores, top, scores[top], top, scores[top])


class PANNsASTFusionTests(unittest.TestCase):
    def setUp(self):
        self.fusion = PANNsASTFusion(
            panns_threshold=0.30,
            panns_margin_threshold=0.15,
            ast_threshold=0.28,
            ast_veto_threshold=0.20,
            ast_veto_margin=0.08,
            ast_margin_threshold=0.06,
            far_panns_threshold=0.08,
            far_panns_margin=-0.04,
            far_ast_threshold=0.38,
            far_ast_margin=0.12,
        )

    def test_normal_real_gunshot_is_accepted(self):
        result = self.fusion.decide(
            0.62, 0.03, 0.05,
            ast_result(gunshot=0.72, other=0.12),
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.reason, "accepted")

    def test_far_clean_gunshot_can_rescue_weak_panns(self):
        result = self.fusion.decide_far(
            0.13, 0.03, 0.14,
            ast_result(gunshot=0.74, other=0.12, click=0.05),
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.reason, "accepted_far_consensus")

    def test_ast_never_alarms_without_panns_firearm_evidence(self):
        result = self.fusion.decide_far(
            0.03, 0.01, 0.02,
            ast_result(gunshot=0.97, other=0.01),
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "far_panns_threshold")

    def test_far_click_and_heels_are_rejected(self):
        result = self.fusion.decide_far(
            0.14, 0.02, 0.15,
            ast_result(gunshot=0.58, click=0.67, other=0.04),
        )
        self.assertFalse(result.accepted)
        self.assertIn("far_ast", result.reason)

    def test_clap_is_rejected(self):
        result = self.fusion.decide(
            0.70, 0.02, 0.04,
            ast_result(gunshot=0.42, clap=0.71, other=0.03),
        )
        self.assertFalse(result.accepted)

    def test_metal_impact_is_rejected(self):
        result = self.fusion.decide(
            0.74, 0.02, 0.03,
            ast_result(gunshot=0.43, metal_impact=0.75, other=0.03),
        )
        self.assertFalse(result.accepted)

    def test_door_slam_is_rejected(self):
        result = self.fusion.decide(
            0.71, 0.02, 0.03,
            ast_result(gunshot=0.41, door_slam=0.79, other=0.03),
        )
        self.assertFalse(result.accepted)

    def test_firecracker_is_rejected(self):
        result = self.fusion.decide(
            0.72, 0.02, 0.03,
            ast_result(gunshot=0.55, firecracker=0.68, other=0.03),
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "ast_firecracker")

    def test_far_ambiguous_ast_is_rejected(self):
        result = self.fusion.decide_far(
            0.13, 0.03, 0.14,
            ast_result(gunshot=0.56, other=0.47),
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "far_ast_margin")

    def test_cap_gun_is_not_a_real_firearm_alarm_class(self):
        self.assertIn("Cap gun", AST_GROUPS["firecracker"])
        self.assertNotIn("Cap gun", AST_GROUPS["gunshot"])


if __name__ == "__main__":
    unittest.main()
