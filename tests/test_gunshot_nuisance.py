import unittest

from audio.gunshot_detector import AST_GROUPS, YAMNetVeto


class GunshotNuisanceClassTests(unittest.TestCase):
    def test_claps_clicks_heels_and_camera_are_veto_classes(self):
        nuisance = (
            "Clapping",
            "Finger snapping",
            "Clicking",
            "Tick",
            "Tap",
            "Camera",
            "Single-lens reflex camera",
            "Walk, footsteps",
            "Clip-clop",
            "Computer keyboard",
            "Knock",
            "Clang",
        )
        for class_name in nuisance:
            self.assertIn(class_name, YAMNetVeto.VETO_CLASSES)
            self.assertNotIn(class_name, YAMNetVeto.GUN_CLASSES)

    def test_real_firearm_labels_remain_gun_classes(self):
        self.assertIn("Gunshot, gunfire", YAMNetVeto.GUN_CLASSES)
        self.assertIn("Machine gun", YAMNetVeto.GUN_CLASSES)
        self.assertNotIn("Cap gun", YAMNetVeto.GUN_CLASSES)

    def test_cap_gun_firecracker_and_burst_are_not_real_firearm_classes(self):
        for name in ("Cap gun", "Firecracker", "Fireworks", "Burst, pop"):
            self.assertIn(name, AST_GROUPS["firecracker"])
            self.assertNotIn(name, AST_GROUPS["gunshot"])


if __name__ == "__main__":
    unittest.main()
