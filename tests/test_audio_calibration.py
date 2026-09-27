import unittest

import numpy as np

from audio.calibration import (
    analyze_ambient_audio,
    analyze_reference_audio,
    microphone_profile_key,
    resolve_audio_profile,
    validate_microphone_profiles,
)
from core.config import AudioConfig


class AmbientCalibrationTests(unittest.TestCase):
    @staticmethod
    def _noise(amplitude: float, seed: int = 1) -> list[np.ndarray]:
        generator = np.random.default_rng(seed)
        return [
            generator.normal(0.0, amplitude, 4_000).astype(np.float32)
            for _ in range(12)
        ]

    def test_noisy_room_gets_higher_trigger_than_quiet_room(self):
        quiet = analyze_ambient_audio(self._noise(0.002), "balanced")
        noisy = analyze_ambient_audio(self._noise(0.05), "balanced")
        self.assertGreater(noisy.trigger_dbfs, quiet.trigger_dbfs)
        self.assertGreater(noisy.rms_min, quiet.rms_min)
        self.assertLessEqual(noisy.trigger_dbfs, -8.0)
        self.assertLess(noisy.rms_min, noisy.ambient_rms)

    def test_noisy_preset_is_stricter_than_sensitive_preset(self):
        chunks = self._noise(0.01)
        sensitive = analyze_ambient_audio(chunks, "sensitive")
        noisy = analyze_ambient_audio(chunks, "noisy")
        self.assertGreater(noisy.trigger_dbfs, sensitive.trigger_dbfs)
        self.assertGreater(noisy.min_snr_db, sensitive.min_snr_db)
        self.assertGreater(noisy.min_crest_factor, sensitive.min_crest_factor)

    def test_profile_is_resolved_for_only_selected_microphone(self):
        config = AudioConfig()
        config.microphone_profiles = {
            "3": {
                "preset": "custom",
                "trigger_dbfs": -8.0,
                "gunshot_threshold": 0.91,
                "min_snr_db": 16.0,
            },
            "4": {"preset": "sensitive", "trigger_dbfs": -22.0},
        }
        first = resolve_audio_profile(config, 3)
        second = resolve_audio_profile(config, 4)
        default = resolve_audio_profile(config, None)
        self.assertEqual(first["trigger_dbfs"], -8.0)
        self.assertEqual(first["gunshot_threshold"], 0.91)
        self.assertEqual(second["trigger_dbfs"], -22.0)
        self.assertEqual(default["preset"], "balanced")
        self.assertEqual(default["trigger_dbfs"], -12.0)
        self.assertEqual(microphone_profile_key(None), "__default__")

    def test_existing_reference_profile_is_upgraded_at_runtime(self):
        config = AudioConfig()
        config.microphone_profiles = {
            "17": {
                "preset": "custom",
                "trigger_dbfs": -16.0,
                "rms_min": 0.0025,
                "panns_threshold": 0.30,
                "panns_margin_threshold": 0.15,
                "yamnet_threshold": 0.30,
                "calibration_mode": "ambient_plus_reference",
                "ambient_peak_dbfs": -32.6,
                "reference_peak_dbfs": -21.9,
                "reference_snr_db": 39.4,
            }
        }
        resolved = resolve_audio_profile(config, 17)
        self.assertLessEqual(resolved["trigger_dbfs"], -24.9)
        self.assertEqual(resolved["panns_threshold"], 0.30)
        self.assertEqual(resolved["panns_margin_threshold"], 0.15)
        self.assertEqual(resolved["yamnet_threshold"], 0.30)
        self.assertLess(resolved["rms_min"], 0.0025)

    def test_manual_override_is_not_clamped_by_reference_profile(self):
        config = AudioConfig()
        config.microphone_profiles = {
            "17": {
                "preset": "custom",
                "trigger_dbfs": -27.0,
                "panns_threshold": 0.22,
                "panns_margin_threshold": 0.08,
                "yamnet_threshold": 0.12,
                "min_snr_db": 8.0,
                "min_crest_factor": 2.5,
                "rms_min": 0.001,
                "calibration_mode": "ambient_plus_reference",
                "manual_override": True,
            }
        }
        resolved = resolve_audio_profile(config, 17)
        self.assertEqual(resolved["panns_threshold"], 0.22)
        self.assertEqual(resolved["panns_margin_threshold"], 0.08)
        self.assertEqual(resolved["yamnet_threshold"], 0.12)
        self.assertEqual(resolved["min_snr_db"], 8.0)

    def test_invalid_profile_is_reported(self):
        errors = validate_microphone_profiles(
            {"3": {"preset": "unknown", "min_snr_db": -2, "unexpected": True}}
        )
        self.assertGreaterEqual(len(errors), 3)

    def test_reference_calibration_uses_the_quietest_verified_take(self):
        ambient = self._noise(0.002, seed=7)
        generator = np.random.default_rng(8)
        takes = []
        for amplitude in (0.75, 0.45, 0.25):
            take = generator.normal(0.0, 0.002, 48_000).astype(np.float32)
            impulse = amplitude * np.hanning(320).astype(np.float32)
            take[24_000:24_320] += impulse
            takes.append([take])
        result = analyze_reference_audio(ambient, takes, "balanced")
        profile = result.profile_values()
        self.assertEqual(result.reference_count, 3)
        self.assertLessEqual(result.trigger_dbfs, result.weakest_peak_dbfs)
        self.assertEqual(profile["calibration_mode"], "ambient_plus_reference")
        self.assertEqual(validate_microphone_profiles({"mic": profile}), [])

    def test_reference_calibration_uses_measured_noise_not_balanced_floor(self):
        ambient = self._noise(0.002, seed=13)
        generator = np.random.default_rng(14)
        takes = []
        for amplitude in (0.42, 0.24, 0.10):
            take = generator.normal(0.0, 0.002, 48_000).astype(np.float32)
            take[24_000:24_320] += amplitude * np.hanning(320).astype(np.float32)
            takes.append([take])
        result = analyze_reference_audio(ambient, takes, "balanced")
        self.assertLess(result.trigger_dbfs, -16.0)
        self.assertLess(result.trigger_dbfs, result.weakest_peak_dbfs)
        self.assertGreater(result.profile_values()["reference_rms"], 0.0)
        profile = result.profile_values()
        self.assertEqual(profile["panns_threshold"], 0.30)
        self.assertEqual(profile["panns_margin_threshold"], 0.15)
        self.assertEqual(profile["yamnet_threshold"], 0.30)

    def test_reference_calibration_requires_three_distances(self):
        with self.assertRaisesRegex(ValueError, "как минимум три"):
            analyze_reference_audio(self._noise(0.002), [self._noise(0.4), self._noise(0.3)])
    def test_calibration_requires_enough_audio(self):
        with self.assertRaises(ValueError):
            analyze_ambient_audio([np.zeros(100, dtype=np.float32)], "balanced")


if __name__ == "__main__":
    unittest.main()
