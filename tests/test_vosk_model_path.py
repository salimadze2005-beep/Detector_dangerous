import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.vosk_model_path import vosk_runtime_path


class VoskModelPathTests(unittest.TestCase):
    @staticmethod
    def _fake_model(root: Path) -> Path:
        model = root / "модель"
        for relative in (
            "am/final.mdl",
            "graph/Gr.fst",
            "graph/HCLr.fst",
            "ivector/final.ie",
        ):
            target = model / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(relative.encode("ascii"))
        return model

    def test_non_ascii_model_falls_back_to_ascii_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._fake_model(root)
            cache = root / "ascii-cache"
            with (
                patch("core.vosk_model_path.os.name", "nt"),
                patch("core.vosk_model_path._short_windows_path", return_value=None),
                patch("core.vosk_model_path._cache_roots", return_value=[cache]),
            ):
                resolved = Path(vosk_runtime_path(source))
            self.assertTrue(resolved.is_dir())
            self.assertTrue((resolved / "am" / "final.mdl").is_file())
            str(resolved).encode("ascii")

    def test_missing_path_is_left_for_vosk_to_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing"
            self.assertEqual(Path(vosk_runtime_path(missing)), missing.resolve())


if __name__ == "__main__":
    unittest.main()
