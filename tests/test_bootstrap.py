import tempfile
import unittest
import zipfile
from pathlib import Path

from tools.bootstrap import (
    PANNS_BYTES,
    resource_status,
    safe_extract_zip,
    valid_faster_whisper,
    valid_panns,
    valid_vosk,
)


class BootstrapTests(unittest.TestCase):
    def test_safe_extract_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("../outside.txt", "bad")
            with self.assertRaises(RuntimeError):
                safe_extract_zip(archive, root / "target")

    def test_resource_status_accepts_panns_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "models" / "panns" / "Cnn14_mAP=0.431.pth"
            checkpoint.parent.mkdir(parents=True)
            with checkpoint.open("wb") as handle:
                handle.truncate(PANNS_BYTES)
            labels = root / "models" / "panns_data" / "class_labels_indices.csv"
            labels.parent.mkdir(parents=True)
            labels.write_text(
                "index,mid,display_name\n"
                + "".join(f"{index},/m/{index},Class {index}\n" for index in range(527)),
                encoding="utf-8",
            )
            self.assertTrue(valid_panns(checkpoint))
            self.assertTrue(resource_status(root)["PANNs Cnn14"])

    def test_faster_whisper_validator_requires_local_model_structure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir)
            with (model / "model.bin").open("wb") as handle:
                handle.truncate(100_000_001)
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "tokenizer.json").write_text("{}", encoding="utf-8")
            self.assertTrue(valid_faster_whisper(model))

    def test_vosk_validator_rejects_marker_only_and_accepts_full_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir)
            files = {
                "am/final.mdl": 10_000_000,
                "conf/mfcc.conf": 50,
                "conf/model.conf": 50,
                "graph/Gr.fst": 1_000_000,
                "graph/HCLr.fst": 1_000_000,
                "graph/phones/word_boundary.int": 100,
                "ivector/final.ie": 1_000_000,
                "ivector/final.dubm": 100_000,
                "ivector/final.mat": 10_000,
            }
            for relative, size in files.items():
                target = model / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("wb") as handle:
                    handle.truncate(size)
            self.assertTrue(valid_vosk(model))
            (model / "graph" / "Gr.fst").unlink()
            self.assertFalse(valid_vosk(model))

    def test_windows_entrypoints_all_reach_required_model_bootstrap(self):
        root = Path(__file__).resolve().parent.parent
        setup = (root / "setup.bat").read_text(encoding="utf-8")
        run = (root / "run.bat").read_text(encoding="utf-8")
        install = (root / "install_and_run.bat").read_text(encoding="utf-8")
        verify = (root / "verify_install.bat").read_text(encoding="utf-8")
        prerequisites = (root / "tools" / "windows_prerequisites.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn("tools\\bootstrap.py", setup)
        self.assertIn("download_ast.py", setup)
        self.assertIn("tools\\bootstrap.py", run)
        self.assertIn("download_ast.py", run)
        self.assertIn("call \"%~dp0setup.bat\"", install)
        self.assertIn("tools\\bootstrap.py", verify)
        self.assertIn("download_ast.py", verify)
        self.assertIn("windows_prerequisites.ps1", setup)
        self.assertIn("vc_redist.x64.exe", prerequisites)
        self.assertNotIn("requires GPU", run)


if __name__ == "__main__":
    unittest.main()
