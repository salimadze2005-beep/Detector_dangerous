import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from video.model_optimizer import select_yolo_runtime_model


class YoloRuntimeSelectionTests(unittest.TestCase):
    def test_cuda_prefers_fresh_tensor_rt_engine(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "yolov8n-pose.pt"
            engine = root / "yolov8n-pose.engine"
            source.write_bytes(b"pt" * 600_000)
            engine.write_bytes(b"engine" * 200_000)
            os.utime(engine, (source.stat().st_atime + 1, source.stat().st_mtime + 1))
            self.assertEqual(select_yolo_runtime_model(source, "cuda:0"), engine.resolve())

    def test_cpu_prefers_fresh_onnx_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "yolov8s-pose.pt"
            onnx = root / "yolov8s-pose.onnx"
            source.write_bytes(b"pt" * 600_000)
            onnx.write_bytes(b"onnx" * 300_000)
            os.utime(onnx, (source.stat().st_atime + 1, source.stat().st_mtime + 1))
            with patch("video.model_optimizer._onnxruntime_available", return_value=True):
                self.assertEqual(select_yolo_runtime_model(source, "cpu"), onnx.resolve())

    def test_stale_artifact_does_not_replace_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "yolov8m-pose.pt"
            engine = root / "yolov8m-pose.engine"
            source.write_bytes(b"pt" * 600_000)
            engine.write_bytes(b"engine" * 200_000)
            os.utime(engine, (source.stat().st_atime - 1, source.stat().st_mtime - 1))
            self.assertEqual(select_yolo_runtime_model(source, "cuda:0"), source.resolve())


if __name__ == "__main__":
    unittest.main()
