from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from tools.doctor import query_nvidia_gpus


class NvidiaDriverProbeTests(unittest.TestCase):
    @patch("tools.doctor.shutil.which", return_value=None)
    def test_cpu_only_machine_has_no_driver_error(self, _which):
        self.assertEqual(query_nvidia_gpus(), (None, None))

    @patch("tools.doctor.subprocess.run")
    @patch("tools.doctor.shutil.which", return_value="nvidia-smi")
    def test_reports_each_adapter_and_driver_version(self, _which, run):
        run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="NVIDIA RTX, 560.12\n", stderr=""
        )
        adapters, error = query_nvidia_gpus()
        self.assertEqual(adapters, ["NVIDIA RTX, 560.12"])
        self.assertIsNone(error)

    @patch("tools.doctor.subprocess.run")
    @patch("tools.doctor.shutil.which", return_value="nvidia-smi")
    def test_driver_command_failure_is_reported(self, _which, run):
        run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="driver/library version mismatch"
        )
        adapters, error = query_nvidia_gpus()
        self.assertIsNone(adapters)
        self.assertIn("driver/library version mismatch", error)


if __name__ == "__main__":
    unittest.main()
