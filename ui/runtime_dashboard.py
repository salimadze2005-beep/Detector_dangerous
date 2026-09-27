from __future__ import annotations

import gc
import logging

from PyQt6.QtWidgets import QApplication, QDialog

from ui.dashboard import SecurityDashboard as _SecurityDashboard
from ui.settings_dialog import SettingsDialog

logger = logging.getLogger(__name__)


class SecurityDashboard(_SecurityDashboard):
    """Dashboard with a deterministic video-runtime restart after settings changes.

    Switching side/top-down mode does not require a different model file, but the
    current worker owns the Ultralytics predictor/tracker and its CUDA tensors.
    Before a replacement worker is created, fully drain the old Qt thread/signals
    and release cached Python/CUDA objects. This prevents YOLO from disappearing
    after repeated settings/mode changes, especially with the Large pose model.
    """

    @staticmethod
    def _release_video_runtime() -> None:
        # STOPPED/frame signals from the old QThreads are queued to the GUI thread.
        # Drain them before new workers start so a late STOPPED signal cannot clear
        # a freshly restarted camera tile.
        app = QApplication.instance()
        if app is not None:
            app.processEvents()

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            # Runtime cleanup must never make the settings dialog unusable on a
            # CPU-only installation or when torch is only partially available.
            logger.debug("CUDA cache cleanup skipped", exc_info=True)

        if app is not None:
            app.processEvents()

    def open_settings(self) -> None:
        was_active = self.monitoring_active
        dialog = SettingsDialog(self.config, self)
        accepted = dialog.exec() == QDialog.DialogCode.Accepted

        # Cancel means exactly that: keep the current detector/model untouched.
        # The previous implementation restarted monitoring even after Cancel.
        if not accepted:
            return

        if was_active:
            self.stop_monitoring()
            if self.monitoring_active:
                self.bridge.log.emit(
                    "error",
                    "Настройки не применены: видеопотоки не остановились",
                )
                return
            self._release_video_runtime()

        self.alert_service.config = self.config.alerts

        if was_active:
            self.start_monitoring()
        else:
            self._prepare_camera_tiles()
