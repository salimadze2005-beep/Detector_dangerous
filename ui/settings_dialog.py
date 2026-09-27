from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QComboBox, QDialog, QFormLayout, QGroupBox, QMessageBox

from core.config import get_resource_path
from ui.settings_dialog_base import *
from ui.settings_dialog_base import SettingsDialog as _BaseSettingsDialog


POSE_MODELS = {
    "large": ("Large — точнее, тяжелее", "models/yolov8l-pose.pt"),
    "middle": ("Middle — баланс", "models/yolov8m-pose.pt"),
    "small": ("Small — быстрее", "models/yolov8s-pose.pt"),
    "nano": ("Nano — самый быстрый", "models/yolov8n-pose.pt"),
}


def pose_model_key(path: str) -> str | None:
    """Return the selector key for known PT/ONNX/Engine pose model filenames.

    Older installations may still store an ONNX model path. Treat that as the
    same model size instead of silently displaying/selecting Large.
    """
    name = Path(str(path)).name.casefold()
    for key, (_label, relative_path) in POSE_MODELS.items():
        stem = Path(relative_path).stem.casefold()
        if name in {f"{stem}.pt", f"{stem}.onnx", f"{stem}.engine"}:
            return key
    return None


class SettingsDialog(_BaseSettingsDialog):
    """Base operator settings plus a persistent YOLO-Pose size selector."""

    def __init__(self, app_config, parent=None):
        self._initial_pose_model_path = str(app_config.video.model_path)
        self._initial_pose_model_key = pose_model_key(self._initial_pose_model_path)
        super().__init__(app_config, parent)

    def _audio_tab(self):
        page = super()._audio_tab()
        fall_group = next(
            (
                group
                for group in page.findChildren(QGroupBox)
                if group.title() == "Детекция падения"
            ),
            None,
        )
        if fall_group is None:
            return page
        form = fall_group.layout()
        if not isinstance(form, QFormLayout):
            return page

        self.pose_model_combo = QComboBox()
        for key, (label, _relative_path) in POSE_MODELS.items():
            # Keep userData deliberately simple. Arbitrary tuple QVariant data
            # can be converted differently across PyQt builds; a plain string
            # key is stable and cannot accidentally fall back to Large.
            self.pose_model_combo.addItem(label, key)

        current = (
            self.pose_model_combo.findData(self._initial_pose_model_key)
            if self._initial_pose_model_key is not None
            else -1
        )
        if current >= 0:
            self.pose_model_combo.setCurrentIndex(current)
        else:
            # Unknown/custom model: do not silently replace it with Large just
            # because the settings dialog was opened. The current path remains
            # active until the operator explicitly chooses a known model.
            self.pose_model_combo.setCurrentIndex(-1)
            self.pose_model_combo.setPlaceholderText(
                f"Текущая: {Path(self._initial_pose_model_path).name or 'другая модель'}"
            )

        self.pose_model_combo.setToolTip(
            "Выбор сохраняется в detector_config.json и используется после смены "
            "ракурса и следующего запуска программы. Large требует больше GPU; "
            "Middle/Small/Nano уменьшают нагрузку."
        )
        form.insertRow(0, "Модель YOLO-Pose", self.pose_model_combo)
        return page

    def _selected_pose_model(self) -> tuple[str | None, str]:
        key = self.pose_model_combo.currentData()
        if isinstance(key, str) and key in POSE_MODELS:
            return key, POSE_MODELS[key][1]
        # If no explicit choice was made, preserve the current/custom model.
        return self._initial_pose_model_key, self._initial_pose_model_path

    def _ensure_pose_weights(self, relative_path: str) -> bool:
        target = Path(get_resource_path(relative_path))
        if target.is_file() and target.stat().st_size > 1024 * 1024:
            return True
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            from ultralytics import YOLO

            target.parent.mkdir(parents=True, exist_ok=True)
            model = YOLO(target.name)
            source = Path(model.ckpt_path).resolve()
            if source != target.resolve():
                shutil.copy2(source, target)
            if not target.is_file() or target.stat().st_size <= 1024 * 1024:
                raise RuntimeError("файл модели не был корректно загружен")
            from video.model_optimizer import prepare_yolo_artifacts

            prepare_yolo_artifacts(target)
            return True
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Не удалось загрузить YOLO-Pose",
                f"Модель {target.name} не готова к работе: {exc}",
            )
            return False
        finally:
            QApplication.restoreOverrideCursor()

    def _persist_pose_model(self, model_path: str) -> None:
        """Persist and verify the selected model path atomically."""
        if not self.config.config_file:
            return
        path = Path(self.config.config_file)
        existing = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                existing = loaded
        video = existing.setdefault("video", {})
        if not isinstance(video, dict):
            video = {}
            existing["video"] = video
        video["model_path"] = model_path

        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

        # Fail loudly instead of silently reverting to the dataclass default on
        # the next program start.
        with path.open("r", encoding="utf-8") as handle:
            persisted = json.load(handle)
        if persisted.get("video", {}).get("model_path") != model_path:
            raise OSError("model_path не сохранился в detector_config.json")

    def _apply_and_accept(self) -> None:
        model_key, model_path = self._selected_pose_model()
        # Config loading resolves bundled paths to absolute paths while the
        # selector stores portable relative paths. Compare model identities so
        # pressing Apply cannot download/export the same model again.
        model_changed = (
            model_key != self._initial_pose_model_key
            or (
                model_key is None
                and Path(model_path).resolve() != Path(self._initial_pose_model_path).resolve()
            )
        )
        if model_changed and model_key is not None and not self._ensure_pose_weights(model_path):
            return

        # The in-memory config is what the freshly restarted VideoSystemWorker
        # receives immediately after this dialog closes.
        self.config.video.model_path = model_path
        super()._apply_and_accept()
        if self.result() != QDialog.DialogCode.Accepted:
            return

        try:
            self._persist_pose_model(model_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            QMessageBox.warning(
                self,
                "Настройки модели",
                "Настройки применены в текущем запуске, но выбранную модель не "
                f"удалось сохранить для следующего запуска: {exc}",
            )
            return

        self._initial_pose_model_path = model_path
        self._initial_pose_model_key = model_key


__all__ = [
    "SettingsDialog",
    "CameraBindingRow",
    "discover_local_cameras",
    "add_microphones",
    "select_device",
    "selected_microphone_source",
    "selected_device",
    "is_network_audio_source",
    "POSE_MODELS",
    "pose_model_key",
]
