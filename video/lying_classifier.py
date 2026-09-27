"""Optional RGB ``lying / not_lying`` classifier for fall detection crops.

The classifier is deliberately optional: a generic ImageNet model must never
be treated as a fall model.  Until a camera-specific checkpoint is supplied,
the detector remains in Pose-only mode and logs that RGB fallback is absent.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from core.config import get_resource_path
from video.model_optimizer import select_yolo_runtime_model

logger = logging.getLogger(__name__)


class LyingClassifier:
    """Run an Ultralytics classification checkpoint on a person crop."""

    def __init__(
        self,
        model_path: str,
        *,
        enabled: bool,
        device: str,
        label: str = "lying",
        imgsz: int = 224,
    ) -> None:
        self.enabled = bool(enabled)
        self.device = str(device)
        self.label = str(label).strip().casefold()
        self.imgsz = min(max(int(imgsz), 64), 1024)
        self.model_path: str | None = None
        self._model = None
        self._label_index: int | None = None

        if not self.enabled:
            return
        configured = Path(get_resource_path(model_path)).resolve()
        if not configured.is_file():
            logger.warning(
                "RGB-классификатор Лежит/Не лежит не найден: %s. "
                "Работаем только по YOLO-Pose.",
                configured,
            )
            return
        try:
            runtime = select_yolo_runtime_model(configured, self.device)
            from ultralytics import YOLO

            self._model = YOLO(str(runtime))
            self.model_path = str(runtime)
            logger.info("RGB-классификатор позы загружен: %s", runtime)
        except Exception:
            logger.warning("Не удалось загрузить RGB-классификатор позы", exc_info=True)

    @property
    def available(self) -> bool:
        return self._model is not None

    @staticmethod
    def _as_scores(value) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value, dtype=np.float32).reshape(-1)

    def _resolve_label_index(self, names) -> int | None:
        if self._label_index is not None:
            return self._label_index
        values = names.values() if isinstance(names, dict) else names
        if values is None:
            return None
        for index, value in enumerate(values):
            if str(value).strip().casefold() == self.label:
                self._label_index = index
                return index
        logger.error(
            "В RGB-классификаторе нет класса %r; ожидается, например, lying/not_lying.",
            self.label,
        )
        return None

    def score(self, crop: np.ndarray) -> float | None:
        """Return probability of the configured ``lying`` class, if available."""
        if not self.available or crop is None or crop.size == 0:
            return None
        try:
            result = self._model.predict(
                crop, device=self.device, imgsz=self.imgsz, verbose=False
            )[0]
            probabilities = getattr(result, "probs", None)
            if probabilities is None:
                logger.error("RGB-модель не вернула classification probabilities")
                return None
            index = self._resolve_label_index(
                getattr(result, "names", None) or getattr(self._model, "names", None)
            )
            if index is None:
                return None
            scores = self._as_scores(getattr(probabilities, "data", probabilities))
            if not 0 <= index < len(scores):
                logger.error("Некорректный индекс класса lying: %s", index)
                return None
            return min(max(float(scores[index]), 0.0), 1.0)
        except Exception:
            logger.warning("Ошибка RGB-классификации crop человека", exc_info=True)
            return None


__all__ = ["LyingClassifier"]
