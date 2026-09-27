"""Prepare and select accelerated YOLO-Pose runtime artifacts.

The PyTorch checkpoint remains the portable source of truth.  ONNX and
TensorRT files are local cache artifacts: they may be rebuilt for the target
GPU and are never required for CPU fallback.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MIN_MODEL_BYTES = 1024 * 1024


def _fresh_artifact(source: Path, artifact: Path) -> bool:
    return (
        artifact.is_file()
        and artifact.stat().st_size > MIN_MODEL_BYTES
        and artifact.stat().st_mtime >= source.stat().st_mtime
    )


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _onnxruntime_available() -> bool:
    try:
        import onnxruntime  # noqa: F401

        return True
    except Exception:
        return False


def _export_result_path(result, expected: Path) -> Path:
    if expected.is_file():
        return expected
    if isinstance(result, (str, Path)):
        candidate = Path(result)
        if candidate.is_file():
            return candidate
    raise RuntimeError(f"Экспорт не создал ожидаемый файл: {expected}")


def prepare_yolo_artifacts(
    model_path: str | Path,
    *,
    imgsz: int = 960,
    build_tensorrt: bool = True,
) -> dict[str, str | None]:
    """Export a YOLO ``.pt`` checkpoint to ONNX and, when possible, TensorRT.

    Export failures are logged and returned to the caller; the original PT
    model remains usable.  TensorRT is attempted only when CUDA is available,
    because an Engine is GPU-specific and cannot be used as a CPU fallback.
    """
    source = Path(model_path).resolve()
    if source.suffix.casefold() != ".pt":
        return {"source": str(source), "onnx": None, "engine": None}
    if not source.is_file():
        raise FileNotFoundError(f"YOLO-веса не найдены: {source}")

    onnx_path = source.with_suffix(".onnx")
    engine_path = source.with_suffix(".engine")
    result: dict[str, str | None] = {
        "source": str(source),
        "onnx": str(onnx_path) if _fresh_artifact(source, onnx_path) else None,
        "engine": str(engine_path) if _fresh_artifact(source, engine_path) else None,
    }
    if result["onnx"] is None or (build_tensorrt and _cuda_available() and result["engine"] is None):
        from ultralytics import YOLO

        model = YOLO(str(source))
        if result["onnx"] is None:
            try:
                logger.info("Экспорт YOLO-Pose в ONNX: %s", source.name)
                exported = model.export(
                    format="onnx",
                    imgsz=int(imgsz),
                    half=bool(_cuda_available()),
                    dynamic=False,
                    simplify=True,
                    opset=17,
                    device=0 if _cuda_available() else "cpu",
                )
                result["onnx"] = str(_export_result_path(exported, onnx_path))
                logger.info("ONNX YOLO-Pose готов: %s", onnx_path)
            except Exception:
                logger.warning("Не удалось экспортировать YOLO-Pose в ONNX", exc_info=True)

        if build_tensorrt and _cuda_available() and result["engine"] is None:
            try:
                logger.info("Сборка TensorRT Engine для YOLO-Pose: %s", source.name)
                exported = model.export(
                    format="engine",
                    imgsz=int(imgsz),
                    half=True,
                    dynamic=False,
                    device=0,
                    workspace=4,
                )
                result["engine"] = str(_export_result_path(exported, engine_path))
                logger.info("TensorRT Engine готов: %s", engine_path)
            except Exception:
                logger.warning(
                    "TensorRT недоступен; будет использован ONNX/PT fallback",
                    exc_info=True,
                )
    return result


def select_yolo_runtime_model(model_path: str | Path, device: str) -> Path:
    """Select a fresh Engine/ONNX artifact without breaking PT fallback."""
    configured = Path(model_path).resolve()
    source = configured.with_suffix(".pt") if configured.suffix.casefold() in {".onnx", ".engine"} else configured
    if source.is_file() and str(device).casefold().startswith("cuda"):
        engine = source.with_suffix(".engine")
        if _fresh_artifact(source, engine):
            return engine
    if source.is_file() and str(device).casefold() == "cpu" and _onnxruntime_available():
        onnx = source.with_suffix(".onnx")
        if _fresh_artifact(source, onnx):
            return onnx
    return configured


__all__ = ["prepare_yolo_artifacts", "select_yolo_runtime_model"]
