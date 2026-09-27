from __future__ import annotations

import argparse
import importlib
import shutil
import struct
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def report(ok: bool, label: str, detail: str = "") -> bool:
    suffix = f": {detail}" if detail else ""
    print(f"[{'OK' if ok else 'FAIL'}] {label}{suffix}")
    return ok


def info(label: str, detail: str = "") -> None:
    suffix = f": {detail}" if detail else ""
    print(f"[INFO] {label}{suffix}")


def query_nvidia_gpus() -> tuple[list[str] | None, str | None]:
    """Return installed NVIDIA adapters and driver versions without raising."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None, None
    try:
        result = subprocess.run(
            [executable, "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"nvidia-smi не запустился: {exc}"
    if result.returncode != 0:
        return None, f"nvidia-smi завершился с кодом {result.returncode}: {result.stderr.strip()}"
    adapters = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return adapters, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Диагностика Detector Danger (AST)")
    parser.add_argument("--full", action="store_true", help="реально загрузить все модели")
    parser.add_argument(
        "--skip-resources", action="store_true", help="проверить только Python-пакеты"
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="считать отсутствие CUDA ошибкой",
    )
    parser.add_argument(
        "--require-gpu-if-nvidia",
        action="store_true",
        help="считать ошибкой CUDA/YOLO только если nvidia-smi видит NVIDIA-карту",
    )
    args = parser.parse_args()
    checks: list[bool] = []
    nvidia_gpus, nvidia_error = query_nvidia_gpus()
    if nvidia_gpus:
        checks.append(report(True, "NVIDIA driver", "; ".join(nvidia_gpus)))
    elif nvidia_error:
        if args.require_gpu_if_nvidia:
            checks.append(report(False, "NVIDIA driver", nvidia_error))
        else:
            info("NVIDIA driver check", nvidia_error)
    else:
        info("NVIDIA driver", "NVIDIA-карта не обнаружена; CPU fallback поддерживается")
    gpu_required = bool(args.require_gpu or (args.require_gpu_if_nvidia and nvidia_gpus))

    checks.append(report(sys.version_info[:2] == (3, 12), "Python", sys.version.split()[0]))
    checks.append(report(struct.calcsize("P") * 8 == 64, "Архитектура", "64-bit"))
    for module in (
        "numpy", "cv2", "librosa", "sounddevice", "scipy", "soxr",
        "torch", "transformers", "safetensors", "huggingface_hub",
        "ultralytics", "vosk", "PyQt6", "onnx", "onnxruntime",
    ):
        try:
            imported = importlib.import_module(module)
            version = getattr(imported, "__version__", "installed")
            checks.append(report(True, module, str(version)))
        except Exception as exc:
            checks.append(report(False, module, str(exc)))

    torch_module = None
    cuda_available = False
    try:
        import torch

        torch_module = torch
        cuda_available = bool(torch.cuda.is_available())
        cuda_build = str(torch.version.cuda or "CPU-only")
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            detail = (
                f"{gpu_name}; torch={torch.__version__}; "
                f"CUDA build={cuda_build}; devices={torch.cuda.device_count()}"
            )
            checks.append(report(True, "PyTorch CUDA", detail))
            try:
                probe = torch.zeros((32, 32), device="cuda:0") + 1
                torch.cuda.synchronize()
                checks.append(report(bool(probe.sum().item() == 1024), "CUDA tensor probe"))
            except Exception as exc:
                checks.append(report(False, "CUDA tensor probe", str(exc)))
        else:
            detail = f"torch={torch.__version__}; CUDA build={cuda_build}"
            if nvidia_gpus:
                detail += (
                    "; NVIDIA-карта обнаружена, но CUDA PyTorch недоступен. "
                    "Обновите драйвер NVIDIA с официального сайта и повторно запустите setup.bat."
                )
            if gpu_required:
                checks.append(report(False, "PyTorch CUDA", detail))
            else:
                info("PyTorch CUDA unavailable; CPU fallback active", detail)
    except Exception as exc:
        if gpu_required:
            checks.append(report(False, "PyTorch CUDA", str(exc)))
        else:
            info("PyTorch CUDA check skipped", str(exc))

    from core.config import config

    validation_errors = [] if args.skip_resources else config.validate()
    if not args.skip_resources:
        checks.append(report(not validation_errors, "Ресурсы", "; ".join(validation_errors)))

    if args.full and not args.skip_resources and not validation_errors:
        try:
            from audio.gunshot_detector_ast import PANNsClassifier

            PANNsClassifier(config.audio.panns_model_path, config.audio.panns_device)
            panns_device = (
                "cuda"
                if cuda_available and config.audio.panns_device == "auto"
                else config.audio.panns_device
            )
            checks.append(report(True, "PANNs Cnn14", f"device={panns_device}"))
        except Exception as exc:
            checks.append(report(False, "PANNs Cnn14", str(exc)))
        try:
            from audio.gunshot_detector_ast import ASTClassifier
            from core.config import get_resource_path

            # Full install verification uses CPU deliberately so small/older
            # NVIDIA GPUs are not rejected merely because AST does not fit VRAM.
            ASTClassifier(get_resource_path("models/ast"), "cpu")
            checks.append(report(True, "AST AudioSet", "device=cpu verification"))
        except Exception as exc:
            checks.append(report(False, "AST AudioSet", str(exc)))
        try:
            import numpy as np
            from video.fall_detector import FallDetector

            detector = FallDetector(
                model_path=config.video.model_path,
                lying_classifier_enabled=config.video.lying_classifier_enabled,
                lying_classifier_model_path=config.video.lying_classifier_model_path,
            )
            blank = np.zeros((320, 320, 3), dtype=np.uint8)
            detector.model.predict(blank, device=detector.device, verbose=False)
            checks.append(
                report(
                    True,
                    "YOLO-Pose inference",
                    f"device={detector.device}; model={Path(detector.model_path).name}",
                )
            )
            if gpu_required:
                checks.append(report(detector.device.startswith("cuda"), "YOLO-Pose CUDA"))
        except Exception as exc:
            checks.append(report(False, "YOLO-Pose inference", str(exc)))
        if config.speech.enabled:
            try:
                from vosk import Model
                from core.vosk_model_path import vosk_runtime_path

                Model(vosk_runtime_path(config.speech.vosk_model_path))
                checks.append(report(True, "Vosk"))
            except Exception as exc:
                checks.append(report(False, "Vosk", str(exc)))
            if config.speech.secondary_enabled:
                try:
                    from faster_whisper import WhisperModel

                    secondary = WhisperModel(
                        config.speech.secondary_model_path,
                        device="cpu",
                        compute_type="int8",
                        local_files_only=True,
                    )
                    del secondary
                    checks.append(report(True, "Faster-Whisper", "device=cpu verification"))
                except Exception as exc:
                    checks.append(report(False, "Faster-Whisper", str(exc)))

    if torch_module is not None and cuda_available:
        try:
            import tensorrt

            info("TensorRT acceleration available", str(tensorrt.__version__))
        except Exception as exc:
            info("TensorRT acceleration is optional; PyTorch fallback active", str(exc))
        info(
            "GPU policy",
            "YOLO-Pose, PANNs и AST используют CUDA в auto-режиме; "
            "диагностика AST выполняется на CPU; Vosk остаётся CPU-компонентом",
        )

    print("\nДиагностика пройдена." if all(checks) else "\nДиагностика обнаружила ошибки.")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
