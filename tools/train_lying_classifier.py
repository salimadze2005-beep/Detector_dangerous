"""Train and export the camera-specific ``lying / not_lying`` RGB model.

Dataset layout::

    dataset/
      train/{lying,not_lying}/
      val/{lying,not_lying}/

Use separate source videos for ``train`` and ``val``.  Do not split adjacent
frames from the same recording between them: that reports an unrealistically
high validation score without proving the model works in the real camera view.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


CLASSES = ("lying", "not_lying")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="dataset directory")
    parser.add_argument("--base-model", default="yolo11n-cls.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--project", type=Path, default=Path("runs/lying_classifier"))
    parser.add_argument("--name", default="camera")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/lying_classifier.pt"),
        help="checkpoint used by detector_config.json",
    )
    return parser.parse_args()


def _validate_dataset(root: Path) -> None:
    missing = []
    for split in ("train", "val"):
        for name in CLASSES:
            directory = root / split / name
            if not directory.is_dir() or not any(path.is_file() for path in directory.iterdir()):
                missing.append(str(directory))
    if missing:
        raise SystemExit("Нет кадров в обязательных каталогах:\n - " + "\n - ".join(missing))


def main() -> None:
    args = _parse_args()
    root = args.data.resolve()
    _validate_dataset(root)

    from ultralytics import YOLO
    from video.model_optimizer import prepare_yolo_artifacts

    requested_device = str(args.device).casefold()
    if requested_device == "auto":
        import torch

        device = 0 if torch.cuda.is_available() else "cpu"
    elif requested_device in {"cuda", "cuda:0", "gpu"}:
        device = 0
    else:
        device = args.device
    model = YOLO(args.base_model)
    model.train(
        data=str(root),
        epochs=max(1, args.epochs),
        imgsz=min(max(args.imgsz, 64), 1024),
        batch=max(1, args.batch),
        device=device,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        seed=42,
    )
    best = args.project / args.name / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit(f"Ultralytics не создал checkpoint: {best}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.output)
    artifacts = prepare_yolo_artifacts(args.output, imgsz=args.imgsz)
    print(f"Готово: {args.output.resolve()}")
    print(f"ONNX: {artifacts['onnx'] or 'не создан'}")
    print(f"TensorRT: {artifacts['engine'] or 'не создан'}")


if __name__ == "__main__":
    main()
