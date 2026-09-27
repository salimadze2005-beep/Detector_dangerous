import logging
import sys
from pathlib import Path

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# --- Configuration ---
MODEL_SOURCE_PATH = Path("models/yolov8n-pose.pt")
MODEL_DEST_PATH = Path(f"{MODEL_SOURCE_PATH.stem}.onnx")
IMAGE_SIZE = 640  # The image size the model was trained on

def main():
    """
    Exports the YOLOv8 pose model to ONNX format.
    """
    if not MODEL_SOURCE_PATH.exists():
        logging.error(f"Model source file not found: {MODEL_SOURCE_PATH}")
        logging.error("Please ensure the model exists and the path is correct.")
        sys.exit(1)

    logging.info(f"Loading model from: {MODEL_SOURCE_PATH}")
    model = YOLO(MODEL_SOURCE_PATH)

    logging.info(f"Exporting model to ONNX: {MODEL_DEST_PATH}")
    logging.info(f"This may take a few moments...")

    try:
        model.export(
            format="onnx",
            imgsz=IMAGE_SIZE,
            verbose=True,
        )
        # The export function saves the file with a name like 'yolov8n-pose.onnx'
        # Let's ensure it's named correctly and moved if necessary.
        exported_file = Path(f"{MODEL_SOURCE_PATH.stem}.onnx")
        if exported_file.exists() and not MODEL_DEST_PATH.exists():
             exported_file.rename(MODEL_DEST_PATH)
        
        logging.info(f"Successfully exported model to: {MODEL_DEST_PATH}")

    except Exception as e:
        logging.error(f"An error occurred during model export: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()