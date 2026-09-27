import os
import shutil
import subprocess
import sys
from pathlib import Path

# --- Configuration ---
APP_NAME = "Detector_Danger"
ENTRY_POINT = "main.py"
DIST_DIR = "dist"
MODEL_EXPORT_SCRIPT = "export_model.py"
YOLO_ONNX_MODEL = "models/yolov8n-pose.onnx"

# --- Nuitka Options ---
nuitka_options = [
    "--standalone",
    "--onefile",
    "--remove-output",
    "--disable-console",
    f"--output-dir={DIST_DIR}",
    f"--output-filename={APP_NAME}",
    
    # --- Plugins ---
    "--plugin-enable=pyqt6",
    "--plugin-enable=numpy",
    "--plugin-enable=tensorflow",

    # --- Data Files & Directories ---
    # Include the generated ONNX model
    f"--include-data-file={YOLO_ONNX_MODEL}={YOLO_ONNX_MODEL}",
    
    # Include other models and data
    "--include-data-file=models/binary_gunshot_full.h5=models/binary_gunshot_full.h5",
    "--include-data-file=models/binary_gunshot.h5=models/binary_gunshot.h5",
    "--include-data-dir=models/yamnet=models/yamnet",
    "--include-data-dir=model-ru=model-ru",
    "--include-data-dir=sounds=sounds",
    "--include-data-dir=ui/assets=ui/assets",
]

def run_command(command, description):
    """Runs a command and exits if it fails."""
    print(f"--- {description} ---")
    print(f"Command: {' '.join(command)}")
    try:
        # Set capture_output to False to stream output in real-time
        process = subprocess.run(command, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during: {description}", file=sys.stderr)
        print(f"Return code: {e.returncode}", file=sys.stderr)
        sys.exit(1)

def main():
    """Main build process."""
    # 1. Install dependencies for Nuitka and model export
    run_command(
        ["pip", "install", "nuitka", "zstandard", "ordered-set"],
        "Installing Nuitka and dependencies"
    )
    
    # 2. Install ONNX dependencies
    run_command(
        ["pip", "install", "--user", "onnx", "onnxslim"],
        "Installing ONNX dependencies"
    )

    # 3. Export the YOLO model to ONNX
    run_command(
        ["python", MODEL_EXPORT_SCRIPT],
        "Exporting YOLO model to ONNX"
    )
    
    # Verify that the ONNX file was created
    if not Path(YOLO_ONNX_MODEL).exists():
        print(f"Error: ONNX model file was not found after export: {YOLO_ONNX_MODEL}", file=sys.stderr)
        sys.exit(1)
    print(f"Successfully created ONNX model: {YOLO_ONNX_MODEL}")

    # 4. Run Nuitka to build the executable
    nuitka_command = ["python", "-m", "nuitka"] + nuitka_options + [ENTRY_POINT]
    run_command(nuitka_command, "Running Nuitka")

    # 5. Copy the configuration file
    print("\n--- Copying Config ---")
    dist_path = Path(DIST_DIR)
    dist_path.mkdir(exist_ok=True)
    shutil.copy("detector_config.example.json", dist_path / "detector_config.json")

    print(f"\n--- Build Complete ---")
    print(f"Executable and config are in the '{DIST_DIR}' directory.")

if __name__ == "__main__":
    main()
