from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from tools.bootstrap import snapshot_download_resilient

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
TARGET = Path(__file__).resolve().parent / "models" / "ast"
STAGED = TARGET.with_name("ast.new")
REQUIRED = ("config.json", "preprocessor_config.json", "model.safetensors")


def valid_ast(path: Path = TARGET) -> bool:
    try:
        if not all((path / name).is_file() for name in REQUIRED):
            return False
        # The official safetensors checkpoint is ~346 MB. A much smaller file
        # is almost certainly an interrupted/download-error artifact.
        if (path / "model.safetensors").stat().st_size <= 300_000_000:
            return False
        json.loads((path / "config.json").read_text(encoding="utf-8"))
        json.loads((path / "preprocessor_config.json").read_text(encoding="utf-8"))
        return True
    except (OSError, UnicodeError, ValueError):
        return False


def main() -> int:
    if valid_ast():
        print(f"[OK] AST already installed: {TARGET}")
        return 0

    print(f"[AST] Downloading {MODEL_ID} -> {TARGET} (resumable)")
    if TARGET.exists() and not STAGED.exists():
        os.replace(TARGET, STAGED)
    STAGED.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download_resilient(MODEL_ID, STAGED, list(REQUIRED))
    except Exception as exc:
        print(f"[ERROR] AST download failed: {exc}")
        return 1

    if not valid_ast(STAGED):
        print("[ERROR] AST files are incomplete. Re-run setup.bat to resume.")
        return 1
    if TARGET.exists():
        shutil.rmtree(TARGET)
    os.replace(STAGED, TARGET)
    print(f"[OK] AST installed: {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
