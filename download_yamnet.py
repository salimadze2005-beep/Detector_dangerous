from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
URL = "https://tfhub.dev/google/yamnet/1?tf-hub-format=compressed"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_yamnet() -> None:
    models_dir = ROOT / "models"
    target = models_dir / "yamnet"
    models_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="yamnet-", dir=models_dir) as temp_dir:
        temp_root = Path(temp_dir)
        archive = temp_root / "yamnet.tar.gz"
        extracted = temp_root / "extracted"
        extracted.mkdir()
        print("Downloading YAMNet from TensorFlow Hub...")
        urllib.request.urlretrieve(URL, archive)

        expected_hash = os.getenv("YAMNET_SHA256", "").casefold().strip()
        actual_hash = _sha256(archive)
        if expected_hash and actual_hash != expected_hash:
            raise RuntimeError(
                f"YAMNet SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
            )
        print(f"Archive SHA-256: {actual_hash}")

        with tarfile.open(archive, "r:gz") as bundle:
            bundle.extractall(extracted, filter="data")
        if not (extracted / "saved_model.pb").is_file():
            raise RuntimeError("Downloaded archive does not contain saved_model.pb")
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(extracted), str(target))
    print(f"YAMNet installed to {target}")


if __name__ == "__main__":
    download_yamnet()

