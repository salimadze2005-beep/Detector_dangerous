from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DOWNLOADS = ROOT / "models" / ".downloads"
VOSK_URL = "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip"
VOSK_ARCHIVE = "vosk-model-small-ru-0.22.zip"
VOSK_MAX_BYTES = 256 * 1024 * 1024
VOSK_SHA256 = "961d5ff98a17f4aa6de69864d0aa71fa5bac682301d2b5d17a3f24c5c99a46d4"
PANNS_URL = (
    "https://zenodo.org/records/3987831/files/"
    "Cnn14_mAP%3D0.431.pth?download=1"
)
PANNS_BYTES = 327_428_481
PANNS_SHA256 = "0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31"
PANNS_LABELS_URL = (
    "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/"
    "class_labels_indices.csv"
)


def supported_python() -> tuple[bool, str]:
    version = sys.version_info
    is_64_bit = struct.calcsize("P") * 8 == 64
    supported = version[:2] == (3, 12) and is_64_bit
    detail = (
        f"Python {version.major}.{version.minor}.{version.micro} "
        f"({64 if is_64_bit else 32}-bit)"
    )
    return supported, detail


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(
    url: str,
    destination: Path,
    max_bytes: int,
    *,
    attempts: int = 6,
) -> None:
    """Download with retries and HTTP Range resume, preserving ``.part`` files."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if destination.is_file():
        return

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        offset = partial.stat().st_size if partial.is_file() else 0
        if offset > max_bytes:
            partial.unlink(missing_ok=True)
            offset = 0
        headers = {"User-Agent": "DetectorDanger/2.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                status = int(getattr(response, "status", 200) or 200)
                resumed = bool(offset and status == 206)
                if offset and not resumed:
                    offset = 0
                declared = int(response.headers.get("Content-Length") or 0)
                expected_total = offset + declared if declared else 0
                if expected_total and expected_total > max_bytes:
                    raise RuntimeError(
                        f"Слишком большой файл: {expected_total} байт (лимит {max_bytes})"
                    )
                mode = "ab" if resumed else "wb"
                received = offset
                next_report = 10
                with partial.open(mode) as output:
                    while chunk := response.read(1024 * 1024):
                        received += len(chunk)
                        if received > max_bytes:
                            raise RuntimeError(
                                f"Превышен лимит загрузки {max_bytes} байт"
                            )
                        output.write(chunk)
                        if expected_total:
                            percent = int(received * 100 / expected_total)
                            if percent >= next_report:
                                print(f"  {destination.name}: {min(percent, 100)}%")
                                next_report = (percent // 10 + 1) * 10
                if expected_total and received != expected_total:
                    raise RuntimeError(
                        f"Загрузка оборвалась: {received} из {expected_total} байт"
                    )
            os.replace(partial, destination)
            return
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt == attempts:
                break
            delay = min(2 ** (attempt - 1), 20)
            kept = partial.stat().st_size if partial.is_file() else 0
            print(
                f"[RETRY {attempt}/{attempts}] {destination.name}: {exc}; "
                f"сохранено {kept // (1024 * 1024)} MB, повтор через {delay} сек."
            )
            time.sleep(delay)
    raise RuntimeError(f"Не удалось скачать {url}: {last_error}")


def safe_extract_zip(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (root / member.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Небезопасный путь в ZIP: {member.filename}")
        bundle.extractall(root)


def _file_at_least(path: Path, size: int) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= size
    except OSError:
        return False


def valid_vosk(path: Path) -> bool:
    """Validate the complete small-model layout, not just two marker files."""
    required = {
        "am/final.mdl": 10_000_000,
        "conf/mfcc.conf": 50,
        "conf/model.conf": 50,
        "graph/Gr.fst": 1_000_000,
        "graph/HCLr.fst": 1_000_000,
        "graph/phones/word_boundary.int": 100,
        "ivector/final.ie": 1_000_000,
        "ivector/final.dubm": 100_000,
        "ivector/final.mat": 10_000,
    }
    return path.is_dir() and all(
        _file_at_least(path / relative, minimum)
        for relative, minimum in required.items()
    )


def loadable_vosk(path: Path) -> bool:
    if not valid_vosk(path):
        return False
    try:
        from core.vosk_model_path import vosk_runtime_path
        from vosk import Model, SetLogLevel

        SetLogLevel(-1)
        model = Model(vosk_runtime_path(path))
        del model
        return True
    except Exception:
        return False


def valid_faster_whisper(path: Path) -> bool:
    model = path / "model.bin"
    try:
        if not (
            _file_at_least(model, 100_000_000)
            and (path / "config.json").is_file()
            and (path / "tokenizer.json").is_file()
        ):
            return False
        json.loads((path / "config.json").read_text(encoding="utf-8"))
        json.loads((path / "tokenizer.json").read_text(encoding="utf-8"))
        return True
    except (OSError, UnicodeError, ValueError):
        return False


def valid_panns(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size == PANNS_BYTES
    except OSError:
        return False


def valid_panns_labels(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle) >= 528
    except (OSError, UnicodeError):
        return False


def _replace_directory(staged: Path, target: Path) -> None:
    backup = target.with_name(f"{target.name}.invalid")
    if backup.exists():
        shutil.rmtree(backup)
    if target.exists():
        os.replace(target, backup)
    try:
        os.replace(staged, target)
    except Exception:
        if backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _adopt_file(candidates: Iterable[Path], target: Path, validator) -> bool:
    for candidate in candidates:
        if candidate.resolve() == target.resolve() or not validator(candidate):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, target)
        print(f"[MOVED] Найден перенесённый файл: {candidate} -> {target}")
        return True
    return False


def install_panns_labels(root: Path = ROOT) -> None:
    target = root / "models" / "panns_data" / "class_labels_indices.csv"
    if valid_panns_labels(target):
        return
    _adopt_file(
        (
            root / "panns_data" / target.name,
            root.parent / "panns_data" / target.name,
        ),
        target,
        valid_panns_labels,
    )
    if valid_panns_labels(target):
        return
    print("[DOWNLOAD] Справочник 527 классов AudioSet для PANNs...")
    download(PANNS_LABELS_URL, target, 2 * 1024 * 1024)
    if not valid_panns_labels(target):
        target.unlink(missing_ok=True)
        raise RuntimeError("Справочник PANNs повреждён или содержит меньше 527 классов")


def install_panns(root: Path = ROOT) -> None:
    target = root / "models" / "panns" / "Cnn14_mAP=0.431.pth"
    if not valid_panns(target):
        _adopt_file(
            (
                root / "panns" / target.name,
                root.parent / "panns" / target.name,
            ),
            target,
            valid_panns,
        )
    if not valid_panns(target):
        if root != ROOT:
            raise RuntimeError("PANNs test root is not supported")
        archive = DOWNLOADS / target.name
        print("[DOWNLOAD] PANNs Cnn14 (~320 MB, загрузка возобновляется)...")
        download(PANNS_URL, archive, 400 * 1024 * 1024)
        actual = sha256(archive)
        if actual != PANNS_SHA256:
            archive.unlink(missing_ok=True)
            raise RuntimeError(
                f"PANNs SHA-256 mismatch: expected {PANNS_SHA256}, got {actual}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(archive, target)
    install_panns_labels(root)
    print(f"[OK] PANNs Cnn14: {target}")


def _find_nested_vosk(target: Path) -> Path | None:
    if not target.is_dir():
        return None
    matches = [item for item in target.iterdir() if item.is_dir() and valid_vosk(item)]
    return matches[0] if len(matches) == 1 else None


def install_vosk(root: Path = ROOT, *, force: bool = False) -> None:
    target = root / "model-ru"
    if not force and loadable_vosk(target):
        print(f"[OK] Vosk: {target}")
        return

    nested = _find_nested_vosk(target)
    if not force and nested is not None and loadable_vosk(nested):
        staged = root / "model-ru.new"
        if staged.exists():
            shutil.rmtree(staged)
        shutil.move(str(nested), staged)
        _replace_directory(staged, target)
        print(f"[FIXED] Убрана лишняя вложенная папка Vosk: {target}")
        return

    if root != ROOT:
        raise RuntimeError("Vosk test root is not supported")
    archive = DOWNLOADS / VOSK_ARCHIVE
    extracted = DOWNLOADS / "vosk-extracted"
    if archive.exists():
        try:
            with zipfile.ZipFile(archive) as bundle:
                if bundle.testzip() is not None:
                    raise zipfile.BadZipFile("CRC error")
        except (OSError, zipfile.BadZipFile):
            archive.unlink(missing_ok=True)
    print("[DOWNLOAD] Русская модель Vosk (~45 MB, загрузка возобновляется)...")
    download(VOSK_URL, archive, VOSK_MAX_BYTES)
    expected = os.getenv("VOSK_SHA256", VOSK_SHA256).strip().casefold()
    actual = sha256(archive)
    if expected and actual != expected:
        archive.unlink(missing_ok=True)
        raise RuntimeError(f"Vosk SHA-256 mismatch: expected {expected}, got {actual}")
    if extracted.exists():
        shutil.rmtree(extracted)
    extracted.mkdir(parents=True)
    safe_extract_zip(archive, extracted)
    candidates = [item for item in extracted.iterdir() if item.is_dir() and valid_vosk(item)]
    if len(candidates) != 1:
        raise RuntimeError("Архив Vosk не содержит ожидаемую структуру модели")
    staged = root / "model-ru.new"
    if staged.exists():
        shutil.rmtree(staged)
    shutil.move(str(candidates[0]), staged)
    if not loadable_vosk(staged):
        shutil.rmtree(staged, ignore_errors=True)
        archive.unlink(missing_ok=True)
        raise RuntimeError("Скачанная модель Vosk не открывается библиотекой Vosk")
    _replace_directory(staged, target)
    shutil.rmtree(extracted, ignore_errors=True)
    print(f"[OK] Vosk: {target} (sha256={actual})")


def snapshot_download_resilient(
    repo_id: str,
    local_dir: Path,
    allow_patterns: list[str],
    *,
    attempts: int = 5,
) -> None:
    """Use the Hugging Face resumable cache and retry transient failures."""
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "180")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                allow_patterns=allow_patterns,
                max_workers=1,
            )
            return
        except Exception as exc:
            last_error = exc
            if attempt == attempts:
                break
            delay = min(2 ** (attempt - 1), 20)
            print(
                f"[RETRY {attempt}/{attempts}] Hugging Face {repo_id}: {exc}; "
                f"повтор через {delay} сек."
            )
            time.sleep(delay)
    raise RuntimeError(f"Hugging Face download failed for {repo_id}: {last_error}")


def install_faster_whisper(root: Path = ROOT) -> None:
    target = root / "models" / "faster-whisper-small"
    if valid_faster_whisper(target):
        print(f"[OK] Faster-Whisper small: {target}")
        return
    if root != ROOT:
        raise RuntimeError("Faster-Whisper test root is not supported")
    staged = root / "models" / "faster-whisper-small.new"
    print("[DOWNLOAD] Faster-Whisper small (<1 GB, загрузка возобновляется)...")
    snapshot_download_resilient(
        "Systran/faster-whisper-small",
        staged,
        [
            "config.json",
            "model.bin",
            "preprocessor_config.json",
            "tokenizer.json",
            "vocabulary.*",
        ],
    )
    total_bytes = sum(item.stat().st_size for item in staged.rglob("*") if item.is_file())
    if total_bytes > 2 * 1024 * 1024 * 1024:
        raise RuntimeError("Faster-Whisper exceeded the 2 GB safety limit")
    if not valid_faster_whisper(staged):
        raise RuntimeError(
            "Faster-Whisper model structure is incomplete; rerun setup to resume"
        )
    _replace_directory(staged, target)
    print(f"[OK] Faster-Whisper small: {target}")


def configured_yolo_target(root: Path = ROOT) -> Path:
    configured: str | None = None
    for config_path in (root / "detector_config.json", root / "detector_config.example.json"):
        if not config_path.is_file():
            continue
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
            value = raw.get("video", {}).get("model_path") if isinstance(raw, dict) else None
            if isinstance(value, str) and value.strip():
                configured = value.strip()
                break
        except (OSError, ValueError, AttributeError):
            continue
    target = Path(configured or "models/yolov8n-pose.pt")
    return target if target.is_absolute() else root / target


def install_yolo(root: Path = ROOT) -> None:
    target = configured_yolo_target(root)
    asset_name = target.name
    if not _file_at_least(target, 1024 * 1024):
        _adopt_file(
            (root / asset_name, root.parent / asset_name),
            target,
            lambda path: _file_at_least(path, 1024 * 1024),
        )
    if _file_at_least(target, 1024 * 1024):
        print(f"[OK] YOLO-Pose: {target}")
    else:
        print(f"[DOWNLOAD] YOLO-Pose ({asset_name})...")
        from ultralytics import YOLO

        model = YOLO(asset_name)
        source = Path(model.ckpt_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if source != target.resolve():
            shutil.copy2(source, target)
        if not _file_at_least(target, 1024 * 1024):
            raise RuntimeError("YOLO-Pose загрузился, но файл весов повреждён")
        print(f"[OK] YOLO-Pose: {target}")

    from video.model_optimizer import prepare_yolo_artifacts

    artifacts = prepare_yolo_artifacts(target)
    print(f"[OK] YOLO ONNX: {artifacts['onnx'] or 'не создан (fallback PT)'}")
    print(f"[OK] YOLO TensorRT: {artifacts['engine'] or 'не создан (fallback ONNX/PT)'}")


def ensure_local_config(root: Path = ROOT) -> None:
    source = root / "detector_config.example.json"
    target = root / "detector_config.json"
    if not target.exists() and source.is_file():
        shutil.copy2(source, target)
        print(f"[OK] Создан локальный конфиг: {target}")


def resource_status(root: Path = ROOT) -> dict[str, bool]:
    return {
        "PANNs Cnn14": (
            valid_panns(root / "models" / "panns" / "Cnn14_mAP=0.431.pth")
            and valid_panns_labels(
                root / "models" / "panns_data" / "class_labels_indices.csv"
            )
        ),
        "YOLO-Pose": _file_at_least(configured_yolo_target(root), 1024 * 1024),
        "Vosk": loadable_vosk(root / "model-ru"),
        "Faster-Whisper small": valid_faster_whisper(
            root / "models" / "faster-whisper-small"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Подготовка моделей Detector Danger")
    parser.add_argument("--check-only", action="store_true", help="ничего не скачивать")
    parser.add_argument("--runtime-only", action="store_true", help="проверить только Python")
    parser.add_argument("--vosk-only", action="store_true", help="переустановить только Vosk")
    parser.add_argument("--force", action="store_true", help="скачать выбранную модель заново")
    args = parser.parse_args()

    supported, detail = supported_python()
    print(f"[{'OK' if supported else 'FAIL'}] {detail}")
    if not supported:
        print("Требуется Python 3.12 x64.")
        return 2
    if args.runtime_only:
        return 0
    if args.vosk_only:
        install_vosk(force=args.force)
        return 0 if loadable_vosk(ROOT / "model-ru") else 1

    if not args.check_only:
        install_vosk()
        install_faster_whisper()
        install_yolo()
        install_panns()
        ensure_local_config()

    status = resource_status()
    for name, ready in status.items():
        print(f"[{'OK' if ready else 'MISSING'}] {name}")
    return 0 if all(status.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
