"""Windows-safe path handling for Vosk's native model loader.

Vosk 0.3.45 can fail on otherwise valid models when any parent directory
contains non-ASCII characters. Prefer an 8.3 path and otherwise maintain an
ASCII-only runtime copy outside the project.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shutil
from pathlib import Path


def _is_ascii(path: Path) -> bool:
    try:
        str(path).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _short_windows_path(path: Path) -> Path | None:
    if os.name != "nt":
        return None
    try:
        function = ctypes.windll.kernel32.GetShortPathNameW
        required = function(str(path), None, 0)
        if not required:
            return None
        buffer = ctypes.create_unicode_buffer(required)
        if not function(str(path), buffer, required):
            return None
        result = Path(buffer.value)
        return result if _is_ascii(result) else None
    except (AttributeError, OSError, ValueError):
        return None


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    for relative in (
        "am/final.mdl",
        "graph/Gr.fst",
        "graph/HCLr.fst",
        "ivector/final.ie",
    ):
        item = path / relative
        stat = item.stat()
        digest.update(relative.encode("ascii"))
        digest.update(str(stat.st_size).encode("ascii"))
        with item.open("rb") as handle:
            digest.update(handle.read(64 * 1024))
    return digest.hexdigest()


def _cache_roots() -> list[Path]:
    roots: list[Path] = []
    if value := os.getenv("DETECTOR_ASCII_CACHE"):
        roots.append(Path(value).expanduser())
    # Public Documents is normally writable by interactive users and, unlike a
    # user profile, has an ASCII path even when the Windows account name does not.
    public = os.getenv("PUBLIC", r"C:\Users\Public")
    roots.append(Path(public) / "Documents" / "DetectorDanger")
    program_data = os.getenv("PROGRAMDATA", r"C:\ProgramData")
    roots.append(Path(program_data) / "DetectorDanger")
    roots.append(Path(r"C:\DetectorDangerData"))
    return [root for root in roots if _is_ascii(root)]


def _sync_ascii_copy(source: Path, cache_root: Path) -> Path:
    target = cache_root / "models" / "vosk-model-small-ru-0.22"
    marker = target / ".detector-source.json"
    fingerprint = _fingerprint(source)
    try:
        if marker.is_file():
            saved = json.loads(marker.read_text(encoding="utf-8"))
            if (
                saved.get("fingerprint") == fingerprint
                and _fingerprint(target) == fingerprint
            ):
                return target
    except (OSError, UnicodeError, ValueError):
        pass

    cache_root.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(f"{target.name}.new-{os.getpid()}")
    if staged.exists():
        shutil.rmtree(staged)
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, staged)
    (staged / ".detector-source.json").write_text(
        json.dumps({"fingerprint": fingerprint}, indent=2) + "\n",
        encoding="utf-8",
    )
    if target.exists():
        shutil.rmtree(target)
    os.replace(staged, target)
    return target


def vosk_runtime_path(path: str | os.PathLike[str]) -> str:
    """Return a path that the native Windows Vosk library can actually open."""
    source = Path(path).expanduser().resolve()
    # Preserve Vosk's own missing-path error (and allow lightweight mocked
    # model paths in unit tests) instead of trying to mirror a non-existent tree.
    if not source.exists():
        return str(source)
    if os.name != "nt" or _is_ascii(source):
        return str(source)

    if short := _short_windows_path(source):
        return str(short)

    errors: list[str] = []
    for cache_root in _cache_roots():
        try:
            return str(_sync_ascii_copy(source, cache_root))
        except (OSError, ValueError, shutil.Error) as exc:
            errors.append(f"{cache_root}: {exc}")
    raise RuntimeError(
        "Vosk не поддерживает кириллицу в пути модели, а ASCII-копию создать "
        "не удалось. Задайте DETECTOR_ASCII_CACHE, например C:\\DetectorDangerData. "
        + "; ".join(errors)
    )


__all__ = ["vosk_runtime_path"]
