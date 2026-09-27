"""Reject runtime state and unsafe artifacts from the version-controlled tree."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import PurePosixPath


MAX_TRACKED_FILE_BYTES = 25 * 1024 * 1024
FORBIDDEN_EXACT = {"detector_config.json"}
FORBIDDEN_SUFFIXES = {".log", ".sqlite", ".sqlite3", ".zip", ".mp4"}


def inspect_path(path: str, size: int | None = None) -> str | None:
    """Return a human-readable policy violation, or ``None`` for a safe path."""
    normalized = PurePosixPath(path.replace("\\", "/"))
    if str(normalized) in FORBIDDEN_EXACT:
        return "рабочая конфигурация объекта запрещена; используйте detector_config.example.json"
    if normalized.parts and normalized.parts[0] == "data":
        return "runtime-данные запрещены; data создаётся при запуске"
    if normalized.suffix.lower() in FORBIDDEN_SUFFIXES:
        return f"runtime-артефакт {normalized.suffix} запрещён"
    if size is not None and size > MAX_TRACKED_FILE_BYTES:
        return f"файл больше {MAX_TRACKED_FILE_BYTES // (1024 * 1024)} MiB; храните его вне Git или как релизный артефакт"
    return None


def git_paths(mode: str) -> list[str]:
    command = ["git", "diff", "--cached", "--name-only"] if mode == "staged" else ["git", "ls-files"]
    return [line for line in subprocess.check_output(command, text=True, encoding="utf-8").splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="проверить только индекс перед коммитом")
    args = parser.parse_args()
    violations: list[str] = []
    for item in git_paths("staged" if args.staged else "tracked"):
        try:
            size = None if args.staged else __import__("os").path.getsize(item)
        except OSError:
            size = None
        if problem := inspect_path(item, size):
            violations.append(f"{item}: {problem}")
    if violations:
        print("[ERROR] Repository hygiene check failed:", file=sys.stderr)
        print(*violations, sep="\n", file=sys.stderr)
        return 1
    print("[OK] Repository hygiene check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
