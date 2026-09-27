from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from imageio_ffmpeg import get_ffmpeg_exe

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DeliveryVideo:
    path: str
    sha256: str
    size_bytes: int
    crf: int
    max_width: int
    max_height: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_delivery_video(
    source_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    fps: float = 10.0,
    max_bytes: int = 2_000_000,
    initial_crf: int = 28,
) -> DeliveryVideo:
    """Transcode an incident clip to a small AVC MP4 named by its SHA-256."""
    source = Path(source_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Исходный клип не найден: {source}")
    target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = get_ffmpeg_exe()

    start_crf = min(max(int(initial_crf), 0), 51)
    attempts: list[tuple[int, int, int]] = []
    for width, height, offsets in (
        (960, 540, (0, 4, 8, 12, 16)),
        (640, 360, (10, 14, 18)),
        (480, 270, (16, 20, 23)),
    ):
        for offset in offsets:
            candidate = min(51, start_crf + offset)
            item = (candidate, width, height)
            if item not in attempts:
                attempts.append(item)

    errors: list[str] = []
    for crf, width, height in attempts:
        temporary = target_dir / f".{uuid.uuid4().hex}.tmp.mp4"
        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-vf",
            (
                f"fps={float(fps):g},"
                f"scale={width}:{height}:force_original_aspect_ratio=decrease:"
                "force_divisible_by=2"
            ),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "48k",
            "-movflags",
            "+faststart",
            str(temporary),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        if completed.returncode != 0 or not temporary.is_file():
            errors.append(completed.stderr.strip()[:500] or f"ffmpeg code {completed.returncode}")
            temporary.unlink(missing_ok=True)
            continue
        size = temporary.stat().st_size
        if size <= 0 or size >= int(max_bytes):
            logger.info(
                "REST-video CRF=%d %dx%d: %d bytes, требуется меньше %d",
                crf,
                width,
                height,
                size,
                max_bytes,
            )
            temporary.unlink(missing_ok=True)
            continue

        digest = _sha256(temporary).lower()
        target = target_dir / f"{digest}.mp4"
        if target.is_file():
            temporary.unlink(missing_ok=True)
        else:
            os.replace(temporary, target)
        logger.info(
            "REST-video подготовлено: %s (%d bytes, AVC, %.1f FPS, CRF=%d)",
            target.name,
            target.stat().st_size,
            fps,
            crf,
        )
        return DeliveryVideo(
            path=str(target),
            sha256=digest,
            size_bytes=target.stat().st_size,
            crf=crf,
            max_width=width,
            max_height=height,
        )

    detail = errors[-1] if errors else "все варианты CRF превысили ограничение"
    raise RuntimeError(
        f"Не удалось подготовить AVC/MP4 меньше {int(max_bytes)} байт: {detail}"
    )
