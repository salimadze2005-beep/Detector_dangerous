"""Measure the live latest-frame video pipeline without changing camera settings."""
from __future__ import annotations

import threading
import time
import json
import tempfile
from types import SimpleNamespace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import load_config
from ui.workers import VideoSystemWorker


class _Emitter:
    def __init__(self, callback=None) -> None:
        self.callback = callback or (lambda *args: None)

    def emit(self, *args) -> None:
        self.callback(*args)


def main() -> int:
    checkpoints = (5, 30, 120)
    metrics: dict[str, object] = {}
    result_path = Path(tempfile.gettempdir()) / "detector-rtsp-realtime-benchmark.jsonl"
    result_path.unlink(missing_ok=True)
    config = load_config()

    def on_metrics(_camera_id: str, values: dict) -> None:
        metrics.update(values)

    bridge = SimpleNamespace(
        component_status=_Emitter(), video_frame=_Emitter(),
        video_metrics=_Emitter(on_metrics), log=_Emitter(),
    )
    worker = VideoSystemWorker(
        bridge=bridge,
        event_bus=SimpleNamespace(publish=lambda _event: None),
        app_config=config,
        video_source=config.video.sources[0],
        frame_store=SimpleNamespace(update=lambda *args, **kwargs: None),
    )
    thread = threading.Thread(target=worker.run, name="rtsp-realtime-benchmark", daemon=True)
    started = time.monotonic()
    thread.start()
    for checkpoint in checkpoints:
        while time.monotonic() - started < checkpoint:
            time.sleep(0.1)
        result = {"elapsed_sec": checkpoint, **metrics}
        print(result, flush=True)
        with result_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    print({"results_file": str(result_path)}, flush=True)
    worker.requestInterruption()
    thread.join(timeout=8.0)
    return 0 if not thread.is_alive() else 2


if __name__ == "__main__":
    raise SystemExit(main())
