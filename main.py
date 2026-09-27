import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMessageBox, QPushButton

from core.config import config
from core.event_clips import EventClipRecorder
from core.logging_setup import configure_logging
from ui.runtime_dashboard import SecurityDashboard
from ui.incident_history import install_incident_history


def _install_incidents_header_button(window, controller) -> None:
    """Keep incident history accessible from the dashboard header, not the menu bar."""
    button = QPushButton("📋  ИСТОРИЯ")
    button.setObjectName("incidentButton")
    button.setToolTip("Открыть историю тревог, фото, видео и заметки")
    button.setStyleSheet(
        "QPushButton { background:#30344d; color:#fff; border:1px solid #59617f; "
        "border-radius:8px; font-size:14px; font-weight:700; padding:11px 16px; }"
        "QPushButton:hover { background:#3c425f; border-color:#7a85ad; }"
        "QPushButton:pressed { background:#262b40; }"
    )
    button.clicked.connect(controller.show_history)

    root = window.centralWidget().layout()
    header = root.itemAt(0).layout() if root is not None and root.count() else None
    if header is not None:
        settings_index = header.indexOf(window.settings_button)
        if settings_index >= 0:
            header.insertWidget(settings_index, button)
        else:
            header.addWidget(button)

    # The menu entry was only an intermediate UI and is intentionally hidden.
    window.menuBar().hide()
    window.incident_history_button = button


def main() -> int:
    configure_logging(config.paths.log_file)
    app = QApplication(sys.argv)
    errors = config.validate()
    if errors:
        QMessageBox.critical(
            None,
            "Detector Danger — ошибка конфигурации",
            "Система не запущена:\n\n" + "\n".join(f"• {error}" for error in errors),
        )
        return 2

    window = SecurityDashboard(config)
    clips_root = Path(config.paths.data_dir) / "event_clips"

    # Gunshot/keyword: 5 s before + 5 s after.
    # Fall: 2 s before lying + configured confirmation interval + 2 s after alarm.
    clip_recorder = EventClipRecorder(
        frame_store=window.frame_store,
        output_dir=str(clips_root),
        pre_sec=5.0,
        post_sec=5.0,
        fall_pre_sec=2.0,
        fall_post_sec=2.0,
        fall_confirmation_sec=float(config.video.fall_duration_sec),
        fps=10.0,
        jpeg_quality=70,
        max_width=960,
        max_height=540,
        on_clip_ready=window.alert_service.prepare_video_delivery,
    )
    window.event_bus.subscribe_all(clip_recorder.capture_event)
    app.aboutToQuit.connect(clip_recorder.close)

    incident_history = install_incident_history(
        window,
        db_path=config.paths.event_db,
        clips_root=str(clips_root),
    )
    _install_incidents_header_button(window, incident_history)

    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
