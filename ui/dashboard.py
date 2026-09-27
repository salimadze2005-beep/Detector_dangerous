from __future__ import annotations

import logging
import math
from collections import deque
from pathlib import Path

import cv2
from PyQt6.QtCore import QDateTime, QEvent, QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QResizeEvent
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import (
    QAbstractButton,
    QDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QStyleOptionButton,
    QStylePainter,
    QVBoxLayout,
    QWidget,
)

from core.alerts import ALERT_EVENT_TYPES, AlertService, describe_event
from core.config import AppConfig, get_resource_path, save_operator_settings
from core.event_bus import Event, EventBus, EventType
from core.frames import FrameStore
from core.health import ComponentState, HealthRegistry
from core.storage import EventStore, OutboxItem
from ui.settings_dialog import SettingsDialog
from ui.workers import AudioSystemWorker, SignalBridge, VideoSystemWorker, camera_id_for_source

logger = logging.getLogger(__name__)

BASE_COLOR = "#26293d"
PANEL_COLOR = "#30344d"
ACCENT_COLOR = "#a7dc00"
ALARM_COLOR = "#e3424f"


class CenteredCrossButton(QPushButton):
    """Draw the exit cross geometrically so font metrics cannot offset it."""

    def paintEvent(self, event) -> None:
        if self.text() != "×":
            super().paintEvent(event)
            return
        if self.objectName() == "viewerClose":
            background = "#e3424f" if self.underMouse() else "#26293d"
            if self.isDown():
                background = "#ba3440"
            button_painter = QPainter(self)
            button_painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            button_painter.setPen(QPen(QColor("#555b78"), 1.2))
            button_painter.setBrush(QColor(background))
            button_painter.drawEllipse(QRectF(2, 2, self.width() - 4, self.height() - 4))
            button_painter.end()
        else:
            option = QStyleOptionButton()
            self.initStyleOption(option)
            option.text = ""
            style_painter = QStylePainter(self)
            style_painter.drawControl(QStyle.ControlElement.CE_PushButton, option)
            style_painter.end()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor("#ffffff"), 2.6)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        center = QPointF(self.width() / 2.0, self.height() / 2.0)
        radius = 5.2
        painter.drawLine(
            QPointF(center.x() - radius, center.y() - radius),
            QPointF(center.x() + radius, center.y() + radius),
        )
        painter.drawLine(
            QPointF(center.x() + radius, center.y() - radius),
            QPointF(center.x() - radius, center.y() + radius),
        )


class CameraModeToggle(QAbstractButton):
    """Stable two-position switch: its labels never change the widget geometry."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFixedSize(144, 28)
        self.setToolTip("Переключить между оригиналом и изображением с разметкой ИИ")

    def sizeHint(self) -> QSize:
        return QSize(144, 28)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bounds = QRectF(0.5, 0.5, self.width() - 1.0, self.height() - 1.0)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(Qt.GlobalColor.transparent)
        painter.setBrush(QColor("#202335"))
        painter.drawRoundedRect(bounds, 14, 14)

        half = self.width() / 2.0
        selected = QRectF(half, 1, half - 1, self.height() - 2) if self.isChecked() else QRectF(1, 1, half - 1, self.height() - 2)
        painter.setBrush(QColor(ACCENT_COLOR))
        painter.drawRoundedRect(selected, 13, 13)

        font = painter.font()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        active = QColor("#1b1d2b")
        inactive = QColor("#aeb4cd")
        left = QRectF(0, 0, half, self.height())
        right = QRectF(half, 0, half, self.height())
        painter.setPen(inactive if self.isChecked() else active)
        painter.drawText(left, Qt.AlignmentFlag.AlignCenter, "ОРИГИНАЛ")
        painter.setPen(active if self.isChecked() else inactive)
        painter.drawText(right, Qt.AlignmentFlag.AlignCenter, "ИИ")


class CameraTile(QFrame):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("cameraTile")
        self.setMinimumSize(360, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(
            """
            QFrame#cameraTile { background:#11131d; border:1px solid #424762; border-radius:10px; }
            QLabel { border:0; }
            """
        )
        self.video = QLabel("Ожидание видеопотока")
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setStyleSheet("color:#8d93ad; font-size:16px; background:#0b0c12;")
        self.title = QLabel(title)
        self.title.setStyleSheet("color:#f4f6ff; font-size:14px; font-weight:600; padding:8px 12px;")
        self.state = QLabel("ОСТАНОВЛЕНА")
        self.state.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.state.setStyleSheet("color:#9ca1b8; font-size:12px; font-weight:700; padding:8px 12px;")
        self.metrics = QLabel("")
        self.metrics.setStyleSheet("color:#9ca1b8; font-size:11px; padding:8px 4px;")
        self.raw_frame = None
        self.annotated_frame = None
        self.display_mode = "annotated"
        self.mode_button = CameraModeToggle()
        self.mode_button.setObjectName("modeToggle")
        self.mode_button.toggled.connect(self._toggle_mode)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.addWidget(self.title)
        footer.addStretch()
        footer.addWidget(self.metrics)
        footer.addWidget(self.mode_button)
        footer.addWidget(self.state)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)
        layout.addWidget(self.video, 1)
        layout.addLayout(footer)

    def set_frames(self, raw_frame, annotated_frame) -> None:
        self.raw_frame = raw_frame
        self.annotated_frame = annotated_frame
        self._render_frame()

    def clear_frame(self, message: str = "Мониторинг остановлен") -> None:
        self.raw_frame = None
        self.annotated_frame = None
        self.video.setPixmap(QPixmap())
        self.video.setText(message)

    def _toggle_mode(self, enabled: bool) -> None:
        self.display_mode = "annotated" if enabled else "raw"
        self._render_frame()

    def current_frame(self):
        return self.annotated_frame if self.display_mode == "annotated" else self.raw_frame

    def _render_frame(self) -> None:
        frame = self.current_frame()
        if frame is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb.shape
        image = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()
        self.video.setPixmap(
            QPixmap.fromImage(image).scaled(
                self.video.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def set_state(self, state: ComponentState, message: str = "") -> None:
        labels = {
            ComponentState.STOPPED: ("ОСТАНОВЛЕНА", "#9ca1b8"),
            ComponentState.STARTING: ("ЗАПУСК", "#f7c85c"),
            ComponentState.RUNNING: ("В ЭФИРЕ", ACCENT_COLOR),
            ComponentState.DEGRADED: ("НЕСТАБИЛЬНО", "#f7a84f"),
            ComponentState.ERROR: ("ОШИБКА", ALARM_COLOR),
            ComponentState.RECONNECTING: ("ПОДКЛЮЧЕНИЕ", "#69a9ff"),
        }
        text, color = labels[state]
        self.state.setText(text)
        self.state.setStyleSheet(
            f"color:{color}; font-size:12px; font-weight:700; padding:8px 12px;"
        )
        self.state.setToolTip(message)
        if state in (ComponentState.ERROR, ComponentState.RECONNECTING) and message:
            self.video.setText(message)
            self.video.setPixmap(QPixmap())

    def set_metrics(self, metrics: dict) -> None:
        self.metrics.setText(
            f"{metrics.get('age_sec', 0):.1f}с · "
            f"in {metrics.get('input_fps', 0):.0f} · "
            f"AI {metrics.get('analysis_fps', 0):.0f} fps · "
            f"drop {metrics.get('dropped_frames', 0)}"
        )
        self.metrics.setToolTip(
            "Возраст кадра: {age:.3f}с\nRTSP p95: {read:.1f} мс\n"
            "Инференс p95: {inference:.1f} мс\nПредпросмотр: {preview:.0f} fps"
            .format(
                age=float(metrics.get("age_sec", 0)),
                read=float(metrics.get("read_p95_ms", 0)),
                inference=float(metrics.get("inference_p95_ms", 0)),
                preview=float(metrics.get("preview_fps", 0)),
            )
        )

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._render_frame()


class CameraViewer(QFrame):
    """Single-camera overlay with mouse and keyboard navigation."""

    previous_requested = pyqtSignal()
    next_requested = pyqtSignal()
    close_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("cameraViewer")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setStyleSheet(
            """
            QFrame#cameraViewer { background:#07080d; }
            QLabel#viewerVideo { background:#07080d; color:#8d93ad; font-size:17px; }
            QLabel#viewerTitle {
                background:transparent; color:#fff; border:0;
                font-size:18px; font-weight:700; padding:0;
            }
            QLabel#viewerCounter {
                background:rgba(20, 22, 34, 205); color:#f4f6ff; border-radius:8px;
                font-size:16px; font-weight:700; padding:8px 14px;
            }
            QPushButton#viewerClose {
                background:rgba(38, 41, 61, 225); color:#fff; border:1px solid #555b78;
                border-radius:25px; font-size:24px; font-weight:700; padding:0;
            }
            QPushButton#viewerClose:hover { background:#e3424f; border-color:#e3424f; }
            QPushButton#viewerNavigation {
                background:transparent; color:rgba(255, 255, 255, 105); border:0;
                font-size:58px; font-weight:300;
            }
            QPushButton#viewerNavigation:hover { background:rgba(38, 41, 61, 120); color:#fff; }
            """
        )
        self._frame = None
        self.video = QLabel("Ожидание видеопотока", self)
        self.video.setObjectName("viewerVideo")
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title = QLabel("", self)
        self.title.setObjectName("viewerTitle")
        title_shadow = QGraphicsDropShadowEffect(self.title)
        title_shadow.setBlurRadius(4)
        title_shadow.setColor(QColor(0, 0, 0, 230))
        title_shadow.setOffset(1, 1)
        self.title.setGraphicsEffect(title_shadow)
        self.counter = QLabel("", self)
        self.counter.setObjectName("viewerCounter")
        self.counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.close_button = CenteredCrossButton("×", self)
        self.close_button.setObjectName("viewerClose")
        self.previous_button = QPushButton("‹", self)
        self.previous_button.setObjectName("viewerNavigation")
        self.next_button = QPushButton("›", self)
        self.next_button.setObjectName("viewerNavigation")
        for button in (self.close_button, self.previous_button, self.next_button):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.close_button.clicked.connect(self.close_requested.emit)
        self.previous_button.clicked.connect(self.previous_requested.emit)
        self.next_button.clicked.connect(self.next_requested.emit)
        self.hide()

    def show_camera(self, title: str, index: int, count: int, frame) -> None:
        self.title.setText(title)
        self.counter.setText(f"{index + 1} / {count}")
        navigation_enabled = count > 1
        self.previous_button.setVisible(navigation_enabled)
        self.next_button.setVisible(navigation_enabled)
        self.set_frame(frame)

    def set_frame(self, frame) -> None:
        self._frame = frame
        if frame is None:
            self.video.setPixmap(QPixmap())
            self.video.setText("Ожидание видеопотока")
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb.shape
        image = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()
        self.video.setText("")
        self.video.setPixmap(QPixmap.fromImage(image).scaled(
            self.video.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        width, height = self.width(), self.height()
        self.video.setGeometry(self.rect())
        self.title.setGeometry(24, 18, min(420, max(180, width - 120)), 42)
        self.close_button.setGeometry(width - 74, 24, 50, 50)
        zone = max(52, min(100, width // 14))
        self.previous_button.setGeometry(0, 0, zone, height)
        self.next_button.setGeometry(width - zone, 0, zone, height)
        self.counter.setGeometry(max(10, width // 2 - 45), height - 60, 90, 38)
        self.video.lower()
        self.title.raise_()
        self.counter.raise_()
        self.close_button.raise_()
        self.set_frame(self._frame)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Left:
            self.previous_requested.emit()
        elif event.key() == Qt.Key.Key_Right:
            self.next_requested.emit()
        elif event.key() == Qt.Key.Key_Escape:
            self.close_requested.emit()
        else:
            super().keyPressEvent(event)


class AlertOverlay(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("alertOverlay")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            """
            QFrame#alertOverlay { background:rgba(10, 11, 18, 205); }
            QFrame#alertCard { background:#26293d; border:2px solid #e3424f; border-radius:16px; }
            QLabel#alertTitle { color:#fff; font-size:28px; font-weight:800; }
            QLabel#alertTime { color:#ffbdc2; font-size:14px; }
            QLabel#alertDescription { color:#f4f6ff; font-size:18px; }
            QPushButton { background:#e3424f; color:white; border:0; border-radius:8px;
                          font-size:16px; font-weight:700; padding:13px 26px; }
            QPushButton:hover { background:#f05260; }
            """
        )
        card = QFrame()
        card.setObjectName("alertCard")
        card.setMaximumWidth(980)
        card.setMinimumWidth(700)
        card_layout = QGridLayout(card)
        card_layout.setContentsMargins(24, 24, 24, 24)
        card_layout.setHorizontalSpacing(24)
        card_layout.setVerticalSpacing(10)

        self.image = QLabel("Снимок недоступен")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setFixedSize(430, 280)
        self.image.setStyleSheet("background:#11131d; color:#9ca1b8; border-radius:8px;")
        self.title = QLabel("ТРЕВОГА")
        self.title.setObjectName("alertTitle")
        self.time = QLabel()
        self.time.setObjectName("alertTime")
        self.description = QLabel()
        self.description.setObjectName("alertDescription")
        self.description.setWordWrap(True)
        self.acknowledge = QPushButton("ПОДТВЕРДИТЬ ТРЕВОГУ")

        details = QVBoxLayout()
        details.addWidget(self.title)
        details.addWidget(self.time)
        details.addSpacing(10)
        details.addWidget(self.description, 1)
        details.addWidget(self.acknowledge, 0, Qt.AlignmentFlag.AlignLeft)
        card_layout.addWidget(self.image, 0, 0)
        card_layout.addLayout(details, 0, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.addStretch()
        layout.addWidget(card, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()
        self.hide()

    def show_incident(self, event: Event, item: OutboxItem | None) -> None:
        titles = {
            EventType.GUNSHOT_DETECTED: "ТРЕВОГА: ВОЗМОЖНЫЙ ВЫСТРЕЛ",
            EventType.KEYWORD_DETECTED: "ТРЕВОГА: КРИК О ПОМОЩИ",
            EventType.FALL_DETECTED: "ТРЕВОГА: ПАДЕНИЕ ЧЕЛОВЕКА",
        }
        self.title.setText(titles.get(event.type, "ТРЕВОГА"))
        self.time.setText(event.timestamp.astimezone().strftime("%d.%m.%Y  %H:%M:%S"))
        self.description.setText(item.description if item else describe_event(event, False))
        image_path = Path(item.image_path) if item and item.image_path else None
        pixmap = QPixmap(str(image_path)) if image_path and image_path.is_file() else QPixmap()
        if pixmap.isNull():
            self.image.setPixmap(QPixmap())
            self.image.setText("Снимок камеры недоступен")
        else:
            self.image.setText("")
            self.image.setPixmap(
                pixmap.scaled(
                    self.image.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        self.show()
        self.raise_()


class SecurityDashboard(QMainWindow):
    def __init__(self, app_config: AppConfig):
        super().__init__()
        self.config = app_config
        self.audio_workers: dict[str, AudioSystemWorker] = {}
        self.video_workers: dict[str, VideoSystemWorker] = {}
        self.camera_tiles: dict[str, CameraTile] = {}
        self.camera_names: dict[str, str] = {}
        self.video_states: dict[str, ComponentState] = {}
        self.monitoring_active = False
        self.alert_queue: deque[tuple[Event, OutboxItem | None]] = deque()
        self.current_alert: Event | None = None
        self.viewer_camera_index = 0
        self._pending_video_frames: dict[str, tuple[object, object]] = {}

        self.bridge = SignalBridge()
        self.health = HealthRegistry()
        self.frame_store = FrameStore()
        self.event_store = EventStore(app_config.paths.event_db)
        self.alert_service = AlertService(
            app_config.alerts,
            self.event_store,
            self.frame_store,
            app_config.paths.screenshots_dir,
            config_persistor=lambda: save_operator_settings(self.config),
        )
        self.event_bus = EventBus()
        self.event_bus.subscribe_all(self.event_store.record)
        self.event_bus.subscribe_all(self.alert_service.handle_event)
        self.event_bus.subscribe_all(self.bridge.event_received.emit)
        self.alert_service.start()
        self.event_bus.start()

        self._build_ui()
        self.bridge.event_received.connect(self._handle_event)
        self.bridge.video_frame.connect(self._queue_video_frame)
        self.bridge.video_metrics.connect(self.update_video_metrics)
        self.bridge.log.connect(self.append_log)
        self.bridge.component_status.connect(self.update_component_status)
        self.alert_overlay.acknowledge.clicked.connect(self.acknowledge_alert)

        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self._update_clock)
        self.clock_timer.start(1000)
        self._update_clock()
        self._prepare_camera_tiles()

    def _build_ui(self) -> None:
        self.setWindowTitle("Центр видеонаблюдения")
        self.resize(1440, 900)
        self.setMinimumSize(1024, 680)
        self.setStyleSheet(
            f"""
            QMainWindow, QWidget#central {{ background:{BASE_COLOR}; color:#f4f6ff;
                font-family:'Segoe UI'; }}
            QScrollArea {{ background:#191b29; border:0; }}
            QScrollBar:vertical {{ background:#202235; width:10px; }}
            QScrollBar::handle:vertical {{ background:#4b506f; border-radius:5px; }}
            QPushButton#controlButton {{ background:{ACCENT_COLOR}; color:#1b1d2b; border:0;
                border-radius:8px; font-size:14px; font-weight:700; padding:12px 22px; }}
            QPushButton#controlButton:hover {{ background:#b9ed15; }}
            QPushButton#settingsButton {{ background:{PANEL_COLOR}; color:#fff; border:1px solid #4b506f;
                border-radius:8px; font-size:14px; font-weight:700; padding:11px 20px; }}
            QPushButton#settingsButton:hover {{ background:#3b405c; }}
            QFrame#displayControls {{ background:#202335; border:1px solid #393e59;
                border-radius:10px; }}
            QPushButton#displayButton {{ background:{PANEL_COLOR}; color:#fff;
                border:1px solid #4b506f; border-radius:7px; font-size:14px;
                font-weight:700; padding:10px 14px; }}
            QPushButton#displayButton:hover {{ background:#3b405c; }}
            QFrame#headerDivider {{ background:#454a67; border:0; }}
            QPushButton#fullscreenButton {{ background:{PANEL_COLOR}; color:#fff; border:1px solid #4b506f;
                border-radius:8px; font-size:20px; font-weight:700; padding:0; }}
            QPushButton#fullscreenButton:hover {{ background:#3b405c; }}
            """
        )
        central = QWidget()
        central.setObjectName("central")
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 10)
        root.setSpacing(8)

        header = QHBoxLayout()
        self.logo = QLabel()
        self.logo.setPixmap(self._render_logo())
        self.logo.setStyleSheet("background:transparent; border:0; padding:2px;")
        self.layout_view_button = QPushButton("▣  ПРОСМОТР")
        self.layout_view_button.setObjectName("displayButton")
        self.layout_view_button.setToolTip("Открыть первую камеру в режиме просмотра")
        self.layout_view_button.clicked.connect(self.open_camera_viewer)
        self.all_mode_label = QLabel("РЕЖИМ ВСЕХ КАМЕР")
        self.all_mode_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.all_mode_label.setStyleSheet(
            "color:#aeb4cd; font-size:9px; font-weight:700; background:transparent;"
        )
        self.all_mode_toggle = CameraModeToggle()
        self.all_mode_toggle.setToolTip("Переключить оригинал / ИИ сразу для всех камер")
        self.all_mode_toggle.toggled.connect(self.set_all_camera_modes)
        self.clock = QLabel()
        self.clock.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.clock.setStyleSheet("color:#fff; font-size:22px; font-weight:700; padding:0 20px;")
        self.control_button = QPushButton("ЗАПУСТИТЬ")
        self.control_button.setObjectName("controlButton")
        self.control_button.clicked.connect(self.toggle_monitoring)
        self.settings_button = QPushButton("⚙  НАСТРОЙКИ")
        self.settings_button.setObjectName("settingsButton")
        self.settings_button.clicked.connect(self.open_settings)
        self.fullscreen_button = CenteredCrossButton("⛶")
        self.fullscreen_button.setObjectName("fullscreenButton")
        self.fullscreen_button.setFixedSize(56, 50)
        self.fullscreen_button.setToolTip("Полноэкранный режим (F11)")
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)

        mode_block = QWidget()
        mode_block.setStyleSheet("background:transparent;")
        mode_layout = QVBoxLayout(mode_block)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(2)
        mode_layout.addWidget(self.all_mode_label)
        mode_layout.addWidget(self.all_mode_toggle)

        display_controls = QFrame()
        display_controls.setObjectName("displayControls")
        display_layout = QHBoxLayout(display_controls)
        display_layout.setContentsMargins(8, 5, 8, 5)
        display_layout.setSpacing(8)
        display_layout.addWidget(mode_block)
        display_layout.addWidget(self.layout_view_button)

        header_divider = QFrame()
        header_divider.setObjectName("headerDivider")
        header_divider.setFixedSize(1, 42)

        header.addWidget(self.logo)
        header.addStretch()
        header.addWidget(self.clock)
        header.addStretch()
        header.addWidget(display_controls)
        header.addWidget(header_divider)
        header.addWidget(self.control_button)
        header.addWidget(self.settings_button)
        header.addWidget(self.fullscreen_button)
        root.addLayout(header)

        self.video_grid_widget = QWidget()
        self.video_grid_widget.setStyleSheet("background:#191b29;")
        self.video_grid = QGridLayout(self.video_grid_widget)
        self.video_grid.setContentsMargins(12, 12, 12, 12)
        self.video_grid.setSpacing(12)
        self.video_scroll = QScrollArea()
        self.video_scroll.setWidgetResizable(True)
        self.video_scroll.setWidget(self.video_grid_widget)
        self.video_scroll.viewport().installEventFilter(self)
        root.addWidget(self.video_scroll, 1)
        self.setCentralWidget(central)

        self.alert_overlay = AlertOverlay(self.video_scroll.viewport())
        self.alert_overlay.setGeometry(self.video_scroll.viewport().rect())
        self.camera_viewer = CameraViewer(central)
        self.camera_viewer.setGeometry(central.rect())
        self.camera_viewer.previous_requested.connect(lambda: self.step_camera_viewer(-1))
        self.camera_viewer.next_requested.connect(lambda: self.step_camera_viewer(1))
        self.camera_viewer.close_requested.connect(self.close_camera_viewer)

    @staticmethod
    def _render_logo() -> QPixmap:
        """Render the compact logo variant from the supplied brand sheet, transparently."""
        renderer = QSvgRenderer(get_resource_path("ui/assets/company_brand.svg"))
        renderer.setViewBox(QRectF(4100, 5500, 12600, 2300))
        pixmap = QPixmap(150, 30)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        renderer.render(painter, QRectF(0, 0, 150, 30))
        painter.end()
        return pixmap

    def _update_clock(self) -> None:
        self.clock.setText(QDateTime.currentDateTime().toString("dd.MM.yyyy   HH:mm:ss"))

    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
            self.fullscreen_button.setText("⛶")
        else:
            self.showFullScreen()
            self.fullscreen_button.setText("×")

    def keyPressEvent(self, event) -> None:
        if self.camera_viewer.isVisible():
            if event.key() == Qt.Key.Key_Left:
                self.step_camera_viewer(-1)
            elif event.key() == Qt.Key.Key_Right:
                self.step_camera_viewer(1)
            elif event.key() == Qt.Key.Key_Escape:
                self.close_camera_viewer()
            else:
                super().keyPressEvent(event)
                return
            event.accept()
            return
        if event.key() == Qt.Key.Key_F11:
            self.toggle_fullscreen()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape and self.isFullScreen():
            self.toggle_fullscreen()
            event.accept()
            return
        super().keyPressEvent(event)

    def _configured_sources(self) -> list[int | str]:
        sources: list[int | str] = []
        for raw in self.config.video.sources or ["0"]:
            text = str(raw).strip()
            if not text:
                continue
            try:
                source: int | str = int(text)
            except ValueError:
                source = text
            if source not in sources:
                sources.append(source)
        return sources or [0]

    def _clear_camera_tiles(self) -> None:
        for tile in self.camera_tiles.values():
            self.video_grid.removeWidget(tile)
            tile.deleteLater()
        self.camera_tiles.clear()
        self.camera_names.clear()

    def _prepare_camera_tiles(self) -> None:
        self._clear_camera_tiles()
        for index in range(16):
            self.video_grid.setColumnStretch(index, 0)
            self.video_grid.setRowStretch(index, 0)
        sources = self._configured_sources()
        columns = max(1, math.ceil(math.sqrt(len(sources))))
        for index, source in enumerate(sources):
            camera_id = camera_id_for_source(source)
            source_text = str(source)
            title = self.config.video.names_by_source.get(source_text, f"Камера {index + 1}")
            tile = CameraTile(title)
            tile.mode_button.setChecked(self.all_mode_toggle.isChecked())
            tile.mode_button.toggled.connect(self._sync_global_mode_control)
            self.camera_tiles[camera_id] = tile
            self.camera_names[camera_id] = title
            self.video_grid.addWidget(tile, index // columns, index % columns)
        for column in range(columns):
            self.video_grid.setColumnStretch(column, 1)
        rows = math.ceil(len(sources) / columns)
        for row in range(rows):
            self.video_grid.setRowStretch(row, 1)

    def set_all_camera_modes(self, enabled: bool) -> None:
        for tile in self.camera_tiles.values():
            tile.mode_button.setChecked(enabled)
        self._refresh_camera_viewer()

    def _sync_global_mode_control(self) -> None:
        states = [tile.mode_button.isChecked() for tile in self.camera_tiles.values()]
        if states and all(state == states[0] for state in states):
            self.all_mode_toggle.blockSignals(True)
            self.all_mode_toggle.setChecked(states[0])
            self.all_mode_toggle.blockSignals(False)
        self._refresh_camera_viewer()

    def open_camera_viewer(self) -> None:
        if not self.camera_tiles:
            return
        self.viewer_camera_index = 0
        self.camera_viewer.setGeometry(self.centralWidget().rect())
        self.camera_viewer.show()
        self.camera_viewer.raise_()
        self._refresh_camera_viewer()
        self.camera_viewer.setFocus(Qt.FocusReason.OtherFocusReason)

    def close_camera_viewer(self) -> None:
        self.camera_viewer.hide()
        self.setFocus(Qt.FocusReason.OtherFocusReason)

    def step_camera_viewer(self, direction: int) -> None:
        camera_count = len(self.camera_tiles)
        if not self.camera_viewer.isVisible() or camera_count == 0:
            return
        self.viewer_camera_index = (self.viewer_camera_index + direction) % camera_count
        self._refresh_camera_viewer()

    def _refresh_camera_viewer(self) -> None:
        if not hasattr(self, "camera_viewer") or not self.camera_viewer.isVisible():
            return
        camera_ids = list(self.camera_tiles)
        if not camera_ids:
            self.close_camera_viewer()
            return
        self.viewer_camera_index %= len(camera_ids)
        camera_id = camera_ids[self.viewer_camera_index]
        tile = self.camera_tiles[camera_id]
        self.camera_viewer.show_camera(
            self.camera_names.get(camera_id, tile.title.text()),
            self.viewer_camera_index,
            len(camera_ids),
            tile.current_frame(),
        )

    def toggle_all_view(self) -> None:
        """Compatibility entry point for the former broken mosaic action."""
        self.open_camera_viewer()

    def toggle_monitoring(self) -> None:
        if self.monitoring_active:
            self.stop_monitoring()
        else:
            self.start_monitoring()

    def start_monitoring(self) -> None:
        if not self.stop_all():
            return
        self._prepare_camera_tiles()
        self._start_audio_workers()
        for source in self._configured_sources():
            camera_id = camera_id_for_source(source)
            worker = VideoSystemWorker(
                self.bridge,
                self.event_bus,
                self.config,
                source,
                self.frame_store,
                camera_id,
            )
            self.video_workers[camera_id] = worker
            worker.start()
        self.monitoring_active = True
        self.control_button.setText("ОСТАНОВИТЬ")
        self.control_button.setStyleSheet(
            f"background:{ALARM_COLOR}; color:#fff; border:0; border-radius:8px;"
            "font-size:14px; font-weight:700; padding:12px 22px;"
        )

    @staticmethod
    def _microphone_key(device) -> str:
        return "default" if device is None else f"device:{device}"

    def _start_audio_workers(self) -> None:
        gunshot_devices = []
        bindings = self.config.video.microphone_by_source
        if bindings:
            for source in self.config.video.sources:
                source_text = str(source)
                if source_text not in bindings:
                    continue
                device = bindings[source_text]
                if device is None:
                    continue
                device = None if device == "__default__" else device
                if device not in gunshot_devices:
                    gunshot_devices.append(device)
        else:
            # Backward-compatible default for configurations created before camera bindings.
            gunshot_devices.append(None)

        speech_device = self.config.speech.microphone_device
        roles: dict[str, dict] = {}
        for device in gunshot_devices:
            roles[self._microphone_key(device)] = {
                "device": device, "gunshot": True, "speech": False, "camera_ids": [],
            }
        for source in self.config.video.sources:
            source_text = str(source)
            if source_text not in bindings or bindings[source_text] is None:
                continue
            device = bindings[source_text]
            device = None if device == "__default__" else device
            role = roles.get(self._microphone_key(device))
            if role is not None:
                role["camera_ids"].append(camera_id_for_source(source))
        if self.config.speech.enabled:
            key = self._microphone_key(speech_device)
            roles.setdefault(
                key, {
                    "device": speech_device,
                    "gunshot": False,
                    "speech": False,
                    "camera_ids": [],
                }
            )["speech"] = True

        for key, role in roles.items():
            worker = AudioSystemWorker(
                self.bridge,
                self.event_bus,
                self.config,
                mic_device=role["device"],
                enable_gunshot=role["gunshot"],
                enable_speech=role["speech"],
                camera_ids=tuple(role["camera_ids"]),
                frame_store=self.frame_store,
            )
            self.audio_workers[key] = worker
            worker.start()

    def stop_monitoring(self) -> None:
        if self.stop_all():
            self.monitoring_active = False
            self.close_camera_viewer()
            self.control_button.setText("ЗАПУСТИТЬ")
            self.control_button.setStyleSheet("")
            for tile in self.camera_tiles.values():
                tile.clear_frame()
                tile.set_state(ComponentState.STOPPED)

    def stop_all(self) -> bool:
        all_stopped = True
        workers = [
            (f"микрофон {key}", worker) for key, worker in self.audio_workers.items()
        ] + [
            (self.camera_names.get(camera_id, camera_id), worker)
            for camera_id, worker in self.video_workers.items()
        ]
        for name, worker in workers:
            if worker and worker.isRunning() and not worker.stop():
                logger.error("Поток %s не завершился за отведённое время", name)
                all_stopped = False
        self.audio_workers = {
            key: worker for key, worker in self.audio_workers.items() if worker.isRunning()
        }
        self.video_workers = {
            camera_id: worker for camera_id, worker in self.video_workers.items() if worker.isRunning()
        }
        return all_stopped

    def open_settings(self) -> None:
        was_active = self.monitoring_active
        # Keep workers running while the dialog is being constructed and shown.
        # Stopping model threads here made the Settings button appear frozen.
        dialog = SettingsDialog(self.config, self)
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        if accepted:
            if was_active:
                self.stop_monitoring()
                if self.monitoring_active:
                    self.bridge.log.emit("error", "Настройки не применены: потоки не остановились")
                    return
            self.alert_service.config = self.config.alerts
            if not was_active:
                self._prepare_camera_tiles()
        if was_active:
            self.start_monitoring()

    def update_video(self, camera_id: str, raw_frame, annotated_frame) -> None:
        tile = self.camera_tiles.get(camera_id)
        if tile is not None:
            tile.set_frames(raw_frame, annotated_frame)
            camera_ids = list(self.camera_tiles)
            if (
                self.camera_viewer.isVisible()
                and camera_ids
                and camera_ids[self.viewer_camera_index % len(camera_ids)] == camera_id
            ):
                self.camera_viewer.set_frame(tile.current_frame())

    def _queue_video_frame(self, camera_id: str, raw_frame, annotated_frame) -> None:
        """Coalesce queued worker frames so the UI renders only the newest."""
        self._pending_video_frames[camera_id] = (raw_frame, annotated_frame)
        if not hasattr(self, "_video_render_timer"):
            self._video_render_timer = QTimer(self)
            self._video_render_timer.setSingleShot(True)
            self._video_render_timer.timeout.connect(self._render_pending_video_frames)
        if not self._video_render_timer.isActive():
            self._video_render_timer.start(0)

    def _render_pending_video_frames(self) -> None:
        frames, self._pending_video_frames = self._pending_video_frames, {}
        for camera_id, (raw_frame, annotated_frame) in frames.items():
            self.update_video(camera_id, raw_frame, annotated_frame)

    def update_video_metrics(self, camera_id: str, metrics: dict) -> None:
        tile = self.camera_tiles.get(camera_id)
        if tile is not None:
            tile.set_metrics(metrics)

    def update_component_status(self, component: str, state_value: str, message: str) -> None:
        state = ComponentState(state_value)
        self.health.update(component, state, message)
        if not component.startswith("video:"):
            return
        camera_id = component.split(":", 1)[1]
        self.video_states[camera_id] = state
        tile = self.camera_tiles.get(camera_id)
        if tile is not None:
            tile.set_state(state, message)
            if state is ComponentState.STOPPED:
                tile.clear_frame()

    def append_log(self, level: str, message: str) -> None:
        getattr(logger, "error" if level == "error" else "warning" if level == "warning" else "info")(
            "%s", message
        )

    def update_level(self, _peak: float) -> None:
        pass

    def _handle_event(self, event: Event) -> None:
        if event.type not in ALERT_EVENT_TYPES:
            return
        item = self.event_store.get_alert(event.id)
        self.alert_queue.append((event, item))
        if self.current_alert is None:
            self._show_next_alert()

    def _show_next_alert(self) -> None:
        if not self.alert_queue:
            self.current_alert = None
            self.alert_overlay.hide()
            return
        event, item = self.alert_queue.popleft()
        self.current_alert = event
        self.close_camera_viewer()
        self.alert_service.replay_local_alarm(event)
        self.alert_overlay.show_incident(event, item)

    def acknowledge_alert(self) -> None:
        self.alert_service.acknowledge_local_alarm()
        self.current_alert = None
        self._show_next_alert()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if hasattr(self, "alert_overlay"):
            self.alert_overlay.setGeometry(self.video_scroll.viewport().rect())
        if hasattr(self, "camera_viewer"):
            self.camera_viewer.setGeometry(self.centralWidget().rect())

    def eventFilter(self, watched, event) -> bool:
        if (
            hasattr(self, "video_scroll")
            and watched is self.video_scroll.viewport()
            and event.type() == QEvent.Type.Resize
            and hasattr(self, "alert_overlay")
        ):
            self.alert_overlay.setGeometry(self.video_scroll.viewport().rect())
        return super().eventFilter(watched, event)

    def closeEvent(self, event) -> None:
        self.alert_service.acknowledge_local_alarm()
        if not self.stop_all():
            event.ignore()
            return
        bus_stopped = self.event_bus.stop(drain=True)
        alerts_stopped = self.alert_service.stop()
        if not bus_stopped or not alerts_stopped:
            logger.error("Не все фоновые сервисы завершились штатно")
        event.accept()

