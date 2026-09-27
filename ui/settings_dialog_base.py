from __future__ import annotations

from copy import deepcopy
import cv2
from pathlib import Path
import re

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QApplication,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from audio.microphone import MicrophoneStream
from core.config import AppConfig, get_resource_path, save_operator_settings
from ui.audio_calibration_dialog import AudioCalibrationDialog, HelpBadge
from vosk import Model

NETWORK_AUDIO_SCHEMES = ("rtsp://", "rtmp://", "http://", "https://")


def is_network_audio_source(value) -> bool:
    return isinstance(value, str) and value.strip().casefold().startswith(
        NETWORK_AUDIO_SCHEMES
    )


def discover_local_cameras(max_index: int = 10) -> list[int]:
    """Probe common OpenCV camera indices without loading the pose model runtime."""
    available: list[int] = []
    for index in range(max(0, max_index)):
        capture = cv2.VideoCapture()
        try:
            if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 350)
            if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 350)
            if capture.open(index, cv2.CAP_ANY):
                ok, _ = capture.read()
                if ok:
                    available.append(index)
        finally:
            capture.release()
    return available


def add_microphones(
    combo: QComboBox, devices: list[tuple[int | str, str]], allow_none: bool
) -> None:
    combo.setEditable(True)
    combo.lineEdit().setPlaceholderText("Системный микрофон или rtsp://…")
    if allow_none:
        combo.addItem("Без микрофона", None)
    combo.addItem("Системный микрофон", "__default__")
    for device_id, name in devices:
        label = f"RTSP: {name}" if is_network_audio_source(device_id) else f"[{device_id}] {name}"
        combo.addItem(label, device_id)


def select_device(combo: QComboBox, device) -> None:
    normalized = "__default__" if device is None else device
    index = combo.findData(normalized)
    if index >= 0:
        combo.setCurrentIndex(index)
    elif is_network_audio_source(device):
        combo.setEditText(str(device).strip())


def selected_microphone_source(combo: QComboBox):
    if (
        combo.currentIndex() >= 0
        and combo.currentText() == combo.itemText(combo.currentIndex())
    ):
        return combo.currentData()
    typed = combo.currentText().strip()
    return typed if typed else None


def selected_device(combo: QComboBox):
    value = selected_microphone_source(combo)
    return None if value == "__default__" else value


class CameraBindingRow(QFrame):
    def __init__(
        self,
        devices: list[tuple[int | str, str]],
        camera_ids: list[int],
        name: str = "",
        source: str = "",
        microphone=None,
        view_mode: str = "side",
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("cameraBinding")
        self.name_edit = QLineEdit(name)
        self.name_edit.setPlaceholderText("Например: Вход")
        self.name_edit.setMinimumWidth(130)
        self.source_combo = QComboBox()
        self.source_combo.setEditable(True)
        self.source_combo.setMinimumWidth(220)
        self.source_combo.setMaximumWidth(380)
        self.source_combo.lineEdit().setPlaceholderText("ID или rtsp://user:password@host/stream")
        for camera_id in camera_ids:
            self.source_combo.addItem(f"Локальная камера {camera_id}", str(camera_id))
        if source:
            index = self.source_combo.findData(str(source))
            if index >= 0:
                self.source_combo.setCurrentIndex(index)
            else:
                self.source_combo.setEditText(source)
        self.microphone_combo = QComboBox()
        self.microphone_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.microphone_combo.setMinimumContentsLength(28)
        self.microphone_combo.setMinimumWidth(280)
        add_microphones(self.microphone_combo, devices, allow_none=True)
        if microphone is None:
            self.microphone_combo.setCurrentIndex(0)
        else:
            select_device(self.microphone_combo, microphone)
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItem("Боковой / обычный ракурс", "side")
        self.view_mode_combo.addItem("Сверху вниз", "top_down")
        mode_index = self.view_mode_combo.findData(view_mode)
        self.view_mode_combo.setCurrentIndex(mode_index if mode_index >= 0 else 0)
        self.remove_button = QPushButton("Удалить")
        self.remove_button.setObjectName("dangerButton")

        layout = QGridLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.addWidget(QLabel("Название камеры"), 0, 0)
        layout.addWidget(QLabel("Источник / RTSP-ссылка"), 0, 1)
        layout.addWidget(QLabel("Ракурс для позы"), 0, 2)
        layout.addWidget(QLabel("Микрофон для выстрела"), 0, 3)
        layout.addWidget(self.name_edit, 1, 0)
        layout.addWidget(self.source_combo, 1, 1)
        layout.addWidget(self.view_mode_combo, 1, 2)
        layout.addWidget(self.microphone_combo, 1, 3)
        layout.addWidget(self.remove_button, 1, 4)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 2)
        layout.setColumnStretch(2, 1)
        layout.setColumnStretch(3, 3)


class SettingsDialog(QDialog):
    """Operator-friendly runtime settings for cameras, microphones and alerts."""

    def __init__(self, app_config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = app_config
        self.microphones = MicrophoneStream.list_input_devices()
        configured_network_sources = {
            value
            for value in [
                *app_config.video.microphone_by_source.values(),
                app_config.speech.microphone_device,
            ]
            if is_network_audio_source(value)
        }
        self.microphones.extend(
            (source, source) for source in sorted(configured_network_sources)
        )
        self._microphone_profiles = deepcopy(app_config.audio.microphone_profiles)
        self._calibration_duration_sec = app_config.audio.calibration_duration_sec
        # Do not probe physical camera devices while opening the modal dialog.
        # OpenCV probing can block for several seconds per unavailable device.
        self.camera_ids = sorted({
            int(str(source).strip())
            for source in (app_config.video.sources or [])
            if str(source).strip().isdigit()
        })
        self._keyword_model: Model | None = None
        self.camera_rows: list[CameraBindingRow] = []
        self.setWindowTitle("Настройки системы")
        self.setMinimumSize(920, 650)
        self.setModal(True)
        stylesheet = (
            """
            QDialog, QWidget { background:#202235; color:#f4f6ff; font-family:'Segoe UI'; }
            QLabel, QCheckBox, QGroupBox { color:#f4f6ff; }
            QTabWidget::pane { border:1px solid #3d415d; border-radius:8px; }
            QGroupBox { border:1px solid #3d415d; border-radius:8px; margin-top:18px; padding-top:14px; }
            QGroupBox::title { subcontrol-origin:margin; subcontrol-position:top left;
                               left:12px; padding:0 7px; background:#202235; color:#f4f6ff; }
            QTabBar::tab { background:#30334b; color:#e9ecf8; padding:11px 20px; margin-right:2px; }
            QTabBar::tab:selected { background:#505675; color:#ffffff; }
            QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {
                background:#171927; color:#ffffff; selection-background-color:#62698f;
                border:1px solid #4b506f; border-radius:6px; padding:8px;
            }
            QDoubleSpinBox, QSpinBox { padding-right:26px; }
            QDoubleSpinBox::up-button, QSpinBox::up-button {
                subcontrol-origin:padding; subcontrol-position:top right;
                width:20px; margin:1px 1px 0 0; border-left:1px solid #4b506f;
                border-bottom:1px solid #353a55; border-top-right-radius:4px;
                background:#30334b;
            }
            QDoubleSpinBox::down-button, QSpinBox::down-button {
                subcontrol-origin:padding; subcontrol-position:bottom right;
                width:20px; margin:0 1px 1px 0; border-left:1px solid #4b506f;
                border-top:1px solid #353a55; border-bottom-right-radius:4px;
                background:#30334b;
            }
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover,
            QSpinBox::up-button:hover, QSpinBox::down-button:hover { background:#505675; }
            QDoubleSpinBox::up-arrow, QSpinBox::up-arrow {
                image:url(__SPIN_UP__); width:9px; height:6px;
            }
            QDoubleSpinBox::down-arrow, QSpinBox::down-arrow {
                image:url(__SPIN_DOWN__); width:9px; height:6px;
            }
            QComboBox { padding-right:30px; }
            QComboBox::drop-down {
                subcontrol-origin:padding; subcontrol-position:center right;
                width:24px; margin:1px 1px 1px 0;
                border-left:1px solid #4b506f;
                border-top-right-radius:4px; border-bottom-right-radius:4px;
                background:#30334b;
            }
            QComboBox::drop-down:hover { background:#505675; }
            QComboBox::down-arrow {
                image:url(__SPIN_DOWN__); width:10px; height:7px;
            }
            QComboBox QAbstractItemView {
                background:#171927; color:#ffffff; selection-background-color:#505675;
                border:1px solid #4b506f; outline:0;
            }
            QPlainTextEdit { background:#10121b; color:#e7eaf5; border:1px solid #4b506f;
                border-radius:6px; padding:8px; font-family:'Consolas'; font-size:12px; }
            QToolTip { background:#171927; color:#ffffff; border:1px solid #62698f; }
            QFrame#cameraBinding { background:#292c42; border:1px solid #454a68; border-radius:8px; }
            QFrame#cameraBinding QLabel { background:transparent; color:#f4f6ff; border:0; }
            QPushButton { background:#505675; color:#ffffff; border:0; border-radius:6px;
                          padding:9px 16px; font-weight:600; }
            QPushButton:hover { background:#62698f; }
            QPushButton#addButton { background:#a7dc00; color:#202235; }
            QPushButton#dangerButton { background:#793b48; color:#ffffff; }
            QPushButton#dangerButton:hover { background:#9c4656; }
            QSlider::groove:horizontal { background:#454a68; height:8px; border-radius:4px; }
            QSlider::sub-page:horizontal { background:#a7dc00; border-radius:4px; }
            QSlider::handle:horizontal { background:#ffffff; width:20px; margin:-6px 0; border-radius:10px; }
            QScrollArea { border:0; background:#202235; }
            """
        )
        self.setStyleSheet(
            stylesheet.replace(
                "__SPIN_UP__", Path(get_resource_path("ui/assets/spin_up.svg")).as_posix()
            ).replace(
                "__SPIN_DOWN__", Path(get_resource_path("ui/assets/spin_down.svg")).as_posix()
            )
        )

        tabs = QTabWidget()
        tabs.addTab(self._camera_tab(), "Камеры и микрофоны")
        tabs.addTab(self._audio_tab(), "Детекторы")
        tabs.addTab(self._alerts_tab(), "Тревоги и REST")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("Применить")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
        buttons.accepted.connect(self._apply_and_accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(tabs)
        layout.addWidget(buttons)

    def _camera_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        intro = QLabel(
            "Добавьте локальную камеру или RTSP-поток и выберите микрофон рядом с ней. "
            "В поле микрофона можно выбрать системное устройство или вставить RTSP-ссылку. "
            "Все назначенные микрофоны будут параллельно анализироваться на выстрел. "
            "Для камеры, смотрящей почти вертикально вниз, выберите «Сверху вниз»: "
            "тогда лежащая поза определяется по поворотно-инвариантной геометрии тела."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#c7cbdb; padding:4px 0 8px 0;")
        layout.addWidget(intro)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.camera_rows_widget = QWidget()
        self.camera_rows_layout = QVBoxLayout(self.camera_rows_widget)
        self.camera_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.camera_rows_layout.setSpacing(10)
        self.camera_rows_layout.addStretch()
        scroll.setWidget(self.camera_rows_widget)
        layout.addWidget(scroll, 1)

        add_button = QPushButton("＋ Добавить камеру")
        add_button.setObjectName("addButton")
        add_button.clicked.connect(lambda: self.add_camera_row(empty=True))
        layout.addWidget(add_button, 0, Qt.AlignmentFlag.AlignLeft)

        for index, source in enumerate(self.config.video.sources or ["0"]):
            source_text = str(source)
            microphone = self.config.video.microphone_by_source.get(source_text)
            self.add_camera_row(
                self.config.video.names_by_source.get(source_text, f"Камера {index + 1}"),
                source_text,
                microphone,
                self.config.video.view_mode_by_source.get(source_text, "side"),
            )
        return page

    def add_camera_row(
        self,
        name: str = "",
        source: str = "",
        microphone=None,
        view_mode: str = "side",
        *,
        empty: bool = False,
    ) -> CameraBindingRow:
        row = CameraBindingRow(self.microphones, self.camera_ids, name, source, microphone, view_mode)
        if empty:
            row.name_edit.clear()
            row.source_combo.setCurrentIndex(-1)
            row.source_combo.setEditText("")
            row.microphone_combo.setCurrentIndex(-1)
            row.microphone_combo.setEditText("")
        row.remove_button.clicked.connect(lambda: self.remove_camera_row(row))
        self.camera_rows.append(row)
        self.camera_rows_layout.insertWidget(self.camera_rows_layout.count() - 1, row)
        return row

    def remove_camera_row(self, row: CameraBindingRow) -> None:
        if len(self.camera_rows) <= 1:
            row.source_combo.setEditText("")
            row.name_edit.clear()
            row.microphone_combo.setCurrentIndex(0)
            return
        self.camera_rows.remove(row)
        row.deleteLater()

    def _audio_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        loudness = QGroupBox("Детекция выстрела")
        loudness_layout = QVBoxLayout(loudness)
        row = QHBoxLayout()
        threshold_label = QLabel("Общий порог для микрофонов без отдельного профиля")
        row.addWidget(threshold_label)
        threshold_help = HelpBadge(
            "Общий порог для микрофонов без отдельного профиля. Ближе к 0 dBFS "
            "строже и меньше ложных запусков; ниже — лучше слышны дальние "
            "импульсы. Для каждого микрофона лучше создать отдельный профиль."
        )
        threshold_help.setObjectName("gunshotThresholdHelp")
        row.addWidget(threshold_help)
        row.addStretch()
        self.db_value_label = QLabel()
        self.db_value_label.setStyleSheet("color:#a7dc00; font-size:18px; font-weight:700;")
        row.addWidget(self.db_value_label)
        loudness_layout.addLayout(row)
        self.db_slider = QSlider(Qt.Orientation.Horizontal)
        self.db_slider.setRange(-60, -5)
        self.db_slider.setValue(round(self.config.audio.trigger_dbfs))
        self.db_slider.valueChanged.connect(self._update_db_label)
        loudness_layout.addWidget(self.db_slider)
        hint = QLabel("Для точной настройки конкретного помещения используйте профиль и замер фона.")
        hint.setStyleSheet("color:#b8bdcf;")
        loudness_layout.addWidget(hint)
        calibration_button = QPushButton("Профили микрофонов и автоматическая калибровка")
        calibration_button.setObjectName("addButton")
        calibration_button.clicked.connect(self._open_audio_calibration)
        loudness_layout.addWidget(calibration_button, 0, Qt.AlignmentFlag.AlignLeft)
        self._update_db_label(self.db_slider.value())
        layout.addWidget(loudness)

        speech = QGroupBox("Детекция ключевого слова")
        speech_form = QFormLayout(speech)
        self.speech_enabled = QCheckBox("Включить распознавание тревожных слов")
        self.speech_enabled.setChecked(self.config.speech.enabled)
        self.speech_microphone_combo = QComboBox()
        add_microphones(self.speech_microphone_combo, self.microphones, allow_none=False)
        select_device(self.speech_microphone_combo, self.config.speech.microphone_device)
        speech_form.addRow("", self.speech_enabled)
        speech_form.addRow("Единственный речевой микрофон", self.speech_microphone_combo)
        self.keywords_list = QListWidget()
        self.keywords_list.addItems(self.config.speech.keywords)
        self.keywords_list.setMinimumHeight(120)
        self.keyword_edit = QLineEdit()
        self.keyword_edit.setPlaceholderText("Новое ключевое слово")
        add_keyword = QPushButton("Добавить")
        remove_keyword = QPushButton("Удалить выбранное")
        add_keyword.clicked.connect(self._add_keyword)
        remove_keyword.clicked.connect(self._remove_keyword)
        keyword_buttons = QHBoxLayout()
        keyword_buttons.addWidget(self.keyword_edit, 1)
        keyword_buttons.addWidget(add_keyword)
        keyword_buttons.addWidget(remove_keyword)
        keywords_widget = QWidget()
        keywords_layout = QVBoxLayout(keywords_widget)
        keywords_layout.setContentsMargins(0, 0, 0, 0)
        keywords_layout.addWidget(self.keywords_list)
        keywords_layout.addLayout(keyword_buttons)
        speech_form.addRow("Тревожные слова", keywords_widget)
        layout.addWidget(speech)

        fall = QGroupBox("Детекция падения")
        fall_form = QFormLayout(fall)
        self.fall_duration_spin = QDoubleSpinBox()
        self.fall_duration_spin.setRange(0.5, 60.0)
        self.fall_duration_spin.setSuffix(" сек")
        self.fall_duration_spin.setValue(self.config.video.fall_duration_sec)
        fall_form.addRow("Поза считается тревожной через", self.fall_duration_spin)
        layout.addWidget(fall)
        layout.addStretch()
        return page

    def _open_audio_calibration(self) -> None:
        original_profiles = self.config.audio.microphone_profiles
        original_duration = self.config.audio.calibration_duration_sec
        self.config.audio.microphone_profiles = deepcopy(self._microphone_profiles)
        self.config.audio.calibration_duration_sec = self._calibration_duration_sec
        initial_device = next(
            (
                selected_microphone_source(row.microphone_combo)
                for row in self.camera_rows
                if selected_microphone_source(row.microphone_combo) is not None
            ),
            selected_microphone_source(self.speech_microphone_combo),
        )
        calibration_devices = list(self.microphones)
        configured_ids = {device_id for device_id, _ in calibration_devices}
        for combo in [
            *(row.microphone_combo for row in self.camera_rows),
            self.speech_microphone_combo,
        ]:
            source = selected_microphone_source(combo)
            if is_network_audio_source(source) and source not in configured_ids:
                calibration_devices.append((source, source))
                configured_ids.add(source)
        dialog = AudioCalibrationDialog(
            self.config.audio,
            calibration_devices,
            self,
            initial_device=initial_device,
        )
        self.config.audio.microphone_profiles = original_profiles
        self.config.audio.calibration_duration_sec = original_duration
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._microphone_profiles = dialog.profiles
            self._calibration_duration_sec = dialog.calibration_duration_sec
            default_profile = dialog.profiles.get("__default__", {})
            if "trigger_dbfs" in default_profile:
                self.db_slider.setValue(round(float(default_profile["trigger_dbfs"])))

    def _update_db_label(self, value: int) -> None:
        self.db_value_label.setText(f"{value} dBFS")

    def _add_keyword(self) -> None:
        keyword = self.keyword_edit.text().casefold().strip()
        existing = {
            self.keywords_list.item(index).text().casefold()
            for index in range(self.keywords_list.count())
        }
        if keyword and keyword not in existing:
            exists_in_model = self._keyword_exists(keyword)
            if exists_in_model is None:
                return
            if not exists_in_model:
                QMessageBox.warning(
                    self,
                    "Слово отсутствует в модели",
                    f"Слово «{keyword}» отсутствует в словаре речевой модели и не может быть назначено ключевым.",
                )
                return
        if keyword and keyword not in existing:
            self.keywords_list.addItem(keyword)
            self.keyword_edit.clear()

    def _keyword_exists(self, keyword: str) -> bool | None:
        try:
            if self._keyword_model is None:
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                from core.vosk_model_path import vosk_runtime_path

                self._keyword_model = Model(
                    vosk_runtime_path(self.config.speech.vosk_model_path)
                )
            words = re.findall(r"[\w-]+", keyword.casefold(), flags=re.UNICODE)
            return bool(words) and all(
                self._keyword_model.vosk_model_find_word(word) >= 0 for word in words
            )
        except Exception as exc:
            QMessageBox.critical(self, "Не удалось проверить словарь", str(exc))
            return None
        finally:
            QApplication.restoreOverrideCursor()

    def _remove_keyword(self) -> None:
        for item in self.keywords_list.selectedItems():
            self.keywords_list.takeItem(self.keywords_list.row(item))

    def _alerts_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        delivery = QGroupBox("Доставка тревог")
        form = QFormLayout(delivery)
        self.webhook_edit = QLineEdit(self.config.alerts.webhook_url)
        self.webhook_edit.setPlaceholderText("http://IP:8080/Trigger")
        self.local_sound_check = QCheckBox("Включить сирену при тревоге")
        self.local_sound_check.setChecked(self.config.alerts.local_sound_enabled)
        self.snapshot_age_spin = QDoubleSpinBox()
        self.snapshot_age_spin.setRange(0.5, 60.0)
        self.snapshot_age_spin.setSuffix(" сек")
        self.snapshot_age_spin.setValue(self.config.alerts.snapshot_max_age_sec)
        form.addRow("REST endpoint", self.webhook_edit)
        form.addRow("", self.local_sound_check)
        form.addRow("Актуальность кадра", self.snapshot_age_spin)
        layout.addWidget(delivery)

        rest_log = QGroupBox("Журнал REST-доставки")
        rest_log_layout = QVBoxLayout(rest_log)
        log_hint = QLabel(
            f"Технические строки REST/Webhook из файла: {self.config.paths.log_file}"
        )
        log_hint.setWordWrap(True)
        log_hint.setStyleSheet("color:#b8bdcf; background:transparent;")
        rest_log_layout.addWidget(log_hint)
        self.rest_log_view = QPlainTextEdit()
        self.rest_log_view.setReadOnly(True)
        self.rest_log_view.document().setMaximumBlockCount(1_000)
        self.rest_log_view.setMinimumHeight(260)
        rest_log_layout.addWidget(self.rest_log_view)
        refresh_logs = QPushButton("Обновить журнал")
        refresh_logs.clicked.connect(self._load_rest_logs)
        rest_log_layout.addWidget(refresh_logs, 0, Qt.AlignmentFlag.AlignRight)
        layout.addWidget(rest_log, 1)
        self._load_rest_logs()
        return page

    def _load_rest_logs(self) -> None:
        if not hasattr(self, "rest_log_view"):
            return
        log_path = Path(self.config.paths.log_file)
        try:
            if not log_path.is_file():
                text = "Файл журнала ещё не создан."
            else:
                with log_path.open("rb") as handle:
                    handle.seek(0, 2)
                    size = handle.tell()
                    handle.seek(max(0, size - 1024 * 1024))
                    raw = handle.read().decode("utf-8", errors="replace")
                lines = raw.splitlines()
                if size > 1024 * 1024 and lines:
                    lines = lines[1:]
                markers = ("rest", "webhook")
                filtered = [
                    line for line in lines
                    if any(marker in line.casefold() for marker in markers)
                ][-1_000:]
                text = "\n".join(filtered) if filtered else "В журнале пока нет записей REST/Webhook."
        except OSError as exc:
            text = f"Не удалось прочитать журнал: {exc}"
        self.rest_log_view.setPlainText(text)
        scrollbar = self.rest_log_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _apply_and_accept(self) -> None:
        sources: list[str] = []
        names: dict[str, str] = {}
        bindings: dict[str, int | str | None] = {}
        view_modes: dict[str, str] = {}
        for index, row in enumerate(self.camera_rows):
            source = row.source_combo.currentData()
            typed = row.source_combo.currentText().strip()
            if source is None or not typed.startswith("Локальная камера "):
                source = typed
            source = str(source).strip()
            if not source or source in sources:
                continue
            sources.append(source)
            names[source] = row.name_edit.text().strip() or f"Камера {index + 1}"
            microphone_source = selected_microphone_source(row.microphone_combo)
            if isinstance(microphone_source, str) and microphone_source not in {"__default__"}:
                if not is_network_audio_source(microphone_source):
                    QMessageBox.warning(
                        self,
                        "Некорректный источник микрофона",
                        "Сетевой микрофон должен начинаться с rtsp://, rtmp://, http:// или https://.",
                    )
                    return
            bindings[source] = microphone_source
            view_modes[source] = str(row.view_mode_combo.currentData() or "side")
        self.config.video.sources = sources or ["0"]
        self.config.video.names_by_source = names
        self.config.video.microphone_by_source = bindings
        self.config.video.view_mode_by_source = view_modes
        self.config.audio.trigger_dbfs = float(self.db_slider.value())
        self.config.audio.microphone_profiles = deepcopy(self._microphone_profiles)
        self.config.audio.calibration_duration_sec = self._calibration_duration_sec
        self.config.speech.enabled = self.speech_enabled.isChecked()
        speech_source = selected_device(self.speech_microphone_combo)
        if isinstance(speech_source, str) and not is_network_audio_source(speech_source):
            QMessageBox.warning(
                self,
                "Некорректный речевой микрофон",
                "Сетевой речевой микрофон должен начинаться с rtsp://, rtmp://, http:// или https://.",
            )
            return
        self.config.speech.microphone_device = speech_source
        self.config.speech.keywords = [
            self.keywords_list.item(index).text().strip()
            for index in range(self.keywords_list.count())
            if self.keywords_list.item(index).text().strip()
        ] or ["помоги"]
        self.config.video.fall_duration_sec = self.fall_duration_spin.value()
        self.config.alerts.webhook_url = self.webhook_edit.text().strip()
        self.config.alerts.local_sound_enabled = self.local_sound_check.isChecked()
        self.config.alerts.snapshot_max_age_sec = self.snapshot_age_spin.value()
        try:
            save_operator_settings(self.config)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Не удалось сохранить настройки", str(exc))
            return
        self.accept()
