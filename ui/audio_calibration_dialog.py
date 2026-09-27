from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QScrollArea,
    QSpinBox,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from audio.calibration import (
    PROFILE_PRESETS,
    RUNTIME_FIELDS,
    AmbientCalibration,
    analyze_reference_audio,
    ReferenceCalibration,
    microphone_profile_key,
    resolve_audio_profile,
)
from ui.workers import (
    AmbientCalibrationWorker,
    CalibrationCaptureWorker,
)


def _compact_help(text: str) -> str:
    """Keep hover help useful without turning it into a large floating panel."""
    compact = " ".join(str(text).split())
    sentences = compact.split(". ")
    return ". ".join(sentences[:2]).strip()[:220]
class HelpBadge(QLabel):
    """A small, immediate hover target for one concise setting hint."""
    def __init__(self, help_text: str, parent=None):
        super().__init__("?", parent)
        self.setObjectName("helpBadge")
        self.setFixedSize(16, 16)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.WhatsThisCursor)
        self.setToolTip(_compact_help(help_text))
        self.setStyleSheet(
            "color:#a7dc00; border:1px solid #a7dc00; border-radius:8px; "
            "background:#292d46; font-weight:800; font-size:11px;"
        )
    def enterEvent(self, event) -> None:
        QToolTip.showText(
            self.mapToGlobal(self.rect().bottomLeft()), self.toolTip(), self
        )
        super().enterEvent(event)
    def leaveEvent(self, event) -> None:
        QToolTip.hideText()
        super().leaveEvent(event)


class _ScrollOnlyWheelFilter(QObject):
    """Route wheel input to the calibration scroll area only."""

    def __init__(self, scroll_area: QScrollArea, parent=None):
        super().__init__(parent)
        self.scroll_area = scroll_area

    def eventFilter(self, watched, event):
        del watched
        if event.type() != QEvent.Type.Wheel:
            return False
        bar = self.scroll_area.verticalScrollBar()
        delta = event.pixelDelta().y()
        if not delta:
            delta = (event.angleDelta().y() // 120) * max(bar.singleStep() * 3, 1)
        if delta:
            bar.setValue(bar.value() - delta)
        # Never let a wheel event reach spin boxes, sliders, combos, etc.
        return True

class AudioCalibrationDialog(QDialog):
    """Per-microphone ambient calibration and expert gunshot thresholds."""

    def __init__(
        self,
        audio_config,
        devices: list[tuple[int, str]],
        parent=None,
        initial_device: int | str | None = None,
    ):
        super().__init__(parent)
        self.audio_config = audio_config
        self.profiles: dict[str, dict[str, Any]] = deepcopy(audio_config.microphone_profiles)
        self.calibration_duration_sec = float(audio_config.calibration_duration_sec)
        self._current_key = "__default__"
        self._loading = False
        self.worker: AmbientCalibrationWorker | None = None
        self.manual_capture_worker: CalibrationCaptureWorker | None = None
        self._manual_capture_target: str | None = None
        self._manual_ambient_chunks = None
        self._manual_reference_takes: dict[str, list] = {}
        self._manual_last_apply_ok = False
        self._manual_last_apply_error = ""
        self.help_badges: list[QLabel] = []

        self.setWindowTitle("Калибровка микрофона для детектора выстрела")
        self.setMinimumSize(780, 720)
        self.setModal(True)
        self.setStyleSheet("""
            QDialog { background: #202235; color: #f4f6ff; }
            QGroupBox { border: 1px solid #3d415d; border-radius: 10px; margin-top: 18px; padding: 18px 12px 12px 12px; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #f4f6ff; font-weight: 700; }
            QScrollArea { border: 0; background: transparent; }
            QComboBox, QSpinBox, QDoubleSpinBox { min-height: 28px; padding: 2px 6px; border: 1px solid #4b5272; border-radius: 6px; background: #292d46; color: #f4f6ff; }
            QSlider::groove:horizontal { height: 6px; background: #363b58; border-radius: 3px; }
            QSlider::handle:horizontal { width: 16px; margin: -5px 0; border-radius: 8px; background: #a7dc00; }
            QProgressBar { height: 8px; border: 0; border-radius: 4px; background: #303550; text-align: center; }
            QProgressBar::chunk { border-radius: 4px; background: #a7dc00; }
            QPushButton { min-height: 30px; padding: 4px 12px; }
            QPushButton#addButton { background: #a7dc00; color: #202235; border: 0; border-radius: 6px; font-weight: 700; }
            QPushButton#addButton:hover { background: #c4f23d; }
            QPushButton#secondaryButton { background: #2a3045; color: #a7dc00; border: 1px solid #a7dc00; border-radius: 6px; font-weight: 700; }
            QPushButton#secondaryButton:hover { background: #35402b; }
            QPushButton#distanceButton { min-height: 44px; background: #a7dc00; color: #202235; border: 0; border-radius: 8px; font-weight: 700; }
            QPushButton#distanceButton:hover { background: #c4f23d; }
            QPushButton#distanceButton:disabled { background: #3a4054; color: #8c92a6; }
            QLabel#dialogTitle { color: #ffffff; font-size: 20px; font-weight: 700; }
            QLabel#sectionHint { color: #b8bdcf; padding: 2px 0 6px 0; }
            QLabel#helpBadge {
                min-width: 18px; max-width: 18px;
                min-height: 18px; max-height: 18px;
                border: 1px solid #a7dc00; border-radius: 9px;
                color: #a7dc00; background: #292d46;
                font-weight: 800;
            }
            QLabel#helpBadge:hover { color: #202235; background: #a7dc00; }
        """)

        title = QLabel("Настройка микрофона и моделей")
        title.setObjectName("dialogTitle")
        intro = QLabel(
            "Профиль хранится отдельно для каждого микрофона. Изменения применяются только к выбранному профилю; "
            "сырые записи не сохраняются."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#c7cbdb; padding:4px 0 10px 0;")

        profile_group = QGroupBox("1. Микрофон и тип помещения")
        profile_form = QFormLayout(profile_group)
        profile_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        profile_form.setHorizontalSpacing(16)
        profile_form.setVerticalSpacing(8)
        profile_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.microphone_combo = QComboBox()
        self.microphone_combo.addItem("Системный микрофон", "__default__")
        for device_id, name in devices:
            self.microphone_combo.addItem(f"[{device_id}] {name}", device_id)
        self.preset_combo = QComboBox()
        for key, values in PROFILE_PRESETS.items():
            self.preset_combo.addItem(str(values["label"]), key)
        profile_form.addRow(
            self._setting_label(
                "Микрофон",
                "Выбирает устройство и его отдельный профиль. После смены микрофона "
                "повторите замер фона и контрольные дистанции: усиление и шум у каждого "
                "устройства отличаются.",
            ),
            self.microphone_combo,
        )
        profile_form.addRow(
            self._setting_label(
                "Тип помещения",
                "Начальная строгость. «Шумное» повышает защиту от ложных тревог, "
                "«Чувствительное» лучше слышит дальние тихие импульсы. После хорошей "
                "калибровки обычно оставляйте автоматически полученный профиль.",
            ),
            self.preset_combo,
        )
        self.microphone_combo.setToolTip("Профиль сохраняется отдельно для каждого выбранного устройства.")
        self.preset_combo.setToolTip("Выберите базовый уровень строгости. Для точной настройки используйте экспериментальную запись дистанций ниже.")

        thresholds = QGroupBox("2. Ручная корректировка порогов")
        thresholds_form = QFormLayout(thresholds)
        thresholds_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        thresholds_form.setHorizontalSpacing(16)
        thresholds_form.setVerticalSpacing(8)
        thresholds_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        trigger_row = QHBoxLayout()
        self.trigger_slider = QSlider(Qt.Orientation.Horizontal)
        self.trigger_slider.setRange(-60, -5)
        self.trigger_label = QLabel()
        self.trigger_label.setMinimumWidth(86)
        self.trigger_label.setStyleSheet("color:#a7dc00; font-weight:700;")
        trigger_row.addWidget(self.trigger_slider, 1)
        trigger_row.addWidget(self.trigger_label)
        thresholds_form.addRow(
            self._setting_label(
                "Пиковая громкость запуска",
                "Самый тихий пик, который запускает модели. Сдвиг к 0 dBFS делает "
                "фильтр строже и убирает тихие щелчки; более отрицательное значение "
                "лучше слышит дальние выстрелы, но чаще анализирует помехи. Сначала "
                "используйте замер фона, затем меняйте по 2–3 dB.",
            ),
            trigger_row,
        )
        self.cnn_threshold = self._double_spin(0.05, 0.80, 0.01, 3)
        self.panns_margin = self._double_spin(0.0, 0.60, 0.01, 3)
        self.snr_spin = self._double_spin(0.0, 24.0, 0.5, 1, " dB")
        self.crest_spin = self._double_spin(1.0, 6.0, 0.1, 2)
        self.yamnet_spin = self._double_spin(0.0, 1.0, 0.01, 2)
        thresholds_form.addRow(
            self._setting_label(
                "Порог PANNs — выстрел",
                "Минимальная уверенность основной аудиомодели в выстреле. Выше — "
                "меньше хлопков и щелчков, но больше риск пропустить дальний звук. "
                "Ниже — чувствительнее. Если PANNs в журнале часто принимает помеху "
                "за firearm, повышайте на 0,03–0,05.",
            ),
            self.cnn_threshold,
        )
        thresholds_form.addRow(
            self._setting_label(
                "Отрыв PANNs от помех",
                "Насколько оценка выстрела должна быть выше оценки хлопка, щелчка, "
                "речи или другого шума. Повышайте, если firearm лишь немного обгоняет "
                "помеху; понижайте только когда реальные дальние выстрелы стабильно "
                "теряются по причине panns_margin.",
            ),
            self.panns_margin,
        )
        thresholds_form.addRow(
            self._setting_label(
                "Минимальный сигнал над фоном",
                "Требуемый запас пика над автоматически измеренным шумом. Больше dB — "
                "строже в шумном помещении; меньше — чувствительнее к дальним звукам. "
                "Если вентилятор или разговор запускает анализ, повышайте на 1–2 dB.",
            ),
            self.snr_spin,
        )
        thresholds_form.addRow(
            self._setting_label(
                "Минимальная импульсность",
                "Отношение короткого пика к средней громкости. Выше отсекает речь, "
                "музыку и ровный гул; ниже помогает при сильно приглушённых выстрелах. "
                "Хлопки тоже импульсные, поэтому этот параметр работает только вместе "
                "с PANNs и YAMNet.",
            ),
            self.crest_spin,
        )
        thresholds_form.addRow(
            self._setting_label(
                "Подтверждение YAMNet",
                "Минимальная независимая уверенность второй модели. Повышайте, если "
                "щелчки проходят PANNs и YAMNet даёт им слабый firearm; понижайте "
                "небольшими шагами, если реальный выстрел отклонён как yamnet_threshold. "
                "Сильный класс помехи всё равно имеет приоритет.",
            ),
            self.yamnet_spin,
        )
        self.trigger_slider.setToolTip("Минимальный пик сигнала для запуска анализа. Более высокое значение уменьшает ложные срабатывания.")
        self.cnn_threshold.setToolTip("Минимальная уверенность PANNs в классе firearm.")
        self.panns_margin.setToolTip("Запас PANNs над наиболее вероятной помехой.")
        self.snr_spin.setToolTip("Минимальное превышение пика над адаптивным шумовым фоном.")
        self.crest_spin.setToolTip("Импульсность сигнала: помогает отсеивать ровные шумы и речь.")
        self.yamnet_spin.setToolTip("Минимальная независимая уверенность YAMNet.")

        calibrate_group = QGroupBox("3. Быстрая калибровка фона")
        calibrate_layout = QVBoxLayout(calibrate_group)
        calibrate_layout.setContentsMargins(12, 10, 12, 10)
        calibrate_layout.setSpacing(6)
        duration_row = QHBoxLayout()
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(3, 60)
        self.duration_spin.setSuffix(" сек")
        self.duration_spin.setValue(round(self.calibration_duration_sec))
        self.calibrate_button = QPushButton("Измерить фон")
        self.calibrate_button.setObjectName("addButton")
        duration_row.addWidget(
            self._setting_label(
                "Длительность",
                "Сколько секунд измерять обычный фон без тестовых импульсов. "
                "10–15 секунд достаточно для тихой комнаты; 20–30 секунд лучше, "
                "если шум меняется. Более долгий замер стабильнее, но не делает "
                "модели чувствительнее сам по себе.",
            )
        )
        duration_row.addWidget(self.duration_spin)
        duration_row.addWidget(self.calibrate_button)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.result_label = QLabel("Калибровка ещё не выполнялась для выбранного профиля.")
        self.result_label.setWordWrap(True)
        self.result_label.setObjectName("sectionHint")
        calibrate_layout.addLayout(duration_row)
        calibrate_layout.addWidget(self.progress)
        calibrate_layout.addWidget(self.result_label)
        manual_group = QGroupBox("4. Ручная запись отдельных дистанций (экспериментальная функция)")
        manual_layout = QVBoxLayout(manual_group)
        manual_layout.setContentsMargins(12, 10, 12, 10)
        manual_layout.setSpacing(6)
        manual_hint = QLabel(
            "Запишите один проверенный тестовый импульс на выбранной дистанции. "
            "После каждой записи профиль обновляется автоматически: 1 запись — черновой расчёт, "
            "3+ — полноценная настройка."
        )
        manual_hint.setWordWrap(True)
        manual_hint.setObjectName("sectionHint")
        manual_controls = QHBoxLayout()
        self.manual_duration_spin = QSpinBox()
        self.manual_duration_spin.setRange(2, 30)
        self.manual_duration_spin.setValue(10)
        self.manual_duration_spin.setSuffix(" сек на запись")
        self.manual_duration_spin.setToolTip("Длина каждого контрольного фрагмента: от 2 до 30 секунд.")
        manual_controls.addWidget(
            self._setting_label(
                "Длина записи",
                "Время одной экспериментальной записи дистанции. Внутри должен быть "
                "один проверенный тестовый импульс и немного обычного фона. "
                "10 секунд обычно достаточно; увеличивайте до 20–30 секунд, если "
                "звук на большой дистанции трудно отделить от меняющегося шума.",
            )
        )
        manual_controls.addWidget(self.manual_duration_spin)
        self.manual_apply_button = QPushButton("Итоговый расчёт по всем")
        self.manual_apply_button.setObjectName("secondaryButton")
        self.manual_apply_button.setToolTip("Пересчитать профиль по всем уже записанным дистанциям.")
        self.manual_apply_button.setEnabled(False)
        manual_controls.addWidget(self.manual_apply_button)
        distance_grid = QGridLayout()
        distance_grid.setHorizontalSpacing(8)
        distance_grid.setVerticalSpacing(8)
        self.manual_distance_buttons: dict[str, QPushButton] = {}
        for index, distance in enumerate(("10–20 см", "50 см", "1 м", "2 м", "5 м")):
            button = QPushButton(f"Записать\n{distance}")
            button.setObjectName("distanceButton")
            button.setToolTip(f"Записать тестовый импульс на дистанции {distance}.")
            button.clicked.connect(lambda _checked=False, value=distance: self._start_manual_capture(value))
            self.manual_distance_buttons[distance] = button
            distance_grid.addWidget(button, 0, index)
            distance_grid.setColumnStretch(index, 1)
        self.manual_status = QLabel("Сначала выполните «Измерить фон», затем выберите дистанцию.")
        self.manual_status.setWordWrap(True)
        self.manual_status.setObjectName("sectionHint")
        manual_layout.addWidget(manual_hint)
        manual_layout.addLayout(manual_controls)
        manual_layout.addLayout(distance_grid)
        manual_layout.addWidget(self.manual_status)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("Сохранить профиль")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
        buttons.accepted.connect(self._save_and_accept)
        buttons.rejected.connect(self.reject)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 12, 16, 16)
        layout.setSpacing(10)
        layout.addWidget(title)
        layout.addWidget(intro)
        layout.addWidget(profile_group)
        layout.addWidget(thresholds)
        layout.addWidget(calibrate_group)
        layout.addWidget(manual_group)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(8, 8, 8, 8)
        outer_layout.setSpacing(8)
        outer_layout.addWidget(scroll, 1)
        outer_layout.addWidget(buttons)

        # All controls keep their value when the pointer is over them;
        # the wheel is exclusively a shortcut for vertical scrolling.
        self._wheel_filter = _ScrollOnlyWheelFilter(scroll, self)
        wheel_targets = [content, scroll, scroll.viewport(), scroll.verticalScrollBar()]
        wheel_targets.extend(content.findChildren(QWidget))
        for target in wheel_targets:
            target.installEventFilter(self._wheel_filter)

        initial_index = self.microphone_combo.findData(initial_device)
        if initial_index >= 0:
            self.microphone_combo.setCurrentIndex(initial_index)

        self.trigger_slider.valueChanged.connect(self._update_trigger_label)
        self.trigger_slider.valueChanged.connect(self._threshold_edited)
        self.cnn_threshold.valueChanged.connect(self._threshold_edited)
        self.panns_margin.valueChanged.connect(self._threshold_edited)
        self.snr_spin.valueChanged.connect(self._threshold_edited)
        self.crest_spin.valueChanged.connect(self._threshold_edited)
        self.yamnet_spin.valueChanged.connect(self._threshold_edited)
        self.microphone_combo.currentIndexChanged.connect(self._microphone_changed)
        self.preset_combo.currentIndexChanged.connect(self._preset_changed)
        self.calibrate_button.clicked.connect(self._start_calibration)
        self.manual_apply_button.clicked.connect(lambda: self._apply_manual_reference_calibration())
        self._load_profile(microphone_profile_key(self.microphone_combo.currentData()))
        self._manual_controls(True)

    def _setting_label(self, text: str, help_text: str) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        compact_help = _compact_help(help_text)
        caption = QLabel(text)
        caption.setToolTip(compact_help)
        badge = HelpBadge(compact_help)
        badge.setAccessibleName(f"Справка: {text}")
        layout.addWidget(badge)
        layout.addWidget(caption)
        self.help_badges.append(badge)
        return container

    @staticmethod
    def _double_spin(
        minimum: float,
        maximum: float,
        step: float,
        decimals: int,
        suffix: str = "",
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setSuffix(suffix)
        return spin

    def _base_settings(self) -> dict[str, Any]:
        return {
            field: getattr(self.audio_config, field)
            for field in RUNTIME_FIELDS
            if hasattr(self.audio_config, field)
        }

    def _resolved_profile(self, key: str) -> tuple[str, dict[str, Any]]:
        # Resolve the same compatibility/migration rules as runtime so
        # the sliders never show stale values from an older calibration.
        config_view = SimpleNamespace(
            **self._base_settings(),
            microphone_profiles=self.profiles,
        )
        values = resolve_audio_profile(config_view, key)
        preset = str(values.pop("preset", "balanced"))
        values.pop("profile_key", None)
        return preset, values

    def _capture_current(self) -> None:
        if self._loading:
            return
        existing = dict(self.profiles.get(self._current_key, {}))
        preset = str(self.preset_combo.currentData() or "custom")
        preset_values = PROFILE_PRESETS.get(preset, {})
        for field, value in preset_values.items():
            if field in RUNTIME_FIELDS:
                existing[field] = value
        existing.update(
            {
                "preset": preset,
                "trigger_dbfs": float(self.trigger_slider.value()),
                "panns_threshold": self.cnn_threshold.value(),
                "panns_margin_threshold": self.panns_margin.value(),
                "min_snr_db": self.snr_spin.value(),
                "min_crest_factor": self.crest_spin.value(),
                "yamnet_threshold": self.yamnet_spin.value(),
            }
        )
        if preset == "custom" and existing.get("calibration_mode") == "ambient_plus_reference":
            # A user edit must win over automatic reference-profile clamps.
            existing["manual_override"] = True
        elif preset != "custom":
            existing.pop("manual_override", None)
        self.profiles[self._current_key] = existing

    def _load_profile(self, key: str) -> None:
        self._loading = True
        try:
            self._current_key = key
            preset, values = self._resolved_profile(key)
            index = self.preset_combo.findData(preset)
            self.preset_combo.setCurrentIndex(index if index >= 0 else self.preset_combo.findData("custom"))
            self.trigger_slider.setValue(round(float(values["trigger_dbfs"])))
            self.cnn_threshold.setValue(float(values["panns_threshold"]))
            self.panns_margin.setValue(float(values["panns_margin_threshold"]))
            self.snr_spin.setValue(float(values["min_snr_db"]))
            self.crest_spin.setValue(float(values["min_crest_factor"]))
            self.yamnet_spin.setValue(float(values["yamnet_threshold"]))
            metadata = self.profiles.get(key, {})
            if "ambient_peak_dbfs" in metadata:
                self.result_label.setText(
                    f"Последний фон: {float(metadata['ambient_peak_dbfs']):.1f} dBFS; "
                    f"порог запуска: {float(values['trigger_dbfs']):.1f} dBFS."
                )
            else:
                self.result_label.setText("Калибровка ещё не выполнялась для выбранного профиля.")
        finally:
            self._loading = False
        self._update_trigger_label(self.trigger_slider.value())

    def _microphone_changed(self) -> None:
        self._capture_current()
        self._reset_manual_recordings()
        self._load_profile(microphone_profile_key(self.microphone_combo.currentData()))
        self._manual_controls(True)

    def _reset_manual_recordings(self) -> None:
        self._manual_ambient_chunks = None
        self._manual_reference_takes.clear()
        self._manual_last_apply_ok = False
        self._manual_last_apply_error = ""
        for distance, button in self.manual_distance_buttons.items():
            button.setText(f"Записать\n{distance}")
        self._update_manual_status()

    def _preset_changed(self) -> None:
        if self._loading:
            return
        preset = str(self.preset_combo.currentData() or "custom")
        values = PROFILE_PRESETS.get(preset, {})
        if preset != "custom":
            self._loading = True
            try:
                self.cnn_threshold.setValue(float(values["panns_threshold"]))
                self.panns_margin.setValue(float(values["panns_margin_threshold"]))
                self.snr_spin.setValue(float(values["min_snr_db"]))
                self.crest_spin.setValue(float(values["min_crest_factor"]))
                self.yamnet_spin.setValue(float(values["yamnet_threshold"]))
                self.trigger_slider.setValue(round(float(values["trigger_dbfs"])))
            finally:
                self._loading = False

    def _threshold_edited(self, *_args) -> None:
        if self._loading or self.preset_combo.currentData() == "custom":
            return
        self._loading = True
        try:
            self.preset_combo.setCurrentIndex(self.preset_combo.findData("custom"))
        finally:
            self._loading = False

    def _update_trigger_label(self, value: int) -> None:
        self.trigger_label.setText(f"{value} dBFS")

    def _start_calibration(self) -> None:
        if any(
            worker is not None and worker.isRunning()
            for worker in (self.worker, self.manual_capture_worker)
        ):
            return
        self._capture_current()
        preset = str(self.preset_combo.currentData() or "balanced")
        if preset == "custom":
            preset = "balanced"
            self.preset_combo.setCurrentIndex(self.preset_combo.findData(preset))
        device = self.microphone_combo.currentData()
        if device == "__default__":
            device = None
        self.calibrate_button.setEnabled(False)
        self.microphone_combo.setEnabled(False)
        self.preset_combo.setEnabled(False)
        self.progress.setValue(0)
        self.result_label.setText(
            "Идёт замер. Создавайте обычные для помещения шумы, но не хлопайте рядом с микрофоном."
        )
        self.worker = AmbientCalibrationWorker(
            device=device,
            duration_sec=float(self.duration_spin.value()),
            preset=preset,
        )
        self.worker.progress.connect(self.progress.setValue)
        self.worker.completed.connect(self._calibration_completed)
        self.worker.failed.connect(self._calibration_failed)
        self.worker.finished.connect(self._calibration_finished)
        self.worker.start()

    def _manual_controls(self, enabled: bool) -> None:
        self.manual_duration_spin.setEnabled(enabled)
        has_background = self._manual_ambient_chunks is not None
        for button in self.manual_distance_buttons.values():
            button.setEnabled(enabled and has_background)
        self.manual_apply_button.setEnabled(
            enabled and has_background and len(self._manual_reference_takes) >= 3
        )

    def _start_manual_capture(self, target: str) -> None:
        if any(
            worker is not None and worker.isRunning()
            for worker in (self.worker, self.manual_capture_worker)
        ):
            return
        if self._manual_ambient_chunks is None:
            QMessageBox.information(
                self,
                "Сначала нужен фон",
                "Сначала выполните шаг 3 «Измерить фон», затем записывайте контрольные дистанции.",
            )
            return
        self._capture_current()
        device = self.microphone_combo.currentData()
        if device == "__default__":
            device = None
        self._manual_capture_target = target
        self.calibrate_button.setEnabled(False)
        self.microphone_combo.setEnabled(False)
        self.preset_combo.setEnabled(False)
        self._manual_controls(False)
        self.progress.setValue(0)
        self.manual_status.setText(
            f"Идёт запись: дистанция {target}. Воспроизведите один тестовый импульс."
        )
        self.manual_capture_worker = CalibrationCaptureWorker(
            device=device,
            duration_sec=float(self.manual_duration_spin.value()),
        )
        self.manual_capture_worker.progress.connect(self.progress.setValue)
        self.manual_capture_worker.completed.connect(self._manual_capture_completed)
        self.manual_capture_worker.failed.connect(self._calibration_failed)
        self.manual_capture_worker.finished.connect(self._manual_capture_finished)
        self.manual_capture_worker.start()

    def _manual_capture_completed(self, chunks: list) -> None:
        target = self._manual_capture_target
        if target:
            self._manual_reference_takes[target] = chunks
            self.manual_distance_buttons[target].setText(f"Записано\n{target} ✓")
            self._update_manual_status()
            # Apply a useful intermediate profile immediately. Every following
            # distance is then evaluated against the latest, quieter reference.
            self._apply_manual_reference_calibration(minimum_references=1, incremental=True)

    def _update_manual_status(self) -> None:
        distances = ", ".join(self._manual_reference_takes) or "нет"
        background = "готов" if self._manual_ambient_chunks is not None else "не записан"
        if self._manual_last_apply_ok:
            tail = "Профиль применён сразу после последней записи; следующая дистанция уточнит его ещё точнее."
        elif self._manual_last_apply_error:
            tail = f"Запись сохранена, но профиль не изменён: {self._manual_last_apply_error}"
        elif self._manual_reference_takes:
            tail = "После каждой записи будет применён промежуточный профиль."
        else:
            tail = "После первой дистанции профиль обновится автоматически."
        self.manual_status.setText(
            f"Фон: {background}. Дистанций: {len(self._manual_reference_takes)} из 5 "
            f"({distances}). {tail}"
        )
        self.manual_apply_button.setEnabled(
            self.manual_capture_worker is None
            and self._manual_ambient_chunks is not None
            and len(self._manual_reference_takes) >= 3
        )

    def _manual_capture_finished(self) -> None:
        self.calibrate_button.setEnabled(True)
        self.microphone_combo.setEnabled(True)
        self.preset_combo.setEnabled(True)
        self.manual_capture_worker = None
        self._manual_controls(True)
        self._update_manual_status()

    def _apply_manual_reference_calibration(
        self, minimum_references: int = 3, incremental: bool = False
    ) -> bool:
        if self._manual_ambient_chunks is None or len(self._manual_reference_takes) < minimum_references:
            if not incremental:
                self._update_manual_status()
            return False
        preset = str(self.preset_combo.currentData() or "balanced")
        if preset == "custom":
            preset = "balanced"
        try:
            result = analyze_reference_audio(
                self._manual_ambient_chunks,
                self._manual_reference_takes.values(),
                preset,
                minimum_references=minimum_references,
            )
        except ValueError as exc:
            if incremental:
                self._manual_last_apply_ok = False
                self._manual_last_apply_error = str(exc)
                self._update_manual_status()
            else:
                self._calibration_failed(str(exc))
            return False
        self._manual_last_apply_ok = True
        self._manual_last_apply_error = ""
        self._reference_completed(result, incremental=incremental)
        if incremental:
            self.manual_status.setText(
                f"Дистанция сохранена. Профиль обновлён по {result.reference_count} записям; "
                "следующая дистанция уточнит пороги ещё точнее."
            )
        else:
            self.manual_status.setText(
                f"Итоговая настройка применена по {result.reference_count} дистанциям: "
                + ", ".join(self._manual_reference_takes)
            )
        return True

    def _reference_completed(
        self, result: ReferenceCalibration, incremental: bool = False
    ) -> None:
        self.profiles[self._current_key] = result.profile_values()
        self._load_profile(self._current_key)
        stage = "Промежуточный профиль" if incremental else "Итоговый профиль"
        self.result_label.setText(
            f"{stage} применена для профиля {self._current_key}: "
            f"порог {result.trigger_dbfs:.1f} dBFS, RMS минимум {result.rms_min:.6f}, "
            f"импульсность {result.min_crest_factor:.2f}. Семантическая проверка "
            "PANNs + YAMNet сохранена для защиты от ложных тревог."
        )

    def _calibration_completed(self, result: AmbientCalibration) -> None:
        self.profiles[self._current_key] = result.profile_values()
        self._load_profile(self._current_key)
        if self.worker is not None and self.worker.captured_chunks:
            # Raw audio is retained in memory only and reused by the manual
            # distance recorder; it is discarded when the dialog is closed or
            # the microphone changes.
            self._manual_ambient_chunks = [chunk.copy() for chunk in self.worker.captured_chunks]
            self._update_manual_status()
        clipping = ""
        if result.clipping_ratio > 0.001:
            clipping = " Обнаружен клиппинг — уменьшите усиление микрофона в Windows и повторите."
        elif result.ambient_peak_dbfs > -12.0:
            clipping = (
                " Фон или усиление микрофона очень высокие — уменьшите Microphone Boost "
                "или отодвиньте микрофон от постоянного источника шума и повторите замер."
            )
        self.result_label.setText(
            f"Готово: фон до {result.ambient_peak_dbfs:.1f} dBFS, новый порог "
            f"{result.trigger_dbfs:.1f} dBFS, SNR {result.min_snr_db:.1f} dB, "
            f"импульсность {result.min_crest_factor:.2f}.{clipping} "
            "Теперь можно записывать дистанции в шаге 4."
        )

    def _calibration_failed(self, message: str) -> None:
        self.result_label.setText(f"Калибровка не выполнена: {message}")
        QMessageBox.critical(self, "Ошибка калибровки", message)

    def _calibration_finished(self) -> None:
        self.calibrate_button.setEnabled(True)
        self.microphone_combo.setEnabled(True)
        self.preset_combo.setEnabled(True)
        self.worker = None
        self._manual_controls(True)
        self._update_manual_status()

    def _save_and_accept(self) -> None:
        self._capture_current()
        self.calibration_duration_sec = float(self.duration_spin.value())
        self.accept()

    def done(self, result: int) -> None:
        for worker in (self.worker, self.manual_capture_worker):
            if worker is not None and worker.isRunning():
                worker.requestInterruption()
                worker.wait(2_000)
        super().done(result)
