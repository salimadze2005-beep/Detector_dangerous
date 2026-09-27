from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QColor, QDesktopServices
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
)

from core.event_bus import Event, EventType
from core.incidents import IncidentRecord, IncidentStore
from video.fall_detector import dismiss_fall_track


_EVENT_LABELS = {
    "FALL_DETECTED": "Падение",
    "GUNSHOT_DETECTED": "Выстрел",
    "KEYWORD_DETECTED": "Ключевое слово",
}

_DECISION_LABELS = {
    "unreviewed": "Не проверено",
    "confirmed": "Подтверждено",
    "false_alarm": "Ложная тревога",
}

def _format_incident_timestamp(value: str) -> str:
    """Render stored UTC/ISO timestamps in the operator local timezone."""
    try:
        timestamp = datetime.fromisoformat(str(value))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone().strftime("%d.%m.%Y %H:%M:%S")
    except (TypeError, ValueError):
        # Keep malformed legacy values visible instead of breaking history.
        return str(value).replace("T", " ")[:19]

_DECISION_COLORS = {
    "unreviewed": QColor("#f2c94c"),
    "confirmed": QColor("#a7dc00"),
    "false_alarm": QColor("#ff7b86"),
}

_DELIVERY_LABELS = {
    "pending": "ожидает отправки",
    "delivered": "доставлено",
    "failed": "ошибка доставки",
    "cancelled": "отменено",
}


class IncidentHistoryDialog(QDialog):
    """Independent, non-modal incident journal optimized for fast operator actions."""

    def __init__(self, store: IncidentStore, parent=None):
        super().__init__(parent)
        self.store = store
        self.records: list[IncidentRecord] = []
        self._media_checked: set[str] = set()
        self._reloading = False
        self._action_busy = False

        # This is intentionally a normal top-level window, not a child dialog.
        # It can sit behind the dashboard when the operator switches back to it.
        self.setWindowTitle("История тревог")
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.resize(1180, 760)
        self.setMinimumSize(900, 620)
        self.setStyleSheet(
            """
            QDialog { background:#202335; color:#f4f6ff; font-family:'Segoe UI'; }
            QFrame#panel { background:#292d43; border:1px solid #3b405c; border-radius:12px; }
            QLabel { color:#f4f6ff; background:transparent; }
            QLabel#title { font-size:24px; font-weight:800; color:#ffffff; }
            QLabel#subtitle { color:#aeb4cd; font-size:12px; }
            QLabel#fieldName { color:#ffffff; font-weight:700; }
            QLabel#fieldValue { color:#f4f6ff; }
            QLabel#status { color:#aeb4cd; padding:4px 2px; }
            QTableWidget { background:#24283b; alternate-background-color:#282c41; color:#f4f6ff;
                border:1px solid #3b405c; border-radius:9px; gridline-color:#353a54; }
            QTableWidget::item { padding:8px; color:#f4f6ff; }
            QTableWidget::item:selected { background:#3d4665; color:#ffffff; }
            QHeaderView::section { background:#30344d; color:#dfe3f4; border:0;
                border-right:1px solid #424760; padding:9px; font-weight:700; }
            QTextEdit { background:#202335; color:#ffffff; border:1px solid #4b506f;
                border-radius:8px; padding:8px; selection-background-color:#4a5275; }
            QPushButton { border:0; border-radius:8px; padding:10px 15px; font-weight:700; }
            QPushButton#primary { background:#a7dc00; color:#1b1d2b; }
            QPushButton#primary:hover { background:#b9ed15; }
            QPushButton#danger { background:#e3424f; color:white; }
            QPushButton#danger:hover { background:#f05260; }
            QPushButton#secondary { background:#343950; color:white; border:1px solid #4b506f; }
            QPushButton#secondary:hover { background:#414760; }
            QPushButton:disabled { background:#30344d; color:#777d98; border-color:#3b405c; }
            """
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        heading = QHBoxLayout()
        titles = QVBoxLayout()
        title = QLabel("История инцидентов")
        title.setObjectName("title")
        subtitle = QLabel("Фото, видео, решения и заметки по тревожным сработкам")
        subtitle.setObjectName("subtitle")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        heading.addLayout(titles)
        heading.addStretch()
        self.refresh = self._button("↻  ОБНОВИТЬ", "secondary")
        self.refresh.setToolTip("Обновить список. Видео проверяется только для выбранного инцидента.")
        heading.addWidget(self.refresh)
        root.addLayout(heading)

        self.table = QTableWidget(0, 5)
        self.table.setAlternatingRowColors(True)
        self.table.setHorizontalHeaderLabels(["Время", "Тип", "Решение", "Уверенность", "Заметка"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setShowGrid(False)
        self.table.setSortingEnabled(False)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        for column in (0, 1, 2, 3):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        root.addWidget(self.table, 1)

        details = QFrame()
        details.setObjectName("panel")
        details_layout = QGridLayout(details)
        details_layout.setContentsMargins(16, 14, 16, 14)
        details_layout.setHorizontalSpacing(14)
        details_layout.setVerticalSpacing(8)

        self.description = self._value_label("Выберите инцидент")
        self.description.setWordWrap(True)
        self.description.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.delivery = self._value_label("—")
        self.media = self._value_label("—")
        self.media.setWordWrap(True)
        self.media.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.note = QTextEdit()
        self.note.setPlaceholderText("Заметка: что произошло, почему сработка подтверждена или отклонена...")
        self.note.setMaximumHeight(90)

        details_layout.addWidget(self._field_label("Описание"), 0, 0)
        details_layout.addWidget(self.description, 0, 1)
        details_layout.addWidget(self._field_label("REST"), 1, 0)
        details_layout.addWidget(self.delivery, 1, 1)
        details_layout.addWidget(self._field_label("Медиа"), 2, 0)
        details_layout.addWidget(self.media, 2, 1)
        details_layout.addWidget(self._field_label("Заметка"), 3, 0)
        details_layout.addWidget(self.note, 3, 1)
        details_layout.setColumnStretch(1, 1)
        root.addWidget(details)

        media_actions = QHBoxLayout()
        self.open_photo = self._button("ОТКРЫТЬ ФОТО", "secondary")
        self.open_video = self._button("ОТКРЫТЬ ВИДЕО", "secondary")
        self.open_folder = self._button("ПАПКА ИНЦИДЕНТА", "secondary")
        self.save_note = self._button("СОХРАНИТЬ ЗАМЕТКУ", "secondary")
        media_actions.addWidget(self.open_photo)
        media_actions.addWidget(self.open_video)
        media_actions.addWidget(self.open_folder)
        media_actions.addStretch()
        media_actions.addWidget(self.save_note)
        root.addLayout(media_actions)

        verdict_actions = QHBoxLayout()
        verdict_actions.addStretch()
        self.confirm = self._button("✓  ПОДТВЕРДИТЬ", "primary")
        self.reject = self._button("✕  ЛОЖНАЯ ТРЕВОГА", "danger")
        verdict_actions.addWidget(self.confirm)
        verdict_actions.addWidget(self.reject)
        root.addLayout(verdict_actions)

        self.status = QLabel("Готово")
        self.status.setObjectName("status")
        root.addWidget(self.status)

        self.table.itemSelectionChanged.connect(self._show_selected)
        self.open_photo.clicked.connect(self._open_photo)
        self.open_video.clicked.connect(self._open_video)
        self.open_folder.clicked.connect(self._open_folder)
        self.save_note.clicked.connect(self._save_note)
        self.confirm.clicked.connect(lambda: self._set_decision("confirmed"))
        self.reject.clicked.connect(lambda: self._set_decision("false_alarm"))
        self.refresh.clicked.connect(self.reload)

        self.reload()

    @staticmethod
    def _button(text: str, object_name: str) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName(object_name)
        return button

    @staticmethod
    def _field_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("fieldName")
        return label

    @staticmethod
    def _value_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("fieldValue")
        return label

    def _set_status(self, message: str, error: bool = False) -> None:
        self.status.setText(message)
        self.status.setStyleSheet("color:#ff8c96;" if error else "color:#aeb4cd;")

    def _set_action_buttons_busy(self, busy: bool) -> None:
        self.save_note.setEnabled(not busy and self.current_record() is not None)
        if busy:
            self.confirm.setEnabled(False)
            self.reject.setEnabled(False)
            return
        record = self.current_record()
        if record is not None:
            self.confirm.setEnabled(record.decision != "confirmed")
            self.reject.setEnabled(record.decision != "false_alarm")

    def reload(self) -> None:
        if self._reloading or self._action_busy:
            return
        self._reloading = True
        self.refresh.setEnabled(False)
        selected = self.current_record()
        selected_id = selected.event_id if selected else None
        try:
            records = self.store.recent(200, resolve_media=False)
            self._media_checked.clear()
            self.table.blockSignals(True)
            self.table.setUpdatesEnabled(False)
            self.records = records
            self.table.setRowCount(len(records))
            restore_row = -1
            for row, record in enumerate(records):
                self._write_row(row, record)
                if selected_id == record.event_id:
                    restore_row = row
            if records:
                self.table.selectRow(restore_row if restore_row >= 0 else 0)
            else:
                self._clear_details()
            self._set_status(f"Показано инцидентов: {len(records)}")
        except Exception as exc:
            self._set_status(f"Не удалось обновить историю: {exc}", error=True)
        finally:
            self.table.setUpdatesEnabled(True)
            self.table.blockSignals(False)
            self.refresh.setEnabled(True)
            self._reloading = False
        if self.records:
            self._show_selected()

    def _write_row(self, row: int, record: IncidentRecord) -> None:
        values = [
            _format_incident_timestamp(record.timestamp),
            _EVENT_LABELS.get(record.event_type, record.event_type),
            _DECISION_LABELS.get(record.decision, record.decision),
            f"{record.confidence:.0%}",
            record.note,
        ]
        for column, value in enumerate(values):
            item = self.table.item(row, column)
            if item is None:
                item = QTableWidgetItem()
                self.table.setItem(row, column, item)
            item.setText(value)
            item.setForeground(
                _DECISION_COLORS.get(record.decision, QColor("#ffffff"))
                if column == 2
                else QColor("#f4f6ff")
            )

    def current_record(self) -> IncidentRecord | None:
        row = self.table.currentRow()
        if 0 <= row < len(self.records):
            return self.records[row]
        return None

    def _load_selected_media_once(self, record: IncidentRecord) -> IncidentRecord:
        if record.event_id in self._media_checked:
            return record
        self._media_checked.add(record.event_id)
        try:
            detailed = self.store.get(record.event_id, resolve_media=True)
        except Exception as exc:
            self._set_status(f"Не удалось прочитать медиа: {exc}", error=True)
            return record
        if detailed is None:
            return record
        row = self.table.currentRow()
        if 0 <= row < len(self.records) and self.records[row].event_id == detailed.event_id:
            self.records[row] = detailed
        return detailed

    def _show_selected(self) -> None:
        if self._reloading:
            return
        record = self.current_record()
        if record is None:
            self._clear_details()
            return
        record = self._load_selected_media_once(record)
        self._render_record(record)

    def _render_record(self, record: IncidentRecord, update_note: bool = True) -> None:
        self.description.setText(
            record.description or f"Событие {_EVENT_LABELS.get(record.event_type, record.event_type)}"
        )
        self.delivery.setText(
            _DELIVERY_LABELS.get(record.delivery_status, record.delivery_status or "нет записи")
        )
        media_lines = []
        if record.image_path:
            media_lines.append(f"Фото: {record.image_path}")
        if record.clip_files:
            media_lines.extend(f"Видео: {path}" for path in record.clip_files)
        elif record.clip_dir:
            media_lines.append(f"Папка: {record.clip_dir}")
        self.media.setText("\n".join(media_lines) if media_lines else "Медиа пока не найдено")
        if update_note:
            self.note.setPlainText(record.note)
        self.open_photo.setEnabled(bool(record.image_path and Path(record.image_path).is_file()))
        self.open_video.setEnabled(bool(record.clip_files and Path(record.clip_files[0]).is_file()))
        self.open_folder.setEnabled(bool(record.clip_dir and Path(record.clip_dir).is_dir()))
        self.save_note.setEnabled(True)
        self.confirm.setEnabled(record.decision != "confirmed")
        self.reject.setEnabled(record.decision != "false_alarm")

    def _clear_details(self) -> None:
        self.description.setText("Выберите инцидент")
        self.delivery.setText("—")
        self.media.setText("—")
        self.note.clear()
        for button in (
            self.open_photo,
            self.open_video,
            self.open_folder,
            self.save_note,
            self.confirm,
            self.reject,
        ):
            button.setEnabled(False)

    def _refresh_current_record(self, preserve_media: bool = True) -> IncidentRecord | None:
        current = self.current_record()
        if current is None:
            return None
        fresh = self.store.get(current.event_id, resolve_media=False)
        if fresh is None:
            return current
        if preserve_media:
            fresh = replace(fresh, clip_dir=current.clip_dir, clip_files=current.clip_files)
        row = self.table.currentRow()
        if 0 <= row < len(self.records):
            self.records[row] = fresh
            self._write_row(row, fresh)
        return fresh

    def refresh_event(self, event_id: str) -> None:
        """Refresh one visible row after an operator verdict without rebuilding the table."""
        for row, record in enumerate(self.records):
            if record.event_id != event_id:
                continue
            try:
                fresh = self.store.get(event_id, resolve_media=False)
            except Exception:
                return
            if fresh is None:
                return
            fresh = replace(fresh, clip_dir=record.clip_dir, clip_files=record.clip_files)
            self.records[row] = fresh
            self._write_row(row, fresh)
            if row == self.table.currentRow():
                self._render_record(fresh)
            return

    def _run_action(self, action, success_message: str) -> None:
        if self._action_busy:
            return
        self._action_busy = True
        self.refresh.setEnabled(False)
        self._set_action_buttons_busy(True)
        try:
            action()
            fresh = self._refresh_current_record(preserve_media=True)
            if fresh is not None:
                # Keep the window, selection, scroll position and media state intact.
                self._render_record(fresh)
            self._set_status(success_message)
        except Exception as exc:
            self._set_status(f"Операция не выполнена: {exc}", error=True)
        finally:
            self._action_busy = False
            self.refresh.setEnabled(True)
            self._set_action_buttons_busy(False)

    def _save_note(self) -> None:
        record = self.current_record()
        if record is None:
            return
        note = self.note.toPlainText()
        self._run_action(
            lambda: self.store.update_note(record.event_id, note),
            "Заметка сохранена",
        )

    def _set_decision(self, decision: str) -> None:
        record = self.current_record()
        if record is None:
            return
        note = self.note.toPlainText()
        label = "Инцидент подтверждён" if decision == "confirmed" else "Инцидент отмечен как ложная тревога"
        self._run_action(
            lambda: self.store.review(record.event_id, decision, note),
            label,
        )

    def _open_path(self, path: str | None, kind: str) -> None:
        try:
            if not path or not Path(path).exists():
                self._set_status(f"{kind} не найдено на диске", error=True)
                return
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).resolve()))):
                self._set_status(f"Windows не смог открыть {kind.lower()}", error=True)
        except Exception as exc:
            self._set_status(f"Не удалось открыть {kind.lower()}: {exc}", error=True)

    def _open_photo(self) -> None:
        record = self.current_record()
        self._open_path(record.image_path if record else None, "Фото")

    def _open_video(self) -> None:
        record = self.current_record()
        path = record.clip_files[0] if record and record.clip_files else None
        self._open_path(path, "Видео")

    def _open_folder(self) -> None:
        record = self.current_record()
        self._open_path(record.clip_dir if record else None, "Папка")


class IncidentHistoryController:
    """Integrates an independent incident window and alert verdict actions."""

    def __init__(self, window, db_path: str, clips_root: str):
        self.window = window
        self.store = IncidentStore(db_path, clips_root)
        self.dialog: IncidentHistoryDialog | None = None
        self._verdict_busy = False

        # The large dashboard header button was intentionally removed.
        # History remains available through the application menu.
        menu = window.menuBar().addMenu("ИСТОРИЯ")
        history_action = menu.addAction("Открыть историю")
        history_action.setShortcut("Ctrl+I")
        history_action.triggered.connect(self.show_history)

        self.false_alarm_button = QPushButton("✕  ЛОЖНАЯ ТРЕВОГА")
        self.false_alarm_button.setToolTip(
            "Отклонить сработку ИИ. Для падения повторная тревога не появится до вставания человека."
        )
        self.false_alarm_button.setStyleSheet(
            "QPushButton { background:#363b53; color:#ffbdc2; border:1px solid #7d4650; "
            "border-radius:8px; font-size:15px; font-weight:700; padding:12px 18px; }"
            "QPushButton:hover { background:#4a3340; color:white; border-color:#e3424f; }"
            "QPushButton:disabled { color:#777d98; border-color:#4b506f; }"
        )
        self.false_alarm_button.clicked.connect(self.reject_current)
        self._install_alert_actions()

        try:
            window.alert_overlay.acknowledge.clicked.disconnect(window.acknowledge_alert)
        except (TypeError, RuntimeError):
            pass
        window.alert_overlay.acknowledge.clicked.connect(self.confirm_current)
        window.bridge.event_received.connect(self._handle_event)
        window.destroyed.connect(self._close_history)

    def _install_alert_actions(self) -> None:
        try:
            card = self.window.alert_overlay.acknowledge.parentWidget()
            grid = card.layout() if card is not None else None
            details = grid.itemAtPosition(0, 1).layout() if isinstance(grid, QGridLayout) else None
            if details is None:
                return
            details.removeWidget(self.window.alert_overlay.acknowledge)
            action_row = QHBoxLayout()
            action_row.setSpacing(10)
            action_row.addWidget(self.window.alert_overlay.acknowledge)
            action_row.addWidget(self.false_alarm_button)
            action_row.addStretch()
            details.addLayout(action_row)
        except Exception:
            self.false_alarm_button.setParent(self.window.alert_overlay)
            self.false_alarm_button.hide()

    def _close_history(self) -> None:
        if self.dialog is not None:
            self.dialog.close()

    def show_history(self) -> None:
        try:
            if self.dialog is None:
                # No parent: ordinary independent Windows window, not always over dashboard.
                self.dialog = IncidentHistoryDialog(self.store, parent=None)
            elif self.dialog.isVisible():
                self.dialog.showNormal()
                self.dialog.activateWindow()
                return
            self.dialog.reload()
            self.dialog.show()
            self.dialog.activateWindow()
        except Exception as exc:
            self.window.statusBar().showMessage(
                f"Не удалось открыть историю инцидентов: {exc}",
                5000,
            )

    def confirm_current(self) -> None:
        if self._verdict_busy:
            return
        event = self.window.current_alert
        if event is None:
            return
        self._verdict_busy = True
        try:
            self.store.review(event.id, "confirmed", "")
            self.window.acknowledge_alert()
            self.window.statusBar().showMessage("Тревога подтверждена", 3500)
            if self.dialog is not None and self.dialog.isVisible():
                self.dialog.refresh_event(event.id)
        except Exception as exc:
            self.window.statusBar().showMessage(
                f"Не удалось подтвердить тревогу: {exc}",
                5000,
            )
        finally:
            self._verdict_busy = False

    def reject_current(self) -> None:
        if self._verdict_busy:
            return
        event = self.window.current_alert
        if event is None:
            return
        note, accepted = QInputDialog.getMultiLineText(
            self.window,
            "Ложная тревога",
            "Комментарий (необязательно):",
            "",
        )
        if not accepted:
            return
        self._verdict_busy = True
        self.false_alarm_button.setEnabled(False)
        try:
            if event.type == EventType.FALL_DETECTED:
                camera_id = str(event.metadata.get("camera_id", "")).strip()
                track_id = event.metadata.get("track_id")
                if camera_id and track_id is not None:
                    dismiss_fall_track(camera_id, int(track_id))
            self.store.review(event.id, "false_alarm", note)
            self.window.alert_service.acknowledge_local_alarm()
            self.window.current_alert = None
            self.window._show_next_alert()
            self.window.statusBar().showMessage("Сработка сохранена как ложная тревога", 4000)
            if self.dialog is not None and self.dialog.isVisible():
                self.dialog.refresh_event(event.id)
        except Exception as exc:
            self.window.statusBar().showMessage(
                f"Не удалось отклонить сработку: {exc}",
                5000,
            )
        finally:
            self.false_alarm_button.setEnabled(True)
            self._verdict_busy = False

    @staticmethod
    def _same_fall(alert: Event, status_event: Event) -> bool:
        if alert.type != EventType.FALL_DETECTED:
            return False
        return (
            str(alert.metadata.get("camera_id", "")) == str(status_event.metadata.get("camera_id", ""))
            and str(alert.metadata.get("track_id", "")) == str(status_event.metadata.get("track_id", ""))
        )

    def _handle_event(self, event: Event) -> None:
        if event.type != EventType.FALL_RECOVERED:
            return
        self.window.alert_queue = type(self.window.alert_queue)(
            pair for pair in self.window.alert_queue if not self._same_fall(pair[0], event)
        )
        current = self.window.current_alert
        if current is not None and self._same_fall(current, event):
            self.window.alert_service.acknowledge_local_alarm()
            self.window.current_alert = None
            self.window._show_next_alert()
            self.window.statusBar().showMessage(
                "Тревога падения снята: человек снова устойчиво стоит",
                4500,
            )


def install_incident_history(window, db_path: str, clips_root: str) -> IncidentHistoryController:
    controller = IncidentHistoryController(window, db_path, clips_root)
    window.incident_history_controller = controller
    return controller
