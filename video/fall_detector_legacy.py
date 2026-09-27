from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.config import get_resource_path
from core.event_bus import Event, EventBus, EventType, Severity
from video.model_optimizer import select_yolo_runtime_model

logger = logging.getLogger(__name__)

_dismiss_lock = threading.Lock()
_dismissed_tracks: set[tuple[str, int]] = set()


def fixed_onnx_input_size(model_path: str) -> int | tuple[int, int] | None:
    """Return static ONNX spatial dimensions, if the model declares them."""
    path = Path(model_path)
    if path.suffix.casefold() != ".onnx" or not path.is_file():
        return None
    try:
        import onnx

        graph = onnx.load(str(path), load_external_data=False).graph
        if not graph.input:
            return None
        dimensions = graph.input[0].type.tensor_type.shape.dim
        if len(dimensions) < 4:
            return None
        height, width = dimensions[-2].dim_value, dimensions[-1].dim_value
        if height <= 0 or width <= 0:
            return None
        return int(height) if height == width else (int(height), int(width))
    except Exception as exc:
        logger.warning("Не удалось прочитать размер входа ONNX %s: %s", path.name, exc)
        return None


def dismiss_fall_track(camera_id: str, track_id: int) -> None:
    """Ask the detector to suppress this fall until the person stands again."""
    with _dismiss_lock:
        _dismissed_tracks.add((str(camera_id), int(track_id)))


def _consume_fall_dismissal(camera_id: str, track_id: int) -> bool:
    key = (str(camera_id), int(track_id))
    with _dismiss_lock:
        if key not in _dismissed_tracks:
            return False
        _dismissed_tracks.remove(key)
        return True


@dataclass
class TrackState:
    seen_upright: bool = False
    fall_since: float | None = None
    standing_since: float | None = None
    missing_since: float | None = None
    alerted: bool = False
    posture: str = "Неизвестно"
    posture_history: deque[str] = field(default_factory=deque)
    transition_since: float | None = None
    rapid_descent_at: float | None = None
    last_center_y: float | None = None
    last_height: float | None = None
    last_observed_at: float | None = None
    vertical_speed: float = 0.0
    suppressed_until_standing: bool = False
    recovered_pending: bool = False


@dataclass(frozen=True, slots=True)
class FrameAnalysis:
    annotated_frame: np.ndarray
    events: tuple[Event, ...]


class FallDetector:
    """Pose-based prolonged horizontal posture detector with per-track state."""

    def __init__(
        self,
        event_bus: EventBus | None = None,
        camera_id: str = "camera-0",
        model_path: str | None = None,
        fall_duration_sec: float = 5.0,
        reset_grace_sec: float = 2.0,
        missing_grace_sec: float = 2.0,
        fps: float = 30.0,
        use_wall_clock: bool = True,
        pose_confidence: float = 0.30,
        fall_min_confidence: float = 0.65,
        keypoint_confidence: float = 0.50,
        partial_pose_confidence: float = 0.25,
        partial_pose_min_keypoints: int = 3,
        partial_lie_min_score: float = 0.72,
        standing_angle_deg: float = 30.0,
        lying_angle_deg: float = 60.0,
        lying_aspect_ratio: float = 1.00,
        seated_knee_drop_ratio: float = 0.12,
        require_upright_transition: bool = True,
        posture_window: int = 5,
        posture_hits: int = 3,
        rapid_descent_heights_per_sec: float = 0.75,
        transition_memory_sec: float = 2.0,
        motion_confirm_sec: float = 1.5,
        require_fall_motion: bool = True,
        view_mode: str = "side",
        device: str = "auto",
    ):
        self.event_bus = event_bus
        self.camera_id = camera_id
        self.view_mode = self._normalize_view_mode(view_mode)
        self.device = self._resolve_device(device)

        base_model_path_str = model_path or "models/yolov8n-pose.pt"
        resolved_model_path = Path(get_resource_path(base_model_path_str))
        resolved_onnx_path = resolved_model_path.with_suffix(".onnx")
        # Prefer the PyTorch weights. Ultralytics can then execute directly on
        # CUDA. ONNX is only a compatibility fallback when the requested model
        # is genuinely absent.
        configured_model_path = (
            resolved_model_path if resolved_model_path.exists() else resolved_onnx_path
        )
        self.model_path = str(select_yolo_runtime_model(configured_model_path, self.device))

        self.fall_duration_sec = fall_duration_sec
        self.reset_grace_sec = reset_grace_sec
        self.missing_grace_sec = missing_grace_sec
        self.frame_duration = 1.0 / max(fps, 1.0)
        self.use_wall_clock = use_wall_clock
        self.pose_confidence = pose_confidence
        self.fall_min_confidence = min(max(float(fall_min_confidence), 0.0), 1.0)
        self.keypoint_confidence = keypoint_confidence
        self.partial_pose_confidence = min(
            max(float(partial_pose_confidence), 0.05), self.keypoint_confidence
        )
        self.partial_pose_min_keypoints = min(
            max(int(partial_pose_min_keypoints), 3), 8
        )
        self.partial_lie_min_score = min(
            max(float(partial_lie_min_score), 0.50), 0.95
        )
        self.standing_angle_deg = standing_angle_deg
        self.lying_angle_deg = lying_angle_deg
        self.lying_aspect_ratio = lying_aspect_ratio
        self.seated_knee_drop_ratio = max(0.0, float(seated_knee_drop_ratio))
        self.require_upright_transition = require_upright_transition
        self.posture_window = max(1, posture_window)
        self.posture_hits = min(max(1, posture_hits), self.posture_window)
        self.rapid_descent_heights_per_sec = rapid_descent_heights_per_sec
        self.transition_memory_sec = transition_memory_sec
        self.motion_confirm_sec = motion_confirm_sec
        self.require_fall_motion = require_fall_motion
        self.current_time = 0.0
        self._last_wall_time: float | None = None
        self._states: dict[int, TrackState] = {}

        logger.info("Загрузка YOLO-Pose: %s; device=%s", self.model_path, self.device)
        from ultralytics import YOLO

        self.model = YOLO(self.model_path)

    @staticmethod
    def _normalize_view_mode(value: str | None) -> str:
        mode = str(value or "side").strip().casefold()
        if mode not in {"side", "top_down"}:
            raise ValueError("Режим ракурса должен быть side или top_down")
        return mode
    @staticmethod
    def _resolve_device(requested: str | None = "auto") -> str:
        """Resolve a deterministic Torch/Ultralytics device with CUDA-first auto mode."""
        import torch

        value = os.getenv("DETECTOR_VIDEO_DEVICE", requested or "auto").strip().casefold()
        if value in {"", "auto"}:
            if torch.cuda.is_available():
                return "cuda:0"
            return "cpu"
        if value == "gpu":
            value = "cuda:0"
        elif value == "cuda":
            value = "cuda:0"
        if value == "cpu":
            return "cpu"
        if value.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Для видеодетектора запрошена CUDA, но PyTorch не видит NVIDIA GPU. "
                    "Проверьте драйвер и переустановите CUDA-сборку PyTorch через setup.bat."
                )
            try:
                index = int(value.split(":", 1)[1]) if ":" in value else 0
            except ValueError as exc:
                raise ValueError(f"Некорректное CUDA-устройство: {value}") from exc
            count = int(torch.cuda.device_count())
            if index < 0 or index >= count:
                raise RuntimeError(
                    f"CUDA-устройство {value} недоступно; найдено GPU: {count}."
                )
            return f"cuda:{index}"
        raise ValueError(
            "DETECTOR_VIDEO_DEVICE должен быть auto, cpu, cuda, cuda:N или gpu"
        )

    def _camera_id(self) -> str:
        """Return a stable camera id, including for lightweight unit-test fixtures."""
        return str(getattr(self, "camera_id", "camera-0"))

    @staticmethod
    def _suppress_state(state: TrackState) -> None:
        state.alerted = False
        state.fall_since = None
        state.standing_since = None
        state.suppressed_until_standing = True
        state.recovered_pending = False

    def _expire_missing_suppression(self) -> None:
        """Re-arm tracks whose dismissed person has left the frame.

        A dismissal is kept while the same person remains visible, so one
        false alarm cannot be recreated on every frame.  It must not survive
        indefinitely after the track has disappeared, otherwise a later
        person/lying episode can inherit a permanently suppressed state.
        """
        grace = max(float(getattr(self, "missing_grace_sec", 2.0)), 0.0)
        for state in self._states.values():
            if (
                state.suppressed_until_standing
                and state.missing_since is not None
                and self.current_time - state.missing_since >= grace
            ):
                state.alerted = False
                state.fall_since = None
                state.standing_since = None
                state.suppressed_until_standing = False
                state.recovered_pending = False

    def _apply_pending_dismissals(self, visible_track_ids: set[int]) -> None:
        """Apply operator verdicts before processing the next tracked frame.

        YOLO can assign a new track id while the same person remains on the
        floor.  When only one person is visible, carry the suppressed state to
        that replacement id instead of starting a fresh confirmation timer.
        """
        self._expire_missing_suppression()
        for track_id, state in list(self._states.items()):
            if _consume_fall_dismissal(self._camera_id(), track_id):
                self._suppress_state(state)

        suppressed_missing = [
            track_id
            for track_id, state in self._states.items()
            if state.suppressed_until_standing and track_id not in visible_track_ids
        ]
        replacement_ids = visible_track_ids.difference(self._states)
        if len(suppressed_missing) == 1 and len(replacement_ids) == 1:
            previous_id = suppressed_missing[0]
            replacement_id = next(iter(replacement_ids))
            self._states[replacement_id] = self._states.pop(previous_id)

    def _tick(self) -> None:
        if not self.use_wall_clock:
            self.current_time += self.frame_duration
            return
        now = time.perf_counter()
        delta = self.frame_duration if self._last_wall_time is None else now - self._last_wall_time
        self._last_wall_time = now
        self.current_time += min(max(delta, 0.0), 1.0)

    @staticmethod
    def calculate_angle(shoulder_center, hip_center) -> float:
        """Torso angle from vertical: 0° upright, 90° horizontal."""
        dx = float(hip_center[0]) - float(shoulder_center[0])
        dy = float(hip_center[1]) - float(shoulder_center[1])
        if dx == 0.0 and dy == 0.0:
            return 0.0
        return math.degrees(math.atan2(abs(dx), abs(dy)))

    def has_seated_leg_geometry(self, hip_center, knees, bbox_height: float) -> bool:
        """Return true when a visible knee is clearly below the hips.

        A horizontal torso and wide box are ambiguous by themselves: a person
        leaning forward on a chair often looks identical to a lying person.
        In a seated pose at least one knee remains substantially below the hip;
        in a side-on lying pose hips and knees are usually at a similar height.
        """
        if bbox_height <= 0:
            return False
        visible = [
            point for point in knees
            if len(point) >= 3 and float(point[2]) >= self.keypoint_confidence
        ]
        if not visible:
            return False
        required_drop = self.seated_knee_drop_ratio * float(bbox_height)
        return any(float(point[1]) - float(hip_center[1]) >= required_drop for point in visible)

    def classify_posture(
        self,
        torso_angle: float,
        aspect_ratio: float = 1.0,
        seated_evidence: bool = False,
    ) -> str:
        if torso_angle < self.standing_angle_deg:
            return "Стоит"
        if seated_evidence:
            return "Сидит"
        if torso_angle >= self.lying_angle_deg and aspect_ratio >= self.lying_aspect_ratio:
            return "Лежит"
        return "Падает"

    def classify_top_down_posture(
        self,
        keypoints: np.ndarray,
        bbox_width: float,
        bbox_height: float,
    ) -> tuple[str, dict[str, float]]:
        """Classify a near-overhead view from rotation-invariant body geometry.

        Screen verticality is meaningless for an overhead camera. A lying body
        instead occupies a long, coherent axis; a standing person is compact.
        Bent legs are treated as seated, not as a fall.
        """
        threshold = float(getattr(self, "keypoint_confidence", 0.5))
        body_indices = (5, 6, 11, 12, 13, 14, 15, 16)
        visible = np.asarray(
            [keypoints[index, :2] for index in body_indices if float(keypoints[index, 2]) >= threshold],
            dtype=np.float32,
        )
        metrics = {"visible_points": float(len(visible)), "elongation": 0.0, "body_extent": 0.0, "leg_alignment": 0.0}
        if len(visible) < 5 or bbox_width <= 0 or bbox_height <= 0:
            return "Стоит", metrics
        covariance = np.cov(visible, rowvar=False)
        eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 1e-6))
        elongation = float(math.sqrt(eigenvalues[-1] / eigenvalues[0]))
        body_span = float(np.linalg.norm(np.ptp(visible, axis=0)))
        bbox_diagonal = math.hypot(float(bbox_width), float(bbox_height))
        extent = body_span / max(bbox_diagonal, 1e-6)
        metrics.update({"elongation": elongation, "body_extent": extent})

        def center(indices: tuple[int, ...]) -> np.ndarray | None:
            points = [keypoints[index, :2] for index in indices if float(keypoints[index, 2]) >= threshold]
            return np.mean(points, axis=0) if points else None

        shoulders, hips, legs = center((5, 6)), center((11, 12)), center((13, 14, 15, 16))
        alignment = 0.0
        if shoulders is not None and hips is not None and legs is not None:
            torso = hips - shoulders
            lower_body = legs - hips
            denominator = float(np.linalg.norm(torso) * np.linalg.norm(lower_body))
            if denominator > 1e-6:
                alignment = abs(float(np.dot(torso, lower_body))) / denominator
        metrics["leg_alignment"] = alignment
        if alignment < 0.35 and extent < 0.65:
            return "Сидит", metrics
        if elongation >= 2.0 and extent >= 0.50 and alignment >= 0.45:
            return "Лежит", metrics
        return "Стоит", metrics
    def _update_state(
        self,
        track_id: int,
        posture: str,
        center_y: float | None = None,
        height: float | None = None,
    ) -> TrackState:
        state = self._states.setdefault(track_id, TrackState())
        if _consume_fall_dismissal(self._camera_id(), track_id):
            self._suppress_state(state)

        state.missing_since = None
        state.posture = posture
        posture_window = max(1, getattr(self, "posture_window", 1))
        posture_hits = min(max(1, getattr(self, "posture_hits", 1)), posture_window)
        state.posture_history.append(posture)
        while len(state.posture_history) > posture_window:
            state.posture_history.popleft()

        if center_y is not None and height is not None and height > 0:
            if state.last_center_y is not None and state.last_observed_at is not None:
                delta = self.current_time - state.last_observed_at
                reference_height = max((height + (state.last_height or height)) / 2.0, 1.0)
                if delta > 1e-6:
                    state.vertical_speed = (center_y - state.last_center_y) / reference_height / delta
                    threshold = getattr(self, "rapid_descent_heights_per_sec", float("inf"))
                    if state.vertical_speed >= threshold:
                        state.rapid_descent_at = self.current_time
            state.last_center_y = center_y
            state.last_height = height
            state.last_observed_at = self.current_time

        safe_postures = {"Стоит", "Сидит"}
        stable_safe = (
            posture in safe_postures
            and sum(item in safe_postures for item in state.posture_history) >= posture_hits
        )
        stable_lying = posture == "Лежит" and state.posture_history.count("Лежит") >= posture_hits

        if stable_safe:
            if posture == "Стоит":
                state.seen_upright = True
            state.transition_since = None
            if state.fall_since is not None or state.alerted or state.suppressed_until_standing:
                state.standing_since = state.standing_since or self.current_time
                if self.current_time - state.standing_since >= self.reset_grace_sec:
                    was_alerted = state.alerted
                    state.fall_since = None
                    state.standing_since = None
                    state.alerted = False
                    state.suppressed_until_standing = False
                    state.recovered_pending = was_alerted
            return state

        state.standing_since = None
        if state.suppressed_until_standing:
            state.fall_since = None
            return state

        if posture == "Падает" and state.transition_since is None:
            state.transition_since = self.current_time

        if stable_lying and state.fall_since is None:
            memory = getattr(self, "transition_memory_sec", 0.0)
            transition_is_recent = (
                state.transition_since is not None
                and self.current_time - state.transition_since <= memory
            )
            state.fall_since = state.transition_since if transition_is_recent else self.current_time
        return state

    def _required_confirmation_sec(self, state: TrackState) -> float:
        memory = getattr(self, "transition_memory_sec", 0.0)
        recent_descent = (
            state.rapid_descent_at is not None
            and self.current_time - state.rapid_descent_at <= memory
        )
        if recent_descent:
            return max(self.fall_duration_sec, getattr(self, "motion_confirm_sec", self.fall_duration_sec))
        return self.fall_duration_sec

    def _detection_basis(self, state: TrackState) -> str:
        memory = getattr(self, "transition_memory_sec", 0.0)
        rapid_descent_recent = (
            state.rapid_descent_at is not None
            and self.current_time - state.rapid_descent_at <= memory
        )
        if rapid_descent_recent:
            return "rapid_descent"
        if state.seen_upright or state.transition_since is not None:
            return "upright_to_lying"
        return "prolonged_lying"

    def _mark_missing(self, active_track_ids: set[int]) -> None:
        for track_id, state in list(self._states.items()):
            if track_id in active_track_ids:
                continue
            state.missing_since = state.missing_since or self.current_time
            if (
                self.current_time - state.missing_since >= self.missing_grace_sec
                and not state.alerted
                and state.fall_since is None
                and not state.suppressed_until_standing
            ):
                del self._states[track_id]

    @staticmethod
    def _label(frame: np.ndarray, text: str, x: int, y: int, color) -> None:
        import cv2

        cv2.putText(
            frame, text, (x, max(20, y)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, color, 2, cv2.LINE_AA,
        )

    def process_frame(self, frame: np.ndarray) -> FrameAnalysis:
        self._tick()
        results = self.model.track(
            frame,
            persist=True,
            conf=self.pose_confidence,
            iou=0.3,
            device=self.device,
            verbose=False,
        )
        result = results[0]
        annotated = result.plot()
        active_track_ids: set[int] = set()
        events: list[Event] = []

        if result.keypoints is not None and result.boxes is not None and result.boxes.id is not None:
            keypoints_batch = result.keypoints.data.cpu().numpy()
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.int().cpu().tolist()
            track_ids = result.boxes.id.int().cpu().tolist()
            visible_person_ids = {
                track_id for track_id, class_id in zip(track_ids, classes) if class_id == 0
            }
            self._apply_pending_dismissals(visible_person_ids)

            for index, keypoints in enumerate(keypoints_batch):
                if classes[index] != 0:
                    continue
                track_id = track_ids[index]
                active_track_ids.add(track_id)
                x1, y1, x2, y2 = boxes_xyxy[index]
                width = max(float(x2 - x1), 1.0)
                height = max(float(y2 - y1), 1.0)
                points = [keypoints[5], keypoints[6], keypoints[11], keypoints[12]]
                if self.view_mode == "top_down":
                    pose_is_visible = sum(
                        float(point[2]) >= self.keypoint_confidence for point in keypoints[5:17]
                    ) >= 5
                else:
                    pose_is_visible = not any(
                        float(point[2]) < self.keypoint_confidence for point in points
                    )
                if not pose_is_visible:
                    state = self._states.setdefault(track_id, TrackState())
                    if _consume_fall_dismissal(self._camera_id(), track_id):
                        self._suppress_state(state)
                    state.missing_since = state.missing_since or self.current_time
                    if not state.alerted and self.current_time - state.missing_since >= self.missing_grace_sec:
                        state.fall_since = None
                        state.standing_since = None
                    self._label(annotated, "Состояние: не видно позу", int(x1), int(y1) - 35, (0, 165, 255))
                    continue

                left_shoulder, right_shoulder, left_hip, right_hip = points
                shoulder_center = (
                    (left_shoulder[0] + right_shoulder[0]) / 2,
                    (left_shoulder[1] + right_shoulder[1]) / 2,
                )
                hip_center = (
                    (left_hip[0] + right_hip[0]) / 2,
                    (left_hip[1] + right_hip[1]) / 2,
                )
                angle = self.calculate_angle(shoulder_center, hip_center)
                top_down_metrics: dict[str, float] = {}
                if self.view_mode == "top_down":
                    posture, top_down_metrics = self.classify_top_down_posture(keypoints, width, height)
                    # Image-space vertical motion has no fall meaning from overhead.
                    state = self._update_state(track_id, posture)
                else:
                    knees = [keypoints[13], keypoints[14]]
                    seated_evidence = self.has_seated_leg_geometry(hip_center, knees, height)
                    posture = self.classify_posture(angle, width / height, seated_evidence)
                    state = self._update_state(
                        track_id, posture, center_y=float(hip_center[1]), height=height
                    )

                # Keep the on-screen label identical for side and top-down views.
                # Geometry diagnostics remain available in event metadata/logs.
                posture_display = posture

                if state.recovered_pending:
                    recovery = Event(
                        EventType.FALL_RECOVERED,
                        confidence=1.0,
                        source="video.fall",
                        severity=Severity.INFO,
                        metadata={
                            "camera_id": self._camera_id(),
                            "track_id": track_id,
                            "posture": posture,
                        },
                    )
                    events.append(recovery)
                    if getattr(self, "event_bus", None) is not None:
                        self.event_bus.publish(recovery)
                    state.recovered_pending = False

                elapsed = 0.0 if state.fall_since is None else self.current_time - state.fall_since
                confirmation_sec = self._required_confirmation_sec(state)
                if state.fall_since is not None and elapsed >= confirmation_sec:
                    color = (0, 0, 255)
                    label = "ТРЕВОГА: возможное падение"
                    if not state.alerted:
                        pose_quality = float(np.mean([point[2] for point in points]))
                        if self.view_mode == "top_down":
                            geometry_quality = min(
                                max((top_down_metrics.get("elongation", 0.0) - 2.0) / 2.0, 0.0),
                                1.0,
                            )
                        else:
                            geometry_quality = min(
                                max((angle - self.standing_angle_deg) / 60.0, 0.0), 1.0
                            )
                        motion_quality = min(max(state.vertical_speed, 0.0), 1.0)
                        confidence = (
                            (pose_quality + geometry_quality) / 2
                            if self.view_mode == "top_down"
                            else (pose_quality + geometry_quality + motion_quality) / 3
                        )
                        event = Event(
                            EventType.FALL_DETECTED,
                            confidence=confidence,
                            source="video.fall",
                            severity=Severity.CRITICAL,
                            metadata={
                                "camera_id": self._camera_id(),
                                "view_mode": self.view_mode,
                                "track_id": track_id,
                                "duration_sec": round(elapsed, 1),
                                "posture": posture,
                                "torso_angle_deg": round(angle, 1),
                                "bbox_aspect_ratio": round(width / height, 2),
                                "top_down_elongation": round(top_down_metrics.get("elongation", 0.0), 2),
                                "top_down_body_extent": round(top_down_metrics.get("body_extent", 0.0), 2),
                                "vertical_speed": round(state.vertical_speed, 2),
                                "confirmation_sec": round(confirmation_sec, 1),
                                "detection_basis": self._detection_basis(state),
                            },
                        )
                        events.append(event)
                        if getattr(self, "event_bus", None) is not None:
                            self.event_bus.publish(event)
                        state.alerted = True
                elif state.suppressed_until_standing:
                    color = (90, 180, 90)
                    label = "ложная тревога — ожидается восстановление"
                elif state.fall_since is not None:
                    color = (0, 165, 255)
                    label = f"{posture_display}: {elapsed:.1f}/{confirmation_sec:.1f}с"
                else:
                    color = (0, 255, 0)
                    label = posture_display
                self._label(annotated, f"Состояние: {label}", int(x1), int(y1) - 35, color)

        self._mark_missing(active_track_ids)
        alarm_active = any(state.alerted for state in self._states.values())
        banner = "!!! ТРЕВОГА: ВОЗМОЖНОЕ ПАДЕНИЕ !!!" if alarm_active else "Статус: наблюдение активно"
        banner_color = (0, 0, 255) if alarm_active else (0, 255, 0)
        self._label(annotated, banner, 20, 50, banner_color)
        return FrameAnalysis(annotated_frame=annotated, events=tuple(events))
