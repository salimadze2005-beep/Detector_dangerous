from __future__ import annotations

import math
import logging

import numpy as np

from core.event_bus import Event, EventType, Severity
from video.fall_detector_temporal import (
    FallDetector as _TemporalFallDetector,
    FrameAnalysis,
    PoseSample,
    TrackState,
    dismiss_fall_track,
)
from video.lying_classifier import LyingClassifier

logger = logging.getLogger(__name__)


class FallDetector(_TemporalFallDetector):
    """Pose detector with robust side-view posture and a direct lying timer.

    Both camera modes keep processing detections even when ByteTrack temporarily
    does not provide IDs. Side mode accepts a geometrically coherent partial
    skeleton, and every confirmed ``Лежит`` state starts the same alert timer
    that the operator sees on screen.
    """

    def __init__(
        self,
        *args,
        lying_classifier_enabled=False,
        lying_classifier_model_path="models/lying_classifier.pt",
        lying_classifier_label="lying",
        lying_classifier_imgsz=224,
        lying_classifier_hz=5.0,
        lying_classifier_threshold=0.75,
        lying_alert_score_threshold=0.55,
        lying_classifier_pose_quality_min=0.45,
        lying_classifier_rgb_weight=0.65,
        lying_stable_duration_sec=1.0,
        lying_stable_motion_heights_per_sec=0.12,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lying_classifier_hz = min(max(float(lying_classifier_hz), 0.2), 30.0)
        self.lying_classifier_threshold = min(
            max(float(lying_classifier_threshold), 0.5), 0.99
        )
        # Kept for config compatibility. The visible posture decision is now the
        # single source of truth for starting/stopping the lying timer.
        self.lying_alert_score_threshold = min(
            max(float(lying_alert_score_threshold), 0.0), 1.0
        )
        self.lying_classifier_pose_quality_min = min(
            max(float(lying_classifier_pose_quality_min), 0.0), 1.0
        )
        self.lying_classifier_rgb_weight = min(
            max(float(lying_classifier_rgb_weight), 0.0), 1.0
        )
        self.lying_stable_duration_sec = min(
            max(float(lying_stable_duration_sec), 0.0), 10.0
        )
        self.lying_stable_motion_heights_per_sec = min(
            max(float(lying_stable_motion_heights_per_sec), 0.01), 2.0
        )
        self.lying_classifier = LyingClassifier(
            lying_classifier_model_path,
            enabled=lying_classifier_enabled,
            device=self.device,
            label=lying_classifier_label,
            imgsz=lying_classifier_imgsz,
        )

    def _rgb_lying_score(self, frame, bbox, state) -> float | None:
        classifier = getattr(self, "lying_classifier", None)
        if classifier is None or not classifier.available:
            return None
        interval = 1.0 / max(float(self.lying_classifier_hz), 0.2)
        if (
            state.last_rgb_at is not None
            and self.current_time - state.last_rgb_at < interval
        ):
            return state.last_rgb_score
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        left, top = max(0, int(math.floor(x1))), max(0, int(math.floor(y1)))
        right, bottom = min(width, int(math.ceil(x2))), min(height, int(math.ceil(y2)))
        state.last_rgb_at = self.current_time
        state.last_rgb_score = classifier.score(frame[top:bottom, left:right])
        return state.last_rgb_score

    def _fuse_side_lying_scores(self, pose_posture, metrics, rgb_score):
        """Fuse reliable Pose geometry with camera-specific RGB evidence."""
        pose_score = float(metrics.get("lying_score", 0.0))
        pose_quality = float(metrics.get("pose_quality", 0.0))
        pose_good = bool(metrics.get("full_pose", 0.0)) and (
            pose_quality >= self.lying_classifier_pose_quality_min
        )
        if rgb_score is None:
            fused_score = pose_score if pose_good else None
        elif pose_good:
            fused_score = max(
                pose_score,
                self.lying_classifier_rgb_weight * float(rgb_score)
                + (1.0 - self.lying_classifier_rgb_weight) * pose_score,
            )
        else:
            fused_score = float(rgb_score)
        metrics.update(
            pose_lying_score=pose_score,
            rgb_lying_score=rgb_score,
            fused_lying_score=fused_score,
            pose_good=float(pose_good),
        )
        if fused_score is not None:
            metrics["lying_score"] = float(fused_score)
        if fused_score is not None and fused_score >= self.lying_classifier_threshold:
            return "Лежит"
        return pose_posture if pose_good else "Неизвестно"

    def _timer_posture(self, posture, metrics):
        """Use exactly the same posture for UI and the alert state machine."""
        del metrics
        return posture

    @staticmethod
    def _resolve_track_ids(boxes_id, count: int) -> list[int]:
        if boxes_id is None:
            return [-(index + 1) for index in range(count)]
        return boxes_id.int().cpu().tolist()

    def _update_state(
        self,
        track_id,
        posture,
        center_x=None,
        center_y=None,
        height=None,
        bbox_area=None,
    ):
        # The legacy temporal layer still tracks descent/upright history for
        # metadata and confidence, but this branch deliberately does not require
        # that history before starting a prolonged-lying alarm timer.
        previous = self._state(track_id)
        previous_lying_since = previous.lying_since
        state = super()._update_state(
            track_id,
            posture,
            center_x=center_x,
            center_y=center_y,
            height=height,
            bbox_area=bbox_area,
        )

        if posture == "Лежит" and not state.suppressed_until_standing:
            state.lying_since = (
                previous_lying_since
                if previous_lying_since is not None
                else self.current_time
            )
            state.fall_since = state.lying_since
            state.sequence_state = "LYING_CONFIRMATION"
            if self._person_is_stable(state):
                if state.stable_since is None:
                    state.stable_since = self.current_time
            else:
                state.stable_since = None
        return state

    def _detection_basis(self, state):
        return super()._detection_basis(state)

    def classify_top_down_posture(self, keypoints, bbox_width, bbox_height):
        """Rotation-invariant posture classification for steep/overhead views."""
        threshold = min(float(getattr(self, "keypoint_confidence", 0.5)), 0.35)
        body_indices = (5, 6, 11, 12, 13, 14, 15, 16)
        visible = np.asarray(
            [
                keypoints[index, :2]
                for index in body_indices
                if float(keypoints[index, 2]) >= threshold
            ],
            dtype=np.float32,
        )
        metrics = {
            "visible_points": float(len(visible)),
            "elongation": 0.0,
            "body_extent": 0.0,
            "leg_alignment": 0.0,
            "lying_score": 0.0,
        }
        if len(visible) < 5 or bbox_width <= 0 or bbox_height <= 0:
            return "Стоит", metrics

        covariance = np.cov(visible, rowvar=False)
        eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 1e-6))
        elongation = float(math.sqrt(eigenvalues[-1] / eigenvalues[0]))
        body_span = float(np.linalg.norm(np.ptp(visible, axis=0)))
        bbox_diagonal = math.hypot(float(bbox_width), float(bbox_height))
        extent = body_span / max(bbox_diagonal, 1e-6)
        metrics.update({"elongation": elongation, "body_extent": extent})

        def center(indices):
            points = [
                keypoints[index, :2]
                for index in indices
                if float(keypoints[index, 2]) >= threshold
            ]
            return np.mean(points, axis=0) if points else None

        shoulders = center((5, 6))
        hips = center((11, 12))
        legs = center((13, 14, 15, 16))
        alignment = 0.0
        if shoulders is not None and hips is not None and legs is not None:
            torso = hips - shoulders
            lower_body = legs - hips
            denominator = float(np.linalg.norm(torso) * np.linalg.norm(lower_body))
            if denominator > 1e-6:
                alignment = abs(float(np.dot(torso, lower_body))) / denominator
        metrics["leg_alignment"] = alignment

        lying_score = sum(
            (
                self._c((elongation - 1.45) / 1.10),
                self._c((extent - 0.38) / 0.24),
                self._c((alignment - 0.25) / 0.45),
            )
        ) / 3.0
        metrics["lying_score"] = lying_score

        if alignment < 0.35 and extent < 0.65:
            return "Сидит", metrics
        if elongation >= 2.0 and extent >= 0.50 and alignment >= 0.45:
            return "Лежит", metrics
        if lying_score >= 0.55:
            return "Падает", metrics
        return "Стоит", metrics

    def _publish_recovery_if_needed(self, state, track_id, posture, events):
        if not state.recovered_pending:
            return
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
        state.recovered_pending = False
        if getattr(self, "event_bus", None) is not None:
            self.event_bus.publish(recovery)

    def _side_event_and_label(
        self,
        keypoints,
        state,
        track_id,
        posture,
        side_metrics,
        width,
        height,
        events,
    ):
        elapsed = (
            0.0 if state.fall_since is None else self.current_time - state.fall_since
        )
        confirmation_sec = self._required_confirmation_sec(state)
        if state.fall_since is not None and elapsed >= confirmation_sec:
            color = (0, 0, 255)
            label = "ТРЕВОГА: возможное падение"
            if not state.alerted:
                visible_confidences = [
                    float(keypoints[index, 2])
                    for index in (5, 6, 11, 12, 13, 14, 15, 16)
                    if float(keypoints[index, 2]) >= self.keypoint_confidence
                ]
                pose_quality = (
                    float(np.mean(visible_confidences))
                    if visible_confidences
                    else 0.0
                )
                geometry_quality = float(side_metrics.get("lying_score", 0.0))
                rgb_quality = side_metrics.get("rgb_lying_score")
                motion_quality = self._c(max(state.vertical_speed, 0.0))
                persistence_quality = self._c(
                    elapsed / max(confirmation_sec, 1e-6)
                )
                temporal_quality = max(motion_quality, persistence_quality)
                if rgb_quality is None:
                    confidence = self._c(
                        0.35 * pose_quality
                        + 0.40 * geometry_quality
                        + 0.25 * temporal_quality
                    )
                else:
                    rgb_quality = float(rgb_quality)
                    confidence = self._c(
                        0.15 * pose_quality
                        + 0.45 * geometry_quality
                        + 0.20 * rgb_quality
                        + 0.20 * temporal_quality
                    )
                event = Event(
                    EventType.FALL_DETECTED,
                    confidence=confidence,
                    source="video.fall",
                    severity=Severity.CRITICAL,
                    metadata={
                        "camera_id": self._camera_id(),
                        "view_mode": "side",
                        "track_id": track_id,
                        "duration_sec": round(elapsed, 1),
                        "posture": posture,
                        "torso_angle_deg": round(
                            float(side_metrics.get("torso_angle", 0.0)), 1
                        ),
                        "bbox_aspect_ratio": round(width / height, 2),
                        "lying_score": round(
                            float(side_metrics.get("lying_score", 0.0)), 3
                        ),
                        "lying_alert_score_threshold": round(
                            self.lying_alert_score_threshold, 2
                        ),
                        "pose_lying_score": round(
                            float(side_metrics.get("pose_lying_score", 0.0)), 3
                        ),
                        "rgb_lying_score": (
                            None
                            if side_metrics.get("rgb_lying_score") is None
                            else round(float(side_metrics["rgb_lying_score"]), 3)
                        ),
                        "pose_layout": (
                            "partial" if side_metrics.get("partial_pose", 0.0) else "full"
                        ),
                        "sitting_score": round(
                            float(side_metrics.get("sitting_score", 0.0)), 3
                        ),
                        "vertical_speed": round(state.vertical_speed, 2),
                        "persistence_quality": round(persistence_quality, 3),
                        "confirmation_sec": round(confirmation_sec, 1),
                        "detection_basis": self._detection_basis(state),
                        "sequence_state": state.sequence_state,
                        "temporal_window_sec": round(self.temporal_window_sec, 2),
                        "lying_since": (
                            None
                            if state.lying_since is None
                            else round(state.lying_since, 2)
                        ),
                        "stable_since": (
                            None
                            if state.stable_since is None
                            else round(state.stable_since, 2)
                        ),
                        "motion_speed": round(state.motion_speed, 3),
                    },
                )
                if confidence >= getattr(self, "fall_min_confidence", 0.65):
                    events.append(event)
                    state.alerted = True
                    state.sequence_state = "FALL_CONFIRMED"
                    if getattr(self, "event_bus", None) is not None:
                        self.event_bus.publish(event)
                else:
                    color = (0, 165, 255)
                    label = f"падение не подтверждено: {confidence:.0%}"
        elif state.suppressed_until_standing:
            color = (90, 180, 90)
            label = "ложная тревога — ожидается восстановление"
        elif state.fall_since is not None:
            color = (0, 165, 255)
            posture_label = (
                "Лежит (частичная поза)"
                if posture == "Лежит" and side_metrics.get("partial_pose", 0.0)
                else posture
            )
            label = f"{posture_label}: {elapsed:.1f}/{confirmation_sec:.1f}с"
        else:
            color = (0, 255, 0)
            label = (
                "Лежит (частичная поза)"
                if posture == "Лежит" and side_metrics.get("partial_pose", 0.0)
                else posture
            )
        return color, label

    def _process_side_frame(self, frame):
        self._tick()
        kwargs = dict(
            persist=True,
            conf=float(self.pose_confidence),
            iou=0.3,
            device=self.device,
            imgsz=self.inference_imgsz,
            verbose=False,
        )
        if self.use_half_precision:
            kwargs["quantize"] = 16

        result = self.model.track(frame, **kwargs)[0]
        annotated = result.plot(kpt_line=True, kpt_radius=4)
        active_track_ids = set()
        events = []

        if result.keypoints is not None and result.boxes is not None:
            keypoints_batch = result.keypoints.data.cpu().numpy()
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.int().cpu().tolist()
            track_ids = self._resolve_track_ids(result.boxes.id, len(boxes_xyxy))
            visible_people = {
                track_ids[index]: (
                    float((boxes_xyxy[index][0] + boxes_xyxy[index][2]) / 2.0),
                    float((boxes_xyxy[index][1] + boxes_xyxy[index][3]) / 2.0),
                    max(float(boxes_xyxy[index][3] - boxes_xyxy[index][1]), 1.0),
                )
                for index, class_id in enumerate(classes)
                if class_id == 0
            }
            self._apply_pending_dismissals(visible_people)

            for index, keypoints in enumerate(keypoints_batch):
                if classes[index] != 0:
                    continue
                track_id = track_ids[index]
                active_track_ids.add(track_id)
                x1, y1, x2, y2 = boxes_xyxy[index]
                width = max(float(x2 - x1), 1.0)
                height = max(float(y2 - y1), 1.0)
                bbox_area = width * height

                side_metrics = self._side_metrics(keypoints, width, height)
                shoulders = self._center(
                    keypoints, (5, 6), float(self.keypoint_confidence)
                )
                hips = self._center(
                    keypoints, (11, 12), float(self.keypoint_confidence)
                )
                pose_usable = self._side_pose_is_usable(side_metrics)
                if pose_usable:
                    pose_posture, side_metrics = self._temporal_side_posture(
                        track_id, keypoints, width, height, metrics=side_metrics
                    )
                else:
                    pose_posture = "Неизвестно"
                state = self._state(track_id)
                rgb_score = self._rgb_lying_score(
                    frame, (x1, y1, x2, y2), state
                )
                posture = self._fuse_side_lying_scores(
                    pose_posture, side_metrics, rgb_score
                )
                if posture == "Неизвестно" and rgb_score is None and not pose_usable:
                    state.missing_since = state.missing_since or self.current_time
                    self._label(
                        annotated,
                        "Состояние: нет Pose и RGB-классификатора",
                        int(x1),
                        int(y1) - 35,
                        (0, 165, 255),
                    )
                    continue
                state = self._update_state(
                    track_id,
                    self._timer_posture(posture, side_metrics),
                    center_x=(
                        float(hips[0])
                        if hips is not None
                        else float((x1 + x2) / 2.0)
                    ),
                    center_y=(
                        float(hips[1])
                        if hips is not None
                        else float((y1 + y2) / 2.0)
                    ),
                    height=height,
                    bbox_area=bbox_area,
                )
                self._publish_recovery_if_needed(
                    state, track_id, posture, events
                )
                color, label = self._side_event_and_label(
                    keypoints,
                    state,
                    track_id,
                    posture,
                    side_metrics,
                    width,
                    height,
                    events,
                )
                self._label(
                    annotated,
                    f"Состояние: {label}",
                    int(x1),
                    int(y1) - 35,
                    color,
                )

        self._mark_missing(active_track_ids)
        alarm_active = any(state.alerted for state in self._states.values())
        self._label(
            annotated,
            "!!! ТРЕВОГА: ВОЗМОЖНОЕ ПАДЕНИЕ !!!"
            if alarm_active
            else "Статус: наблюдение активно",
            20,
            50,
            (0, 0, 255) if alarm_active else (0, 255, 0),
        )
        return FrameAnalysis(annotated_frame=annotated, events=tuple(events))

    def _process_top_down_frame(self, frame):
        self._tick()
        kwargs = dict(
            persist=True,
            conf=min(float(self.pose_confidence), 0.15),
            iou=0.3,
            device=self.device,
            imgsz=self.inference_imgsz,
            verbose=False,
        )
        if self.use_half_precision:
            kwargs["quantize"] = 16

        result = self.model.track(frame, **kwargs)[0]
        annotated = result.plot(kpt_line=True, kpt_radius=4)
        active_track_ids = set()
        events = []

        if result.keypoints is not None and result.boxes is not None:
            keypoints_batch = result.keypoints.data.cpu().numpy()
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.int().cpu().tolist()
            track_ids = self._resolve_track_ids(result.boxes.id, len(boxes_xyxy))
            visible_people = {
                track_ids[index]: (
                    float((boxes_xyxy[index][0] + boxes_xyxy[index][2]) / 2.0),
                    float((boxes_xyxy[index][1] + boxes_xyxy[index][3]) / 2.0),
                    max(float(boxes_xyxy[index][3] - boxes_xyxy[index][1]), 1.0),
                )
                for index, class_id in enumerate(classes)
                if class_id == 0
            }
            self._apply_pending_dismissals(visible_people)

            visibility_threshold = min(float(self.keypoint_confidence), 0.35)
            for index, keypoints in enumerate(keypoints_batch):
                if classes[index] != 0:
                    continue
                track_id = track_ids[index]
                active_track_ids.add(track_id)
                x1, y1, x2, y2 = boxes_xyxy[index]
                width = max(float(x2 - x1), 1.0)
                height = max(float(y2 - y1), 1.0)
                bbox_area = width * height

                pose_is_visible = sum(
                    float(point[2]) >= visibility_threshold
                    for point in keypoints[5:17]
                ) >= 5
                if not pose_is_visible:
                    state = self._state(track_id)
                    state.missing_since = state.missing_since or self.current_time
                    self._label(
                        annotated,
                        "Состояние: не видно позу",
                        int(x1),
                        int(y1) - 35,
                        (0, 165, 255),
                    )
                    continue

                posture, metrics = self.classify_top_down_posture(
                    keypoints, width, height
                )
                posture = self._refine_top_down_posture(
                    track_id, posture, metrics, bbox_area
                )
                state = self._update_state(
                    track_id,
                    posture,
                    center_x=float((x1 + x2) / 2.0),
                    bbox_area=bbox_area,
                )
                self._publish_recovery_if_needed(
                    state, track_id, posture, events
                )

                elapsed = (
                    0.0
                    if state.fall_since is None
                    else self.current_time - state.fall_since
                )
                confirmation_sec = self._required_confirmation_sec(state)
                if state.fall_since is not None and elapsed >= confirmation_sec:
                    color = (0, 0, 255)
                    label = "ТРЕВОГА: возможное падение"
                    if not state.alerted:
                        visible_confidences = [
                            float(keypoints[i, 2])
                            for i in (5, 6, 11, 12, 13, 14, 15, 16)
                            if float(keypoints[i, 2]) >= visibility_threshold
                        ]
                        pose_quality = (
                            float(np.mean(visible_confidences))
                            if visible_confidences
                            else 0.0
                        )
                        geometry_quality = float(metrics.get("lying_score", 0.0))
                        persistence_quality = self._c(
                            elapsed / max(confirmation_sec, 1e-6)
                        )
                        confidence = self._c(
                            0.35 * pose_quality
                            + 0.40 * geometry_quality
                            + 0.25 * persistence_quality
                        )
                        event = Event(
                            EventType.FALL_DETECTED,
                            confidence=confidence,
                            source="video.fall",
                            severity=Severity.CRITICAL,
                            metadata={
                                "camera_id": self._camera_id(),
                                "view_mode": "top_down",
                                "track_id": track_id,
                                "duration_sec": round(elapsed, 1),
                                "posture": posture,
                                "top_down_elongation": round(
                                    float(metrics.get("elongation", 0.0)), 2
                                ),
                                "top_down_body_extent": round(
                                    float(metrics.get("body_extent", 0.0)), 2
                                ),
                                "top_down_leg_alignment": round(
                                    float(metrics.get("leg_alignment", 0.0)), 2
                                ),
                                "lying_score": round(
                                    float(metrics.get("lying_score", 0.0)), 3
                                ),
                                "persistence_quality": round(
                                    persistence_quality, 3
                                ),
                                "confirmation_sec": round(confirmation_sec, 1),
                                "detection_basis": self._detection_basis(state),
                                "sequence_state": state.sequence_state,
                            },
                        )
                        if confidence >= getattr(self, "fall_min_confidence", 0.65):
                            events.append(event)
                            state.alerted = True
                            state.sequence_state = "FALL_CONFIRMED"
                            if getattr(self, "event_bus", None) is not None:
                                self.event_bus.publish(event)
                        else:
                            color = (0, 165, 255)
                            label = f"падение не подтверждено: {confidence:.0%}"
                elif state.suppressed_until_standing:
                    color = (90, 180, 90)
                    label = "ложная тревога — ожидается восстановление"
                elif state.fall_since is not None:
                    color = (0, 165, 255)
                    label = f"{posture}: {elapsed:.1f}/{confirmation_sec:.1f}с"
                else:
                    color = (0, 255, 0)
                    label = posture

                self._label(
                    annotated,
                    f"Состояние: {label}",
                    int(x1),
                    int(y1) - 35,
                    color,
                )

        self._mark_missing(active_track_ids)
        alarm_active = any(state.alerted for state in self._states.values())
        self._label(
            annotated,
            "!!! ТРЕВОГА: ВОЗМОЖНОЕ ПАДЕНИЕ !!!"
            if alarm_active
            else "Статус: наблюдение активно",
            20,
            50,
            (0, 0, 255) if alarm_active else (0, 255, 0),
        )
        return FrameAnalysis(annotated_frame=annotated, events=tuple(events))

    def process_frame(self, frame):
        if getattr(self, "view_mode", "side") == "top_down":
            return self._process_top_down_frame(frame)
        return self._process_side_frame(frame)


__all__ = [
    "FallDetector",
    "FrameAnalysis",
    "TrackState",
    "PoseSample",
    "dismiss_fall_track",
]
