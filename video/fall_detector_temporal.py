from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field, fields

import numpy as np

from core.event_bus import Event, EventType, Severity
from video.fall_detector_legacy import (
    FallDetector as LegacyFallDetector,
    FrameAnalysis,
    TrackState as LegacyTrackState,
    _consume_fall_dismissal,
    dismiss_fall_track,
    fixed_onnx_input_size,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PoseSample:
    t: float
    angle: float
    aspect: float
    lying: float
    sitting: float
    hip_y: float


@dataclass
class TrackState(LegacyTrackState):
    last_center_x: float | None = None
    safe_bbox_area_ema: float | None = None
    last_bbox_area: float | None = None
    footprint_growth: float = 1.0
    expansion_at: float | None = None
    stable_posture: str = "Неизвестно"
    pose_samples: deque[PoseSample] = field(default_factory=deque)
    motion_samples: deque[tuple[float, float, float, float]] = field(default_factory=deque)
    lying_score: float = 0.0
    sitting_score: float = 0.0
    sequence_state: str = "OBSERVING"
    lying_since: float | None = None
    stable_since: float | None = None
    motion_speed: float = 0.0
    last_rgb_score: float | None = None
    last_rgb_at: float | None = None


class FallDetector(LegacyFallDetector):
    """Legacy-stable pose baseline plus temporal smoothing and fall sequencing."""

    def __init__(self, *args, inference_imgsz=640, use_half_precision=True,
                 temporal_window_sec=0.75, **kwargs):
        self.inference_imgsz = min(max(320, int(inference_imgsz)), 1280)
        self.temporal_window_sec = min(max(float(temporal_window_sec), 0.35), 1.5)
        self.use_half_precision = bool(use_half_precision)
        super().__init__(*args, **kwargs)
        if model_size := fixed_onnx_input_size(self.model_path):
            if model_size != self.inference_imgsz:
                logger.warning(
                    "ONNX требует imgsz=%s; настройка imgsz=%s заменена автоматически",
                    model_size,
                    self.inference_imgsz,
                )
                self.inference_imgsz = model_size
        self.use_half_precision = bool(use_half_precision) and str(self.device).startswith("cuda")

    @staticmethod
    def _c(v):
        return min(max(float(v), 0.0), 1.0)

    @staticmethod
    def _center(k, ids, threshold):
        pts = [k[i, :2] for i in ids if float(k[i, 2]) >= threshold]
        return np.mean(pts, axis=0) if pts else None

    @staticmethod
    def _joint_angle(a, b, c):
        a, b, c = map(lambda x: np.asarray(x, dtype=float), (a, b, c))
        u, v = a - b, c - b
        d = float(np.linalg.norm(u) * np.linalg.norm(v))
        if d <= 1e-6:
            return 180.0
        return math.degrees(math.acos(min(1.0, max(-1.0, float(np.dot(u, v)) / d))))

    def _state(self, track_id):
        old = self._states.get(track_id)
        if isinstance(old, TrackState):
            return old
        state = TrackState()
        if old is not None:
            for f in fields(LegacyTrackState):
                value = getattr(old, f.name)
                setattr(state, f.name, deque(value) if f.name == "posture_history" else value)
        self._states[track_id] = state
        return state

    def _side_metrics(self, k, width, height):
        q = float(getattr(self, "keypoint_confidence", 0.5))
        width, height = max(float(width), 1.0), max(float(height), 1.0)
        sh, hip = self._center(k, (5, 6), q), self._center(k, (11, 12), q)
        knees, ankles = self._center(k, (13, 14), q), self._center(k, (15, 16), q)
        m = dict(
            torso_angle=0.0,
            aspect_ratio=width / height,
            body_axis_angle=0.0,
            leg_axis_angle=0.0,
            knee_drop_ratio=0.0,
            leg_bend_score=0.0,
            lying_score=0.0,
            sitting_score=0.0,
            end_on_lying_score=0.0,
            torso_length_ratio=0.0,
            leg_length_ratio=0.0,
            segment_imbalance=1.0,
            axis_alignment=0.0,
            pose_quality=0.0,
            hip_x=0.0,
            hip_y=0.0,
            visible_points=0.0,
            full_pose=0.0,
            partial_pose=0.0,
            partial_extent=0.0,
        )

        partial_q = min(
            q, max(float(getattr(self, "partial_pose_confidence", 0.25)), 0.05)
        )
        partial_points = np.asarray(
            [point[:2] for point in k[5:17] if float(point[2]) >= partial_q],
            dtype=np.float32,
        )
        partial_confidences = [
            float(point[2]) for point in k[5:17] if float(point[2]) >= partial_q
        ]
        m["visible_points"] = float(len(partial_points))
        if sh is None or hip is None:
            minimum = int(getattr(self, "partial_pose_min_keypoints", 3))
            if len(partial_points) < minimum:
                return m

            centered = partial_points - np.mean(partial_points, axis=0)
            covariance = centered.T @ centered
            values, vectors = np.linalg.eigh(covariance)
            direction = vectors[:, int(np.argmax(values))]
            axis_angle = math.degrees(math.atan2(abs(float(direction[0])), abs(float(direction[1]))))
            horizontal_extent = float(np.ptp(partial_points[:, 0])) / width
            quality = float(np.mean(partial_confidences)) if partial_confidences else 0.0
            extent_score = self._c((horizontal_extent - 0.45) / 0.25)
            axis_score = self._c((axis_angle - 52.0) / 28.0)
            aspect_score = self._c((m["aspect_ratio"] - 0.85) / 0.45)
            coverage_score = self._c((len(partial_points) - minimum + 1) / 5.0)
            quality_score = self._c(quality / max(q, 1e-6))
            m.update(
                body_axis_angle=axis_angle,
                pose_quality=quality,
                partial_pose=1.0,
                partial_extent=horizontal_extent,
                lying_score=self._c(
                    0.35 * axis_score
                    + 0.25 * aspect_score
                    + 0.20 * extent_score
                    + 0.10 * coverage_score
                    + 0.10 * quality_score
                ),
            )
            return m

        m["full_pose"] = 1.0
        m["torso_angle"] = self.calculate_angle(sh, hip)
        m["hip_x"], m["hip_y"] = float(hip[0]), float(hip[1])
        vis = [
            float(k[i, 2])
            for i in (5, 6, 11, 12, 13, 14, 15, 16)
            if float(k[i, 2]) >= q
        ]
        m["pose_quality"] = float(np.mean(vis)) if vis else 0.0
        lower = ankles if ankles is not None else knees
        if lower is not None:
            m["body_axis_angle"] = self.calculate_angle(sh, lower)
            m["leg_axis_angle"] = self.calculate_angle(hip, lower)

        vk = [k[i] for i in (13, 14) if float(k[i, 2]) >= q]
        if vk:
            m["knee_drop_ratio"] = max(
                0.0,
                max(float(p[1]) - float(hip[1]) for p in vk) / height,
            )

        bends = []
        for h, n, a in ((11, 13, 15), (12, 14, 16)):
            if all(float(k[i, 2]) >= q for i in (h, n, a)):
                bends.append(
                    self._c(
                        (165.0 - self._joint_angle(k[h, :2], k[n, :2], k[a, :2]))
                        / 75.0
                    )
                )
        m["leg_bend_score"] = max(bends, default=0.0)

        # A person lying along the optical axis can look vertically oriented in 2-D.
        # In that case the body is strongly foreshortened: one projected segment
        # (torso or legs) becomes much shorter than the other.  Standing and normal
        # sitting keep far more stable anatomical proportions.  This cue is scale
        # independent because every length is normalized by the person's bbox.
        if lower is not None:
            torso_vec = np.asarray(hip, dtype=np.float32) - np.asarray(sh, dtype=np.float32)
            leg_vec = np.asarray(lower, dtype=np.float32) - np.asarray(hip, dtype=np.float32)
            torso_len = float(np.linalg.norm(torso_vec)) / height
            leg_len = float(np.linalg.norm(leg_vec)) / height
            m["torso_length_ratio"] = torso_len
            m["leg_length_ratio"] = leg_len
            shortest = max(min(torso_len, leg_len), 0.04)
            imbalance = max(torso_len, leg_len) / shortest
            m["segment_imbalance"] = imbalance

            denominator = float(np.linalg.norm(torso_vec) * np.linalg.norm(leg_vec))
            alignment = (
                abs(float(np.dot(torso_vec, leg_vec))) / denominator
                if denominator > 1e-6
                else 0.0
            )
            m["axis_alignment"] = alignment

            imbalance_score = self._c((imbalance - 1.75) / 1.75)
            compact_torso = (
                self._c((0.20 - torso_len) / 0.12)
                * self._c((leg_len - 0.30) / 0.22)
            )
            compact_legs = (
                self._c((0.24 - leg_len) / 0.14)
                * self._c((torso_len - 0.18) / 0.18)
            )
            foreshortening = max(imbalance_score, compact_torso, compact_legs)
            straight_legs = 1.0 - m["leg_bend_score"]
            alignment_score = self._c((alignment - 0.70) / 0.25)
            narrow_box = 1.0 - self._c((m["aspect_ratio"] - 0.78) / 0.50)
            body_extent = self._c((torso_len + leg_len - 0.32) / 0.28)
            support = (
                0.35
                + 0.20 * straight_legs
                + 0.15 * alignment_score
                + 0.10 * narrow_box
                + 0.10 * m["pose_quality"]
                + 0.10 * body_extent
            )
            m["end_on_lying_score"] = self._c(foreshortening * support)

        horizontal = (
            0.35 * self._c((m["torso_angle"] - 38) / 37)
            + 0.25 * self._c((m["aspect_ratio"] - 0.72) / 0.63)
            + 0.20 * self._c((m["body_axis_angle"] - 38) / 37)
            + 0.10 * self._c((m["leg_axis_angle"] - 32) / 43)
            + 0.10 * m["pose_quality"]
        )
        m["lying_score"] = self._c(max(horizontal, m["end_on_lying_score"]))

        knee = self._c((m["knee_drop_ratio"] - 0.05) / 0.22)
        vertical_torso = 1 - self._c((m["torso_angle"] - 20) / 50)
        vertical_box = 1 - self._c((m["aspect_ratio"] - 0.55) / 0.70)
        sh_above = self._c((float(hip[1]) - float(sh[1])) / height / 0.25)
        sitting = (
            0.28 * knee
            + 0.30 * m["leg_bend_score"]
            + 0.20 * vertical_torso
            + 0.12 * vertical_box
            + 0.10 * sh_above
        )

        # End-on lying was the main source of false "Сидит": knees are below the
        # hips, but the legs are actually straight and the torso is foreshortened.
        # Suppress sitting evidence when perspective evidence says this is a body
        # lying into/out of the camera axis.  A seated override now needs either
        # visible leg bend or a normally projected torso.
        end_on = float(m["end_on_lying_score"])
        sitting *= 1.0 - 0.70 * end_on
        seated_geometry = self.has_seated_leg_geometry(hip, vk, height)
        credible_seated_shape = (
            m["leg_bend_score"] >= 0.08
            or (
                m["torso_length_ratio"] >= 0.22
                and m["axis_alignment"] < 0.85
            )
        )
        if (
            seated_geometry
            and m["aspect_ratio"] < 1.15
            and end_on < 0.45
            and credible_seated_shape
        ):
            sitting = max(sitting, 0.70)
        m["sitting_score"] = self._c(sitting)
        return m

    def _posture(self, m, previous="Неизвестно"):
        a, ar = m["torso_angle"], m["aspect_ratio"]
        lying, sitting = m["lying_score"], m["sitting_score"]
        end_on = float(m.get("end_on_lying_score", 0.0))

        # Perspective-first expert: lying toward/away from the camera is allowed
        # to be vertically oriented in image coordinates.  Requiring a horizontal
        # bbox here is exactly what caused the false "Сидит" from CCTV viewpoints.
        if (
            end_on >= 0.50
            and lying >= 0.50
            and m.get("leg_bend_score", 0.0) < 0.45
        ):
            return "Лежит"

        lying_votes = 0
        if ar >= 1.05:
            lying_votes += 1
        if a >= 50.0:
            lying_votes += 1
        if m.get("body_axis_angle", 0.0) >= 50.0:
            lying_votes += 1
        if lying >= 0.52:
            lying_votes += 1
        if lying_votes >= 3:
            return "Лежит"
        if (
            m.get("partial_pose", 0.0) >= 1.0
            and lying >= float(getattr(self, "partial_lie_min_score", 0.72))
            and m.get("body_axis_angle", 0.0) >= self.lying_angle_deg - 5.0
            and ar >= max(0.9, self.lying_aspect_ratio - 0.10)
        ):
            return "Лежит"
        if previous == "Стоит" and a <= self.standing_angle_deg + 5 and lying < 0.66 and sitting < 0.64:
            return "Стоит"
        if previous == "Лежит" and lying >= 0.50 and sitting < 0.70:
            return "Лежит"
        if previous == "Сидит" and sitting >= 0.48 and ar <= 1.15:
            return "Сидит"
        if sitting >= 0.68 and ar <= 1.15:
            return "Сидит"
        if self.classify_posture(a, ar, False) == "Лежит":
            return "Лежит"
        if lying >= 0.62 and m.get("body_axis_angle", 0) >= self.lying_angle_deg and a >= 50 and ar >= 0.78:
            return "Лежит"
        return "Стоит" if a < self.standing_angle_deg else "Падает"

    def classify_side_posture(self, keypoints, bbox_width, bbox_height):
        m = self._side_metrics(keypoints, bbox_width, bbox_height)
        return self._posture(m), m

    def _side_pose_is_usable(self, metrics):
        return bool(
            metrics.get("full_pose", 0.0) >= 1.0
            or (
                metrics.get("partial_pose", 0.0) >= 1.0
                and metrics.get("lying_score", 0.0)
                >= float(getattr(self, "partial_lie_min_score", 0.72))
            )
        )

    def _temporal_side_posture(self, track_id, keypoints, width, height, metrics=None):
        state = self._state(track_id)
        m = self._side_metrics(keypoints, width, height) if metrics is None else metrics
        state.pose_samples.append(
            PoseSample(
                self.current_time,
                m["torso_angle"],
                m["aspect_ratio"],
                m["lying_score"],
                m["sitting_score"],
                m["hip_y"],
            )
        )
        cutoff = self.current_time - getattr(self, "temporal_window_sec", 0.75)
        while state.pose_samples and state.pose_samples[0].t < cutoff:
            state.pose_samples.popleft()
        samples = list(state.pose_samples)
        for key, attr in (
            ("torso_angle", "angle"),
            ("aspect_ratio", "aspect"),
            ("lying_score", "lying"),
            ("sitting_score", "sitting"),
            ("hip_y", "hip_y"),
        ):
            m[key] = float(np.median([getattr(s, attr) for s in samples]))
        prev = state.stable_posture if state.stable_posture != "Неизвестно" else state.posture
        posture = self._posture(m, prev)
        state.stable_posture = posture
        state.lying_score = m["lying_score"]
        state.sitting_score = m["sitting_score"]
        return posture, m

    def _refine_top_down_posture(self, track_id, posture, metrics, bbox_area):
        state = self._state(track_id)
        score = float(metrics.get("lying_score", 0.0))
        if any(k in metrics for k in ("elongation", "body_extent", "leg_alignment")):
            score = max(
                score,
                0.45 * self._c((metrics.get("elongation", 0) - 1.4) / 1.4)
                + 0.35 * self._c((metrics.get("body_extent", 0) - 0.35) / 0.35)
                + 0.20 * self._c((metrics.get("leg_alignment", 0) - 0.30) / 0.45),
            )
        metrics["lying_score"] = score
        if state.safe_bbox_area_ema is None or bbox_area <= 0:
            return posture
        growth = bbox_area / max(state.safe_bbox_area_ema, 1.0)
        state.footprint_growth = metrics["footprint_growth"] = growth
        return "Лежит" if posture != "Лежит" and growth >= 1.40 and score >= 0.55 else posture

    def _apply_pending_dismissals(self, visible):
        self._expire_missing_suppression()
        if not isinstance(visible, dict):
            return super()._apply_pending_dismissals(set(visible))
        ids = set(visible)
        for tid, state in list(self._states.items()):
            if _consume_fall_dismissal(self._camera_id(), tid):
                self._suppress_state(state)
        missing = [
            tid
            for tid, s in self._states.items()
            if tid not in ids
            and (s.alerted or s.fall_since is not None or s.suppressed_until_standing)
        ]
        new = sorted(ids.difference(self._states))
        pairs = []
        for nid in new:
            x, y, h = visible[nid]
            for oid in missing:
                s = self._states[oid]
                ox = getattr(s, "last_center_x", None)
                if ox is None or s.last_center_y is None:
                    continue
                d = math.hypot(x - ox, y - s.last_center_y)
                if d <= max(80.0, 2 * max(h, s.last_height or h)):
                    pairs.append((d, oid, nid))
        used_old, used_new = set(), set()
        for _, oid, nid in sorted(pairs):
            if oid in used_old or nid in used_new:
                continue
            self._states[nid] = self._states.pop(oid)
            self._states[nid].missing_since = None
            used_old.add(oid)
            used_new.add(nid)

    def _update_motion(self, state, x, y, h):
        state.motion_samples.append((self.current_time, float(x), float(y), float(h)))
        cutoff = self.current_time - getattr(self, "temporal_window_sec", 0.75)
        while state.motion_samples and state.motion_samples[0][0] < cutoff:
            state.motion_samples.popleft()
        speeds = []
        motion_speeds = []
        samples = list(state.motion_samples)
        for (t0, x0, y0, h0), (t1, x1, y1, h1) in zip(samples, samples[1:]):
            if t1 > t0:
                reference_height = max((h0 + h1) / 2, 1.0)
                seconds = t1 - t0
                speeds.append((y1 - y0) / reference_height / seconds)
                motion_speeds.append(
                    math.hypot(x1 - x0, y1 - y0) / reference_height / seconds
                )
        state.vertical_speed = float(np.percentile(speeds, 75)) if speeds else 0.0
        state.motion_speed = float(np.percentile(motion_speeds, 75)) if motion_speeds else 0.0
        if state.vertical_speed >= getattr(self, "rapid_descent_heights_per_sec", float("inf")):
            state.rapid_descent_at = self.current_time

    def _person_is_stable(self, state) -> bool:
        return bool(
            state.motion_samples
            and state.motion_speed
            <= float(getattr(self, "lying_stable_motion_heights_per_sec", 0.12))
        )

    def _stable_for_fall(self, state) -> bool:
        if float(getattr(self, "lying_stable_duration_sec", 1.0)) <= 0.0:
            return True
        return bool(
            state.stable_since is not None
            and self.current_time - state.stable_since
            >= float(getattr(self, "lying_stable_duration_sec", 1.0))
        )

    def _recent_descent(self, state):
        return (
            state.rapid_descent_at is not None
            and self.current_time - state.rapid_descent_at
            <= getattr(self, "transition_memory_sec", 0.0)
        )

    def _update_state(self, track_id, posture, center_x=None, center_y=None, height=None, bbox_area=None):
        state = self._state(track_id)
        if _consume_fall_dismissal(self._camera_id(), track_id):
            self._suppress_state(state)
        state.missing_since = None
        state.posture = posture
        state.posture_history.append(posture)
        while len(state.posture_history) > max(3, int(getattr(self, "posture_window", 5))):
            state.posture_history.popleft()
        if center_x is not None:
            state.last_center_x = float(center_x)
        if center_y is not None and height is not None and height > 0:
            self._update_motion(
                state,
                state.last_center_x if state.last_center_x is not None else 0.0,
                center_y,
                height,
            )
            state.last_center_y = float(center_y)
            state.last_height = float(height)
            state.last_observed_at = self.current_time
        if bbox_area is not None and bbox_area > 0:
            state.last_bbox_area = float(bbox_area)
        if posture in {"Стоит", "Сидит"}:
            state.lying_since = None
            state.stable_since = None
            if bbox_area:
                state.safe_bbox_area_ema = (
                    float(bbox_area)
                    if state.safe_bbox_area_ema is None
                    else 0.92 * state.safe_bbox_area_ema + 0.08 * float(bbox_area)
                )
            if posture == "Стоит":
                state.seen_upright = True
                state.sequence_state = "STANDING"
            else:
                state.sequence_state = "SITTING"
            state.transition_since = None
            if state.fall_since is not None or state.alerted or state.suppressed_until_standing:
                state.standing_since = state.standing_since or self.current_time
                if self.current_time - state.standing_since >= self.reset_grace_sec:
                    was = state.alerted
                    state.fall_since = None
                    state.standing_since = None
                    state.alerted = False
                    state.suppressed_until_standing = False
                    state.recovered_pending = was
            return state
        state.standing_since = None
        if state.suppressed_until_standing:
            state.fall_since = None
            state.lying_since = None
            state.stable_since = None
            return state
        descent = self._recent_descent(state)
        if posture == "Падает":
            state.lying_since = None
            state.stable_since = None
            state.sequence_state = "POSSIBLE_FALL" if descent else "OBSERVING"
            if descent and state.transition_since is None:
                state.transition_since = self.current_time
            return state
        if posture == "Неизвестно":
            state.fall_since = None
            state.lying_since = None
            state.stable_since = None
            state.sequence_state = "OBSERVING"
            return state
        if posture != "Лежит":
            return state
        require_upright = bool(getattr(self, "require_upright_transition", False))
        require_motion = bool(getattr(self, "require_fall_motion", False))
        if getattr(self, "view_mode", "side") == "top_down":
            require_motion = False
        if (require_upright and not state.seen_upright) or (require_motion and not descent):
            state.sequence_state = "LYING_NO_TRANSITION"
            state.fall_since = None
            state.lying_since = None
            state.stable_since = None
            return state
        state.sequence_state = "LYING_CONFIRMATION"
        if state.lying_since is None:
            recent_transition = (
                state.transition_since is not None
                and self.current_time - state.transition_since
                <= getattr(self, "transition_memory_sec", 0.0)
            )
            state.lying_since = state.transition_since if recent_transition else self.current_time
        state.fall_since = state.lying_since
        if self._person_is_stable(state):
            if state.stable_since is None:
                state.stable_since = self.current_time
        else:
            state.stable_since = None
        return state

    def _required_confirmation_sec(self, state):
        return float(getattr(self, "fall_duration_sec", 5.0))

    def _detection_basis(self, state):
        if self._recent_descent(state):
            return "rapid_descent"
        if state.seen_upright or state.transition_since is not None:
            return "upright_to_lying"
        return "prolonged_lying"

    def process_frame(self, frame):
        self._tick()
        kwargs = dict(
            persist=True,
            conf=self.pose_confidence,
            iou=0.3,
            device=self.device,
            imgsz=self.inference_imgsz,
            verbose=False,
        )
        if self.use_half_precision and str(self.device).startswith("cuda"):
            kwargs["half"] = True
        elif self.use_half_precision:
            kwargs["quantize"] = 16
        result = self.model.track(frame, **kwargs)[0]
        annotated = result.plot(kpt_line=True, kpt_radius=4)
        active, events = set(), []
        if result.keypoints is not None and result.boxes is not None and result.boxes.id is not None:
            ks = result.keypoints.data.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.int().cpu().tolist()
            tids = result.boxes.id.int().cpu().tolist()
            visible = {
                tid: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, max(float(b[3] - b[1]), 1.0))
                for tid, c, b in zip(tids, classes, boxes)
                if c == 0
            }
            self._apply_pending_dismissals(visible)
            for i, k in enumerate(ks):
                if classes[i] != 0:
                    continue
                tid = tids[i]
                active.add(tid)
                x1, y1, x2, y2 = boxes[i]
                w = max(float(x2 - x1), 1.0)
                h = max(float(y2 - y1), 1.0)
                area = w * h
                pts = [k[5], k[6], k[11], k[12]]
                pose_visible = (
                    sum(float(p[2]) >= self.keypoint_confidence for p in k[5:17]) >= 5
                    if self.view_mode == "top_down"
                    else not any(float(p[2]) < self.keypoint_confidence for p in pts)
                )
                if not pose_visible:
                    s = self._state(tid)
                    s.missing_since = s.missing_since or self.current_time
                    self._label(
                        annotated,
                        "Состояние: не видно позу",
                        int(x1),
                        int(y1) - 35,
                        (0, 165, 255),
                    )
                    continue
                ls, rs, lh, rh = pts
                sh = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
                hip = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
                angle = self.calculate_angle(sh, hip)
                side = {}
                top = {}
                if self.view_mode == "top_down":
                    posture, top = self.classify_top_down_posture(k, w, h)
                    posture = self._refine_top_down_posture(tid, posture, top, area)
                    state = self._update_state(tid, posture, center_x=(x1 + x2) / 2, bbox_area=area)
                else:
                    posture, side = self._temporal_side_posture(tid, k, w, h)
                    angle = side["torso_angle"]
                    state = self._update_state(
                        tid,
                        posture,
                        center_x=side["hip_x"],
                        center_y=side["hip_y"],
                        height=h,
                        bbox_area=area,
                    )
                if state.recovered_pending:
                    e = Event(
                        EventType.FALL_RECOVERED,
                        confidence=1.0,
                        source="video.fall",
                        severity=Severity.INFO,
                        metadata={
                            "camera_id": self._camera_id(),
                            "track_id": tid,
                            "posture": posture,
                        },
                    )
                    events.append(e)
                    state.recovered_pending = False
                    if getattr(self, "event_bus", None) is not None:
                        self.event_bus.publish(e)
                elapsed = 0.0 if state.fall_since is None else self.current_time - state.fall_since
                confirm = self._required_confirmation_sec(state)
                if state.fall_since is not None and elapsed >= confirm:
                    color, label = (0, 0, 255), "ТРЕВОГА: возможное падение"
                    if not state.alerted:
                        pq = float(np.mean([p[2] for p in pts]))
                        geom = float((side or top).get("lying_score", 0.0))
                        motion = self._c(max(state.vertical_speed, 0.0))
                        e = Event(
                            EventType.FALL_DETECTED,
                            confidence=self._c(0.45 * pq + 0.40 * geom + 0.15 * motion),
                            source="video.fall",
                            severity=Severity.CRITICAL,
                            metadata={
                                "camera_id": self._camera_id(),
                                "view_mode": self.view_mode,
                                "track_id": tid,
                                "duration_sec": round(elapsed, 1),
                                "posture": posture,
                                "torso_angle_deg": round(angle, 1),
                                "bbox_aspect_ratio": round(w / h, 2),
                                "lying_score": round(float((side or top).get("lying_score", 0)), 3),
                                "sitting_score": round(float(side.get("sitting_score", 0)), 3),
                                "vertical_speed": round(state.vertical_speed, 2),
                                "confirmation_sec": round(confirm, 1),
                                "detection_basis": self._detection_basis(state),
                                "sequence_state": state.sequence_state,
                                "temporal_window_sec": round(self.temporal_window_sec, 2),
                            },
                        )
                        events.append(e)
                        state.alerted = True
                        state.sequence_state = "FALL_CONFIRMED"
                        if getattr(self, "event_bus", None) is not None:
                            self.event_bus.publish(e)
                elif state.suppressed_until_standing:
                    color, label = (90, 180, 90), "ложная тревога — ожидается восстановление"
                elif state.fall_since is not None:
                    color, label = (0, 165, 255), f"{posture}: {elapsed:.1f}/{confirm:.1f}с"
                else:
                    color, label = (0, 255, 0), posture
                self._label(
                    annotated,
                    f"Состояние: {label}",
                    int(x1),
                    int(y1) - 35,
                    color,
                )
        self._mark_missing(active)
        alarm = any(s.alerted for s in self._states.values())
        self._label(
            annotated,
            "!!! ТРЕВОГА: ВОЗМОЖНОЕ ПАДЕНИЕ !!!" if alarm else "Статус: наблюдение активно",
            20,
            50,
            (0, 0, 255) if alarm else (0, 255, 0),
        )
        return FrameAnalysis(annotated_frame=annotated, events=tuple(events))


__all__ = ["FallDetector", "FrameAnalysis", "TrackState", "PoseSample", "dismiss_fall_track"]
