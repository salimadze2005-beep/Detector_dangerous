from __future__ import annotations

"""Enhanced runtime wrapper for the branch fall detector.

The branch already contains a large side-view implementation in
``video/fall_detector.py``.  Keeping that implementation untouched avoids
regressions in the side mode while this package overrides only the top-down
posture expert.  Python prefers this package over the sibling module for
``import video.fall_detector``, so all existing imports keep working.
"""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np


_ORIGINAL_NAME = "video._fall_detector_branch_impl"
_ORIGINAL_PATH = Path(__file__).resolve().parent.parent / "fall_detector.py"

_original = sys.modules.get(_ORIGINAL_NAME)
if _original is None:
    spec = importlib.util.spec_from_file_location(_ORIGINAL_NAME, _ORIGINAL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Не удалось загрузить базовый fall detector: {_ORIGINAL_PATH}")
    _original = importlib.util.module_from_spec(spec)
    sys.modules[_ORIGINAL_NAME] = _original
    spec.loader.exec_module(_original)


class FallDetector(_original.FallDetector):
    """Branch detector with a perspective-aware top-down posture expert."""

    @staticmethod
    def _top_down_center(keypoints, indices, threshold):
        points = [
            keypoints[index, :2]
            for index in indices
            if float(keypoints[index, 2]) >= threshold
        ]
        return np.mean(points, axis=0) if points else None

    def classify_top_down_posture(self, keypoints, bbox_width, bbox_height):
        """Classify overhead/steep-camera poses without depending on image angle.

        The strongest cue is projected anatomical length relative to shoulder/
        hip width.  A standing person's vertical body segments are foreshortened
        by a steep camera, while a body lying on the floor is projected almost in
        full.  Knee bend then separates sitting from a compact standing pose.
        """
        threshold = min(float(getattr(self, "keypoint_confidence", 0.5)), 0.35)
        bbox_width = max(float(bbox_width), 1.0)
        bbox_height = max(float(bbox_height), 1.0)
        bbox_diagonal = math.hypot(bbox_width, bbox_height)
        body_indices = (5, 6, 11, 12, 13, 14, 15, 16)

        visible_indices = [
            index
            for index in body_indices
            if float(keypoints[index, 2]) >= threshold
        ]
        visible = np.asarray(
            [keypoints[index, :2] for index in visible_indices],
            dtype=np.float32,
        )
        confidences = [float(keypoints[index, 2]) for index in visible_indices]
        metrics = {
            "visible_points": float(len(visible)),
            "pose_quality": float(np.mean(confidences)) if confidences else 0.0,
            "elongation": 0.0,
            "body_extent": 0.0,
            "body_span_ratio": 0.0,
            "body_width_px": 0.0,
            "torso_ratio": 0.0,
            "leg_ratio": 0.0,
            "axial_ratio": 0.0,
            "leg_alignment": 0.0,
            "leg_bend_score": 0.0,
            "lying_score": 0.0,
            "sitting_score": 0.0,
            "standing_score": 0.0,
            "footprint_growth": 1.0,
        }
        if len(visible) < 5:
            return "Неизвестно", metrics

        centered = visible - np.mean(visible, axis=0)
        covariance = centered.T @ centered / max(len(visible), 1)
        eigenvalues = np.sort(
            np.maximum(np.linalg.eigvalsh(covariance), 1e-6)
        )
        elongation = float(math.sqrt(eigenvalues[-1] / eigenvalues[0]))
        body_span = float(np.linalg.norm(np.ptp(visible, axis=0)))
        body_extent = body_span / max(bbox_diagonal, 1e-6)
        metrics["elongation"] = elongation
        metrics["body_extent"] = body_extent

        shoulders = self._top_down_center(keypoints, (5, 6), threshold)
        hips = self._top_down_center(keypoints, (11, 12), threshold)
        knees = self._top_down_center(keypoints, (13, 14), threshold)
        ankles = self._top_down_center(keypoints, (15, 16), threshold)
        lower = ankles if ankles is not None else knees

        anatomical_widths = []
        for left, right in ((5, 6), (11, 12)):
            if (
                float(keypoints[left, 2]) >= threshold
                and float(keypoints[right, 2]) >= threshold
            ):
                anatomical_widths.append(
                    float(
                        np.linalg.norm(
                            keypoints[left, :2] - keypoints[right, :2]
                        )
                    )
                )
        # A small bbox-derived floor keeps noisy/overlapping shoulder points from
        # exploding ratios, while normal cases use actual anatomical width.
        width_scale = max(
            float(np.median(anatomical_widths)) if anatomical_widths else 0.0,
            0.08 * bbox_diagonal,
            1.0,
        )
        metrics["body_width_px"] = width_scale
        metrics["body_span_ratio"] = body_span / width_scale

        torso_length = 0.0
        leg_length = 0.0
        alignment = 0.0
        if shoulders is not None and hips is not None:
            torso_vector = np.asarray(hips) - np.asarray(shoulders)
            torso_length = float(np.linalg.norm(torso_vector))
        else:
            torso_vector = None
        if hips is not None and lower is not None:
            leg_vector = np.asarray(lower) - np.asarray(hips)
            leg_length = float(np.linalg.norm(leg_vector))
        else:
            leg_vector = None
        if torso_vector is not None and leg_vector is not None:
            denominator = float(
                np.linalg.norm(torso_vector) * np.linalg.norm(leg_vector)
            )
            if denominator > 1e-6:
                alignment = abs(
                    float(np.dot(torso_vector, leg_vector)) / denominator
                )

        torso_ratio = torso_length / width_scale
        leg_ratio = leg_length / width_scale
        axial_ratio = torso_ratio + leg_ratio
        metrics.update(
            torso_ratio=torso_ratio,
            leg_ratio=leg_ratio,
            axial_ratio=axial_ratio,
            leg_alignment=alignment,
        )

        bends = []
        for hip_index, knee_index, ankle_index in (
            (11, 13, 15),
            (12, 14, 16),
        ):
            if all(
                float(keypoints[index, 2]) >= threshold
                for index in (hip_index, knee_index, ankle_index)
            ):
                angle = self._joint_angle(
                    keypoints[hip_index, :2],
                    keypoints[knee_index, :2],
                    keypoints[ankle_index, :2],
                )
                bends.append(self._c((165.0 - angle) / 75.0))
        leg_bend = max(bends, default=0.0)
        metrics["leg_bend_score"] = leg_bend

        quality = float(metrics["pose_quality"])
        projection = self._c((axial_ratio - 1.8) / 3.2)
        torso_floor_projection = self._c((torso_ratio - 1.25) / 1.4)
        spread = self._c((metrics["body_span_ratio"] - 2.2) / 3.5)
        elongation_score = self._c((elongation - 1.3) / 2.0)
        alignment_score = self._c((alignment - 0.30) / 0.60)
        straight_legs = 1.0 - leg_bend

        lying_score = self._c(
            0.28 * projection
            + 0.25 * torso_floor_projection
            + 0.16 * spread
            + 0.12 * elongation_score
            + 0.08 * alignment_score
            + 0.05 * straight_legs
            + 0.06 * quality
        )
        # Curled/fetal lying can have bent legs and weaker PCA elongation, but the
        # torso itself is still projected into the floor plane.  This rescue path
        # deliberately does not require straight legs or one image orientation.
        curled_lying = self._c(
            0.42 * torso_floor_projection
            + 0.25 * projection
            + 0.18 * spread
            + 0.15 * quality
        )
        lying_score = max(lying_score, curled_lying)

        torso_mid = 1.0 - self._c(abs(torso_ratio - 1.35) / 0.90)
        leg_mid = 1.0 - self._c(abs(leg_ratio - 2.30) / 1.80)
        sitting_score = self._c(
            0.46 * leg_bend
            + 0.22 * torso_mid
            + 0.12 * leg_mid
            + 0.10 * (1.0 - torso_floor_projection)
            + 0.10 * quality
        )

        compact_torso = 1.0 - self._c((torso_ratio - 0.70) / 1.10)
        compact_axial = 1.0 - self._c((axial_ratio - 1.40) / 2.00)
        standing_score = self._c(
            0.38 * compact_torso
            + 0.25 * straight_legs
            + 0.17 * compact_axial
            + 0.10 * (1.0 - spread)
            + 0.10 * quality
        )

        metrics["lying_score"] = lying_score
        metrics["sitting_score"] = sitting_score
        metrics["standing_score"] = standing_score

        # Sitting gets priority only with an actual knee bend and a torso that is
        # still compact in perspective.  This prevents a curled person on the
        # floor from being called sitting merely because the knees are bent.
        if (
            sitting_score >= 0.60
            and sitting_score >= lying_score - 0.05
            and leg_bend >= 0.35
            and torso_ratio < 2.0
        ):
            return "Сидит", metrics
        if lying_score >= 0.60 and lying_score >= sitting_score + 0.03:
            return "Лежит", metrics
        if standing_score >= 0.58 and lying_score < 0.56:
            return "Стоит", metrics
        if lying_score >= 0.50:
            return "Падает", metrics
        if sitting_score >= 0.52:
            return "Сидит", metrics
        return "Стоит", metrics

    def _refine_top_down_posture(self, track_id, posture, metrics, bbox_area):
        """Apply footprint change and temporal hysteresis to top-down scores."""
        state = self._state(track_id)
        lying = float(metrics.get("lying_score", 0.0))
        sitting = float(metrics.get("sitting_score", 0.0))
        standing = float(metrics.get("standing_score", 0.0))

        growth = 1.0
        if state.safe_bbox_area_ema is not None and bbox_area > 0:
            growth = float(bbox_area) / max(state.safe_bbox_area_ema, 1.0)
            growth_score = self._c((growth - 1.12) / 0.60)
            lying = self._c(max(lying, lying + 0.18 * growth_score))
            standing = self._c(standing * (1.0 - 0.25 * growth_score))
            metrics["lying_score"] = lying
            metrics["standing_score"] = standing
        metrics["footprint_growth"] = growth

        previous = (
            state.stable_posture
            if state.stable_posture != "Неизвестно"
            else state.posture
        )

        refined = posture
        if previous == "Лежит" and lying >= 0.46 and sitting < 0.72:
            refined = "Лежит"
        elif previous == "Сидит" and sitting >= 0.50 and lying < 0.64:
            refined = "Сидит"
        elif previous == "Стоит" and standing >= 0.46 and lying < 0.58:
            refined = "Стоит"
        elif growth >= 1.28 and lying >= 0.48 and sitting < 0.70:
            refined = "Лежит"

        state.stable_posture = refined
        return refined


FrameAnalysis = _original.FrameAnalysis
TrackState = _original.TrackState
PoseSample = _original.PoseSample
dismiss_fall_track = _original.dismiss_fall_track

__all__ = [
    "FallDetector",
    "FrameAnalysis",
    "TrackState",
    "PoseSample",
    "dismiss_fall_track",
]
