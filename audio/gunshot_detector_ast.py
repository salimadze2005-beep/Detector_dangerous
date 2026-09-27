from __future__ import annotations

import logging
import math
import queue
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from core.config import get_resource_path
from core.event_bus import Event, EventBus, EventType, Severity
from audio.gunshot_detector_runtime import EnergyGate, FusionDecision
from audio.gunshot_detector_runtime import PANNsClassifier as _BasePANNsClassifier

logger = logging.getLogger(__name__)

REAL_GUNSHOT_LABELS = {
    "Gunshot, gunfire",
    "Machine gun",
    "Fusillade",
    "Artillery fire",
}
FIRECRACKER_LABELS = {
    "Explosion",
    "Fireworks",
    "Firecracker",
    "Cap gun",
    "Boom",
    "Burst, pop",
}
CLAP_LABELS = {
    "Clapping",
    "Applause",
    "Finger snapping",
    "Slap, smack",
    "Whack, smack",
    "Whack, thwack",
}
CLICK_LABELS = {
    "Clicking",
    "Clickety-clack",
    "Tick",
    "Tick-tock",
    "Tap",
    "Camera",
    "Single-lens reflex camera",
    "Typing",
    "Typewriter",
    "Computer keyboard",
    "Keys jangling",
    "Walk, footsteps",
    "Run",
    "Shuffle",
    "Clip-clop",
    "Patter",
}
METAL_LABELS = {
    "Hammer",
    "Dishes, pots, and pans",
    "Cutlery, silverware",
    "Chink, clink",
    "Clang",
    "Clatter",
    "Coin (dropping)",
    "Smash, crash",
    "Breaking",
}
DOOR_LABELS = {
    "Door",
    "Door slam",
    "Slam",
    "Knock",
    "Sliding door",
    "Cupboard open or close",
    "Drawer open or close",
}
PANNS_NUISANCE_LABELS = (
    CLAP_LABELS
    | CLICK_LABELS
    | METAL_LABELS
    | DOOR_LABELS
    | {
        "Crackle",
        "Wood",
        "Chop",
        "Glass",
        "Shatter",
        "Crack",
        "Speech",
        "Conversation",
        "Narration, monologue",
        "Vehicle",
        "Engine",
    }
)
AST_GROUPS = {
    "gunshot": REAL_GUNSHOT_LABELS,
    "firecracker": FIRECRACKER_LABELS,
    "clap": CLAP_LABELS,
    "click": CLICK_LABELS,
    "metal_impact": METAL_LABELS,
    "door_slam": DOOR_LABELS,
}
AST_NAMES = (
    "gunshot",
    "firecracker",
    "clap",
    "click",
    "metal_impact",
    "door_slam",
    "other",
)
NUISANCE_BUCKETS = ("clap", "click", "metal_impact", "door_slam")


class PANNsClassifier(_BasePANNsClassifier):
    """Cnn14 summary tuned for firearm-vs-hard-negative discrimination."""

    def _summarize(self, names, scores):
        values = np.asarray(scores).reshape(-1)
        index = {name: i for i, name in enumerate(names)}

        def maximum(group):
            return max(
                (float(values[index[x]]) for x in group if x in index),
                default=0.0,
            )

        top = int(np.argmax(values))
        return (
            maximum(REAL_GUNSHOT_LABELS),
            maximum(FIRECRACKER_LABELS),
            maximum(PANNS_NUISANCE_LABELS),
            names[top],
            float(values[top]),
        )


@dataclass(frozen=True, slots=True)
class ASTResult:
    scores: Mapping[str, float]
    top_class: str
    top_score: float
    source_label: str
    source_score: float

    def score(self, name: str) -> float:
        return float(self.scores.get(name, 0.0))

    @property
    def gunshot(self):
        return self.score("gunshot")

    @property
    def firecracker(self):
        return self.score("firecracker")

    @property
    def nuisance(self):
        return max(self.score(x) for x in NUISANCE_BUCKETS)


class ASTClassifier:
    """AudioSet AST reduced to firearm + requested hard-negative buckets."""

    TARGET_RATE = 16_000

    def __init__(self, path: str, device: str = "auto"):
        import torch
        from transformers import (
            AutoFeatureExtractor,
            AutoModelForAudioClassification,
        )

        model_path = Path(path).expanduser().resolve()
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"AST model not found: {model_path}. "
                "Run setup.bat or python download_ast.py"
            )
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("AST configured for CUDA, but CUDA is unavailable")

        self.torch = torch
        self.device = torch.device(device)
        # The official MIT/ast-finetuned-audioset checkpoint intentionally uses
        # 128 mel filters with a 512-point FFT. Transformers warns about one
        # zero-valued edge filter while constructing the extractor, but changing
        # this checkpoint-owned geometry would corrupt its expected input.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"At least one mel filter has all zero values.*",
                category=UserWarning,
                module=r"transformers\.audio_utils",
            )
            self.extractor = AutoFeatureExtractor.from_pretrained(
                str(model_path), local_files_only=True
            )
        self.model = AutoModelForAudioClassification.from_pretrained(
            str(model_path), local_files_only=True
        )
        self.model.to(self.device)
        self.model.eval()

        id2label = dict(getattr(self.model.config, "id2label", {}) or {})
        self.labels = [
            str(id2label.get(i, id2label.get(str(i), i)))
            for i in range(self.model.config.num_labels)
        ]
        self.index = {name: i for i, name in enumerate(self.labels)}
        self.group_indices = {
            name: [self.index[x] for x in labels if x in self.index]
            for name, labels in AST_GROUPS.items()
        }
        assigned = {
            i for indices in self.group_indices.values() for i in indices
        }
        self.other_indices = [
            i for i in range(len(self.labels)) if i not in assigned
        ]
        if not self.group_indices["gunshot"]:
            raise RuntimeError(
                "AST checkpoint has no expected AudioSet gunshot labels"
            )

    @staticmethod
    def _resample(
        x: np.ndarray, source_rate: int, target_rate: int
    ) -> np.ndarray:
        if source_rate == target_rate:
            return x
        from math import gcd
        from scipy.signal import resample_poly

        divisor = gcd(int(source_rate), int(target_rate))
        return resample_poly(
            x,
            target_rate // divisor,
            source_rate // divisor,
        ).astype(np.float32, copy=False)

    def check(self, audio, sample_rate: int = TARGET_RATE) -> ASTResult:
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        x = np.clip(
            self._resample(x, int(sample_rate), self.TARGET_RATE), -1, 1
        )
        # Transformers emits this warning for the official AST AudioSet
        # preprocessor (128 mel filters / 512 FFT). The checkpoint expects that
        # exact configuration, so changing it would be wrong; suppress only the
        # known benign warning while leaving every other warning visible.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"At least one mel filter has all zero values.*",
                category=UserWarning,
                module=r"transformers\.audio_utils",
            )
            inputs = self.extractor(
                x,
                sampling_rate=self.TARGET_RATE,
                return_tensors="pt",
            )
        input_values = inputs["input_values"].to(self.device)
        with self.torch.inference_mode():
            logits = self.model(input_values=input_values).logits[0].float()
            probabilities = (
                self.torch.softmax(logits, dim=-1).detach().cpu().numpy()
            )

        source_index = int(np.argmax(probabilities))
        # AST can split firearm evidence across adjacent AudioSet labels.
        # Sum within a semantic family, but keep "other" as max: summing
        # hundreds of unrelated labels would otherwise make other always win.
        scores = {}
        for name, indices in self.group_indices.items():
            scores[name] = (
                min(1.0, float(np.sum(probabilities[indices])))
                if indices
                else 0.0
            )
        scores["other"] = max(
            (float(probabilities[i]) for i in self.other_indices),
            default=0.0,
        )
        top = max(AST_NAMES, key=lambda name: scores[name])
        return ASTResult(
            scores=scores,
            top_class=top,
            top_score=float(scores[top]),
            source_label=self.labels[source_index],
            source_score=float(probabilities[source_index]),
        )


class PANNsASTFusion:
    """Normal consensus + conservative low-SNR/distant-shot rescue."""

    def __init__(
        self,
        panns_threshold=0.30,
        panns_margin_threshold=0.15,
        ast_threshold=0.28,
        ast_veto_threshold=0.20,
        ast_veto_margin=0.08,
        ast_margin_threshold=0.06,
        far_panns_threshold=0.08,
        far_panns_margin=-0.04,
        far_ast_threshold=0.38,
        far_ast_margin=0.12,
        **_,
    ):
        self.p = float(panns_threshold)
        self.pm = float(panns_margin_threshold)
        self.a = max(0.22, float(ast_threshold))
        self.av = max(0.16, float(ast_veto_threshold))
        self.avm = float(ast_veto_margin)
        self.am = float(ast_margin_threshold)
        self.fp = min(self.p, float(far_panns_threshold))
        self.fpm = min(self.pm, float(far_panns_margin))
        self.fa = max(self.a + 0.08, float(far_ast_threshold))
        self.fam = max(self.am + 0.04, float(far_ast_margin))

    @staticmethod
    def margin(firearm, explosive, nuisance):
        return float(firearm) - max(
            float(explosive), float(nuisance)
        )

    def strong_candidate(self, firearm, explosive, nuisance):
        margin = self.margin(firearm, explosive, nuisance)
        if firearm < self.p:
            return FusionDecision(False, "panns_threshold", margin)
        if margin < self.pm:
            return FusionDecision(False, "panns_margin", margin)
        return FusionDecision(True, "panns_candidate", margin)

    def far_candidate(self, firearm, explosive, nuisance):
        margin = self.margin(firearm, explosive, nuisance)
        if firearm < self.fp:
            return FusionDecision(False, "far_panns_threshold", margin)
        if margin < self.fpm:
            return FusionDecision(False, "far_panns_margin", margin)
        if explosive > firearm + 0.12:
            return FusionDecision(False, "far_panns_explosive", margin)
        return FusionDecision(True, "panns_far_candidate", margin)

    def _semantic_reject(
        self,
        ast: ASTResult,
        threshold: float,
        margin: float,
        veto_margin: float,
    ):
        if ast.gunshot < threshold:
            return "ast_threshold"
        if ast.firecracker >= max(
            self.av, ast.gunshot - veto_margin
        ):
            return "ast_firecracker"
        if (
            ast.nuisance >= self.av
            and ast.nuisance >= ast.gunshot - veto_margin
        ):
            return (
                f"ast_veto_{ast.top_class}"
                if ast.top_class in NUISANCE_BUCKETS
                else "ast_veto_nuisance"
            )
        if ast.top_class != "gunshot":
            return f"ast_{ast.top_class}"
        strongest_other = max(
            ast.score(x) for x in AST_NAMES if x != "gunshot"
        )
        if ast.gunshot < strongest_other + margin:
            return "ast_margin"
        return None

    def decide(self, firearm, explosive, nuisance, ast: ASTResult):
        candidate = self.strong_candidate(firearm, explosive, nuisance)
        if not candidate.accepted:
            return candidate
        reason = self._semantic_reject(
            ast, self.a, self.am, self.avm
        )
        return FusionDecision(
            reason is None,
            "accepted" if reason is None else reason,
            candidate.margin,
        )

    def decide_far(self, firearm, explosive, nuisance, ast: ASTResult):
        candidate = self.far_candidate(firearm, explosive, nuisance)
        if not candidate.accepted:
            return candidate
        reason = self._semantic_reject(
            ast,
            self.fa,
            self.fam,
            min(self.avm, 0.07),
        )
        return FusionDecision(
            reason is None,
            (
                "accepted_far_consensus"
                if reason is None
                else f"far_{reason}"
            ),
            candidate.margin,
        )


class GunshotDetector:
    """Async detector used by the existing AudioSystemWorker.

    YAMNet-named constructor arguments are accepted only as compatibility
    input; production inference is PANNs + AST.
    """

    def __init__(
        self,
        event_bus: EventBus,
        model_path: str | None = None,
        threshold=0.30,
        cooldown_sec=2.0,
        analysis_step_sec=0.5,
        rms_min=0.008,
        peak_min=0.2,
        veto_thresh=0.20,
        gun_thresh=0.28,
        sample_rate=32_000,
        yamnet_sample_rate=16_000,
        yamnet_model_path=None,
        ast_sample_rate=None,
        ast_model_path=None,
        ast_device="auto",
        adaptive_noise=True,
        noise_alpha=0.05,
        min_snr_db=8.0,
        min_crest_factor=2.5,
        veto_margin=0.08,
        cnn_override_threshold=0.15,
        microphone_source="default",
        camera_ids=(),
        **kwargs,
    ):
        del yamnet_model_path
        self.event_bus = event_bus
        self.sample_rate = int(sample_rate)
        self.ast_rate = int(ast_sample_rate or yamnet_sample_rate or 16_000)
        self.source = microphone_source
        self.camera_ids = tuple(camera_ids)
        self.step = float(analysis_step_sec)
        self.cooldown = float(cooldown_sec)
        self.window_sec = float(kwargs.get("analysis_window_sec", 3.0))
        self.rescue_sec = min(
            max(float(kwargs.get("rescue_window_sec", 1.0)), 0.6),
            self.window_sec,
        )
        self.multiscale = bool(kwargs.get("multiscale_rescue", True))
        self.window = max(1, int(self.sample_rate * self.window_sec))
        self.ast_window = max(1, int(self.ast_rate * self.window_sec))
        self.buf = np.zeros(self.window, dtype=np.float32)
        self.ast_buf = np.zeros(self.ast_window, dtype=np.float32)
        self.t = self.since = 0.0
        self.last = -1e9

        self.gate = EnergyGate(
            rms_min,
            peak_min,
            adaptive_noise,
            noise_alpha,
            min_snr_db,
            min_crest_factor,
        )
        # The old absolute peak gate was a major source of lost distant shots.
        # This quieter gate only opens model inference; it never alarms alone.
        self.far_gate = EnergyGate(
            min(float(rms_min), 0.0025),
            min(float(peak_min), 0.04),
            adaptive_noise,
            noise_alpha,
            min(float(min_snr_db), 5.0),
            min(float(min_crest_factor), 2.0),
        )
        self.rescue_gate = EnergyGate(
            min(float(rms_min), 0.0035),
            min(float(peak_min), 0.06),
            False,
            min_snr_db=min(float(min_snr_db), 6.0),
            min_crest_factor=min(float(min_crest_factor), 2.1),
        )

        raw_threshold = float(threshold)
        panns_threshold = float(
            kwargs.get(
                "panns_threshold",
                raw_threshold if raw_threshold <= 0.60 else 0.30,
            )
        )
        raw_margin = float(cnn_override_threshold)
        panns_margin = float(
            kwargs.get(
                "panns_margin_threshold",
                raw_margin if raw_margin <= 0.60 else 0.15,
            )
        )
        self.prefilter = float(
            kwargs.get("panns_prefilter_threshold", 0.10)
        )
        # Legacy YAMNet profiles may contain values like 0.10. Do not blindly
        # reuse those as AST confidence thresholds.
        ast_threshold = max(
            0.24,
            min(float(kwargs.get("ast_threshold", gun_thresh)), 0.60),
        )
        ast_veto = max(
            0.18,
            min(float(kwargs.get("ast_veto_threshold", veto_thresh)), 0.60),
        )
        self.fusion = PANNsASTFusion(
            panns_threshold=panns_threshold,
            panns_margin_threshold=panns_margin,
            ast_threshold=ast_threshold,
            ast_veto_threshold=ast_veto,
            ast_veto_margin=float(
                kwargs.get("ast_veto_margin", veto_margin)
            ),
            ast_margin_threshold=float(
                kwargs.get("ast_margin_threshold", 0.06)
            ),
            far_panns_threshold=float(
                kwargs.get("far_panns_threshold", 0.08)
            ),
            far_panns_margin=float(
                kwargs.get("far_panns_margin", -0.04)
            ),
            far_ast_threshold=float(
                kwargs.get("far_ast_threshold", 0.38)
            ),
            far_ast_margin=float(
                kwargs.get("far_ast_margin", 0.12)
            ),
        )

        panns_path = kwargs.get("panns_model_path") or self._panns_path(
            model_path
        )
        self.panns = PANNsClassifier(
            panns_path, kwargs.get("panns_device", "auto")
        )
        candidate_ast = ast_model_path or get_resource_path("models/ast")
        self.ast = ASTClassifier(candidate_ast, ast_device)

        self._queue = queue.Queue(maxsize=1)
        self._closed = threading.Event()
        self._worker = threading.Thread(
            target=self._loop,
            name=f"gunshot-panns-ast-{self.source}",
            daemon=True,
        )
        self._worker.start()

    @staticmethod
    def _panns_path(model_path):
        if (
            not model_path
            or Path(str(model_path)).suffix.casefold() == ".h5"
        ):
            return get_resource_path(
                "models/panns/Cnn14_mAP=0.431.pth"
            )
        return str(model_path)

    @staticmethod
    def _append(buffer, chunk):
        if not chunk.size:
            return buffer
        if len(chunk) >= len(buffer):
            return chunk[-len(buffer):].copy()
        buffer = np.roll(buffer, -len(chunk))
        buffer[-len(chunk):] = chunk
        return buffer

    @staticmethod
    def _resample(x, source_rate, target_rate):
        if source_rate == target_rate:
            return x
        from math import gcd
        from scipy.signal import resample_poly

        divisor = gcd(int(source_rate), int(target_rate))
        return resample_poly(
            x,
            target_rate // divisor,
            source_rate // divisor,
        ).astype(np.float32, copy=False)

    @staticmethod
    def _peak_crop(x: np.ndarray, sample_rate: int, seconds: float):
        count = max(1, min(len(x), int(sample_rate * seconds)))
        if len(x) <= count:
            return x
        peak = int(np.argmax(np.abs(x)))
        start = min(
            max(0, peak - count // 3),
            len(x) - count,
        )
        return x[start : start + count]

    def _loop(self):
        while not self._closed.is_set():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                return
            try:
                self._analyze(*item)
            except Exception:
                logger.exception("PANNs/AST gunshot inference failed")

    def _queue_latest(self, item):
        try:
            self._queue.put_nowait(item)
            return
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            pass

    def _evaluate(self, panns_audio, ast_audio, allow_far=True):
        pf, pe, pn, ptop, _ = self.panns.check(
            panns_audio, self.sample_rate
        )
        strong = self.fusion.strong_candidate(pf, pe, pn)
        far = self.fusion.far_candidate(pf, pe, pn)
        if not strong.accepted and not (
            allow_far and far.accepted
        ):
            return pf, pe, pn, ptop, None, strong
        ast = self.ast.check(ast_audio, self.ast_rate)
        decision = (
            self.fusion.decide(pf, pe, pn, ast)
            if strong.accepted
            else self.fusion.decide_far(pf, pe, pn, ast)
        )
        return pf, pe, pn, ptop, ast, decision

    def _analyze(self, timestamp, panns_audio, ast_audio):
        px = np.clip(
            np.asarray(panns_audio, dtype=np.float32).reshape(-1), -1, 1
        )
        ax = np.clip(
            np.asarray(ast_audio, dtype=np.float32).reshape(-1), -1, 1
        )
        selected = None

        ok, rms, peak = self.gate.check(ax)
        acoustic_path = "standard"
        if not ok:
            ok, rms, peak = self.far_gate.check(ax)
            acoustic_path = "far_gate"
        if ok:
            pf, pe, pn, ptop, ast, decision = self._evaluate(
                px, ax, allow_far=True
            )
            self._log(
                acoustic_path,
                pf,
                pe,
                pn,
                ptop,
                ast,
                decision,
                rms,
                peak,
            )
            if ast is not None and decision.accepted:
                selected = (
                    pf,
                    ptop,
                    ast,
                    decision,
                    rms,
                    peak,
                    (
                        "far_consensus"
                        if decision.reason == "accepted_far_consensus"
                        else "baseline_consensus"
                    ),
                    self.window_sec,
                )

        if selected is None and self.multiscale:
            sp = self._peak_crop(
                px, self.sample_rate, self.rescue_sec
            )
            sa = self._peak_crop(
                ax, self.ast_rate, self.rescue_sec
            )
            rescue_ok, rescue_rms, rescue_peak = self.rescue_gate.check(sa)
            if rescue_ok:
                pf, pe, pn, ptop, ast, decision = self._evaluate(
                    sp, sa, allow_far=False
                )
                if ast is not None and decision.accepted:
                    strongest_non_gunshot = max(
                        ast.score(x)
                        for x in AST_NAMES
                        if x != "gunshot"
                    )
                    strong_short = (
                        pf >= max(self.fusion.p, 0.26)
                        and decision.margin >= max(self.fusion.pm, 0.10)
                        and ast.gunshot >= max(self.fusion.a, 0.40)
                        and ast.gunshot
                        >= strongest_non_gunshot
                        + max(self.fusion.am, 0.10)
                    )
                    if strong_short:
                        selected = (
                            pf,
                            ptop,
                            ast,
                            FusionDecision(
                                True,
                                "accepted_peak_rescue",
                                decision.margin,
                            ),
                            rescue_rms,
                            rescue_peak,
                            "peak_rescue",
                            self.rescue_sec,
                        )
                self._log(
                    "peak_rescue",
                    pf,
                    pe,
                    pn,
                    ptop,
                    ast,
                    decision,
                    rescue_rms,
                    rescue_peak,
                )

        if (
            selected is None
            or timestamp - self.last < self.cooldown
        ):
            return
        pf, ptop, ast, decision, rms, peak, path, window_sec = selected
        self.last = timestamp
        # The geometric mean reported 0% for accepted independent-rescue
        # decisions when one classifier was intentionally weak.  Report the
        # strongest accepted evidence instead of hiding a valid decision.
        confidence_scores = [max(float(pf), 0.0), max(float(ast.gunshot), 0.0)]
        impulse = getattr(self, "_last_impulse", None)
        if impulse is not None and getattr(impulse, "gunshot_like", False):
            confidence_scores.append(max(float(impulse.score), 0.0))
        confidence = min(1.0, max(confidence_scores))
        self.event_bus.publish(
            Event(
                type=EventType.GUNSHOT_DETECTED,
                confidence=confidence,
                source=f"audio.gunshot.{self.source}",
                severity=Severity.CRITICAL,
                metadata={
                    "model": "panns_cnn14+ast_audioset",
                    "pipeline": "PANNs candidate -> AST semantic consensus",
                    "detection_path": path,
                    "analysis_window_sec": round(window_sec, 2),
                    "p_panns": round(pf, 3),
                    "panns_margin": round(decision.margin, 3),
                    "panns_top": ptop,
                    "panns_below_prefilter": pf < self.prefilter,
                    "p_ast_gunshot": round(ast.gunshot, 3),
                    "ast_top": ast.top_class,
                    "ast_source_label": ast.source_label,
                    "ast_source_score": round(ast.source_score, 3),
                    "ast_scores": {
                        name: round(ast.score(name), 3)
                        for name in AST_NAMES
                    },
                    "rms": round(rms, 4),
                    "peak": round(peak, 3),
                    "microphone": self.source,
                    "camera_ids": list(self.camera_ids),
                },
            )
        )

    @staticmethod
    def _log(path, pf, pe, pn, ptop, ast, decision, rms, peak):
        if ast is None:
            logger.info(
                "[GUNSHOT] path=%s reject=%s PANNs=%.3f exp=%.3f "
                "nuisance=%.3f margin=%+.3f top=%s AST=skipped "
                "peak=%.3f rms=%.4f",
                path,
                decision.reason,
                pf,
                pe,
                pn,
                decision.margin,
                ptop,
                peak,
                rms,
            )
            return
        logger.info(
            "[GUNSHOT] path=%s %s reason=%s PANNs=%.3f exp=%.3f "
            "nuisance=%.3f margin=%+.3f top=%s AST gun=%.3f "
            "firecracker=%.3f clap=%.3f click=%.3f metal=%.3f "
            "door=%.3f other=%.3f top=%s source=%s(%.3f) "
            "peak=%.3f rms=%.4f",
            path,
            "accept" if decision.accepted else "reject",
            decision.reason,
            pf,
            pe,
            pn,
            decision.margin,
            ptop,
            ast.gunshot,
            ast.score("firecracker"),
            ast.score("clap"),
            ast.score("click"),
            ast.score("metal_impact"),
            ast.score("door_slam"),
            ast.score("other"),
            ast.top_class,
            ast.source_label,
            ast.source_score,
            peak,
            rms,
        )

    def process_audio(self, panns_chunk, ast_chunk=None):
        px = np.clip(
            np.asarray(panns_chunk, dtype=np.float32).reshape(-1), -1, 1
        )
        if not px.size:
            return
        ax = (
            self._resample(px, self.sample_rate, self.ast_rate)
            if ast_chunk is None
            else np.clip(
                np.asarray(ast_chunk, dtype=np.float32).reshape(-1), -1, 1
            )
        )
        duration = len(px) / self.sample_rate
        self.t += duration
        self.since += duration
        self.buf = self._append(self.buf, px)
        self.ast_buf = self._append(self.ast_buf, ax)
        if self.since >= self.step:
            self.since %= self.step
            self._queue_latest(
                (self.t, self.buf.copy(), self.ast_buf.copy())
            )

    def close(self):
        self._closed.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
