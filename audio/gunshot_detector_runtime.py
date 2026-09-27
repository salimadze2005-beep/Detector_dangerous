from __future__ import annotations

import csv
import logging
import math
import queue
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.config import get_resource_path
from core.event_bus import Event, EventBus, EventType, Severity

logger = logging.getLogger(__name__)
_PANNS_IMPORT_LOCK = threading.Lock()

FIREARM_CLASSES = {"Gunshot, gunfire", "Machine gun", "Fusillade", "Artillery fire", "Cap gun"}
EXPLOSIVE_CLASSES = {"Explosion", "Fireworks", "Firecracker"}
NUISANCE_CLASSES = {
    "Clapping", "Applause", "Finger snapping", "Slap, smack", "Whack, smack",
    "Door", "Door slam", "Slam", "Knock", "Tick", "Tick-tock", "Clicking",
    "Clickety-clack", "Crackle", "Hammer", "Wood", "Chop", "Glass",
    "Chink, clink", "Shatter", "Burst, pop", "Crack", "Speech", "Conversation",
    "Narration, monologue", "Vehicle", "Engine", "Clang", "Computer keyboard",
}


class YAMNetVeto:
    VETO_CLASSES = NUISANCE_CLASSES
    GUN_CLASSES = FIREARM_CLASSES | EXPLOSIVE_CLASSES


@dataclass(frozen=True)
class TemporalEvidenceResult:
    matched: bool
    cnn: float = 0.0
    gun: float = 0.0
    cnn_time: float = 0.0
    gun_time: float = 0.0
    vetoed: bool = False


class TemporalEvidenceTracker:
    def __init__(self, window_sec, cnn_threshold, yamnet_threshold):
        self.window_sec = float(window_sec)
        self.cnn_threshold = float(cnn_threshold)
        self.yamnet_threshold = float(yamnet_threshold)
        self.history = deque()

    def reset(self):
        self.history.clear()

    def update(self, timestamp, cnn, gun, semantic_veto=False):
        t = float(timestamp)
        self.history.append((t, float(cnn), float(gun), bool(semantic_veto)))
        while self.history and self.history[0][0] < t - self.window_sec:
            self.history.popleft()
        cs = [x for x in self.history if x[1] >= self.cnn_threshold]
        gs = [x for x in self.history if x[2] >= self.yamnet_threshold]
        pairs = [(c, g) for c in cs for g in gs if abs(c[0] - g[0]) <= self.window_sec]
        if not pairs:
            return TemporalEvidenceResult(False)
        c, g = max(pairs, key=lambda x: x[0][1] * x[1][2])
        a, b = sorted((c[0], g[0]))
        veto = any(v and a <= tt <= b for tt, _, _, v in self.history)
        return TemporalEvidenceResult(True, c[1], g[2], c[0], g[0], veto)


@dataclass(frozen=True)
class FusionResult:
    score: float
    yamnet_confirmed: bool
    semantic_veto: bool


class GunshotFusion:
    def __init__(self, detection_mode, yamnet_threshold, veto_threshold, veto_margin=0.12, cnn_override_threshold=0.92):
        m = detection_mode.casefold()
        self.cnn_only = "только cnn" in m or "only cnn" in m
        self.yamnet_only = "только yamnet" in m or "only yamnet" in m
        self.sensitive = "чувствительный" in m or " or " in m or "или" in m
        self.y = float(yamnet_threshold)
        self.v = float(veto_threshold)
        self.vm = float(veto_margin)
        self.co = float(cnn_override_threshold)

    def combine(self, cnn, gun, veto):
        score = float(cnn) if self.cnn_only else (float(gun) if self.yamnet_only else (max(cnn, gun) if self.sensitive else float(cnn)))
        sv = veto >= self.v and veto >= gun + self.vm and cnn < self.co
        return FusionResult(score, gun >= self.y or cnn >= self.co, sv)


class EnergyGate:
    def __init__(self, rms_min=0.008, peak_min=0.2, adaptive=True, noise_alpha=0.05, min_snr_db=8.0, min_crest_factor=2.5):
        self.rms_min = float(rms_min)
        self.peak_min = float(peak_min)
        self.adaptive = bool(adaptive)
        self.noise_alpha = float(noise_alpha)
        self.min_snr_db = float(min_snr_db)
        self.min_crest_factor = float(min_crest_factor)
        self.noise_rms = max(self.rms_min / 2, 1e-6)
        self.last_metrics = {}

    def check(self, w):
        x = np.asarray(w, dtype=np.float32).reshape(-1)
        x = x - float(np.mean(x)) if x.size else x
        if not x.size:
            return False, 0.0, 0.0
        rms = float(np.sqrt(np.mean(x * x)))
        peak = float(np.max(np.abs(x)))
        crest = peak / max(rms, 1e-9)
        if self.adaptive and (peak < self.peak_min or crest < self.min_crest_factor):
            self.noise_rms = self.noise_alpha * min(rms, max(self.rms_min * 4, self.noise_rms * 2)) + (1 - self.noise_alpha) * self.noise_rms
        dyn = max(self.peak_min, self.noise_rms * 10 ** (self.min_snr_db / 20)) if self.adaptive else self.peak_min
        self.last_metrics = {
            "noise_rms": self.noise_rms,
            "snr_db": 20 * math.log10(max(peak, 1e-9) / max(self.noise_rms, 1e-9)),
            "crest_factor": crest,
            "dynamic_peak_min": dyn,
        }
        return rms >= self.rms_min and peak >= dyn and crest >= self.min_crest_factor, rms, peak


@dataclass(frozen=True)
class FusionDecision:
    accepted: bool
    reason: str
    margin: float


class PANNsYAMNetFusion:
    """Conservative two-model policy with a narrow high-confidence rescue.

    PANNs is the primary firearm-vs-explosion/nuisance discriminator. YAMNet
    must independently confirm the event and is never ignored when it strongly
    prefers a nuisance class. A very strong YAMNet firearm score may rescue a
    borderline PANNs score, but only when both PANNs margins are still positive.
    """

    RESCUE_PANNS_FIREARM = 0.15
    RESCUE_PANNS_MARGIN = 0.10
    RESCUE_YAMNET_FIREARM = 0.70
    INDEPENDENT_YAMNET_FIREARM = 0.70
    INDEPENDENT_YAMNET_MARGIN = 0.12
    # A very high YAMNet score alone is not enough: clicks/claps can
    # occasionally produce a firearm-like YAMNet spike while PANNs
    # correctly reports almost no firearm evidence.
    INDEPENDENT_PANNS_FIREARM_FLOOR = 0.05

    def __init__(self, panns_threshold=0.30, panns_margin_threshold=0.15, yamnet_threshold=0.30, yamnet_veto_threshold=0.45, yamnet_veto_margin=0.10):
        self.p = float(panns_threshold)
        self.m = float(panns_margin_threshold)
        self.y = float(yamnet_threshold)
        self.yv = float(yamnet_veto_threshold)
        self.ym = float(yamnet_veto_margin)

    def decide(self, panns_firearm, panns_explosive, panns_nuisance, yamnet_firearm, yamnet_explosive, yamnet_nuisance, yamnet_veto=False):
        margin = float(panns_firearm) - max(float(panns_explosive), float(panns_nuisance))
        veto = bool(yamnet_veto) or (
            yamnet_nuisance >= self.yv
            and yamnet_nuisance >= yamnet_firearm + self.ym
        )

        normal_pair = panns_firearm >= self.p and margin >= self.m
        rescue_pair = (
            panns_firearm >= min(self.p, self.RESCUE_PANNS_FIREARM)
            and margin >= min(self.m, self.RESCUE_PANNS_MARGIN)
            and yamnet_firearm >= max(self.y, self.RESCUE_YAMNET_FIREARM)
        )
        # PANNs sometimes labels reverberant/distant gunshots as Speech or
        # Whip. Do not let that single model suppress a clean, very strong
        # independent YAMNet firearm decision. The explicit nuisance margin is
        # intentionally stricter than the regular pair and the veto still has
        # priority, so finger snaps/claps remain rejected.
        independent_yamnet = (
            panns_firearm >= self.INDEPENDENT_PANNS_FIREARM_FLOOR
            and yamnet_firearm >= max(self.y, self.INDEPENDENT_YAMNET_FIREARM)
            and yamnet_firearm >= yamnet_nuisance + self.INDEPENDENT_YAMNET_MARGIN
        )

        if not normal_pair:
            if rescue_pair and not veto:
                return FusionDecision(True, "accepted_yamnet_rescue", margin)
            if independent_yamnet and not veto:
                return FusionDecision(True, "accepted_yamnet_independent", margin)
            reason = "panns_threshold" if panns_firearm < self.p else "panns_margin"
            return FusionDecision(False, reason, margin)
        if yamnet_firearm < self.y:
            return FusionDecision(False, "yamnet_threshold", margin)
        # Do not reject merely because YAMNet's explosion score is slightly
        # higher than firearm. Reproduced and distant shots often blur that
        # boundary, while PANNs remains responsible for firearm-vs-explosion.
        if veto:
            return FusionDecision(False, "yamnet_veto", margin)
        return FusionDecision(True, "accepted", margin)


class _AudioSetModel:
    def _summarize(self, names, scores):
        s = np.asarray(scores).reshape(-1)
        idx = {n: i for i, n in enumerate(names)}

        def mx(group):
            return max([float(s[idx[n]]) for n in group if n in idx] or [0.0])

        top = int(np.argmax(s))
        return mx(FIREARM_CLASSES), mx(EXPLOSIVE_CLASSES), mx(NUISANCE_CLASSES), names[top], float(s[top])


class PANNsClassifier(_AudioSetModel):
    def __init__(self, path, device="auto"):
        import torch

        checkpoint = Path(path).resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"PANNs checkpoint не найден: {path}")
        labels_path = checkpoint.parent.parent / "panns_data" / "class_labels_indices.csv"
        labels_count = 0
        if labels_path.is_file():
            with labels_path.open("r", encoding="utf-8") as labels_file:
                labels_count = sum(1 for _ in labels_file)
        if labels_count < 528:
            raise FileNotFoundError(
                "Справочник 527 классов PANNs не найден; повторно запустите setup.bat"
            )
        # panns-inference 0.1.1 hardcodes Path.home()/panns_data and otherwise
        # tries to run wget during import. Point only that import at our
        # project-managed model directory, without changing HOME globally.
        from unittest.mock import patch

        with _PANNS_IMPORT_LOCK, patch.object(
            Path, "home", return_value=checkpoint.parent.parent
        ):
            from panns_inference import AudioTagging, labels

        dev = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        self.model = AudioTagging(checkpoint_path=str(checkpoint), device=dev)
        self.names = list(labels)

    def check(self, audio, sample_rate: int = 32_000):
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        source_rate = int(sample_rate)
        if source_rate != 32_000:
            from math import gcd
            from scipy.signal import resample_poly

            divisor = gcd(source_rate, 32_000)
            x = resample_poly(x, 32_000 // divisor, source_rate // divisor)
        x = np.clip(x, -1, 1)
        clip, _ = self.model.inference(x[None, :])
        return self._summarize(self.names, clip[0])


class YAMNetClassifier(_AudioSetModel):
    def __init__(self, path):
        import tensorflow_hub as hub

        self.model = hub.load(path)
        mp = self.model.class_map_path().numpy().decode("utf-8")
        with open(mp, "r", encoding="utf-8") as f:
            self.names = [r["display_name"] for r in csv.DictReader(f)]

    def check(self, audio16):
        scores, _, _ = self.model(np.asarray(audio16, dtype=np.float32))
        m = np.max(scores.numpy(), axis=0)
        return self._summarize(self.names, m)


class GunshotDetector:
    def __init__(self, event_bus: EventBus, model_path: str, threshold=0.30, ema_alpha=0.6, cooldown_sec=2.0, analysis_step_sec=1.0, rms_min=0.008, peak_min=0.2, sustain_windows=1, sustain_hits=1, sustain_thresh=0.5, use_signature=True, use_veto=True, use_centroid=True, veto_thresh=0.45, gun_thresh=0.30, detection_mode="PANNs + YAMNet (Строгий)", stats_path=None, sample_rate=16000, yamnet_sample_rate=16000, yamnet_model_path=None, adaptive_noise=True, noise_alpha=0.05, min_snr_db=8.0, min_crest_factor=2.5, veto_margin=0.10, cnn_override_threshold=0.15, evidence_window_sec=2.0, microphone_source="default", camera_ids=(), **kwargs):
        del sustain_windows, sustain_hits, sustain_thresh, use_signature, use_centroid, evidence_window_sec, stats_path, ema_alpha
        self.event_bus = event_bus
        self.sample_rate = int(sample_rate)
        self.yamnet_sample_rate = int(yamnet_sample_rate)
        if self.sample_rate <= 0 or self.yamnet_sample_rate <= 0:
            raise ValueError("Частоты PANNs/YAMNet должны быть положительными")
        self.step = float(analysis_step_sec)
        self.cooldown = float(cooldown_sec)
        self.source = microphone_source
        self.camera_ids = tuple(camera_ids)
        self.t = 0.0
        self.since = 0.0
        self.last = -1e9
        self.window_sec = float(kwargs.get("analysis_window_sec", 3.0))
        self.window = int(self.sample_rate * self.window_sec)
        self.yamnet_window = int(self.yamnet_sample_rate * self.window_sec)
        self.buf = np.zeros(self.window, dtype=np.float32)
        self.yamnet_buf = np.zeros(self.yamnet_window, dtype=np.float32)
        self.gate = EnergyGate(
            rms_min,
            peak_min,
            adaptive_noise,
            noise_alpha,
            min_snr_db,
            min_crest_factor,
        )
        self.multiscale_rescue = bool(kwargs.get("multiscale_rescue", True))
        self.rescue_window_sec = min(
            max(float(kwargs.get("rescue_window_sec", 1.0)), 0.5),
            self.window_sec,
        )
        # This independent gate never updates the baseline adaptive noise state.
        # It only concentrates brief impulses that were diluted by the full
        # three-second window.
        self.rescue_gate = EnergyGate(
            rms_min,
            peak_min,
            adaptive=False,
            min_snr_db=min_snr_db,
            min_crest_factor=min_crest_factor,
        )
        pth = float(kwargs.get("panns_threshold", threshold))
        pm = float(kwargs.get("panns_margin_threshold", cnn_override_threshold))
        # Retained in event telemetry and old configs; it is no longer a gate.
        self.pref = float(kwargs.get("panns_prefilter_threshold", 0.10))
        self.fusion = PANNsYAMNetFusion(pth, pm, gun_thresh, veto_thresh, veto_margin)
        self.use_veto = bool(use_veto)
        self.panns = PANNsClassifier(kwargs.get("panns_model_path") or model_path, kwargs.get("panns_device", "auto"))
        self.yamnet = YAMNetClassifier(yamnet_model_path or get_resource_path("models/yamnet"))

        # Heavy inference must never block microphone draining or Vosk.
        self._inference_queue: queue.Queue[tuple[float, np.ndarray, np.ndarray] | None] = queue.Queue(maxsize=1)
        self._closed = threading.Event()
        self._worker = threading.Thread(
            target=self._inference_loop,
            name=f"gunshot-inference-{self.source}",
            daemon=True,
        )
        self._worker.start()

    def _inference_loop(self) -> None:
        while not self._closed.is_set():
            try:
                item = self._inference_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                return
            timestamp, panns_audio, yamnet_audio = item
            try:
                self._analyze(timestamp, panns_audio, yamnet_audio)
            except Exception:
                logger.exception("Ошибка PANNs/YAMNet inference")

    def _queue_latest(
        self,
        timestamp: float,
        panns_audio: np.ndarray,
        yamnet_audio: np.ndarray,
    ) -> None:
        item = (timestamp, panns_audio, yamnet_audio)
        try:
            self._inference_queue.put_nowait(item)
            return
        except queue.Full:
            pass
        # If inference is slower than the analysis cadence, discard the stale
        # pending window, not microphone samples, and keep the newest window.
        try:
            self._inference_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._inference_queue.put_nowait(item)
        except queue.Full:
            pass

    def _strict_short_rescue_decision(
        self,
        panns_firearm: float,
        panns_explosive: float,
        panns_nuisance: float,
        yamnet_firearm: float,
        yamnet_explosive: float,
        yamnet_nuisance: float,
        yamnet_veto: bool,
    ) -> FusionDecision:
        """Accept an extra short window only on strict two-model consensus."""
        del yamnet_explosive
        margin = float(panns_firearm) - max(
            float(panns_explosive), float(panns_nuisance)
        )
        panns_min = max(self.fusion.p, 0.38)
        margin_min = max(self.fusion.m, 0.20)
        yamnet_min = max(self.fusion.y, 0.45)
        accepted = (
            not yamnet_veto
            and panns_firearm >= panns_min
            and margin >= margin_min
            and yamnet_firearm >= yamnet_min
            and yamnet_firearm >= yamnet_nuisance + 0.12
        )
        return FusionDecision(
            accepted,
            "accepted_strict_short_rescue" if accepted else "short_consensus",
            margin,
        )

    def _analyze(
        self,
        timestamp: float,
        panns_audio: np.ndarray,
        yamnet_audio: np.ndarray | None = None,
    ) -> None:
        panns_x = np.clip(
            np.asarray(panns_audio, dtype=np.float32).reshape(-1), -1, 1
        )
        if yamnet_audio is None:
            if self.sample_rate == self.yamnet_sample_rate:
                yamnet_x = panns_x
            else:
                from math import gcd
                from scipy.signal import resample_poly
                divisor = gcd(self.sample_rate, self.yamnet_sample_rate)
                yamnet_x = resample_poly(
                    panns_x,
                    self.yamnet_sample_rate // divisor,
                    self.sample_rate // divisor,
                ).astype(np.float32, copy=False)
        else:
            yamnet_x = np.clip(
                np.asarray(yamnet_audio, dtype=np.float32).reshape(-1),
                -1,
                1,
            )
        # Baseline path: intentionally the same full-window gate, models and
        # fusion policy as before. A positive baseline result is authoritative.
        selected = None
        ok, rms, peak = self.gate.check(yamnet_x)
        if ok:
            pf, pe, pn, ptop, _ = self.panns.check(panns_x, self.sample_rate)
            yf, ye, yn, ytop, _ = self.yamnet.check(yamnet_x)
            veto = (
                self.use_veto
                and yn >= self.fusion.yv
                and yn >= yf + self.fusion.ym
            )
            decision = self.fusion.decide(pf, pe, pn, yf, ye, yn, veto)
            logger.info(
                "[GUNSHOT] path=baseline_3s result=%s reason=%s PANNs firearm=%.3f explosive=%.3f nuisance=%.3f margin=%+.3f top=%s | YAMNet firearm=%.3f explosive=%.3f nuisance=%.3f top=%s | peak=%.3f rms=%.4f",
                "accept" if decision.accepted else "reject",
                decision.reason,
                pf,
                pe,
                pn,
                decision.margin,
                ptop,
                yf,
                ye,
                yn,
                ytop,
                peak,
                rms,
            )
            if decision.accepted:
                selected = (
                    pf,
                    yf,
                    ptop,
                    ytop,
                    decision,
                    rms,
                    peak,
                    "baseline_3s",
                    getattr(
                        self,
                        "window_sec",
                        len(panns_x) / max(self.sample_rate, 1),
                    ),
                )
        # Additive path: it is evaluated only after the established path did
        # not accept. Its higher thresholds and explicit nuisance margin make
        # it a narrow rescue for a brief, diluted impulse rather than a looser
        # replacement for the current detector.
        if selected is None and self.multiscale_rescue:
            panns_count = max(1, int(self.sample_rate * self.rescue_window_sec))
            yamnet_count = max(
                1, int(self.yamnet_sample_rate * self.rescue_window_sec)
            )
            short_panns = panns_x[-panns_count:]
            short_yamnet = yamnet_x[-yamnet_count:]
            dynamic_peak = self.gate.last_metrics.get(
                "dynamic_peak_min", self.gate.peak_min
            )
            self.rescue_gate.peak_min = max(
                self.gate.peak_min, float(dynamic_peak)
            )
            rescue_ok, rescue_rms, rescue_peak = self.rescue_gate.check(
                short_yamnet
            )
            if rescue_ok:
                rpf, rpe, rpn, rptop, _ = self.panns.check(
                    short_panns, self.sample_rate
                )
                ryf, rye, ryn, rytop, _ = self.yamnet.check(short_yamnet)
                rveto = (
                    self.use_veto
                    and ryn >= self.fusion.yv
                    and ryn >= ryf + self.fusion.ym
                )
                rescue = self._strict_short_rescue_decision(
                    rpf, rpe, rpn, ryf, rye, ryn, rveto
                )
                logger.info(
                    "[GUNSHOT] path=strict_short_rescue result=%s reason=%s PANNs firearm=%.3f explosive=%.3f nuisance=%.3f margin=%+.3f top=%s | YAMNet firearm=%.3f explosive=%.3f nuisance=%.3f top=%s | peak=%.3f rms=%.4f",
                    "accept" if rescue.accepted else "reject",
                    rescue.reason,
                    rpf,
                    rpe,
                    rpn,
                    rescue.margin,
                    rptop,
                    ryf,
                    rye,
                    ryn,
                    rytop,
                    rescue_peak,
                    rescue_rms,
                )
                if rescue.accepted:
                    selected = (
                        rpf,
                        ryf,
                        rptop,
                        rytop,
                        rescue,
                        rescue_rms,
                        rescue_peak,
                        "strict_short_rescue",
                        self.rescue_window_sec,
                    )
        if selected is None or timestamp - self.last < self.cooldown:
            return
        pf, yf, ptop, ytop, decision, rms, peak, path, window_sec = selected
        self.last = timestamp
        confidence = min(1.0, math.sqrt(max(pf, 0) * max(yf, 0)))
        self.event_bus.publish(
            Event(
                type=EventType.GUNSHOT_DETECTED,
                confidence=confidence,
                source=f"audio.gunshot.{self.source}",
                severity=Severity.CRITICAL,
                metadata={
                    "model": "panns_cnn14+yamnet",
                    "detection_path": path,
                    "analysis_window_sec": round(window_sec, 2),
                    "p_panns": round(pf, 3),
                    "panns_margin": round(decision.margin, 3),
                    "p_yamnet": round(yf, 3),
                    "panns_top": ptop,
                    "yamnet_top": ytop,
                    "panns_below_prefilter": pf < self.pref,
                    "rms": round(rms, 4),
                    "peak": round(peak, 3),
                    "microphone": self.source,
                    "camera_ids": list(self.camera_ids),
                    "panns_sample_rate": self.sample_rate,
                    "yamnet_sample_rate": self.yamnet_sample_rate,
                },
            )
        )
    @staticmethod
    def _append_buffer(buffer: np.ndarray, chunk: np.ndarray) -> np.ndarray:
        if not chunk.size:
            return buffer
        if len(chunk) >= len(buffer):
            return chunk[-len(buffer):].copy()
        buffer = np.roll(buffer, -len(chunk))
        buffer[-len(chunk):] = chunk
        return buffer

    def process_audio(self, panns_chunk, yamnet_chunk=None):
        panns_x = np.clip(np.asarray(panns_chunk, dtype=np.float32).reshape(-1), -1, 1)
        if not panns_x.size:
            return
        if yamnet_chunk is None:
            if self.sample_rate == self.yamnet_sample_rate:
                yamnet_x = panns_x
            else:
                from math import gcd
                from scipy.signal import resample_poly

                divisor = gcd(self.sample_rate, self.yamnet_sample_rate)
                yamnet_x = resample_poly(
                    panns_x,
                    self.yamnet_sample_rate // divisor,
                    self.sample_rate // divisor,
                ).astype(np.float32, copy=False)
        else:
            yamnet_x = np.clip(
                np.asarray(yamnet_chunk, dtype=np.float32).reshape(-1),
                -1,
                1,
            )
        duration = len(panns_x) / self.sample_rate
        self.t += duration
        self.since += duration
        self.buf = self._append_buffer(self.buf, panns_x)
        self.yamnet_buf = self._append_buffer(self.yamnet_buf, yamnet_x)
        if self.since >= self.step:
            self.since %= self.step
            self._queue_latest(self.t, self.buf.copy(), self.yamnet_buf.copy())

    def close(self):
        self._closed.set()
        try:
            self._inference_queue.put_nowait(None)
        except queue.Full:
            try:
                self._inference_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._inference_queue.put_nowait(None)
            except queue.Full:
                pass
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
