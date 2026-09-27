from __future__ import annotations

import logging
import math
import os
import time
import hashlib
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from audio.calibration import (
    analyze_ambient_audio,
    analyze_reference_audio,
    microphone_profile_key,
    resolve_audio_profile,
)
from audio.microphone import (
    DualRateAudioResampler,
    MicrophoneStream,
    StreamingAudioResampler,
    audio_source_label,
    dbfs_to_amplitude,
)
from core.config import AppConfig
from core.event_bus import EventBus
from core.frames import FrameStore
from core.health import ComponentState

logger = logging.getLogger(__name__)


def safe_source_label(source: int | str) -> str:
    if not isinstance(source, str) or "://" not in source:
        return str(source)
    parts = urlsplit(source)
    hostname = parts.hostname or ""
    if parts.port:
        hostname += f":{parts.port}"
    user = f"{parts.username}:***@" if parts.username else ""
    return urlunsplit((parts.scheme, user + hostname, parts.path, parts.query, parts.fragment))


def camera_id_for_source(source: int | str) -> str:
    digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:10]
    return f"camera-{digest}"


class SignalBridge(QObject):
    event_received = pyqtSignal(object)
    video_frame = pyqtSignal(str, object, object)
    video_metrics = pyqtSignal(str, object)
    log = pyqtSignal(str, str)
    component_status = pyqtSignal(str, str, str)
    mic_level = pyqtSignal(float)


class StoppableThread(QThread):
    def stop(self, timeout_ms: int = 6_000) -> bool:
        self.requestInterruption()
        return self.wait(timeout_ms)

    def _pause(self, seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.isInterruptionRequested():
                return False
            self.msleep(min(100, max(1, int((deadline - time.monotonic()) * 1000))))
        return True


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    frame: np.ndarray
    captured_at: float
    sequence: int


class LatestFrameSlot:
    """A one-frame handoff: producers replace stale work instead of queuing it."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._latest: CapturedFrame | None = None
        self._sequence = 0
        self._consumed_sequence = 0
        self.dropped = 0

    def publish(self, frame: np.ndarray, captured_at: float | None = None) -> CapturedFrame:
        with self._condition:
            if self._sequence > self._consumed_sequence:
                self.dropped += 1
            self._sequence += 1
            packet = CapturedFrame(
                frame=frame,
                captured_at=time.monotonic() if captured_at is None else captured_at,
                sequence=self._sequence,
            )
            self._latest = packet
            self._condition.notify_all()
            return packet

    def take_after(self, sequence: int, timeout_sec: float) -> CapturedFrame | None:
        deadline = time.monotonic() + max(0.0, timeout_sec)
        with self._condition:
            while self._sequence <= sequence:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            assert self._latest is not None
            self._consumed_sequence = self._latest.sequence
            return self._latest


class AudioSystemWorker(StoppableThread):
    def __init__(
        self,
        bridge: SignalBridge,
        event_bus: EventBus,
        app_config: AppConfig,
        mic_device=None,
        detection_mode: str | None = None,
        peak_min: float | None = None,
        use_signature: bool | None = None,
        use_veto: bool | None = None,
        yamnet_threshold: float | None = None,
        enable_gunshot: bool = True,
        enable_speech: bool = True,
        camera_ids: tuple[str, ...] = (),
        frame_store: FrameStore | None = None,
    ):
        super().__init__()
        self.bridge = bridge
        self.event_bus = event_bus
        self.config = app_config
        self.mic_device = mic_device
        self.detection_mode = detection_mode or app_config.audio.detection_mode
        self.peak_min = app_config.audio.peak_min if peak_min is None else peak_min
        self.use_signature = app_config.audio.use_signature if use_signature is None else use_signature
        self.use_veto = app_config.audio.use_veto if use_veto is None else use_veto
        self.yamnet_threshold = (
            app_config.audio.yamnet_threshold if yamnet_threshold is None else yamnet_threshold
        )
        self.enable_gunshot = enable_gunshot
        self.enable_speech = enable_speech
        self.camera_ids = tuple(camera_ids)
        self.frame_store = frame_store

    def _status(self, state: ComponentState, message: str = "") -> None:
        self.bridge.component_status.emit("audio", state.value, message)

    def run(self) -> None:
        self._status(ComponentState.STARTING, "Загрузка моделей")
        microphone = None
        gunshot = None
        speech = None
        dual_resampler = None
        semantic_resampler = None
        try:
            # Heavy ML runtimes are imported in the worker, not during GUI startup.
            from audio.gunshot_detector import GunshotDetector
            from audio.keyword_detector import KeywordDetector

            audio = self.config.audio
            profile = resolve_audio_profile(audio, self.mic_device)
            if self.enable_gunshot:
                stats_path = Path(self.config.paths.detector_stats)
                device_suffix = hashlib.sha256(
                    str(self.mic_device).encode("utf-8")
                ).hexdigest()[:8]
                stats_path = stats_path.with_name(
                    f"{stats_path.stem}_{device_suffix}{stats_path.suffix}"
                )
                gunshot = GunshotDetector(
                    event_bus=self.event_bus,
                    model_path=audio.gunshot_model_path,
                    threshold=float(profile["gunshot_threshold"]),
                    ema_alpha=audio.ema_alpha,
                    cooldown_sec=audio.cooldown_sec,
                    analysis_step_sec=audio.analysis_step_sec,
                    rms_min=float(profile["rms_min"]),
                    peak_min=dbfs_to_amplitude(float(profile["trigger_dbfs"])),
                    sustain_windows=int(profile["sustain_windows"]),
                    sustain_hits=int(profile["sustain_hits"]),
                    sustain_thresh=float(profile["sustain_threshold"]),
                    use_signature=self.use_signature,
                    use_veto=self.use_veto,
                    veto_thresh=float(profile["yamnet_veto_threshold"]),
                    gun_thresh=float(profile["yamnet_threshold"]),
                    detection_mode=self.detection_mode,
                    stats_path=str(stats_path),
                    sample_rate=audio.panns_sample_rate,
                    yamnet_sample_rate=audio.sample_rate,
                    yamnet_model_path=audio.yamnet_model_path,
                    adaptive_noise=audio.adaptive_noise,
                    noise_alpha=audio.noise_alpha,
                    min_snr_db=float(profile["min_snr_db"]),
                    min_crest_factor=float(profile["min_crest_factor"]),
                    veto_margin=float(profile["yamnet_veto_margin"]),
                    cnn_override_threshold=float(profile["cnn_override_threshold"]),
                    panns_threshold=float(profile["panns_threshold"]),
                    panns_margin_threshold=float(profile["panns_margin_threshold"]),
                    panns_prefilter_threshold=audio.panns_prefilter_threshold,
                    panns_model_path=audio.panns_model_path,
                    panns_device=audio.panns_device,
                    analysis_window_sec=audio.analysis_window_sec,
                    multiscale_rescue=audio.multiscale_rescue,
                    rescue_window_sec=audio.rescue_window_sec,
                    microphone_source=audio_source_label(self.mic_device),
                    camera_ids=self.camera_ids,
                )
            speech = None
            if self.config.speech.enabled and self.enable_speech:
                speech = KeywordDetector(
                    event_bus=self.event_bus,
                    model_path=self.config.speech.vosk_model_path,
                    keyword=self.config.speech.keywords,
                    cooldown_sec=self.config.speech.cooldown_sec,
                    min_confidence=self.config.speech.min_confidence,
                    fuzzy_match=self.config.speech.fuzzy_match,
                    microphone_source=audio_source_label(self.mic_device),
                    camera_ids=self.camera_ids,
                    secondary_enabled=self.config.speech.secondary_enabled,
                    secondary_model_path=self.config.speech.secondary_model_path,
                    secondary_device=self.config.speech.secondary_device,
                    secondary_compute_type=self.config.speech.secondary_compute_type,
                    secondary_min_confidence=(
                        self.config.speech.secondary_min_confidence
                    ),
                    secondary_max_utterance_sec=(
                        self.config.speech.secondary_max_utterance_sec
                    ),
                )
            microphone = MicrophoneStream(
                sample_rate=audio.sample_rate,
                chunk_duration_sec=audio.chunk_duration_sec,
                device=self.mic_device,
                max_buffer_sec=audio.max_buffer_sec,
                allow_fallback=self.mic_device is None,
                preserve_native=True,
            )
            microphone.start()
            if gunshot is not None:
                dual_resampler = DualRateAudioResampler(
                    microphone.sample_rate,
                    audio.panns_sample_rate,
                    audio.sample_rate,
                )
            else:
                semantic_resampler = StreamingAudioResampler(
                    microphone.sample_rate,
                    audio.sample_rate,
                )
            self._status(
                ComponentState.RUNNING,
                f"Микрофон {audio_source_label(microphone.device)}",
            )
            if self.enable_gunshot:
                self.bridge.log.emit(
                    "info",
                    f"ПРИМЕНЕНЫ настройки микрофона {profile['profile_key']}: "
                    f"профиль={profile['preset']}, "
                    f"запуск={float(profile['trigger_dbfs']):.1f} dBFS, "
                    f"PANNs={float(profile['panns_threshold']):.3f}, "
                    f"margin={float(profile['panns_margin_threshold']):.3f}, "
                    f"SNR={float(profile['min_snr_db']):.1f} dB, "
                    f"импульсность={float(profile['min_crest_factor']):.2f}, "
                    f"YAMNet={float(profile['yamnet_threshold']):.2f}, "
                    f"частоты=native {microphone.sample_rate} → "
                    f"PANNs {audio.panns_sample_rate} / YAMNet+Vosk {audio.sample_rate}, "
                    f"режим={self.detection_mode}",
                )
            self.bridge.log.emit("info", "Аудиодетекторы активны")
            warned_silence = False
            silent_chunks = 0
            reported_drops = 0
            while not self.isInterruptionRequested():
                native_chunk = microphone.get_chunk(timeout=0.1)
                if native_chunk is None:
                    continue
                if dual_resampler is not None:
                    panns_chunk, semantic_chunk = dual_resampler.process(native_chunk)
                elif semantic_resampler is not None:
                    semantic_chunk = semantic_resampler.process(native_chunk)
                    panns_chunk = np.empty(0, dtype=np.float32)
                else:
                    panns_chunk = semantic_chunk = np.empty(0, dtype=np.float32)
                if self.frame_store is not None and semantic_chunk.size:
                    self.frame_store.update_audio(
                        self.camera_ids, semantic_chunk, audio.sample_rate
                    )
                peak = float(np.max(np.abs(native_chunk)))
                self.bridge.mic_level.emit(peak)
                silent_chunks = silent_chunks + 1 if peak <= 0.01 else 0
                if silent_chunks * audio.chunk_duration_sec >= 3.0 and not warned_silence:
                    warned_silence = True
                    self.bridge.log.emit("warning", "Микрофон открыт, но сигнал отсутствует")
                    self._status(ComponentState.DEGRADED, "Нет входного сигнала")
                elif peak > 0.01 and warned_silence:
                    warned_silence = False
                    self._status(ComponentState.RUNNING, "Сигнал восстановлен")
                if gunshot is not None and panns_chunk.size:
                    gunshot.process_audio(panns_chunk, semantic_chunk)
                if speech is not None and semantic_chunk.size:
                    speech.process_audio(semantic_chunk)
                if microphone.dropped_samples > reported_drops:
                    reported_drops = microphone.dropped_samples
                    self.bridge.log.emit(
                        "warning", f"Аудио не успевает обрабатываться: потеряно {reported_drops} сэмплов"
                    )
                    self._status(ComponentState.DEGRADED, "Переполнение аудиобуфера")
        except Exception as exc:
            logger.exception("Аудиоподсистема остановлена с ошибкой")
            self.bridge.log.emit("error", f"Аудиоподсистема: {exc}")
            self._status(ComponentState.ERROR, str(exc))
        finally:
            if dual_resampler is not None:
                dual_resampler.clear()
            if semantic_resampler is not None:
                semantic_resampler.clear()
            if microphone is not None:
                microphone.stop()
            if gunshot is not None:
                gunshot.close()
            if speech is not None:
                speech.close()
            if not self.isInterruptionRequested():
                return
            self._status(ComponentState.STOPPED, "Остановлено пользователем")


class VideoSystemWorker(StoppableThread):
    STREAM_SCHEMES = ("rtsp://", "rtmp://", "http://", "https://")
    # Rendering every input frame can overwhelm Qt and make its queued signals
    # display seconds-old video.  Detection remains uncapped; only preview is.
    PREVIEW_FPS = 10.0

    def __init__(
        self,
        bridge: SignalBridge,
        event_bus: EventBus,
        app_config: AppConfig,
        video_source: int | str,
        frame_store: FrameStore,
        camera_id: str | None = None,
    ):
        super().__init__()
        self.bridge = bridge
        self.event_bus = event_bus
        self.config = app_config
        self.video_source = video_source
        self.frame_store = frame_store
        self.camera_id = camera_id or camera_id_for_source(video_source)
        self._last_preview_emit = 0.0
        self._frames = LatestFrameSlot()
        self._reader_stop = threading.Event()
        self._reader_thread: threading.Thread | None = None
        self._reader_finished = threading.Event()
        self._reader_error = ""
        self._reader_fps = 30.0
        self._metric_lock = threading.Lock()
        self._read_times: deque[float] = deque(maxlen=120)
        self._input_times: deque[float] = deque(maxlen=120)

    def _status(self, state: ComponentState, message: str = "") -> None:
        self.bridge.component_status.emit(f"video:{self.camera_id}", state.value, message)

    def _is_stream(self) -> bool:
        return isinstance(self.video_source, str) and self.video_source.casefold().startswith(self.STREAM_SCHEMES)

    def _is_file(self) -> bool:
        return isinstance(self.video_source, str) and not self._is_stream()

    def _open_capture(self):
        capture = cv2.VideoCapture()
        if self._is_stream() and self.config.video.rtsp_low_latency:
            # OpenCV forwards these to its FFmpeg backend before ``open``.
            # TCP is reliable on the local network; nobuffer/low_delay make
            # the decoder discard latency rather than showing old frames.
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"
            )
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5_000)
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            capture.set(
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                self.config.video.rtsp_read_timeout_msec if self._is_stream() else 5_000,
            )
        capture.open(self.video_source, cv2.CAP_FFMPEG if self._is_stream() else cv2.CAP_ANY)
        if self._is_stream() and hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            # Not every FFmpeg build honours this property, so apply it after
            # open and continue when unsupported.
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _reader_loop(self) -> None:
        """Continuously drain capture into a single replaceable latest frame."""
        reconnect_delay = self.config.video.reconnect_initial_sec
        try:
            while not self._reader_stop.is_set():
                capture = self._open_capture()
                if not capture.isOpened():
                    capture.release()
                    self._reader_error = "Не удалось открыть видеопоток"
                    if self._is_file():
                        return
                    self._reader_stop.wait(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, self.config.video.reconnect_max_sec)
                    continue
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                self._reader_fps = fps if fps > 0 and not math.isnan(fps) else 30.0
                reconnect_delay = self.config.video.reconnect_initial_sec
                frame_started = time.perf_counter()
                while not self._reader_stop.is_set():
                    started = time.monotonic()
                    ok, frame = capture.read()
                    finished = time.monotonic()
                    with self._metric_lock:
                        self._read_times.append(finished - started)
                    if not ok:
                        self._reader_error = "Видеопоток прерван"
                        break
                    self._frames.publish(frame, finished)
                    with self._metric_lock:
                        self._input_times.append(finished)
                    if self._is_file():
                        elapsed = time.perf_counter() - frame_started
                        delay = max(0.0, 1.0 / self._reader_fps - elapsed)
                        if self._reader_stop.wait(delay):
                            break
                        frame_started = time.perf_counter()
                capture.release()
                if self._is_file():
                    return
                self._reader_stop.wait(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, self.config.video.reconnect_max_sec)
        finally:
            self._reader_finished.set()

    def _start_reader(self) -> None:
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name=f"rtsp-reader-{self.camera_id}",
            daemon=True,
        )
        self._reader_thread.start()

    def _stop_reader(self) -> None:
        self._reader_stop.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)

    @staticmethod
    def _p95(values: deque[float]) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(math.ceil(len(ordered) * 0.95)) - 1)]

    def _emit_metrics(self, inference_times: deque[float], preview_count: int, last_age: float) -> None:
        now = time.monotonic()
        with self._metric_lock:
            input_times = tuple(self._input_times)
            read_times = deque(self._read_times)
        input_fps = 0.0
        if len(input_times) > 1 and input_times[-1] > input_times[0]:
            input_fps = (len(input_times) - 1) / (input_times[-1] - input_times[0])
        analysis_fps = 0.0
        if len(inference_times) > 1:
            total = sum(inference_times)
            analysis_fps = len(inference_times) / total if total else 0.0
        self.bridge.video_metrics.emit(self.camera_id, {
            "age_sec": round(last_age, 3),
            "input_fps": round(input_fps, 1),
            "analysis_fps": round(analysis_fps, 1),
            "preview_fps": self.PREVIEW_FPS,
            "dropped_frames": self._frames.dropped,
            "read_p95_ms": round(self._p95(read_times) * 1000, 1),
            "inference_p95_ms": round(self._p95(inference_times) * 1000, 1),
        })

    def run(self) -> None:
        source_label = safe_source_label(self.video_source)
        self._status(ComponentState.STARTING, source_label)
        failed = False
        try:
            from video.fall_detector import FallDetector

            if self._is_file() and not Path(str(self.video_source)).is_file():
                raise FileNotFoundError(f"Видеофайл не найден: {self.video_source}")
            self._start_reader()
            first = None
            while not self.isInterruptionRequested() and first is None:
                first = self._frames.take_after(0, 0.5)
                if first is None and self._reader_finished.is_set():
                    raise RuntimeError(self._reader_error or "Видеопоток не дал ни одного кадра")
            if first is None:
                return

            video = self.config.video
            detector = FallDetector(
                camera_id=self.camera_id, model_path=video.model_path,
                view_mode=video.view_mode_by_source.get(str(self.video_source), "side"),
                fall_duration_sec=video.fall_duration_sec, reset_grace_sec=video.reset_grace_sec,
                missing_grace_sec=video.missing_grace_sec, fps=self._reader_fps,
                use_wall_clock=not self._is_file(), pose_confidence=video.pose_confidence,
                fall_min_confidence=video.fall_min_confidence, keypoint_confidence=video.keypoint_confidence,
                partial_pose_confidence=video.partial_pose_confidence,
                partial_pose_min_keypoints=video.partial_pose_min_keypoints,
                partial_lie_min_score=video.partial_lie_min_score,
                standing_angle_deg=video.standing_angle_deg, lying_angle_deg=video.lying_angle_deg,
                lying_aspect_ratio=video.lying_aspect_ratio,
                seated_knee_drop_ratio=video.seated_knee_drop_ratio,
                require_upright_transition=video.require_upright_transition,
                posture_window=video.posture_window, posture_hits=video.posture_hits,
                rapid_descent_heights_per_sec=video.rapid_descent_heights_per_sec,
                transition_memory_sec=video.transition_memory_sec,
                motion_confirm_sec=video.motion_confirm_sec, require_fall_motion=video.require_fall_motion,
                inference_imgsz=video.inference_imgsz, use_half_precision=video.use_half_precision,
                lying_classifier_enabled=video.lying_classifier_enabled,
                lying_classifier_model_path=video.lying_classifier_model_path,
                lying_classifier_label=video.lying_classifier_label,
                lying_classifier_imgsz=video.lying_classifier_imgsz,
                lying_classifier_hz=video.lying_classifier_hz,
                lying_classifier_threshold=video.lying_classifier_threshold,
                lying_alert_score_threshold=video.lying_alert_score_threshold,
                lying_classifier_pose_quality_min=video.lying_classifier_pose_quality_min,
                lying_classifier_rgb_weight=video.lying_classifier_rgb_weight,
                lying_stable_duration_sec=video.lying_stable_duration_sec,
                lying_stable_motion_heights_per_sec=video.lying_stable_motion_heights_per_sec,
            )
            self._status(ComponentState.RUNNING, source_label)
            sequence = first.sequence
            inference_times: deque[float] = deque(maxlen=120)
            preview_count = 0
            last_metrics_at = 0.0
            while not self.isInterruptionRequested():
                packet = self._frames.take_after(sequence, 1.0)
                if packet is None:
                    if self._reader_finished.is_set():
                        break
                    continue
                sequence = packet.sequence
                started = time.monotonic()
                analysis = detector.process_frame(packet.frame)
                finished = time.monotonic()
                inference_times.append(finished - started)
                annotated = analysis.annotated_frame
                self.frame_store.update(self.camera_id, source_label, annotated, captured_at=packet.captured_at)
                if finished - self._last_preview_emit >= 1.0 / self.PREVIEW_FPS:
                    self._last_preview_emit = finished
                    preview_count += 1
                    self.bridge.video_frame.emit(self.camera_id, packet.frame, annotated)
                for event in analysis.events:
                    self.event_bus.publish(event)
                age = finished - packet.captured_at
                if age > 1.0:
                    self._status(ComponentState.DEGRADED, f"Аналитика отстаёт: кадр {age:.1f}с")
                elif last_metrics_at and finished - last_metrics_at >= 1.0:
                    self._status(ComponentState.RUNNING, source_label)
                if finished - last_metrics_at >= 1.0:
                    self._emit_metrics(inference_times, preview_count, age)
                    preview_count = 0
                    last_metrics_at = finished
        except Exception as exc:
            failed = True
            logger.exception("Видеоподсистема остановлена с ошибкой")
            self.bridge.log.emit("error", f"Видеоподсистема: {exc}")
            self._status(ComponentState.ERROR, str(exc))
        finally:
            self._stop_reader()
        if not failed:
            self._status(ComponentState.STOPPED, "Остановлено")


class MicCheckWorker(QThread):
    completed = pyqtSignal(bool)

    def __init__(self, bridge: SignalBridge, device, device_label: str):
        super().__init__()
        self.bridge = bridge
        self.device = device
        self.device_label = device_label

    def run(self) -> None:
        try:
            peak = MicrophoneStream.test_device(self.device, seconds=2.0)
            if peak < 0.01:
                self.bridge.log.emit("warning", f"{self.device_label}: устройство открыто, но сигнал отсутствует")
                self.completed.emit(False)
            else:
                self.bridge.log.emit("info", f"{self.device_label}: микрофон работает, peak={peak:.2f}")
                self.completed.emit(True)
        except Exception as exc:
            self.bridge.log.emit("error", f"{self.device_label}: устройство не открылось: {exc}")
            self.completed.emit(False)


class AmbientCalibrationWorker(QThread):
    progress = pyqtSignal(int)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, device, duration_sec: float, preset: str):
        super().__init__()
        self.device = device
        self.duration_sec = min(max(float(duration_sec), 3.0), 60.0)
        self.preset = preset
        # Kept only in memory so the manual-distance step can reuse the
        # background captured in step 3 without asking the user to record it
        # a second time.
        self.captured_chunks: list[np.ndarray] | None = None

    def run(self) -> None:
        microphone = None
        try:
            microphone = MicrophoneStream(
                sample_rate=16_000,
                chunk_duration_sec=0.25,
                device=self.device,
                max_buffer_sec=2.0,
                allow_fallback=self.device is None,
            )
            microphone.start()
            chunks: list[np.ndarray] = []
            started = time.monotonic()
            while not self.isInterruptionRequested():
                elapsed = time.monotonic() - started
                if elapsed >= self.duration_sec:
                    break
                chunk = microphone.get_chunk(timeout=0.2)
                if chunk is not None:
                    chunks.append(chunk.copy())
                self.progress.emit(min(99, round(elapsed / self.duration_sec * 100)))
            if self.isInterruptionRequested():
                return
            result = analyze_ambient_audio(chunks, self.preset)
            self.captured_chunks = chunks
            self.progress.emit(100)
            self.completed.emit(result)
        except Exception as exc:
            logger.exception(
                "Не удалось откалибровать микрофон %s",
                microphone_profile_key(self.device),
            )
            self.failed.emit(str(exc))
        finally:
            if microphone is not None:
                microphone.stop()


class ReferenceCalibrationWorker(QThread):
    """Guided, in-memory background plus reference-impulse recording."""
    progress = pyqtSignal(int)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)
    stage = pyqtSignal(str)

    def __init__(
        self, device, ambient_duration_sec: float, preset: str, reference_count: int = 3
    ):
        super().__init__()
        self.device = device
        self.ambient_duration_sec = min(max(float(ambient_duration_sec), 3.0), 60.0)
        self.preset = preset
        self.reference_count = min(max(int(reference_count), 3), 4)
        self.reference_duration_sec = 3.0
        self.preparation_sec = 4.0

    def _collect(
        self, microphone, duration_sec: float, started: float, total_sec: float, keep: bool
    ):
        chunks: list[np.ndarray] = []
        phase_started = time.monotonic()
        while not self.isInterruptionRequested():
            if time.monotonic() - phase_started >= duration_sec:
                break
            chunk = microphone.get_chunk(timeout=0.2)
            if keep and chunk is not None:
                chunks.append(chunk.copy())
            elapsed = time.monotonic() - started
            self.progress.emit(min(99, round(elapsed / total_sec * 100)))
        return chunks

    def run(self) -> None:
        microphone = None
        try:
            microphone = MicrophoneStream(
                sample_rate=16_000,
                chunk_duration_sec=0.25,
                device=self.device,
                max_buffer_sec=2.0,
                allow_fallback=self.device is None,
            )
            microphone.start()
            total_sec = self.ambient_duration_sec + self.reference_count * (
                self.preparation_sec + self.reference_duration_sec
            )
            started = time.monotonic()
            self.stage.emit(
                "Запись обычного фона: говорите и включите типичные шумы помещения."
            )
            ambient = self._collect(microphone, self.ambient_duration_sec, started, total_sec, True)
            if self.isInterruptionRequested():
                return
            takes: list[list[np.ndarray]] = []
            for index in range(self.reference_count):
                self.stage.emit(
                    f"Контроль {index + 1}/{self.reference_count}: подготовьте тестовый "
                    "источник на выбранной дистанции."
                )
                self._collect(microphone, self.preparation_sec, started, total_sec, False)
                if self.isInterruptionRequested():
                    return
                self.stage.emit(
                    f"Контроль {index + 1}/{self.reference_count}: "
                    "идёт запись тестового импульса."
                )
                takes.append(
                    self._collect(
                        microphone, self.reference_duration_sec, started, total_sec, True
                    )
                )
            if self.isInterruptionRequested():
                return
            result = analyze_reference_audio(ambient, takes, self.preset)
            self.progress.emit(100)
            self.completed.emit(result)
        except Exception as exc:
            logger.exception(
                "Не удалось выполнить полную калибровку микрофона %s",
                microphone_profile_key(self.device),
            )
            self.failed.emit(str(exc))
        finally:
            if microphone is not None:
                microphone.stop()
class CalibrationCaptureWorker(QThread):
    """Capture one labelled calibration segment without persisting raw audio."""

    progress = pyqtSignal(int)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, device, duration_sec: float):
        super().__init__()
        self.device = device
        self.duration_sec = min(max(float(duration_sec), 2.0), 60.0)

    def run(self) -> None:
        microphone = None
        try:
            microphone = MicrophoneStream(
                sample_rate=16_000,
                chunk_duration_sec=0.25,
                device=self.device,
                max_buffer_sec=2.0,
                allow_fallback=self.device is None,
            )
            microphone.start()
            chunks: list[np.ndarray] = []
            started = time.monotonic()
            while not self.isInterruptionRequested():
                elapsed = time.monotonic() - started
                if elapsed >= self.duration_sec:
                    break
                chunk = microphone.get_chunk(timeout=0.2)
                if chunk is not None:
                    chunks.append(chunk.copy())
                self.progress.emit(min(99, round(elapsed / self.duration_sec * 100)))
            if not self.isInterruptionRequested():
                self.progress.emit(100)
                self.completed.emit(chunks)
        except Exception as exc:
            logger.exception("Не удалось записать сегмент калибровки микрофона")
            self.failed.emit(str(exc))
        finally:
            if microphone is not None:
                microphone.stop()
