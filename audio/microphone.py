from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from math import gcd
from urllib.parse import urlsplit, urlunsplit

import numpy as np

logger = logging.getLogger(__name__)
NETWORK_AUDIO_SCHEMES = ("rtsp://", "rtmp://", "http://", "https://")


def is_network_audio_source(value) -> bool:
    return isinstance(value, str) and value.strip().casefold().startswith(
        NETWORK_AUDIO_SCHEMES
    )


def audio_source_label(value) -> str:
    """Return an operator-facing source label without exposing URL credentials."""
    if not is_network_audio_source(value):
        return "default" if value is None else str(value)
    parsed = urlsplit(str(value).strip())
    host = parsed.hostname or "network-audio"
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def dbfs_to_amplitude(dbfs: float) -> float:
    """Convert a full-scale decibel threshold to linear floating-point amplitude."""
    return 10.0 ** (float(dbfs) / 20.0)


class StreamingAudioResampler:
    """Stateful high-quality mono resampler without per-block edge artifacts."""

    def __init__(self, source_rate: int, target_rate: int):
        self.source_rate = int(source_rate)
        self.target_rate = int(target_rate)
        if self.source_rate <= 0 or self.target_rate <= 0:
            raise ValueError("Частоты дискретизации должны быть положительными")
        self._stream = None
        if self.source_rate != self.target_rate:
            import soxr

            self._stream = soxr.ResampleStream(
                self.source_rate,
                self.target_rate,
                1,
                dtype="float32",
                quality="HQ",
            )

    def process(self, audio_data: np.ndarray, last: bool = False) -> np.ndarray:
        data = np.asarray(audio_data, dtype=np.float32).reshape(-1)
        if not data.size:
            if self._stream is not None and last:
                return np.asarray(
                    self._stream.resample_chunk(data, last=True),
                    dtype=np.float32,
                ).reshape(-1)
            return np.empty(0, dtype=np.float32)
        if self._stream is None:
            return data.copy()
        return np.asarray(
            self._stream.resample_chunk(data, last=bool(last)),
            dtype=np.float32,
        ).reshape(-1)

    def clear(self) -> None:
        if self._stream is not None:
            self._stream.clear()


class DualRateAudioResampler:
    """Keep PANNs and semantic branches sample-aligned without dropping data.

    Stateful resamplers may buffer a fractional frame and return an empty output
    from one branch while the other branch already has samples. Keeping pending
    samples until both branches can form the same duration prevents a short
    impulse from being silently lost at startup or around rate boundaries.
    """

    def __init__(
        self,
        source_rate: int,
        panns_rate: int = 32_000,
        semantic_rate: int = 16_000,
    ):
        self.panns_rate = int(panns_rate)
        self.semantic_rate = int(semantic_rate)
        if self.panns_rate <= 0 or self.semantic_rate <= 0:
            raise ValueError("Частоты веток должны быть положительными")
        common_rate = gcd(self.panns_rate, self.semantic_rate)
        self._panns_quantum = self.panns_rate // common_rate
        self._semantic_quantum = self.semantic_rate // common_rate
        self._panns_resampler = StreamingAudioResampler(source_rate, self.panns_rate)
        self._semantic_resampler = StreamingAudioResampler(source_rate, self.semantic_rate)
        self._panns_pending = np.empty(0, dtype=np.float32)
        self._semantic_pending = np.empty(0, dtype=np.float32)

    @staticmethod
    def _append(pending: np.ndarray, chunk: np.ndarray) -> np.ndarray:
        if not chunk.size:
            return pending
        if not pending.size:
            return chunk.copy()
        return np.concatenate((pending, chunk))

    def process(self, audio_data: np.ndarray, last: bool = False) -> tuple[np.ndarray, np.ndarray]:
        self._panns_pending = self._append(
            self._panns_pending,
            self._panns_resampler.process(audio_data, last=last),
        )
        self._semantic_pending = self._append(
            self._semantic_pending,
            self._semantic_resampler.process(audio_data, last=last),
        )
        quanta = min(
            len(self._panns_pending) // self._panns_quantum,
            len(self._semantic_pending) // self._semantic_quantum,
        )
        if quanta <= 0:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
        panns_count = quanta * self._panns_quantum
        semantic_count = quanta * self._semantic_quantum
        panns = self._panns_pending[:panns_count].copy()
        semantic = self._semantic_pending[:semantic_count].copy()
        self._panns_pending = self._panns_pending[panns_count:]
        self._semantic_pending = self._semantic_pending[semantic_count:]
        return panns, semantic

    def clear(self) -> None:
        self._panns_resampler.clear()
        self._semantic_resampler.clear()
        self._panns_pending = np.empty(0, dtype=np.float32)
        self._semantic_pending = np.empty(0, dtype=np.float32)


class MicrophoneStream:
    """Bounded microphone stream in native or normalized sample rate."""

    def __init__(
        self,
        sample_rate: int = 16_000,
        chunk_duration_sec: float = 0.25,
        device=None,
        max_buffer_sec: float = 3.0,
        allow_fallback: bool = True,
        preserve_native: bool = False,
    ):
        self.requested_sample_rate = int(sample_rate)
        self.sample_rate = int(sample_rate)
        self.chunk_duration_sec = float(chunk_duration_sec)
        self.max_buffer_sec = float(max_buffer_sec)
        self.chunk_size = int(self.sample_rate * self.chunk_duration_sec)
        self.max_buffer_samples = max(self.chunk_size, int(self.sample_rate * self.max_buffer_sec))
        self.requested_device = device
        self.allow_fallback = allow_fallback
        self.preserve_native = bool(preserve_native)
        self.device = device
        self.device_sample_rate = self.sample_rate
        self._capture_resampler = StreamingAudioResampler(self.device_sample_rate, self.sample_rate)
        self.dropped_samples = 0
        self._chunks: deque[np.ndarray] = deque()
        self._available_samples = 0
        self._condition = threading.Condition()
        self._stream = None
        self._network_capture = None
        self._network_thread: threading.Thread | None = None
        self._network_stop = threading.Event()
        self._stopped = True

    @staticmethod
    def list_input_devices() -> list[tuple[int, str]]:
        result = []
        try:
            import sounddevice as sd

            devices = list(sd.query_devices())
            preferred_hostapi = None
            if os.name == "nt":
                for index, hostapi in enumerate(sd.query_hostapis()):
                    if "wasapi" in str(hostapi.get("name", "")).casefold():
                        preferred_hostapi = index
                        break
            for idx, device in enumerate(devices):
                if device.get("max_input_channels", 0) > 0:
                    if preferred_hostapi is not None and device.get("hostapi") != preferred_hostapi:
                        continue
                    result.append((idx, str(device["name"])))
            if not result and preferred_hostapi is not None:
                result = [
                    (idx, str(device["name"]))
                    for idx, device in enumerate(devices)
                    if device.get("max_input_channels", 0) > 0
                ]
        except Exception:
            logger.exception("Не удалось получить список микрофонов")
        return result

    @staticmethod
    def test_device(device=None, seconds: float = 2.0) -> float:
        if is_network_audio_source(device):
            stream = MicrophoneStream(device=device, allow_fallback=False)
            peak = 0.0
            stream.start()
            deadline = time.monotonic() + seconds
            try:
                while time.monotonic() < deadline:
                    chunk = stream.get_chunk(timeout=0.2)
                    if chunk is not None and len(chunk):
                        peak = max(peak, float(np.max(np.abs(chunk))))
            finally:
                stream.stop()
            return peak

        import sounddevice as sd

        state = {"peak": 0.0}
        info = sd.query_devices(device, "input")
        native_rate = int(info.get("default_samplerate", 16_000))

        def callback(indata, frames, time_info, status):
            del frames, time_info
            if status:
                logger.warning("Проверка микрофона: %s", status)
            if len(indata):
                state["peak"] = max(state["peak"], float(np.max(np.abs(indata[:, 0]))))

        stream = sd.InputStream(
            samplerate=native_rate, channels=1, dtype="float32", device=device, callback=callback
        )
        try:
            stream.start()
            time.sleep(seconds)
        finally:
            stream.stop()
            stream.close()
        return state["peak"]

    def _configure_device_rate(self, native_rate: int) -> None:
        self.device_sample_rate = int(native_rate) if int(native_rate) > 0 else self.requested_sample_rate
        self.sample_rate = (
            self.device_sample_rate if self.preserve_native else self.requested_sample_rate
        )
        self.chunk_size = max(1, int(round(self.sample_rate * self.chunk_duration_sec)))
        self.max_buffer_samples = max(
            self.chunk_size,
            int(round(self.sample_rate * self.max_buffer_sec)),
        )
        self._capture_resampler = StreamingAudioResampler(
            self.device_sample_rate,
            self.sample_rate,
        )

    def _resample(self, audio_data: np.ndarray) -> np.ndarray:
        expected = (self.device_sample_rate, self.sample_rate)
        actual = (
            self._capture_resampler.source_rate,
            self._capture_resampler.target_rate,
        )
        if actual != expected:
            self._capture_resampler = StreamingAudioResampler(*expected)
        return self._capture_resampler.process(audio_data)

    def _callback(self, indata, frames, time_info, status) -> None:
        del frames, time_info
        if status:
            logger.warning("[Microphone] status: %s", status)
        if not len(indata):
            return
        audio_data = self._resample(indata[:, 0].astype(np.float32, copy=True))
        self._enqueue(audio_data)

    def _enqueue(self, audio_data: np.ndarray) -> None:
        if not len(audio_data):
            return
        with self._condition:
            self._chunks.append(audio_data)
            self._available_samples += len(audio_data)
            while self._available_samples > self.max_buffer_samples and self._chunks:
                removed = self._chunks.popleft()
                self._available_samples -= len(removed)
                self.dropped_samples += len(removed)
            self._condition.notify()

    @staticmethod
    def _network_audio_to_float(audio) -> np.ndarray:
        data = np.asarray(audio)
        if data.size == 0:
            return np.empty(0, dtype=np.float32)
        source_dtype = data.dtype
        if data.ndim > 1:
            # OpenCV normally returns channels x samples, but accept the
            # transposed form as well for backend compatibility.
            channel_axis = 0 if data.shape[0] <= 8 else 1
            data = data.astype(np.float32).mean(axis=channel_axis)
        if np.issubdtype(source_dtype, np.unsignedinteger):
            info = np.iinfo(source_dtype)
            midpoint = (info.max + 1) / 2.0
            normalized = (data.astype(np.float32) - midpoint) / midpoint
        elif np.issubdtype(source_dtype, np.signedinteger):
            info = np.iinfo(source_dtype)
            scale = float(max(abs(info.min), info.max))
            normalized = data.astype(np.float32) / scale
        else:
            normalized = data.astype(np.float32, copy=False)
            peak = float(np.max(np.abs(normalized))) if normalized.size else 0.0
            if peak > 2.0:
                normalized = normalized / 32768.0
        return np.clip(normalized.reshape(-1), -1.0, 1.0)

    def _read_network_audio(self) -> bool:
        capture = self._network_capture
        if capture is None or not capture.grab():
            return False
        import cv2

        base_index = int(capture.get(cv2.CAP_PROP_AUDIO_BASE_INDEX))
        ok, audio = capture.retrieve(None, base_index)
        if not ok or audio is None:
            return True
        audio_data = self._network_audio_to_float(audio)
        self._enqueue(self._resample(audio_data))
        return True

    def _network_loop(self) -> None:
        try:
            while not self._network_stop.is_set() and self._read_network_audio():
                pass
            if not self._network_stop.is_set():
                logger.error(
                    "RTSP-аудиопоток завершился: %s",
                    audio_source_label(self.requested_device),
                )
        except Exception:
            logger.exception(
                "Ошибка чтения RTSP-аудио: %s",
                audio_source_label(self.requested_device),
            )
        finally:
            self._stopped = True
            with self._condition:
                self._condition.notify_all()

    def _open_network_stream(self, source: str) -> None:
        import cv2

        required = (
            "CAP_PROP_AUDIO_STREAM",
            "CAP_PROP_AUDIO_BASE_INDEX",
            "CAP_PROP_AUDIO_TOTAL_STREAMS",
            "CAP_PROP_AUDIO_SAMPLES_PER_SECOND",
        )
        if not all(hasattr(cv2, name) for name in required):
            raise RuntimeError("Эта сборка OpenCV не поддерживает аудиодорожки RTSP")
        params = [
            cv2.CAP_PROP_VIDEO_STREAM,
            -1,
            cv2.CAP_PROP_AUDIO_STREAM,
            0,
        ]
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            params.extend((cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5_000))
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            params.extend((cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2_000))
        capture = cv2.VideoCapture()
        if not capture.open(source, cv2.CAP_FFMPEG, params):
            capture.release()
            raise RuntimeError(
                f"Не удалось открыть RTSP-аудиопоток: {audio_source_label(source)}"
            )
        if int(capture.get(cv2.CAP_PROP_AUDIO_TOTAL_STREAMS)) < 1:
            capture.release()
            raise RuntimeError("В указанном RTSP-потоке нет аудиодорожки")
        native_rate = int(capture.get(cv2.CAP_PROP_AUDIO_SAMPLES_PER_SECOND))
        self._configure_device_rate(native_rate)
        self.device = source
        self._network_capture = capture
        self._network_stop.clear()
        self._stopped = False
        self._network_thread = threading.Thread(
            target=self._network_loop,
            name="rtsp-audio-reader",
            daemon=True,
        )
        self._network_thread.start()
        logger.info(
            "RTSP-микрофон запущен: source=%s, native_sr=%s",
            audio_source_label(source),
            self.device_sample_rate,
        )

    def _open_stream(self, device) -> None:
        import sounddevice as sd

        info = sd.query_devices(device, "input")
        self._configure_device_rate(int(info.get("default_samplerate", self.requested_sample_rate)))
        stream = sd.InputStream(
            samplerate=self.device_sample_rate,
            channels=1,
            dtype="float32",
            device=device,
            callback=self._callback,
        )
        stream.start()
        self._stream = stream
        self.device = device

    def start(self) -> None:
        if is_network_audio_source(self.requested_device):
            self._open_network_stream(str(self.requested_device).strip())
            return
        candidates = []
        fallback_devices = [None, *[idx for idx, _ in self.list_input_devices()]]
        requested = [self.requested_device]
        for candidate in requested + (fallback_devices if self.allow_fallback else []):
            if candidate not in candidates:
                candidates.append(candidate)
        last_error: Exception | None = None
        for device in candidates:
            try:
                self._open_stream(device)
                self._stopped = False
                if device != self.requested_device:
                    logger.warning(
                        "Микрофон %s недоступен; выбран %s", self.requested_device, device
                    )
                logger.info(
                    "Микрофон запущен: device=%s, native_sr=%s", device, self.device_sample_rate
                )
                return
            except Exception as exc:
                last_error = exc
                if self._stream is not None:
                    try:
                        self._stream.close()
                    except Exception:
                        pass
                    self._stream = None
        raise RuntimeError("Не найдено доступного устройства записи") from last_error

    def get_chunk(self, timeout: float = 0.1) -> np.ndarray | None:
        with self._condition:
            if self._available_samples < self.chunk_size and not self._stopped:
                self._condition.wait(timeout)
            if self._available_samples < self.chunk_size:
                return None

            parts: list[np.ndarray] = []
            needed = self.chunk_size
            while needed > 0:
                first = self._chunks.popleft()
                if len(first) <= needed:
                    parts.append(first)
                    needed -= len(first)
                else:
                    parts.append(first[:needed])
                    self._chunks.appendleft(first[needed:])
                    needed = 0
            self._available_samples -= self.chunk_size
        return np.concatenate(parts) if len(parts) > 1 else parts[0].copy()

    def stop(self) -> None:
        self._stopped = True
        self._network_stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._network_capture is not None:
            try:
                self._network_capture.release()
            except Exception:
                logger.exception("Ошибка остановки RTSP-микрофона")
            finally:
                self._network_capture = None
        if self._network_thread is not None:
            self._network_thread.join(3.0)
            self._network_thread = None
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                logger.exception("Ошибка остановки микрофона")
            finally:
                self._stream = None
        self._capture_resampler.clear()
        logger.info("Микрофонный поток остановлен; потеряно сэмплов: %d", self.dropped_samples)
