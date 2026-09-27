from __future__ import annotations

import json
import logging
import math
import queue
import re
import threading
import time
import unicodedata
from pathlib import Path

import numpy as np

from core.config import config
from core.event_bus import Event, EventBus, EventType, Severity
from core.vosk_model_path import vosk_runtime_path

logger = logging.getLogger(__name__)


def normalize_speech_text(text: str) -> str:
    """Normalize recognizer output without losing Russian word boundaries."""
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("ё", "е")
    normalized = re.sub(r"[^\w-]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def _edit_distance_at_most_one(left: str, right: str) -> bool:
    """Fast bounded Levenshtein check used only for long alarm words."""
    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) > len(right):
        left, right = right, left
    if len(left) == len(right):
        mismatches = sum(a != b for a, b in zip(left, right))
        return mismatches <= 1
    index_left = index_right = differences = 0
    while index_left < len(left) and index_right < len(right):
        if left[index_left] == right[index_right]:
            index_left += 1
        else:
            differences += 1
            if differences > 1:
                return False
        index_right += 1
    return True


class KeywordDetector:
    def __init__(
        self,
        event_bus: EventBus,
        model_path: str | None = None,
        keyword=None,
        cooldown_sec: float | None = None,
        min_confidence: float | None = None,
        fuzzy_match: bool | None = None,
        microphone_source: str = "default",
        camera_ids: tuple[str, ...] = (),
        secondary_enabled: bool = False,
        secondary_model_path: str | None = None,
        secondary_device: str = "auto",
        secondary_compute_type: str = "auto",
        secondary_min_confidence: float = 0.30,
        secondary_max_utterance_sec: float = 12.0,
    ):
        self.event_bus = event_bus
        keywords = config.speech.keywords if keyword is None else keyword
        if isinstance(keywords, str):
            keywords = [keywords]
        self.keywords = [normalize_speech_text(word) for word in keywords if word.strip()]
        if not self.keywords:
            raise ValueError("Список тревожных слов пуст")
        self._patterns = {
            word: re.compile(rf"(?<![\w-]){re.escape(word)}(?![\w-])", re.IGNORECASE)
            for word in self.keywords
        }
        self.cooldown_sec = cooldown_sec if cooldown_sec is not None else config.speech.cooldown_sec
        self.min_confidence = (
            config.speech.min_confidence if min_confidence is None else min_confidence
        )
        self.fuzzy_match = config.speech.fuzzy_match if fuzzy_match is None else fuzzy_match
        self.last_event_time = -float("inf")
        self.sample_rate = config.audio.sample_rate
        self.microphone_source = microphone_source
        self.camera_ids = tuple(camera_ids)
        self.secondary_enabled = bool(secondary_enabled)
        self.secondary_model_path = str(secondary_model_path or "")
        self.secondary_device = str(secondary_device or "auto")
        self.secondary_compute_type = str(secondary_compute_type or "auto")
        self.secondary_min_confidence = min(
            max(float(secondary_min_confidence), 0.0), 1.0
        )
        self._max_utterance_samples = max(
            self.sample_rate,
            int(self.sample_rate * max(float(secondary_max_utterance_sec), 2.0)),
        )
        self._utterance_chunks: list[np.ndarray] = []
        self._utterance_samples = 0
        self._secondary_queue: queue.Queue[
            tuple[np.ndarray, tuple[tuple[str, float], ...]] | None
        ] | None = None
        self._secondary_closed = threading.Event()
        self._secondary_ready = threading.Event()
        self._secondary_worker: threading.Thread | None = None

        selected_path = model_path or config.speech.vosk_model_path
        logger.info("Загрузка модели Vosk: %s", selected_path)
        try:
            from vosk import KaldiRecognizer, Model

            runtime_path = vosk_runtime_path(selected_path)
            if runtime_path != str(Path(selected_path).resolve()):
                logger.info("Vosk использует совместимый ASCII-путь: %s", runtime_path)
            self.model = Model(runtime_path)
            # Do not constrain Vosk to alarm words. A restricted grammar turns
            # unrelated short sounds into the nearest allowed command (for
            # example, «бдыщ» into «помощь»), which is unacceptable for alarms.
            # We recognise normal speech first and then apply exact keyword
            # matching below.
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
            if hasattr(self.recognizer, "SetWords"):
                self.recognizer.SetWords(True)
        except Exception as exc:
            raise RuntimeError(f"Не удалось загрузить обязательную модель Vosk: {selected_path}") from exc

        secondary_path = Path(self.secondary_model_path)
        if self.secondary_enabled and secondary_path.is_dir():
            self._secondary_queue = queue.Queue(maxsize=4)
            self._secondary_worker = threading.Thread(
                target=self._secondary_loop,
                name=f"speech-secondary-{self.microphone_source}",
                daemon=True,
            )
            self._secondary_worker.start()
        elif self.secondary_enabled:
            logger.warning(
                "Резервная модель Faster-Whisper не найдена: %s; Vosk продолжит работу",
                secondary_path,
            )

    def find_keywords(self, text: str) -> list[tuple[str, int]]:
        normalized = normalize_speech_text(text)
        exact = [
            (word, len(pattern.findall(normalized)))
            for word, pattern in self._patterns.items()
            if pattern.search(normalized)
        ]
        if exact or not self.fuzzy_match:
            return exact

        tokens = normalized.split()
        fuzzy: list[tuple[str, int]] = []
        for keyword in self.keywords:
            keyword_tokens = keyword.split()
            if len(keyword_tokens) != 1 or len(keyword) < 5:
                continue
            count = sum(_edit_distance_at_most_one(keyword, token) for token in tokens)
            if count:
                fuzzy.append((keyword, count))
        return fuzzy

    @staticmethod
    def _confidence(result: dict) -> float:
        words = result.get("result") or []
        values = [float(item["conf"]) for item in words if "conf" in item]
        return sum(values) / len(values) if values else 0.5

    def _keyword_confidence(self, result: dict, found: list[tuple[str, int]]) -> float:
        keyword_tokens = {token for keyword, _ in found for token in keyword.split()}
        matched = []
        for item in result.get("result") or []:
            word = normalize_speech_text(str(item.get("word", "")))
            if any(_edit_distance_at_most_one(word, expected) for expected in keyword_tokens):
                if "conf" in item:
                    matched.append(float(item["conf"]))
        return sum(matched) / len(matched) if matched else self._confidence(result)

    def _append_utterance_audio(self, audio: np.ndarray) -> None:
        chunk = np.asarray(audio, dtype=np.float32).reshape(-1).copy()
        if not chunk.size:
            return
        self._utterance_chunks.append(chunk)
        self._utterance_samples += len(chunk)
        overflow = self._utterance_samples - self._max_utterance_samples
        while overflow > 0 and self._utterance_chunks:
            first = self._utterance_chunks[0]
            if len(first) <= overflow:
                self._utterance_chunks.pop(0)
                self._utterance_samples -= len(first)
                overflow -= len(first)
            else:
                self._utterance_chunks[0] = first[overflow:].copy()
                self._utterance_samples -= overflow
                overflow = 0

    def _take_utterance_audio(self) -> np.ndarray:
        if not self._utterance_chunks:
            return np.empty(0, dtype=np.float32)
        audio = np.concatenate(self._utterance_chunks)
        self._utterance_chunks.clear()
        self._utterance_samples = 0
        return audio

    def _queue_secondary(
        self,
        audio: np.ndarray,
        vosk_candidates: tuple[tuple[str, float], ...],
    ) -> bool:
        """Queue only a current Vosk alarm candidate for independent confirmation."""
        target = self._secondary_queue
        if target is None or not self._secondary_ready.is_set():
            return False
        if not vosk_candidates:
            # Normal speech never occupies the confirmation worker. Otherwise a
            # quiet room can fill the queue before an actual alarm word arrives.
            return True
        if len(audio) < self.sample_rate // 2:
            return False
        try:
            target.put_nowait((audio, vosk_candidates))
            return True
        except queue.Full:
            # Keep the newest candidate: an older phrase is no longer relevant
            # and must not delay a current "помоги"/"помощь" confirmation.
            try:
                target.get_nowait()
            except queue.Empty:
                return False
            try:
                target.put_nowait((audio, vosk_candidates))
                logger.debug("Faster-Whisper: устаревшая фраза заменена новой")
                return True
            except queue.Full:
                return False
    def _secondary_loop(self) -> None:
        try:
            from faster_whisper import WhisperModel
            requested = self.secondary_device.strip().casefold()
            if requested in {"", "auto"}:
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    device = "cpu"
            else:
                device = requested
            compute_type = self.secondary_compute_type.strip().casefold()
            if compute_type in {"", "auto"}:
                compute_type = "float16" if device == "cuda" else "int8"
            try:
                model = WhisperModel(
                    self.secondary_model_path,
                    device=device,
                    compute_type=compute_type,
                    local_files_only=True,
                )
            except Exception:
                if device != "cuda":
                    raise
                logger.warning(
                    "Faster-Whisper CUDA is unavailable; falling back to CPU int8",
                    exc_info=True,
                )
                device = "cpu"
                compute_type = "int8"
                model = WhisperModel(
                    self.secondary_model_path,
                    device=device,
                    compute_type=compute_type,
                    local_files_only=True,
                )
            logger.info(
                "Faster-Whisper ready: model=%s device=%s compute=%s",
                self.secondary_model_path,
                device,
                compute_type,
            )
            self._secondary_ready.set()
        except Exception:
            logger.exception(
                "Faster-Whisper could not start; the unchanged Vosk path remains active"
            )
            return
        while not self._secondary_closed.is_set():
            target = self._secondary_queue
            if target is None:
                return
            try:
                item = target.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                return
            audio, vosk_candidates = item
            options = {
                "language": "ru",
                "beam_size": 5,
                "vad_filter": True,
                "condition_on_previous_text": False,
                "word_timestamps": True,
            }
            try:
                try:
                    segments, _ = model.transcribe(
                        audio,
                        hotwords=" ".join(self.keywords),
                        **options,
                    )
                except TypeError:
                    segments, _ = model.transcribe(audio, **options)
                segments = list(segments)
                text = " ".join(
                    str(segment.text).strip()
                    for segment in segments
                    if str(segment.text).strip()
                ).strip()
                word_probabilities = [
                    float(word.probability)
                    for segment in segments
                    for word in (getattr(segment, "words", None) or [])
                    if getattr(word, "probability", None) is not None
                ]
                if word_probabilities:
                    confidence = sum(word_probabilities) / len(word_probabilities)
                else:
                    probabilities = [
                        math.exp(min(0.0, float(segment.avg_logprob)))
                        for segment in segments
                        if getattr(segment, "avg_logprob", None) is not None
                    ]
                    confidence = (
                        sum(probabilities) / len(probabilities)
                        if probabilities
                        else 0.0
                    )
                self._publish_secondary(text, confidence, vosk_candidates)
            except Exception:
                logger.exception("Faster-Whisper transcription failed")

    def _publish_secondary(
        self,
        text: str,
        confidence: float,
        vosk_candidates: tuple[tuple[str, float], ...] = (),
    ) -> None:
        text = str(text).strip()
        confidence = min(max(float(confidence), 0.0), 1.0)
        if not text:
            return
        logger.info(
            "Faster-Whisper recognized: %s (%.0f%%)", text, confidence * 100
        )
        self.event_bus.publish(
            Event(
                EventType.SPEECH_RECOGNIZED,
                confidence=confidence,
                source=f"audio.speech.{self.microphone_source}",
                metadata={
                    "text": text,
                    "is_final": True,
                    "recognizer": "faster_whisper",
                    "microphone": self.microphone_source,
                    "camera_ids": list(self.camera_ids),
                },
            )
        )
        normalized = normalize_speech_text(text)
        found = [
            (word, len(pattern.findall(normalized)))
            for word, pattern in self._patterns.items()
            if pattern.search(normalized)
        ]
        if not found:
            return
        matching_vosk = [
            candidate_confidence
            for keyword, _ in found
            for candidate, candidate_confidence in vosk_candidates
            if candidate == keyword
        ]
        if matching_vosk and confidence >= max(0.30, self.secondary_min_confidence):
            # A Vosk alarm candidate is accepted only when the independent
            # recognizer heard the same complete word. This specifically blocks
            # short sound imitations such as «бдыщ» being mapped to «помощь».
            alarm_confidence = math.sqrt(confidence * max(matching_vosk))
            recognizer = "vosk+faster_whisper"
        elif not vosk_candidates and confidence >= max(
            0.78, self.secondary_min_confidence + 0.12
        ):
            # High-confidence exact Whisper output can rescue a Vosk miss.
            alarm_confidence = confidence
            recognizer = "faster_whisper_rescue"
        else:
            logger.info(
                "Тревожное слово не подтверждено двумя распознавателями: %s",
                text,
            )
            return
        now = time.monotonic()
        if now - self.last_event_time < self.cooldown_sec:
            return
        self.last_event_time = now
        keyword, _ = found[0]
        self.event_bus.publish(
            Event(
                EventType.KEYWORD_DETECTED,
                confidence=alarm_confidence,
                source=f"audio.speech.{self.microphone_source}",
                severity=Severity.CRITICAL,
                metadata={
                    "keyword": keyword,
                    "text": text,
                    "matches": sum(count for _, count in found),
                    "recognizer": recognizer,
                    "recognizer_confidence": round(alarm_confidence, 3),
                    "vosk_candidates": [word for word, _ in vosk_candidates],
                    "microphone": self.microphone_source,
                    "camera_ids": list(self.camera_ids),
                },
            )
        )

    def _publish_final(
        self,
        result: dict,
        *,
        allow_keyword_alarm: bool = True,
        publish_speech: bool = True,
    ) -> tuple[tuple[str, float], ...]:
        text = str(result.get("text", "")).strip()
        if not text:
            return ()
        confidence = self._confidence(result)
        logger.info("Распознано: %s (%.0f%%)", text, confidence * 100)
        if publish_speech:
            self.event_bus.publish(
                Event(
                    EventType.SPEECH_RECOGNIZED,
                    confidence=confidence,
                    source=f"audio.speech.{self.microphone_source}",
                    metadata={
                        "text": text,
                        "is_final": True,
                        "recognizer": "vosk",
                        "microphone": self.microphone_source,
                        "camera_ids": list(self.camera_ids),
                    },
                )
            )
        found = self.find_keywords(text)
        if not found:
            return ()
        keyword_confidence = self._keyword_confidence(result, found)
        if keyword_confidence < self.min_confidence:
            logger.info(
                "Тревожное слово отклонено: confidence=%.0f%% < %.0f%%",
                keyword_confidence * 100,
                self.min_confidence * 100,
            )
            return ()
        candidates = tuple((word, keyword_confidence) for word, _ in found)
        if not allow_keyword_alarm:
            return candidates
        now = time.monotonic()
        if now - self.last_event_time < self.cooldown_sec:
            return candidates
        self.last_event_time = now
        keyword, _ = found[0]
        self.event_bus.publish(
            Event(
                EventType.KEYWORD_DETECTED,
                confidence=keyword_confidence,
                source=f"audio.speech.{self.microphone_source}",
                severity=Severity.CRITICAL,
                metadata={
                    "keyword": keyword,
                    "text": text,
                    "matches": sum(count for _, count in found),
                    "recognizer": "vosk",
                    "recognizer_confidence": round(confidence, 3),
                    "microphone": self.microphone_source,
                    "camera_ids": list(self.camera_ids),
                },
            )
        )
        return candidates

    def process_audio(self, chunk_float32: np.ndarray) -> None:
        audio = np.clip(
            np.asarray(chunk_float32, dtype=np.float32).reshape(-1), -1.0, 1.0
        )
        self._append_utterance_audio(audio)
        audio_int16 = (audio * 32767).astype(np.int16)
        try:
            accepted = self.recognizer.AcceptWaveform(audio_int16.tobytes())
            if accepted:
                result = json.loads(self.recognizer.Result())
                utterance = self._take_utterance_audio()
                secondary_ready = self._secondary_ready.is_set()
                vosk_candidates = self._publish_final(
                    result,
                    allow_keyword_alarm=not secondary_ready,
                )
                if (
                    secondary_ready
                    and vosk_candidates
                    and not self._queue_secondary(utterance, vosk_candidates)
                ):
                    # The second model is unavailable only for this short
                    # utterance, so preserve the established Vosk alarm path.
                    self._publish_final(
                        result,
                        allow_keyword_alarm=True,
                        publish_speech=False,
                    )
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("Vosk вернул некорректный результат: %s", exc)
    def close(self) -> None:
        self._secondary_ready.clear()
        self._secondary_closed.set()
        target = self._secondary_queue
        if target is not None:
            try:
                target.put_nowait(None)
            except queue.Full:
                try:
                    target.get_nowait()
                except queue.Empty:
                    pass
                try:
                    target.put_nowait(None)
                except queue.Full:
                    pass
        worker = self._secondary_worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)
