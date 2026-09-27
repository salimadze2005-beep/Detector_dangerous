from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_resource_path(relative_path: str | os.PathLike[str]) -> str:
    """Return an absolute path for source and PyInstaller builds."""
    base = Path(getattr(sys, "_MEIPASS", PROJECT_ROOT))
    return str((base / relative_path).resolve())


def _runtime_path(relative_path: str) -> str:
    return str((PROJECT_ROOT / relative_path).resolve())


def _default_gunshot_model_path() -> str:
    """Legacy path kept only so old user configs still parse."""
    for name in ("binary_gunshot_full.h5", "binary_gunshot.h5"):
        candidate = Path(get_resource_path(f"models/{name}"))
        if candidate.exists():
            return str(candidate)
    return get_resource_path("models/binary_gunshot_full.h5")


def _default_panns_model_path() -> str:
    return get_resource_path("models/panns/Cnn14_mAP=0.431.pth")


@dataclass
class PathsConfig:
    data_dir: str = field(default_factory=lambda: _runtime_path("data"))
    log_file: str = field(default_factory=lambda: _runtime_path("data/detector.log"))
    event_db: str = field(default_factory=lambda: _runtime_path("data/events.sqlite3"))
    detector_stats: str = field(default_factory=lambda: _runtime_path("data/detector_stats.csv"))
    screenshots_dir: str = field(default_factory=lambda: _runtime_path("data/screenshots"))


@dataclass
class AudioConfig:
    sample_rate: int = 16_000
    panns_sample_rate: int = 32_000
    chunk_duration_sec: float = 0.25
    gunshot_threshold: float = 0.75
    ema_alpha: float = 0.6
    cooldown_sec: float = 2.0
    analysis_step_sec: float = 0.5
    rms_min: float = 0.008
    peak_min: float = 0.30
    trigger_dbfs: float = -10.5
    adaptive_noise: bool = True
    noise_alpha: float = 0.05
    min_snr_db: float = 8.0
    min_crest_factor: float = 2.5
    sustain_windows: int = 3
    sustain_hits: int = 2
    sustain_threshold: float = 0.50
    yamnet_threshold: float = 0.10
    yamnet_veto_threshold: float = 0.30
    yamnet_veto_margin: float = 0.12
    cnn_override_threshold: float = 0.92
    panns_threshold: float = 0.30
    panns_margin_threshold: float = 0.15
    panns_prefilter_threshold: float = 0.10
    panns_model_path: str = field(default_factory=_default_panns_model_path)
    panns_device: str = "auto"
    analysis_window_sec: float = 3.0
    multiscale_rescue: bool = True
    rescue_window_sec: float = 1.0
    detection_mode: str = "PANNs + YAMNet (Строгий)"
    use_signature: bool = True
    use_veto: bool = True
    max_buffer_sec: float = 3.0
    calibration_duration_sec: float = 10.0
    microphone_profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    gunshot_model_path: str = field(default_factory=_default_gunshot_model_path)
    yamnet_model_path: str = field(default_factory=lambda: get_resource_path("models/yamnet"))


@dataclass
class SpeechConfig:
    enabled: bool = True
    vosk_model_path: str = field(default_factory=lambda: get_resource_path("model-ru"))
    keywords: list[str] = field(default_factory=lambda: ["помоги", "помогите", "помощь", "спасите"])
    cooldown_sec: float = 3.0
    min_confidence: float = 0.30
    fuzzy_match: bool = True
    microphone_device: int | str | None = None
    secondary_enabled: bool = True
    secondary_model_path: str = field(default_factory=lambda: get_resource_path("models/faster-whisper-small"))
    secondary_device: str = "auto"
    secondary_compute_type: str = "auto"
    secondary_min_confidence: float = 0.30
    secondary_max_utterance_sec: float = 12.0


@dataclass
class VideoConfig:
    model_path: str = field(default_factory=lambda: get_resource_path("models/yolov8n-pose.pt"))
    sources: list[str] = field(default_factory=lambda: ["0"])
    names_by_source: dict[str, str] = field(default_factory=dict)
    microphone_by_source: dict[str, int | str | None] = field(default_factory=dict)
    view_mode_by_source: dict[str, str] = field(default_factory=dict)
    # Prefer FFmpeg's low-delay RTSP path for live network cameras.  It avoids
    # accumulating stale frames when inference briefly takes longer than a
    # camera frame interval.
    rtsp_low_latency: bool = True
    rtsp_read_timeout_msec: int = 1_000
    fall_duration_sec: float = 5.0
    reset_grace_sec: float = 2.0
    missing_grace_sec: float = 2.0
    reconnect_initial_sec: float = 1.0
    reconnect_max_sec: float = 15.0
    loop_video_files: bool = False
    pose_confidence: float = 0.30
    fall_min_confidence: float = 0.65
    keypoint_confidence: float = 0.50
    partial_pose_confidence: float = 0.25
    partial_pose_min_keypoints: int = 3
    partial_lie_min_score: float = 0.72
    standing_angle_deg: float = 30.0
    lying_angle_deg: float = 60.0
    lying_aspect_ratio: float = 1.00
    seated_knee_drop_ratio: float = 0.12
    require_upright_transition: bool = True
    posture_window: int = 5
    posture_hits: int = 3
    rapid_descent_heights_per_sec: float = 0.75
    transition_memory_sec: float = 2.0
    motion_confirm_sec: float = 5.0
    require_fall_motion: bool = True
    # The bundled yolov8n-pose.onnx has a fixed 640x640 input tensor.
    inference_imgsz: int = 640
    use_half_precision: bool = True
    # This optional RGB classifier requires a separately trained private model.
    # Keep the portable pose-based detector enabled without advertising a file
    # that is not part of the distribution.
    lying_classifier_enabled: bool = False
    lying_classifier_model_path: str = "models/lying_classifier.pt"
    lying_classifier_label: str = "lying"
    lying_classifier_imgsz: int = 224
    lying_classifier_hz: float = 5.0
    lying_classifier_threshold: float = 0.75
    lying_alert_score_threshold: float = 0.55
    lying_classifier_pose_quality_min: float = 0.45
    lying_classifier_rgb_weight: float = 0.65
    lying_stable_duration_sec: float = 1.0
    lying_stable_motion_heights_per_sec: float = 0.12


@dataclass
class AlertsConfig:
    local_sound_enabled: bool = True
    webhook_url: str = ""
    webhook_token: str = ""
    webhook_timeout_sec: float = 5.0
    webhook_retries: int = 3
    webhook_payload_mode: str = "multipart"
    video_delivery_enabled: bool = True
    video_upload_url: str = ""
    delivery_video_fps: float = 10.0
    delivery_video_max_bytes: int = 2_000_000
    delivery_video_initial_crf: int = 28
    retry_initial_sec: float = 1.0
    retry_max_sec: float = 300.0
    snapshot_max_age_sec: float = 5.0
    snapshot_retention_days: int = 30
    jpeg_quality: int = 85
    image_max_width: int = 1920
    image_max_height: int = 1080
    gunshot_sound: str = field(default_factory=lambda: get_resource_path("sounds/gunshot_alert.wav"))
    keyword_sound: str = field(default_factory=lambda: get_resource_path("sounds/help_alert.wav"))
    fall_sound: str = field(default_factory=lambda: get_resource_path("sounds/fall_alert.wav"))


@dataclass
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    alerts: AlertsConfig = field(default_factory=AlertsConfig)
    gui_log_max_lines: int = 1_000
    config_file: str = field(default="", repr=False)

    def ensure_runtime_dirs(self) -> None:
        Path(self.paths.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.paths.log_file).parent.mkdir(parents=True, exist_ok=True)
        Path(self.paths.event_db).parent.mkdir(parents=True, exist_ok=True)
        Path(self.paths.detector_stats).parent.mkdir(parents=True, exist_ok=True)
        Path(self.paths.screenshots_dir).mkdir(parents=True, exist_ok=True)

    def validate(self) -> list[str]:
        errors: list[str] = []
        # YAMNet-named config fields remain for backward compatibility, but the
        # AST experiment does not require TensorFlow/YAMNet at runtime.
        required = {
            "Vosk": self.speech.vosk_model_path,
            "AST": get_resource_path("models/ast"),
        }
        for name, value in required.items():
            if value and not Path(value).exists():
                errors.append(f"{name}: не найден путь {value}")
        return errors


def _merge_dataclass(target, values: dict[str, Any], prefix: str = "") -> None:
    known = {item.name for item in fields(target)}
    for name, value in values.items():
        if name not in known:
            raise ValueError(f"Неизвестный параметр конфигурации: {prefix}{name}")
        current = getattr(target, name)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"Параметр {prefix}{name} должен быть объектом")
            _merge_dataclass(current, value, f"{prefix}{name}.")
        else:
            setattr(target, name, value)


def _resolve_bundled_path(value: str, fallback: str) -> str:
    """Make model/sound paths independent from the launch directory and old PC."""
    configured = Path(str(value)).expanduser()
    if configured.is_absolute() and configured.exists():
        return str(configured.resolve())
    if configured.is_absolute():
        # A copied config often contains the previous computer's absolute path.
        # Prefer the equivalent bundled resource when it exists.
        bundled = Path(get_resource_path(fallback))
        return str(bundled) if bundled.exists() else str(configured)
    return get_resource_path(str(configured))


def _resolve_portable_resources(result: AppConfig) -> None:
    result.audio.panns_model_path = _resolve_bundled_path(
        result.audio.panns_model_path, "models/panns/Cnn14_mAP=0.431.pth"
    )
    result.audio.gunshot_model_path = _resolve_bundled_path(
        result.audio.gunshot_model_path, "models/binary_gunshot_full.h5"
    )
    result.audio.yamnet_model_path = _resolve_bundled_path(
        result.audio.yamnet_model_path, "models/yamnet"
    )
    result.speech.vosk_model_path = _resolve_bundled_path(
        result.speech.vosk_model_path, "model-ru"
    )
    result.speech.secondary_model_path = _resolve_bundled_path(
        result.speech.secondary_model_path, "models/faster-whisper-small"
    )
    result.video.model_path = _resolve_bundled_path(
        result.video.model_path, "models/yolov8n-pose.pt"
    )
    result.video.lying_classifier_model_path = _resolve_bundled_path(
        result.video.lying_classifier_model_path, "models/lying_classifier.pt"
    )
    result.alerts.gunshot_sound = _resolve_bundled_path(
        result.alerts.gunshot_sound, "sounds/gunshot_alert.wav"
    )
    result.alerts.keyword_sound = _resolve_bundled_path(
        result.alerts.keyword_sound, "sounds/help_alert.wav"
    )
    result.alerts.fall_sound = _resolve_bundled_path(
        result.alerts.fall_sound, "sounds/fall_alert.wav"
    )


def load_config(path: str | os.PathLike[str] | None = None) -> AppConfig:
    result = AppConfig()
    config_path = Path(path or os.getenv("DETECTOR_CONFIG", PROJECT_ROOT / "detector_config.json"))
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError("Корень detector_config.json должен быть JSON-объектом")
        _merge_dataclass(result, raw)
    _resolve_portable_resources(result)
    result.config_file = str(config_path.resolve())
    if value := os.getenv("DETECTOR_WEBHOOK_URL"):
        result.alerts.webhook_url = value
    if value := os.getenv("DETECTOR_WEBHOOK_TOKEN"):
        result.alerts.webhook_token = value
    if value := os.getenv("DETECTOR_DATA_DIR"):
        data_dir = Path(value).resolve()
        result.paths.data_dir = str(data_dir)
        result.paths.log_file = str(data_dir / "detector.log")
        result.paths.event_db = str(data_dir / "events.sqlite3")
        result.paths.detector_stats = str(data_dir / "detector_stats.csv")
        result.paths.screenshots_dir = str(data_dir / "screenshots")
    result.ensure_runtime_dirs()
    return result


config = load_config()


def save_operator_settings(app_config: AppConfig) -> None:
    """Atomically persist only settings exposed by the operator settings dialog."""
    if not app_config.config_file:
        return
    path = Path(app_config.config_file)
    existing: dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            existing = loaded

    sections = {
        "audio": {
            "gunshot_threshold": app_config.audio.gunshot_threshold,
            "panns_threshold": app_config.audio.panns_threshold,
            "panns_margin_threshold": app_config.audio.panns_margin_threshold,
            "panns_prefilter_threshold": app_config.audio.panns_prefilter_threshold,
            "trigger_dbfs": app_config.audio.trigger_dbfs,
            "calibration_duration_sec": app_config.audio.calibration_duration_sec,
            "microphone_profiles": app_config.audio.microphone_profiles,
        },
        "speech": {
            "enabled": app_config.speech.enabled,
            "microphone_device": app_config.speech.microphone_device,
            "keywords": app_config.speech.keywords,
        },
        "video": {
            "model_path": app_config.video.model_path,
            "sources": app_config.video.sources,
            "names_by_source": app_config.video.names_by_source,
            "microphone_by_source": app_config.video.microphone_by_source,
            "view_mode_by_source": app_config.video.view_mode_by_source,
            "fall_duration_sec": app_config.video.fall_duration_sec,
            "lying_classifier_enabled": app_config.video.lying_classifier_enabled,
            "lying_classifier_model_path": app_config.video.lying_classifier_model_path,
            "lying_classifier_label": app_config.video.lying_classifier_label,
            "lying_classifier_imgsz": app_config.video.lying_classifier_imgsz,
            "lying_classifier_hz": app_config.video.lying_classifier_hz,
            "lying_classifier_threshold": app_config.video.lying_classifier_threshold,
            "lying_alert_score_threshold": app_config.video.lying_alert_score_threshold,
            "lying_classifier_pose_quality_min": app_config.video.lying_classifier_pose_quality_min,
            "lying_classifier_rgb_weight": app_config.video.lying_classifier_rgb_weight,
            "lying_stable_duration_sec": app_config.video.lying_stable_duration_sec,
            "lying_stable_motion_heights_per_sec": app_config.video.lying_stable_motion_heights_per_sec,
        },
        "alerts": {
            "local_sound_enabled": app_config.alerts.local_sound_enabled,
            "webhook_url": app_config.alerts.webhook_url,
            "snapshot_max_age_sec": app_config.alerts.snapshot_max_age_sec,
            "webhook_payload_mode": app_config.alerts.webhook_payload_mode,
            "video_delivery_enabled": app_config.alerts.video_delivery_enabled,
            "video_upload_url": app_config.alerts.video_upload_url,
        },
    }
    for section, values in sections.items():
        target = existing.setdefault(section, {})
        if not isinstance(target, dict):
            target = {}
            existing[section] = target
        target.update(values)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
