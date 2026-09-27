from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np

PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "sensitive": {
        "label": "Чувствительный — тихое помещение",
        "trigger_dbfs": -18.0,
        "gunshot_threshold": 0.76,
        "cnn_override_threshold": 0.96,
        "panns_threshold": 0.22,
        "panns_margin_threshold": 0.10,
        "min_snr_db": 8.0,
        "min_crest_factor": 2.5,
        "yamnet_threshold": 0.25,
        "yamnet_veto_threshold": 0.50,
        "yamnet_veto_margin": 0.12,
        "sustain_windows": 3,
        "sustain_hits": 2,
        "sustain_threshold": 0.52,
        "noise_margin_db": 8.0,
        "trigger_floor_dbfs": -20.0,
        "trigger_ceiling_dbfs": -10.0,
    },
    "balanced": {
        "label": "Сбалансированный — офис/дом",
        "trigger_dbfs": -12.0,
        "gunshot_threshold": 0.82,
        "cnn_override_threshold": 0.98,
        "panns_threshold": 0.30,
        "panns_margin_threshold": 0.15,
        "min_snr_db": 11.0,
        "min_crest_factor": 3.0,
        "yamnet_threshold": 0.30,
        "yamnet_veto_threshold": 0.45,
        "yamnet_veto_margin": 0.10,
        "sustain_windows": 3,
        "sustain_hits": 2,
        "sustain_threshold": 0.60,
        "noise_margin_db": 11.0,
        "trigger_floor_dbfs": -16.0,
        "trigger_ceiling_dbfs": -8.0,
    },
    "noisy": {
        "label": "Строгий — улица/цех/шумное место",
        "trigger_dbfs": -8.0,
        "gunshot_threshold": 0.88,
        "cnn_override_threshold": 0.995,
        "panns_threshold": 0.40,
        "panns_margin_threshold": 0.20,
        "min_snr_db": 14.0,
        "min_crest_factor": 3.5,
        "yamnet_threshold": 0.35,
        "yamnet_veto_threshold": 0.40,
        "yamnet_veto_margin": 0.08,
        "sustain_windows": 4,
        "sustain_hits": 3,
        "sustain_threshold": 0.68,
        "noise_margin_db": 14.0,
        "trigger_floor_dbfs": -12.0,
        "trigger_ceiling_dbfs": -5.0,
    },
    "custom": {"label": "Пользовательский"},
}

RUNTIME_FIELDS = {
    "trigger_dbfs", "rms_min",
    "gunshot_threshold", "cnn_override_threshold",
    "panns_threshold", "panns_margin_threshold",
    "min_snr_db", "min_crest_factor",
    "yamnet_threshold", "yamnet_veto_threshold", "yamnet_veto_margin",
    "sustain_windows", "sustain_hits", "sustain_threshold",
}
PROFILE_METADATA_FIELDS = {
    "preset", "calibrated_at", "ambient_peak_dbfs", "ambient_rms", "ambient_crest_p95",
    "clipping_ratio", "calibration_windows",
    "calibration_mode", "reference_count", "reference_peak_dbfs",
    "reference_snr_db", "reference_crest_factor", "reference_rms", "manual_override",
}


def microphone_profile_key(device: int | str | None) -> str:
    if device is None or device == "__default__":
        return "__default__"
    return str(device)


def amplitude_to_dbfs(amplitude: float) -> float:
    return 20.0 * math.log10(max(float(amplitude), 1e-9))


def dbfs_to_amplitude(dbfs: float) -> float:
    return 10.0 ** (float(dbfs) / 20.0)


def _legacy_to_panns_threshold(value: float) -> float:
    value = float(value)
    return value if value <= 0.60 else 0.30


def _legacy_to_panns_margin(value: float) -> float:
    value = float(value)
    return value if value <= 0.60 else 0.15


@dataclass(frozen=True, slots=True)
class AmbientCalibration:
    preset: str
    windows: int
    ambient_rms: float
    ambient_peak_dbfs: float
    ambient_crest_p95: float
    clipping_ratio: float
    trigger_dbfs: float
    rms_min: float
    min_snr_db: float
    min_crest_factor: float

    def profile_values(self) -> dict[str, Any]:
        preset_values = PROFILE_PRESETS.get(self.preset, PROFILE_PRESETS["balanced"])
        result = {k: v for k, v in preset_values.items() if k in RUNTIME_FIELDS}
        result.update({
            "preset": self.preset,
            "trigger_dbfs": round(self.trigger_dbfs, 1),
            "rms_min": round(self.rms_min, 6),
            "min_snr_db": round(self.min_snr_db, 1),
            "min_crest_factor": round(self.min_crest_factor, 2),
            "calibrated_at": datetime.now(timezone.utc).isoformat(),
            "ambient_peak_dbfs": round(self.ambient_peak_dbfs, 1),
            "ambient_rms": round(self.ambient_rms, 6),
            "ambient_crest_p95": round(self.ambient_crest_p95, 2),
            "clipping_ratio": round(self.clipping_ratio, 6),
            "calibration_windows": self.windows,
        })
        return result


def analyze_ambient_audio(chunks: Iterable[np.ndarray], preset: str = "balanced") -> AmbientCalibration:
    if preset not in PROFILE_PRESETS or preset == "custom":
        preset = "balanced"
    rms_values: list[float] = []
    peak_values: list[float] = []
    crest_values: list[float] = []
    clipped_samples = 0
    sample_count = 0
    for chunk in chunks:
        x = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if x.size == 0:
            continue
        x = x - float(np.mean(x))
        rms = float(np.sqrt(np.mean(x ** 2)))
        peak = float(np.max(np.abs(x)))
        rms_values.append(rms)
        peak_values.append(peak)
        crest_values.append(peak / max(rms, 1e-9))
        clipped_samples += int(np.count_nonzero(np.abs(x) >= 0.98))
        sample_count += int(x.size)
    if len(rms_values) < 4:
        raise ValueError("Для калибровки получено слишком мало аудиоданных")
    values = PROFILE_PRESETS[preset]
    ambient_rms = float(np.percentile(rms_values, 95))
    ambient_peak = float(np.percentile(peak_values, 99))
    ambient_peak_dbfs = amplitude_to_dbfs(ambient_peak)
    ambient_crest = float(np.percentile(crest_values, 95))
    trigger = max(float(values["trigger_floor_dbfs"]), ambient_peak_dbfs + float(values["noise_margin_db"]))
    trigger = min(max(trigger, -50.0), float(values["trigger_ceiling_dbfs"]))
    crest_margin = {"sensitive": 0.20, "balanced": 0.35, "noisy": 0.65}[preset]
    crest_ceiling = {"sensitive": 4.0, "balanced": 4.5, "noisy": 5.0}[preset]
    min_crest = max(float(values["min_crest_factor"]), min(ambient_crest + crest_margin, crest_ceiling))
    rms_min = max(dbfs_to_amplitude(trigger - 36.0), min(ambient_rms * 0.25, 0.02), 1e-5)
    return AmbientCalibration(
        preset, len(rms_values), ambient_rms, ambient_peak_dbfs, ambient_crest,
        clipped_samples / max(sample_count, 1), trigger, rms_min,
        float(values["min_snr_db"]), min_crest,
    )


@dataclass(frozen=True, slots=True)
class ReferenceCalibration:
    """Result of calibration from ambient audio and labelled reference impulses."""
    ambient: AmbientCalibration
    reference_count: int
    weakest_peak_dbfs: float
    weakest_snr_db: float
    weakest_crest_factor: float
    weakest_window_rms: float
    trigger_dbfs: float
    rms_min: float
    min_snr_db: float
    min_crest_factor: float

    def profile_values(self) -> dict[str, Any]:
        values = self.ambient.profile_values()
        values.update({
            "trigger_dbfs": round(self.trigger_dbfs, 1),
            "rms_min": round(self.rms_min, 6),
            "min_snr_db": round(self.min_snr_db, 1),
            "min_crest_factor": round(self.min_crest_factor, 2),
            "calibration_mode": "ambient_plus_reference",
            "reference_count": self.reference_count,
            "reference_peak_dbfs": round(self.weakest_peak_dbfs, 1),
            "reference_snr_db": round(self.weakest_snr_db, 1),
            "reference_crest_factor": round(self.weakest_crest_factor, 2),
            "reference_rms": round(self.weakest_window_rms, 6),
        })
        return values


def _reference_impulse_metrics(
    chunks: Iterable[np.ndarray], ambient_rms: float
) -> tuple[float, float, float, float]:
    parts = [np.asarray(chunk, dtype=np.float32).reshape(-1) for chunk in chunks]
    non_empty = [part for part in parts if part.size]
    x = np.concatenate(non_empty) if non_empty else np.empty(0, dtype=np.float32)
    if x.size < 160:
        raise ValueError("Одна из контрольных записей слишком короткая")
    x = x - float(np.mean(x))
    window_rms = float(np.sqrt(np.mean(x * x)))
    strongest = int(np.argmax(np.abs(x)))
    radius = min(1_280, max(80, x.size // 8))
    local = x[max(0, strongest - radius):min(x.size, strongest + radius)]
    peak = float(np.max(np.abs(local)))
    local_rms = float(np.sqrt(np.mean(local * local)))
    crest = peak / max(local_rms, 1e-9)
    return (
        amplitude_to_dbfs(peak),
        amplitude_to_dbfs(peak / max(float(ambient_rms), 1e-9)),
        crest,
        window_rms,
    )


def analyze_reference_audio(
    ambient_chunks: Iterable[np.ndarray],
    reference_takes: Iterable[Iterable[np.ndarray]],
    preset: str = "balanced",
    minimum_references: int = 3,
) -> ReferenceCalibration:
    """Tune the acoustic gate from room noise plus labelled test takes.

    The quietest accepted take sets the gate; raw audio remains in memory only.
    `minimum_references` stays at three for the guided wizard, while the
    manual recorder can use one or two takes for an intermediate profile that
    is refined as additional distances are captured.
    """
    ambient_list = [np.asarray(chunk, dtype=np.float32).copy() for chunk in ambient_chunks]
    ambient = analyze_ambient_audio(ambient_list, preset)
    metrics = [
        _reference_impulse_metrics(take, ambient.ambient_rms)
        for take in reference_takes
    ]
    required_references = max(1, int(minimum_references))
    if len(metrics) < required_references:
        if required_references == 3:
            raise ValueError("Нужны как минимум три контрольные записи на разных расстояниях")
        raise ValueError(
            f"Нужны как минимум {required_references} контрольные записи на разных расстояниях"
        )
    if any(peak_dbfs >= -0.2 for peak_dbfs, _, _, _ in metrics):
        raise ValueError(
            "Контрольная запись перегружена (клиппинг). Уменьшите усиление микрофона "
            "и повторите полную автонастройку."
        )
    weak_peak_dbfs, weak_snr_db, weak_crest, weak_window_rms = min(
        metrics, key=lambda item: item[0]
    )
    if weak_snr_db < 12.0:
        raise ValueError(
            "Контрольный импульс слишком близок к фону; нужно не менее 12 dB SNR. "
            "Повторите запись с более чистым тестовым сигналом или уменьшите дистанцию."
        )
    if weak_crest < 1.8:
        raise ValueError(
            "Контрольная запись не похожа на короткий импульс. "
            "Повторите её с проверенным тестовым звуком."
        )
    preset_values = PROFILE_PRESETS.get(
        ambient.preset, PROFILE_PRESETS["balanced"]
    )
    # The ambient trigger contains a conservative preset floor. Reusing it
    # here can reject a valid distant reference in a quiet room.
    noise_gate = ambient.ambient_peak_dbfs + float(preset_values["noise_margin_db"])
    # Keep the resulting gate below the weakest reference so the recorded
    # example is guaranteed to be accepted with a small safety reserve.
    trigger = min(weak_peak_dbfs - 3.0, max(noise_gate, weak_peak_dbfs - 6.0))
    trigger = max(trigger, -50.0)
    ceiling = float(preset_values["trigger_ceiling_dbfs"])
    if trigger > ceiling:
        raise ValueError(
            "Самый тихий контрольный импульс слишком слаб для выбранного микрофона: "
            "безопасный порог превысил допустимый предел."
        )
    min_crest = max(1.8, min(ambient.min_crest_factor, weak_crest * 0.60))
    min_snr = max(ambient.min_snr_db, min(ambient.min_snr_db, weak_snr_db - 6.0))
    rms_min = max(
        dbfs_to_amplitude(trigger - 36.0),
        # Runtime evaluates a several-second ring buffer. Set the RMS floor
        # from the quietest complete reference window rather than room noise.
        min(weak_window_rms * 0.20, 0.01),
        1e-5,
    )
    return ReferenceCalibration(
        ambient=ambient,
        reference_count=len(metrics),
        weakest_peak_dbfs=weak_peak_dbfs,
        weakest_snr_db=weak_snr_db,
        weakest_crest_factor=weak_crest,
        weakest_window_rms=weak_window_rms,
        trigger_dbfs=trigger,
        rms_min=rms_min,
        min_snr_db=min_snr,
        min_crest_factor=min_crest,
    )

def resolve_audio_profile(audio_config, device: int | str | None) -> dict[str, Any]:
    settings = {
        field: getattr(audio_config, field)
        for field in RUNTIME_FIELDS
        if hasattr(audio_config, field)
    }
    key = microphone_profile_key(device)
    profile = dict(getattr(audio_config, "microphone_profiles", {}).get(key, {}))
    preset = str(profile.get("preset", "balanced" if not profile else "custom"))
    if preset in PROFILE_PRESETS:
        for field, value in PROFILE_PRESETS[preset].items():
            if field in RUNTIME_FIELDS:
                settings[field] = value
    for field in RUNTIME_FIELDS:
        if field in profile:
            settings[field] = profile[field]

    # Preserve legacy values exactly for UI/tests. Only derive the new PANNs
    # values when a profile has not specified them explicitly.
    if "panns_threshold" not in profile and "gunshot_threshold" in profile:
        settings["panns_threshold"] = _legacy_to_panns_threshold(profile["gunshot_threshold"])
    if "panns_margin_threshold" not in profile and "cnn_override_threshold" in profile:
        settings["panns_margin_threshold"] = _legacy_to_panns_margin(profile["cnn_override_threshold"])

    # Upgrade profiles produced by the previous reference calibrator. Raw
    # takes are intentionally not persisted, but the saved metrics are
    # sufficient to prevent the old conservative semantic/RMS defaults
    # from returning after an application restart.
    if profile.get("calibration_mode") == "ambient_plus_reference" and not profile.get("manual_override"):
        try:
            reference_snr = float(profile.get("reference_snr_db", 0.0))
            reference_peak = float(profile.get("reference_peak_dbfs"))
        except (TypeError, ValueError):
            reference_snr = 0.0
            reference_peak = None
        if reference_peak is not None:
            calibration_preset = preset if preset in {"sensitive", "balanced", "noisy"} else "balanced"
            preset_values = PROFILE_PRESETS[calibration_preset]
            # Reference takes calibrate the acoustic gate, not the
            # semantic classifiers. Keep the preset defaults here so
            # clicks/claps cannot become alarms just because the room was
            # quiet during recording.
            settings["panns_threshold"] = max(
                float(settings["panns_threshold"]),
                float(preset_values["panns_threshold"]),
            )
            settings["panns_margin_threshold"] = max(
                float(settings["panns_margin_threshold"]),
                float(preset_values["panns_margin_threshold"]),
            )
            settings["yamnet_threshold"] = max(
                float(settings["yamnet_threshold"]),
                float(preset_values["yamnet_threshold"]),
            )
            settings["min_snr_db"] = max(
                float(settings["min_snr_db"]),
                float(preset_values["min_snr_db"]),
            )
            settings["min_crest_factor"] = max(
                float(settings["min_crest_factor"]),
                float(preset_values["min_crest_factor"]),
            )
            noise_peak = float(profile.get("ambient_peak_dbfs", -60.0))
            noise_gate = noise_peak + float(preset_values["noise_margin_db"])
            auto_trigger = min(
                reference_peak - 3.0,
                max(noise_gate, reference_peak - 6.0),
            )
            settings["trigger_dbfs"] = min(
                float(settings["trigger_dbfs"]), max(auto_trigger, -50.0)
            )
            if "reference_rms" not in profile:
                settings["rms_min"] = min(
                    float(settings["rms_min"]),
                    dbfs_to_amplitude(reference_peak - 36.0),
                )
    settings["preset"] = preset
    settings["profile_key"] = key
    # Reference calibration already measured the RMS of the complete
    # capture window. Do not reintroduce the old, stricter -30 dBFS
    # floor, otherwise the quietest distant take is discarded again.
    rms_margin_db = (-36.0 if profile.get("calibration_mode") == "ambient_plus_reference" and not profile.get("manual_override") else -30.0)
    settings["rms_min"] = max(
        float(settings.get("rms_min", audio_config.rms_min)),
        dbfs_to_amplitude(float(settings["trigger_dbfs"]) + rms_margin_db),
    )
    return settings


def validate_microphone_profiles(profiles: Any) -> list[str]:
    if not isinstance(profiles, dict):
        return ["audio.microphone_profiles должен быть JSON-объектом"]
    errors: list[str] = []
    allowed = RUNTIME_FIELDS | PROFILE_METADATA_FIELDS
    ranges = {
        "trigger_dbfs": (-80.0, 0.0),
        "rms_min": (0.0, 1.0),
        "panns_threshold": (0.0, 1.0),
        "panns_margin_threshold": (0.0, 1.0),
        "gunshot_threshold": (0.0, 1.0),
        "cnn_override_threshold": (0.0, 1.0),
        "min_snr_db": (0.0, 60.0),
        "min_crest_factor": (1.0, 20.0),
        "yamnet_threshold": (0.0, 1.0),
        "yamnet_veto_threshold": (0.0, 1.0),
        "yamnet_veto_margin": (0.0, 1.0),
        "sustain_threshold": (0.0, 1.0),
        "sustain_windows": (1, 20),
        "sustain_hits": (1, 20),
    }
    for key, profile in profiles.items():
        if not isinstance(profile, dict):
            errors.append(f"audio.microphone_profiles.{key} должен быть объектом")
            continue
        unknown = sorted(set(profile) - allowed)
        if unknown:
            errors.append(f"Неизвестные параметры профиля микрофона {key}: {', '.join(unknown)}")
        preset = profile.get("preset", "custom")
        if preset not in PROFILE_PRESETS:
            errors.append(f"Неизвестный preset микрофона {key}: {preset}")
        for field, (minimum, maximum) in ranges.items():
            if field not in profile:
                continue
            value = profile[field]
            if not isinstance(value, (int, float)) or not minimum <= value <= maximum:
                errors.append(
                    f"audio.microphone_profiles.{key}.{field} должен быть в диапазоне [{minimum}, {maximum}]"
                )
        windows = int(profile.get("sustain_windows", 3))
        hits = int(profile.get("sustain_hits", 2))
        if hits > windows:
            errors.append(
                f"audio.microphone_profiles.{key}.sustain_hits не может быть больше sustain_windows"
            )
    return errors
