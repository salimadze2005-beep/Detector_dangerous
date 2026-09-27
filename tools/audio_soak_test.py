from __future__ import annotations

import argparse
import sys
import time
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from audio.calibration import PROFILE_PRESETS, dbfs_to_amplitude, microphone_profile_key, resolve_audio_profile
from audio.microphone import DualRateAudioResampler, MicrophoneStream
from core.config import config
from core.event_bus import EventBus, EventType


def parse_device(value: str):
    normalized = value.strip()
    if normalized.casefold() in {"default", "__default__", "none"}:
        return None
    try:
        return int(normalized)
    except ValueError:
        return normalized


def build_detector(event_bus: EventBus, audio_config, device, stats_path: str):
    from audio.gunshot_detector import GunshotDetector

    profile = resolve_audio_profile(audio_config, device)
    return GunshotDetector(
        event_bus=event_bus,
        model_path=audio_config.gunshot_model_path,
        threshold=float(profile["gunshot_threshold"]),
        ema_alpha=audio_config.ema_alpha,
        cooldown_sec=audio_config.cooldown_sec,
        analysis_step_sec=audio_config.analysis_step_sec,
        rms_min=float(profile["rms_min"]),
        peak_min=dbfs_to_amplitude(float(profile["trigger_dbfs"])),
        sustain_windows=int(profile["sustain_windows"]),
        sustain_hits=int(profile["sustain_hits"]),
        sustain_thresh=float(profile["sustain_threshold"]),
        use_signature=audio_config.use_signature,
        use_veto=audio_config.use_veto,
        veto_thresh=float(profile["yamnet_veto_threshold"]),
        gun_thresh=float(profile["yamnet_threshold"]),
        detection_mode=audio_config.detection_mode,
        stats_path=stats_path,
        sample_rate=audio_config.panns_sample_rate,
        yamnet_sample_rate=audio_config.sample_rate,
        yamnet_model_path=audio_config.yamnet_model_path,
        adaptive_noise=audio_config.adaptive_noise,
        noise_alpha=audio_config.noise_alpha,
        min_snr_db=float(profile["min_snr_db"]),
        min_crest_factor=float(profile["min_crest_factor"]),
        veto_margin=float(profile["yamnet_veto_margin"]),
        cnn_override_threshold=float(profile["cnn_override_threshold"]),
        panns_threshold=float(profile["panns_threshold"]),
        panns_margin_threshold=float(profile["panns_margin_threshold"]),
        panns_prefilter_threshold=audio_config.panns_prefilter_threshold,
        panns_model_path=audio_config.panns_model_path,
        panns_device=audio_config.panns_device,
        analysis_window_sec=audio_config.analysis_window_sec,
        microphone_source=microphone_profile_key(device),
    ), profile


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Фоновый тест ложных тревог детектора выстрелов без сирены и REST"
    )
    parser.add_argument("--minutes", type=float, default=30.0, help="длительность теста")
    parser.add_argument("--device", default="default", help="ID микрофона или default")
    parser.add_argument(
        "--preset",
        choices=[name for name in PROFILE_PRESETS if name != "custom"],
        help="временно проверить preset вместо сохранённого профиля",
    )
    parser.add_argument("--list-devices", action="store_true", help="показать микрофоны и выйти")
    parser.add_argument(
        "--fail-on-event",
        action="store_true",
        help="вернуть ненулевой код, если была хотя бы одна тревога",
    )
    args = parser.parse_args()

    if args.list_devices:
        print("default: системный микрофон")
        for device_id, name in MicrophoneStream.list_input_devices():
            print(f"{device_id}: {name}")
        return 0
    if not 0.1 <= args.minutes <= 24 * 60:
        parser.error("--minutes должен быть в диапазоне 0.1–1440")

    device = parse_device(args.device)
    audio_config = deepcopy(config.audio)
    if args.preset:
        audio_config.microphone_profiles[microphone_profile_key(device)] = {
            "preset": args.preset
        }

    event_bus = EventBus()
    events = []
    event_bus.subscribe(EventType.GUNSHOT_DETECTED, events.append)
    stats_path = str(Path(config.paths.data_dir) / "audio_soak_stats.csv")
    detector = microphone = dual_resampler = None
    started = time.monotonic()
    try:
        print("Загрузка CNN и YAMNet...")
        detector, profile = build_detector(event_bus, audio_config, device, stats_path)
        print(
            "Профиль: {preset}; trigger={trigger:.1f} dBFS; CNN={cnn:.3f}; "
            "SNR={snr:.1f} dB; crest={crest:.2f}".format(
                preset=profile["preset"],
                trigger=float(profile["trigger_dbfs"]),
                cnn=float(profile["gunshot_threshold"]),
                snr=float(profile["min_snr_db"]),
                crest=float(profile["min_crest_factor"]),
            )
        )
        microphone = MicrophoneStream(
            sample_rate=audio_config.sample_rate,
            chunk_duration_sec=audio_config.chunk_duration_sec,
            device=device,
            max_buffer_sec=audio_config.max_buffer_sec,
            allow_fallback=device is None,
            preserve_native=True,
        )
        microphone.start()
        dual_resampler = DualRateAudioResampler(
            microphone.sample_rate,
            audio_config.panns_sample_rate,
            audio_config.sample_rate,
        )
        duration_sec = args.minutes * 60.0
        next_report = 30.0
        print(
            f"Тест запущен на {args.minutes:.1f} мин. Создавайте обычные шумы помещения. "
            "Аудио не сохраняется. Ctrl+C — закончить раньше."
        )
        started = time.monotonic()
        while time.monotonic() - started < duration_sec:
            native_chunk = microphone.get_chunk(timeout=0.2)
            if native_chunk is not None:
                panns_chunk, semantic_chunk = dual_resampler.process(native_chunk)
                if panns_chunk.size:
                    detector.process_audio(panns_chunk, semantic_chunk)
            elapsed = time.monotonic() - started
            if elapsed >= next_report:
                print(f"{elapsed / 60:.1f} мин; тревог: {len(events)}")
                next_report += 30.0
    except KeyboardInterrupt:
        print("Тест остановлен пользователем.")
    finally:
        elapsed = max(time.monotonic() - started, 0.001)
        if dual_resampler is not None:
            dual_resampler.clear()
        if microphone is not None:
            microphone.stop()
        if detector is not None:
            detector.close()
        event_bus.stop()

    alarms_per_hour = len(events) / (elapsed / 3600.0)
    print("\nРезультат фонового теста")
    print(f"Время: {elapsed / 60:.2f} мин")
    print(f"Тревог: {len(events)}")
    print(f"Тревог в час: {alarms_per_hour:.2f}")
    print(f"Подробные решения: {stats_path}")
    if events:
        print("Это кандидаты на ложные срабатывания: проверьте время и звук, который тогда возник.")
    return 3 if args.fail_on_event and events else 0


if __name__ == "__main__":
    raise SystemExit(main())
