import librosa
import logging
from core.event_bus import EventBus, Event
from audio.gunshot_detector import GunshotDetector
from core.config import config

logging.basicConfig(level=logging.INFO)

print(">>> TEST_WAV v5: EMA + 3of4 + STATS <<<")


def on_event(event: Event):
    print("\n=========================")
    print(f"{event.type}")
    print(f"Confidence (EMA): {event.confidence:.0%}")
    print(f"Details: {event.metadata}")
    print("=========================\n")


def test_file(filepath: str):
    bus = EventBus()
    bus.subscribe("GUNSHOT_DETECTED", on_event)

    detector = GunshotDetector(
        event_bus=bus,
        model_path="models/binary_gunshot_full.h5",
        threshold=0.75
    )

    print(f"\nAnalyzing: {filepath}")
    waveform, sr = librosa.load(filepath, sr=config.audio.sample_rate, mono=True)
    print(f"File length: {len(waveform) / sr:.2f} sec")

    window_size = config.audio.sample_rate
    step = window_size // 2

    n = 0
    for i in range(0, max(len(waveform) - window_size, 0) + 1, step):
        detector.process_window(waveform[i:i + window_size], advance_sec=step / sr)
        n += 1

    if len(waveform) <= window_size:
        detector.process_window(waveform, advance_sec=len(waveform) / sr)
        n += 1
    elif (len(waveform) - window_size) % step != 0:
        detector.process_window(waveform[-window_size:], advance_sec=step / sr)
        n += 1

    detector.close()
    print(f"Processed windows: {n}")
    print("Analysis complete. Телеметрия: detector_stats.csv\n")


if __name__ == "__main__":
    test_file("tests/gun4.wav")