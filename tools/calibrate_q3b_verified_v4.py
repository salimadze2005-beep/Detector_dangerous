#!/usr/bin/env python3
"""Automatic gunshot calibration from a WAV and known positive search zones.

Designed for Detector_danger. The script:
1) treats operator timings as search zones and finds short transients inside them;
2) builds safe 3-second positive/negative runtime-like windows;
3) estimates acoustic gates from the same recording;
4) runs the project's PANNs + YAMNet models (unless --no-models);
5) grid-searches thresholds and writes a ready microphone profile.

Default timings are the Q3B test supplied by the operator:
0.22-0.32, 0.39-0.41, 0.45-0.48, 0.58-1.04, 1.10-1.17,
1.22-1.25, 1.30-1.40, 1.52-2.02, 2.06-2.07, 2.15-2.17,
2.25-2.40, 3.02-3.06

Run from the Detector_danger repository root, for example:
  .venv\\Scripts\\python.exe tools\\calibrate_from_zones_v3.py "recording (2).wav" \
      --device-key 12 --panns-device cuda --output-dir q3b_auto
"""

from __future__ import annotations

CALIBRATOR_VERSION = "q3b-v5-native-multirate"

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    import soundfile as sf
except ImportError as exc:
    raise SystemExit("Нужен пакет soundfile: pip install soundfile") from exc

try:
    from scipy.signal import find_peaks, medfilt, resample_poly
except ImportError as exc:
    raise SystemExit("Нужен пакет scipy: pip install scipy") from exc

EPS = 1e-9
TARGET_SR = 16_000
PANNS_SR = 32_000
DEFAULT_SHOTS = (
    "0.22-0.32,0.39-0.41,0.45-0.48,0.58-1.04,1.10-1.17,"
    "1.22-1.25,1.30-1.40,1.52-2.02,2.06-2.07,2.15-2.17,"
    "2.25-2.40,3.02-3.06"
)

# ВАЖНО: эти тайминги трактуются как MM.SS, а не десятичные секунды.
# Например: 0.22 = 00:22, 1.10 = 01:10, 3.02 = 03:02.


@dataclass(frozen=True)
class Interval:
    start: float
    end: float


@dataclass(frozen=True)
class AcousticSettings:
    trigger_dbfs: float
    rms_min: float
    min_snr_db: float
    min_crest_factor: float
    ambient_rms: float
    ambient_peak_dbfs: float
    positive_window_recall: float
    negative_pass_rate: float


@dataclass(frozen=True)
class ModelSettings:
    panns_prefilter_threshold: float
    panns_threshold: float
    panns_margin_threshold: float
    yamnet_threshold: float
    yamnet_veto_threshold: float
    yamnet_veto_margin: float
    shot_event_recall: float
    false_positives_per_minute: float
    accepted_negative_windows: int


def dbfs(amplitude: float) -> float:
    return 20.0 * math.log10(max(float(amplitude), EPS))


def amp(db_value: float) -> float:
    return 10.0 ** (float(db_value) / 20.0)


def parse_mmss(value: str) -> float:
    """MM.SS -> seconds. Examples: 0.22=22s, 1.10=70s, 3.02=182s.

    Также принимает MM:SS и чистое число секунд с суффиксом s, например 22s.
    """
    value = value.strip()
    if not value:
        raise ValueError("пустой тайминг")

    if value.lower().endswith("s"):
        return float(value[:-1].strip())

    if ":" in value:
        mm, ss = value.split(":", 1)
        minutes = int(mm.strip())
        seconds = float(ss.strip())
    elif "." in value:
        mm, ss = value.split(".", 1)
        minutes = int(mm.strip())
        # После точки это СЕКУНДЫ, а не десятичная дробь минуты.
        seconds = float(ss.strip())
    else:
        # Без разделителя считаем значение секундами.
        return float(value)

    if minutes < 0 or seconds < 0 or seconds >= 60:
        raise ValueError(f"неверный MM.SS тайминг: {value!r}")
    return minutes * 60.0 + seconds


def parse_timings(text: str) -> list[Interval]:
    result: list[Interval] = []
    normalized = text.replace(";", ",").replace("–", "-").replace("—", "-")
    for token in normalized.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            left, right = token.split("-", 1)
            start = parse_mmss(left)
            end = parse_mmss(right)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Неверный интервал {token!r}; нужен MM.SS-MM.SS, например 1.10-1.17"
            ) from exc
        if start < 0 or end <= start:
            raise argparse.ArgumentTypeError(f"Неверный интервал {token!r}")
        result.append(Interval(start, end))
    if not result:
        raise argparse.ArgumentTypeError("Список выстрелов пуст")
    return sorted(result, key=lambda item: item.start)


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load a mono WAV without discarding the native frequency content."""
    data, source_sr = sf.read(path, dtype="float32", always_2d=True)
    if not data.size:
        raise ValueError(f"Пустой WAV: {path}")
    mono = np.mean(data, axis=1, dtype=np.float32)
    mono = np.nan_to_num(mono, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(mono, -1.0, 1.0), int(source_sr)


def resample_audio(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    """Create one model-specific branch directly from the native WAV."""
    if int(source_sr) == int(target_sr):
        return np.asarray(audio, dtype=np.float32).copy()
    ratio = Fraction(int(target_sr), int(source_sr)).limit_denominator()
    return resample_poly(audio, ratio.numerator, ratio.denominator).astype(np.float32)


def metrics(segment: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(segment, dtype=np.float32).reshape(-1)
    if not x.size:
        return 0.0, 0.0, 0.0
    x = x - float(np.mean(x))
    rms = float(np.sqrt(np.mean(x * x)))
    peak = float(np.max(np.abs(x)))
    crest = peak / max(rms, EPS)
    return rms, peak, crest


def overlaps(start: float, end: float, intervals: Sequence[Interval]) -> bool:
    return any(start < item.end and end > item.start for item in intervals)


def merge_intervals(intervals: Sequence[Interval], gap: float = 0.0) -> list[Interval]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda x: x.start)
    merged = [ordered[0]]
    for item in ordered[1:]:
        prev = merged[-1]
        if item.start <= prev.end + gap:
            merged[-1] = Interval(prev.start, max(prev.end, item.end))
        else:
            merged.append(item)
    return merged


def extract(audio: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    a = max(0, int(round(start * sr)))
    b = min(len(audio), int(round(end * sr)))
    return audio[a:b]


def detect_transients(
    audio: np.ndarray,
    sr: int,
    zones: Sequence[Interval],
    min_separation_sec: float = 0.12,
) -> list[dict[str, object]]:
    """Find short impulsive events INSIDE operator-provided search zones.

    The zones are not treated as gunshots themselves. They are only places where
    gunshots are known to exist. We search for sharp local transients and return
    their exact times for model windows and short clips.
    """
    frame = max(32, int(round(0.010 * sr)))   # 10 ms
    hop = max(16, int(round(0.005 * sr)))     # 5 ms

    events: list[dict[str, object]] = []
    for zone_id, zone in enumerate(zones, 1):
        segment = extract(audio, sr, zone.start, zone.end)
        if len(segment) < frame:
            continue

        n = 1 + (len(segment) - frame) // hop
        shape = (n, frame)
        strides = (segment.strides[0] * hop, segment.strides[0])
        frames = np.lib.stride_tricks.as_strided(
            segment, shape=shape, strides=strides, writeable=False
        )

        peaks = np.max(np.abs(frames), axis=1)
        rms = np.sqrt(np.mean(frames * frames, axis=1) + EPS)
        crest = peaks / np.maximum(rms, EPS)
        rough = np.mean(np.abs(np.diff(frames, axis=1)), axis=1)

        peak_db = 20.0 * np.log10(np.maximum(peaks, EPS))
        rough_db = 20.0 * np.log10(np.maximum(rough, EPS))

        kernel = min(41, n if n % 2 else n - 1)
        kernel = max(3, kernel)
        if kernel % 2 == 0:
            kernel -= 1
        baseline_db = medfilt(peak_db, kernel_size=kernel)
        onset_db = np.maximum(0.0, peak_db - baseline_db)

        # Robust score: loudness + high temporal roughness + impulsiveness + onset.
        score = (
            (peak_db - np.median(peak_db))
            + 0.70 * (rough_db - np.median(rough_db))
            + 2.0 * np.maximum(crest - 2.0, 0.0)
            + 0.70 * onset_db
        )

        distance_frames = max(1, int(round(min_separation_sec * sr / hop)))
        peak_ids, props = find_peaks(score, distance=distance_frames, prominence=3.0)

        # Keep events reasonably close to the loudest transient in the zone.
        # The floor is deliberately permissive because distant shots can be quieter.
        zone_floor_db = max(-28.0, float(np.max(peak_db)) - 20.0)
        selected = [
            int(i)
            for i in peak_ids
            if float(peak_db[i]) >= zone_floor_db and float(crest[i]) >= 2.25
        ]

        # Never silently lose a zone: if the detector found nothing, keep the
        # strongest impulsive frame as a low-confidence fallback.
        if not selected:
            best = int(np.argmax(score))
            selected = [best]
        else:
            # Calibration needs representative impulses, not every waveform peak.
            # Keep strongest events with temporal diversity; long automatic-fire
            # zones still get more representatives than short single-shot zones.
            max_events = max(2, min(12, int(math.ceil((zone.end - zone.start) * 1.5))))
            ranked = sorted(selected, key=lambda idx: float(score[idx]), reverse=True)
            kept: list[int] = []
            diversity_frames = max(1, int(round(0.25 * sr / hop)))
            for idx in ranked:
                if all(abs(idx - other) >= diversity_frames for other in kept):
                    kept.append(idx)
                if len(kept) >= max_events:
                    break
            if not kept:
                kept = [ranked[0]]
            selected = sorted(kept)

        for local_id, i in enumerate(selected, 1):
            center = zone.start + (i * hop + frame / 2.0) / sr
            events.append(
                {
                    "zone_id": zone_id,
                    "event_id": f"z{zone_id:02d}_e{local_id:03d}",
                    "center_sec": float(center),
                    "start_sec": max(zone.start, float(center) - 0.06),
                    "end_sec": min(zone.end, float(center) + 0.12),
                    "peak_dbfs": float(peak_db[i]),
                    "crest": float(crest[i]),
                    "roughness_db": float(rough_db[i]),
                    "onset_db": float(onset_db[i]),
                    "transient_score": float(score[i]),
                    "fallback": len(peak_ids) == 0,
                }
            )

    return events


def save_short_clips(
    audio: np.ndarray,
    sr: int,
    events: Sequence[dict[str, object]],
    out_dir: Path,
    pre_sec: float = 0.12,
    post_sec: float = 0.48,
) -> None:
    shot_dir = out_dir / "clips" / "shots"
    shot_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    duration = len(audio) / sr

    for index, event in enumerate(events, 1):
        center = float(event["center_sec"])
        start = max(0.0, center - pre_sec)
        end = min(duration, center + post_sec)
        clip = extract(audio, sr, start, end)

        name = (
            f"shot_candidate_{index:03d}_"
            f"{center:.3f}s_zone{int(event['zone_id']):02d}.wav"
        )
        sf.write(shot_dir / name, clip, sr)

        rms, peak, crest = metrics(clip)
        row = dict(event)
        row.update(
            {
                "clip_start_sec": start,
                "clip_end_sec": end,
                "clip_duration_sec": end - start,
                "clip_rms": rms,
                "clip_peak": peak,
                "clip_peak_dbfs": dbfs(peak),
                "clip_crest": crest,
                "file": name,
            }
        )
        manifest.append(row)

    write_csv(manifest, out_dir / "transients.csv")

def quiet_stats(
    audio: np.ndarray, sr: int, shots: Sequence[Interval]
) -> tuple[float, float, float]:
    frame_sec = 0.25
    frame = int(round(frame_sec * sr))
    rms_values: list[float] = []
    peaks: list[float] = []
    crests: list[float] = []
    for start_sample in range(0, len(audio) - frame + 1, frame):
        start = start_sample / sr
        end = (start_sample + frame) / sr
        if overlaps(start - 0.50, end + 0.50, shots):
            continue
        rms, peak, crest = metrics(audio[start_sample : start_sample + frame])
        rms_values.append(rms)
        peaks.append(peak)
        crests.append(crest)
    if not rms_values:
        rms, peak, crest = metrics(audio)
        return max(rms, EPS), peak, crest
    rms_arr = np.asarray(rms_values)
    peak_arr = np.asarray(peaks)
    crest_arr = np.asarray(crests)
    quiet_cut = np.quantile(rms_arr, 0.35)
    quiet_mask = rms_arr <= quiet_cut
    ambient_rms = float(np.percentile(rms_arr[quiet_mask], 95))
    ambient_peak = float(np.percentile(peak_arr[quiet_mask], 99))
    ambient_crest_median = float(np.percentile(crest_arr[quiet_mask], 50))
    return max(ambient_rms, EPS), ambient_peak, ambient_crest_median

def make_runtime_windows(
    audio: np.ndarray,
    sr: int,
    events: Sequence[dict[str, object]],
    search_zones: Sequence[Interval],
    window_sec: float,
    positive_step: float,
    negative_step: float,
    negative_guard: float,
) -> list[dict[str, object]]:
    """Build runtime-like windows around detected transients, not whole zones."""
    duration = len(audio) / sr
    size = int(round(window_sec * sr))

    positive_specs: dict[tuple[float, int], str] = {}
    for event in events:
        center = float(event["center_sec"])
        zone_id = int(event["zone_id"])

        # Put the transient at several realistic relative positions in the
        # runtime buffer. Relative positions keep non-default window sizes
        # valid as well (the old hard-coded 3 s offsets could place the known
        # transient outside a shorter positive window).
        for fraction in (0.12, 0.27, 0.50, 0.73, 0.88):
            position = window_sec * fraction
            start = center - position
            end = start + window_sec
            if start < 0:
                end -= start
                start = 0.0
            if end > duration:
                start -= end - duration
                end = duration
            if start < 0 or end - start < window_sec * 0.95:
                continue
            positive_specs[(round(end, 6), zone_id)] = str(event["event_id"])

    # Do not use anything in/near the operator's known-positive search zones as noise.
    expanded = [
        Interval(
            max(0.0, zone.start - negative_guard),
            min(duration, zone.end + negative_guard),
        )
        for zone in search_zones
    ]

    negative_ends: set[float] = set()
    end = window_sec
    while end <= duration + 1e-9:
        start = end - window_sec
        if not overlaps(start, end, expanded):
            negative_ends.add(round(float(end), 6))
        end += negative_step

    rows: list[dict[str, object]] = []

    for (end, zone_id), event_id in sorted(positive_specs.items()):
        start = max(0.0, float(end) - window_sec)
        segment = extract(audio, sr, start, float(end))
        if len(segment) < size:
            segment = np.pad(segment, (size - len(segment), 0))
        elif len(segment) > size:
            segment = segment[-size:]
        rms, peak, crest = metrics(segment)
        rows.append(
            {
                "start_sec": start,
                "end_sec": float(end),
                "label": "shot",
                "zone_id": zone_id,
                "event_id": event_id,
                "rms": rms,
                "peak": peak,
                "peak_dbfs": dbfs(peak),
                "crest": crest,
                "audio": segment,
            }
        )

    for end in sorted(negative_ends):
        start = max(0.0, float(end) - window_sec)
        segment = extract(audio, sr, start, float(end))
        if len(segment) < size:
            segment = np.pad(segment, (size - len(segment), 0))
        elif len(segment) > size:
            segment = segment[-size:]
        rms, peak, crest = metrics(segment)
        rows.append(
            {
                "start_sec": start,
                "end_sec": float(end),
                "label": "noise",
                "zone_id": "",
                "event_id": "",
                "rms": rms,
                "peak": peak,
                "peak_dbfs": dbfs(peak),
                "crest": crest,
                "audio": segment,
            }
        )

    rows.sort(key=lambda row: (float(row["start_sec"]), str(row["label"])))
    return rows

def tune_acoustic(
    rows: Sequence[dict[str, object]],
    ambient_rms: float,
    ambient_peak: float,
    ambient_crest_median: float,
) -> AcousticSettings:
    positives = [row for row in rows if row["label"] == "shot"]
    negatives = [row for row in rows if row["label"] == "noise"]
    if not positives:
        raise ValueError("Не удалось построить положительные 3-секундные окна")

    # Do not fit these gates directly to the loud positive recording. A single
    # close/loud burst would otherwise push trigger/RMS unrealistically high.
    # Instead anchor them to the measured background, matching the runtime
    # calibration philosophy and preserving headroom for quieter/distant shots.
    trigger = float(np.clip(dbfs(ambient_peak) + 10.0, -20.0, -8.0))
    rms_min = float(np.clip(ambient_rms * 0.45, 1e-5, 0.02))
    snr_min = 8.0
    crest_min = float(np.clip(ambient_crest_median - 1.0, 2.5, 3.5))

    for row in rows:
        row["snr_db"] = 20.0 * math.log10(max(float(row["peak"]), EPS) / ambient_rms)

    def passed(row: dict[str, object]) -> bool:
        return (
            float(row["peak"]) >= amp(trigger)
            and float(row["rms"]) >= rms_min
            and float(row["snr_db"]) >= snr_min
            and float(row["crest"]) >= crest_min
        )

    pos_pass = sum(passed(x) for x in positives)
    neg_pass = sum(passed(x) for x in negatives)
    recall = pos_pass / len(positives)
    neg_rate = neg_pass / max(len(negatives), 1)

    return AcousticSettings(
        trigger_dbfs=round(trigger, 1),
        rms_min=round(rms_min, 6),
        min_snr_db=snr_min,
        min_crest_factor=round(crest_min, 2),
        ambient_rms=round(ambient_rms, 6),
        ambient_peak_dbfs=round(dbfs(ambient_peak), 2),
        positive_window_recall=round(float(recall), 4),
        negative_pass_rate=round(float(neg_rate), 4),
    )

def import_models(config_path: Path, panns_device: str):
    root = Path.cwd()
    if not (root / "audio").is_dir() or not (root / "core").is_dir():
        raise RuntimeError(
            "Запускайте скрипт из корня Detector_danger (рядом должны быть audio/ и core/)"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from core.config import load_config  # type: ignore
    from audio.gunshot_detector_runtime import PANNsClassifier, YAMNetClassifier  # type: ignore

    config = load_config(config_path)
    panns = PANNsClassifier(config.audio.panns_model_path, panns_device)
    yamnet = YAMNetClassifier(config.audio.yamnet_model_path)
    return panns, yamnet


def _window_for_rate(
    row: dict[str, object],
    audio: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    start = float(row["start_sec"])
    end = float(row["end_sec"])
    expected = int(round((end - start) * sample_rate))
    segment = extract(audio, sample_rate, start, end)
    if len(segment) < expected:
        return np.pad(segment, (expected - len(segment), 0))
    return segment[-expected:]


def score_models(
    rows: list[dict[str, object]],
    panns_audio: np.ndarray,
    panns_sr: int,
    panns,
    yamnet,
) -> None:
    total = len(rows)
    for index, row in enumerate(rows, 1):
        semantic_segment = np.asarray(row["audio"], dtype=np.float32)
        panns_segment = _window_for_rate(row, panns_audio, panns_sr)
        pf, pe, pn, ptop, ptop_score = panns.check(panns_segment, panns_sr)
        yf, ye, yn, ytop, ytop_score = yamnet.check(semantic_segment)
        row.update(
            {
                "panns_input_sample_rate": int(panns_sr),
                "panns_firearm": float(pf),
                "panns_explosive": float(pe),
                "panns_nuisance": float(pn),
                "panns_margin": float(pf - max(pe, pn)),
                "panns_top": str(ptop),
                "panns_top_score": float(ptop_score),
                "yamnet_firearm": float(yf),
                "yamnet_explosive": float(ye),
                "yamnet_nuisance": float(yn),
                "yamnet_top": str(ytop),
                "yamnet_top_score": float(ytop_score),
            }
        )
        print(
            f"\rМодели {index:3d}/{total}: t={float(row['end_sec']):7.2f}s "
            f"PANNs={pf:.3f} YAMNet={yf:.3f}",
            end="",
            flush=True,
        )
    print()


def acoustic_pass(row: dict[str, object], settings: AcousticSettings) -> bool:
    return (
        float(row["peak"]) >= amp(settings.trigger_dbfs)
        and float(row["rms"]) >= settings.rms_min
        and float(row["snr_db"]) >= settings.min_snr_db
        and float(row["crest"]) >= settings.min_crest_factor
    )


def fusion_accept(
    row: dict[str, object],
    acoustic: AcousticSettings,
    prefilter: float,
    panns_threshold: float,
    margin_threshold: float,
    yamnet_threshold: float,
    veto_threshold: float,
    veto_margin: float,
) -> bool:
    if not acoustic_pass(row, acoustic):
        return False
    pf = float(row["panns_firearm"])
    pe = float(row["panns_explosive"])
    pn = float(row["panns_nuisance"])
    yf = float(row["yamnet_firearm"])
    yn = float(row["yamnet_nuisance"])
    margin = pf - max(pe, pn)
    # Kept in the function signature for compatibility with older reports.
    # Runtime no longer lets a PANNs-only prefilter suppress YAMNet inference.
    _ = prefilter
    veto = yn >= veto_threshold and yn >= yf + veto_margin
    normal = pf >= panns_threshold and margin >= margin_threshold and yf >= yamnet_threshold
    rescue = (
        pf >= min(panns_threshold, 0.15)
        and margin >= min(margin_threshold, 0.10)
        and yf >= max(yamnet_threshold, 0.70)
    )
    independent_yamnet = (
        yf >= max(yamnet_threshold, 0.70)
        and yf >= yn + 0.12
    )
    return bool((normal or rescue or independent_yamnet) and not veto)


def event_recall(
    rows: Sequence[dict[str, object]],
    accepted: Sequence[bool],
    shots: Sequence[Interval],
) -> float:
    hits = 0
    for shot in shots:
        if any(
            ok
            and float(row["start_sec"]) < shot.end
            and float(row["end_sec"]) > shot.start
            for row, ok in zip(rows, accepted)
        ):
            hits += 1
    return hits / len(shots)


def tune_models(
    rows: Sequence[dict[str, object]],
    shots: Sequence[Interval],
    acoustic: AcousticSettings,
) -> ModelSettings:
    negative_minutes = sum(
        float(row["end_sec"]) - float(row["start_sec"])
        for row in rows
        if row["label"] == "noise"
    ) / 60.0
    # Windows overlap by design; use elapsed negative coverage rather than the sum
    # for the FP/min denominator.
    negative_rows = [row for row in rows if row["label"] == "noise"]
    if negative_rows:
        negative_coverage_min = (
            max(float(x["end_sec"]) for x in negative_rows)
            - min(float(x["start_sec"]) for x in negative_rows)
        ) / 60.0
    else:
        negative_coverage_min = max(negative_minutes, EPS)

    episodes = merge_intervals(shots, gap=1.0)
    one_episode = len(episodes) <= 1

    # Compatibility field only; the current runtime always executes YAMNet.
    prefilters = (0.0,)
    panns_values = (0.15, 0.20, 0.22, 0.25, 0.30) if one_episode else (0.15, 0.20, 0.22, 0.25, 0.30, 0.35)
    margin_values = (0.05, 0.08, 0.10, 0.12, 0.15) if one_episode else (0.05, 0.08, 0.10, 0.12, 0.15, 0.20)
    yamnet_values = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45) if one_episode else (0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60)
    veto_values = (0.45, 0.50, 0.55, 0.65, 0.75)

    best = None
    for prefilter in prefilters:
        for panns_threshold in panns_values:
            if prefilter > panns_threshold:
                continue
            for margin_threshold in margin_values:
                for yamnet_threshold in yamnet_values:
                    for veto_threshold in veto_values:
                        accepted = [
                            fusion_accept(
                                row,
                                acoustic,
                                prefilter,
                                panns_threshold,
                                margin_threshold,
                                yamnet_threshold,
                                veto_threshold,
                                0.10,
                            )
                            for row in rows
                        ]
                        recall = event_recall(rows, accepted, shots)
                        fp_count = sum(
                            ok and row["label"] == "noise"
                            for row, ok in zip(rows, accepted)
                        )
                        fp_per_min = fp_count / max(negative_coverage_min, EPS)

                        # Recall is dominant. Then suppress false alarms. With only
                        # one positive burst, regularize toward moderate thresholds
                        # instead of overfitting to a single unusually loud sample.
                        regularizer = (
                            abs(panns_threshold - 0.25) * 1.5
                            + abs(margin_threshold - 0.10) * 1.0
                            + abs(yamnet_threshold - 0.35) * 1.2
                        )
                        score = recall * 1000.0 - fp_per_min * 35.0 - fp_count * 1.0 - regularizer
                        candidate = (
                            score,
                            recall,
                            -fp_per_min,
                            -fp_count,
                            prefilter,
                            panns_threshold,
                            margin_threshold,
                            yamnet_threshold,
                            veto_threshold,
                        )
                        if best is None or candidate > best:
                            best = candidate

    if best is None:
        raise RuntimeError("Не удалось подобрать модельные пороги")
    _, recall, neg_fp, neg_count, prefilter, panns_t, margin_t, yamnet_t, veto_t = best

    # A profile with zero/very low recall is not a recommendation. The previous
    # version still wrote its regularized defaults, which was misleading.
    if float(recall) < 0.70:
        raise RuntimeError(
            f"Модели не разделяют выстрелы и фон достаточно хорошо: "
            f"лучший recall={float(recall):.1%}. "
            "Не применяйте автоматически подобранные PANNs/YAMNet пороги."
        )
    return ModelSettings(
        panns_prefilter_threshold=float(prefilter),
        panns_threshold=float(panns_t),
        panns_margin_threshold=float(margin_t),
        yamnet_threshold=float(yamnet_t),
        yamnet_veto_threshold=float(veto_t),
        yamnet_veto_margin=0.10,
        shot_event_recall=round(float(recall), 4),
        false_positives_per_minute=round(float(-neg_fp), 4),
        accepted_negative_windows=int(-neg_count),
    )


def write_csv(rows: Sequence[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    clean_rows = []
    for row in rows:
        clean_rows.append({k: v for k, v in row.items() if k != "audio"})
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(clean_rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(clean_rows)


def save_noise_examples(
    rows: Sequence[dict[str, object]], sr: int, out_dir: Path, limit: int = 20
) -> None:
    noise_dir = out_dir / "clips" / "noise"
    noise_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for row in rows:
        if row["label"] != "noise":
            continue
        count += 1
        if count > limit:
            break
        start = float(row["start_sec"])
        end = float(row["end_sec"])
        sf.write(noise_dir / f"noise_{count:02d}_{start:.2f}-{end:.2f}.wav", row["audio"], sr)


def make_profile(device_key: str, acoustic: AcousticSettings, model: ModelSettings | None) -> dict[str, object]:
    profile: dict[str, object] = {
        "preset": "custom",
        "trigger_dbfs": acoustic.trigger_dbfs,
        "rms_min": acoustic.rms_min,
        "min_snr_db": acoustic.min_snr_db,
        "min_crest_factor": acoustic.min_crest_factor,
    }
    panns_prefilter = 0.05
    if model is not None:
        panns_prefilter = model.panns_prefilter_threshold
        profile.update(
            {
                "panns_threshold": model.panns_threshold,
                "panns_margin_threshold": model.panns_margin_threshold,
                "yamnet_threshold": model.yamnet_threshold,
                "yamnet_veto_threshold": model.yamnet_veto_threshold,
                "yamnet_veto_margin": model.yamnet_veto_margin,
            }
        )
    return {
        "audio": {
            "panns_prefilter_threshold": panns_prefilter,
            "microphone_profiles": {str(device_key): profile},
        }
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Автокалибровка Detector_danger: тайминги задают зоны поиска, внутри них автоматически находятся короткие импульсы"
    )
    parser.add_argument("wav", type=Path, help="Оригинальная WAV-запись")
    parser.add_argument(
        "--shots",
        default=DEFAULT_SHOTS,
        help="Зоны поиска MM.SS-MM.SS; 1.10 означает 01:10. Скрипт сам найдёт короткие импульсы внутри зон.",
    )
    parser.add_argument("--config", type=Path, default=Path("detector_config.json"))
    parser.add_argument("--device-key", default="12")
    parser.add_argument("--output-dir", type=Path, default=Path("q3b_auto_calibration"))
    parser.add_argument("--panns-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--window-sec", type=float, default=3.0)
    parser.add_argument("--positive-step", type=float, default=0.25)
    parser.add_argument("--negative-step", type=float, default=2.0)
    parser.add_argument("--negative-guard", type=float, default=0.50)
    parser.add_argument("--clip-padding", type=float, default=0.08)
    parser.add_argument(
        "--no-models",
        action="store_true",
        help="Только акустические параметры; не запускать PANNs/YAMNet",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    print(f"Calibrator version: {CALIBRATOR_VERSION}")
    print(f"Script path: {Path(__file__).resolve()}")
    args = build_parser().parse_args(argv)
    if not args.wav.is_file():
        raise SystemExit(f"Файл не найден: {args.wav}")
    if args.window_sec <= 0 or args.positive_step <= 0 or args.negative_step <= 0:
        raise SystemExit("Размер окна и шаги должны быть > 0")

    shots = parse_timings(args.shots)
    native_audio, source_sr = load_audio(args.wav)
    audio = resample_audio(native_audio, source_sr, TARGET_SR)
    panns_audio = resample_audio(native_audio, source_sr, PANNS_SR)
    sr = TARGET_SR
    duration = len(audio) / sr
    for shot in shots:
        if shot.end > duration:
            raise SystemExit(
                f"Интервал {shot.start:.2f}-{shot.end:.2f} выходит за длительность {duration:.2f} с"
            )

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    events = detect_transients(audio, sr, shots)
    if not events:
        raise SystemExit("В заданных зонах не найдено ни одного импульсного события")
    save_short_clips(audio, sr, events, out)

    ambient_rms, ambient_peak, ambient_crest_median = quiet_stats(audio, sr, shots)
    rows = make_runtime_windows(
        audio,
        sr,
        events,
        shots,
        args.window_sec,
        args.positive_step,
        args.negative_step,
        args.negative_guard,
    )
    acoustic = tune_acoustic(rows, ambient_rms, ambient_peak, ambient_crest_median)
    save_noise_examples(rows, sr, out)

    print(f"WAV: {args.wav}")
    print(f"Длительность: {duration:.2f} с, source_sr={source_sr}, PANNs_sr={PANNS_SR}, YAMNet_sr={sr}")
    print(f"Зон поиска: {len(shots)}")
    print(f"Найдено коротких импульсов: {len(events)}")
    print("Формат таймингов: MM.SS (например 1.10 = 70 секунд)")
    print(f"Runtime-окон: {len(rows)}")
    print("Акустические настройки:")
    print(json.dumps(asdict(acoustic), ensure_ascii=False, indent=2))

    model: ModelSettings | None = None
    model_error: str | None = None
    if not args.no_models:
        try:
            panns, yamnet = import_models(args.config, args.panns_device)
            score_models(rows, panns_audio, PANNS_SR, panns, yamnet)
            model = tune_models(rows, shots, acoustic)
            print("Модельные настройки:")
            print(json.dumps(asdict(model), ensure_ascii=False, indent=2))
        except Exception as exc:
            model_error = f"{type(exc).__name__}: {exc}"
            print(f"Модельный этап не выполнен: {model_error}")

    write_csv(rows, out / "window_scores.csv")

    recommended_path = out / "recommended_profile.json"
    acoustic_path = out / "acoustic_profile.json"

    # Always make acoustic-only output explicit.
    acoustic_profile = make_profile(args.device_key, acoustic, None)
    acoustic_path.write_text(
        json.dumps(acoustic_profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # A recommended model profile exists only when the model stage genuinely passed.
    if model is not None and model_error is None:
        profile = make_profile(args.device_key, acoustic, model)
        recommended_path.write_text(
            json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        profile = None
        if recommended_path.exists():
            recommended_path.unlink()

    episodes = merge_intervals(shots, gap=1.0)
    clipping = np.abs(audio) >= 0.999
    report = {
        "calibrator_version": CALIBRATOR_VERSION,
        "input": {
            "wav": str(args.wav.resolve()),
            "duration_sec": round(duration, 3),
            "source_sample_rate": source_sr,
            "analysis_sample_rate": sr,
            "panns_analysis_sample_rate": PANNS_SR,
            "search_zones": [asdict(item) for item in shots],
            "positive_episodes": len(episodes),
            "detected_transients": len(events),
        },
        "acoustic": asdict(acoustic),
        "model": asdict(model) if model else None,
        "model_error": model_error,
        "clipping": {
            "clipped_samples": int(np.count_nonzero(clipping)),
            "ratio": float(np.mean(clipping)),
        },
        "profile_fragment": profile,
        "confidence": (
            "model_invalid"
            if model_error and not args.no_models
            else ("limited_one_positive_burst" if len(episodes) <= 1 else "multi_episode")
        ),
        "note": (
            (
                "Модельный профиль не выдан: PANNs/YAMNet не достигли минимального "
                "recall 70% на заданных зонах. Акустические параметры можно анализировать "
                "отдельно, но модельные пороги автоматически применять нельзя."
            )
            if model_error and not args.no_models
            else (
                f"Калибровка построена по {len(episodes)} независимым положительным зонам "
                f"и {len(events)} автоматически найденным коротким импульсам."
            )
        ),
    }
    (out / "calibration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out / "CALIBRATOR_VERSION.txt").write_text(
        CALIBRATOR_VERSION + "\n" + str(Path(__file__).resolve()) + "\n",
        encoding="utf-8",
    )

    print(f"\nГотово: {out}")
    print(f"Короткие импульсы: {out / 'clips' / 'shots'}")
    print(f"Фоновые примеры:   {out / 'clips' / 'noise'}")
    print(f"Оценки окон:       {out / 'window_scores.csv'}")
    print(f"Готовый профиль:   {out / 'recommended_profile.json'}")
    print(f"Полный отчёт:      {out / 'calibration_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
