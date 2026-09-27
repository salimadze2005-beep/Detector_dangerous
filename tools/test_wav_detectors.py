#!/usr/bin/env python3
"""Run the current acoustic, PANNs, YAMNet and fusion detectors on a WAV.

This is a read-only diagnostic runner. It does not publish EventBus events,
send REST requests, or change detector_config.json. It writes a per-window CSV
and a JSON summary to the selected output directory.

Example:
    .venv\\Scripts\\python.exe tools\\test_wav_detectors.py \
        "recording (2).wav" --device-key 12 --panns-device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio.calibration import dbfs_to_amplitude, resolve_audio_profile
from audio.gunshot_detector_runtime import (
    EnergyGate,
    PANNsClassifier,
    PANNsYAMNetFusion,
    YAMNetClassifier,
)
from audio.microphone import StreamingAudioResampler
from core.config import load_config


LOGGER = logging.getLogger("test_wav_detectors")
CSV_FIELDS = (
    "time_sec",
    "warmup",
    "acoustic_pass",
    "rms",
    "peak",
    "snr_db",
    "crest_factor",
    "noise_rms",
    "panns_firearm",
    "panns_explosive",
    "panns_nuisance",
    "panns_margin",
    "panns_top",
    "yamnet_firearm",
    "yamnet_explosive",
    "yamnet_nuisance",
    "yamnet_top",
    "yamnet_veto",
    "panns_below_prefilter",
    "accepted",
    "event_emitted",
    "cooldown_suppressed",
    "reason",
)


def append_buffer(buffer: np.ndarray, chunk: np.ndarray) -> np.ndarray:
    if not chunk.size:
        return buffer
    if len(chunk) >= len(buffer):
        return chunk[-len(buffer):].copy()
    buffer = np.roll(buffer, -len(chunk))
    buffer[-len(chunk):] = chunk
    return buffer


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=False, dtype="float32")
    data = np.asarray(audio, dtype=np.float32)
    if data.ndim == 2:
        data = np.mean(data, axis=1, dtype=np.float32)
    data = np.clip(data.reshape(-1), -1.0, 1.0)
    if not data.size:
        raise ValueError("WAV не содержит аудиосэмплов")
    return data, int(sample_rate)


def test_wav(args: argparse.Namespace) -> int:
    wav_path = Path(args.wav).expanduser().resolve()
    if not wav_path.is_file():
        raise SystemExit(f"WAV не найден: {wav_path}")

    config = load_config(args.config)
    errors = config.validate()
    if errors:
        raise SystemExit("Конфигурация невалидна:\n- " + "\n- ".join(errors))

    audio, source_rate = load_mono(wav_path)
    panns_rate = int(config.audio.panns_sample_rate)
    semantic_rate = int(config.audio.sample_rate)
    panns_audio = StreamingAudioResampler(source_rate, panns_rate).process(audio, last=True)
    semantic_audio = StreamingAudioResampler(source_rate, semantic_rate).process(audio, last=True)

    profile = resolve_audio_profile(config.audio, args.device_key)
    panns = PANNsClassifier(config.audio.panns_model_path, args.panns_device)
    yamnet = YAMNetClassifier(config.audio.yamnet_model_path)
    fusion = PANNsYAMNetFusion(
        panns_threshold=float(profile["panns_threshold"]),
        panns_margin_threshold=float(profile["panns_margin_threshold"]),
        yamnet_threshold=float(profile["yamnet_threshold"]),
        yamnet_veto_threshold=float(profile["yamnet_veto_threshold"]),
        yamnet_veto_margin=float(profile["yamnet_veto_margin"]),
    )
    gate = EnergyGate(
        rms_min=float(profile["rms_min"]),
        peak_min=dbfs_to_amplitude(float(profile["trigger_dbfs"])),
        adaptive=bool(config.audio.adaptive_noise),
        noise_alpha=float(config.audio.noise_alpha),
        min_snr_db=float(profile["min_snr_db"]),
        min_crest_factor=float(profile["min_crest_factor"]),
    )

    window_sec = float(args.window_sec or config.audio.analysis_window_sec)
    step_sec = float(args.step_sec or config.audio.analysis_step_sec)
    if window_sec <= 0 or step_sec <= 0:
        raise SystemExit("window-sec и step-sec должны быть больше нуля")
    panns_window = max(1, round(panns_rate * window_sec))
    semantic_window = max(1, round(semantic_rate * window_sec))
    panns_buffer = np.zeros(panns_window, dtype=np.float32)
    semantic_buffer = np.zeros(semantic_window, dtype=np.float32)

    total_sec = len(semantic_audio) / semantic_rate
    rows: list[dict[str, object]] = []
    reasons: Counter[str] = Counter()
    accepted_count = 0
    emitted_count = 0
    cooldown_suppressed_count = 0
    last_event_sec = -1e9
    acoustic_pass_count = 0
    next_analysis = step_sec
    chunk_sec = float(config.audio.chunk_duration_sec)
    chunk_count = max(1, int(np.ceil(total_sec / chunk_sec)))

    LOGGER.info(
        "WAV=%s duration=%.2fs source=%d native→PANNs=%d native→YAMNet=%d",
        wav_path,
        total_sec,
        source_rate,
        panns_rate,
        semantic_rate,
    )
    LOGGER.info(
        "profile=%s trigger=%.1f dBFS rms_min=%.6f crest_min=%.2f "
        "PANNs=%.3f margin=%.3f YAMNet=%.3f veto=%.3f",
        profile["profile_key"],
        float(profile["trigger_dbfs"]),
        float(profile["rms_min"]),
        float(profile["min_crest_factor"]),
        float(profile["panns_threshold"]),
        float(profile["panns_margin_threshold"]),
        float(profile["yamnet_threshold"]),
        float(profile["yamnet_veto_threshold"]),
    )

    for chunk_index in range(chunk_count):
        start_sec = chunk_index * chunk_sec
        end_sec = min(total_sec, start_sec + chunk_sec)
        p_start = round(start_sec * panns_rate)
        p_end = min(len(panns_audio), round(end_sec * panns_rate))
        y_start = round(start_sec * semantic_rate)
        y_end = min(len(semantic_audio), round(end_sec * semantic_rate))
        panns_chunk = panns_audio[p_start:p_end]
        semantic_chunk = semantic_audio[y_start:y_end]
        if not panns_chunk.size or not semantic_chunk.size:
            continue
        panns_buffer = append_buffer(panns_buffer, panns_chunk)
        semantic_buffer = append_buffer(semantic_buffer, semantic_chunk)
        current_sec = end_sec
        if current_sec + 1e-9 < next_analysis:
            continue

        while current_sec + 1e-9 >= next_analysis:
            ok, rms, peak = gate.check(semantic_buffer)
            metrics = dict(gate.last_metrics)
            row: dict[str, object] = {
                "time_sec": round(next_analysis, 3),
                "warmup": next_analysis < window_sec,
                "acoustic_pass": bool(ok),
                "rms": round(float(rms), 6),
                "peak": round(float(peak), 6),
                "snr_db": round(float(metrics.get("snr_db", 0.0)), 3),
                "crest_factor": round(float(metrics.get("crest_factor", 0.0)), 3),
                "noise_rms": round(float(metrics.get("noise_rms", 0.0)), 6),
                "panns_firearm": "",
                "panns_explosive": "",
                "panns_nuisance": "",
                "panns_margin": "",
                "panns_top": "",
                "yamnet_firearm": "",
                "yamnet_explosive": "",
                "yamnet_nuisance": "",
                "yamnet_top": "",
                "yamnet_veto": "",
                "panns_below_prefilter": "",
                "accepted": False,
                "event_emitted": False,
                "cooldown_suppressed": False,
                "reason": "acoustic_gate" if not ok else "",
            }
            if ok:
                acoustic_pass_count += 1
                pf, pe, pn, ptop, _ = panns.check(panns_buffer, panns_rate)
                yf, ye, yn, ytop, _ = yamnet.check(semantic_buffer)
                margin = float(pf) - max(float(pe), float(pn))
                veto = bool(config.audio.use_veto and yn >= fusion.yv and yn >= yf + fusion.ym)
                decision = fusion.decide(pf, pe, pn, yf, ye, yn, veto)
                row.update(
                    {
                        "panns_firearm": round(float(pf), 6),
                        "panns_explosive": round(float(pe), 6),
                        "panns_nuisance": round(float(pn), 6),
                        "panns_margin": round(margin, 6),
                        "panns_top": ptop,
                        "yamnet_firearm": round(float(yf), 6),
                        "yamnet_explosive": round(float(ye), 6),
                        "yamnet_nuisance": round(float(yn), 6),
                        "yamnet_top": ytop,
                        "yamnet_veto": bool(veto),
                        "panns_below_prefilter": bool(pf < float(config.audio.panns_prefilter_threshold)),
                        "accepted": bool(decision.accepted),
                        "reason": decision.reason,
                    }
                )
                if decision.accepted:
                    accepted_count += 1
                    if next_analysis - last_event_sec >= float(config.audio.cooldown_sec):
                        row["event_emitted"] = True
                        last_event_sec = next_analysis
                        emitted_count += 1
                    else:
                        row["cooldown_suppressed"] = True
                        cooldown_suppressed_count += 1
                reasons[decision.reason] += 1
            else:
                reasons["acoustic_gate"] += 1
            rows.append(row)
            next_analysis += step_sec

    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "data" / "wav_tests" / wav_path.stem
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "window_scores.csv"
    summary_path = output_dir / "summary.json"
    write_csv(csv_path, rows)
    summary = {
        "wav": str(wav_path),
        "source_sample_rate": source_rate,
        "panns_sample_rate": panns_rate,
        "semantic_sample_rate": semantic_rate,
        "duration_sec": round(total_sec, 3),
        "profile": profile,
        "window_sec": window_sec,
        "step_sec": step_sec,
        "windows": len(rows),
        "acoustic_pass_windows": acoustic_pass_count,
        "accepted_windows": accepted_count,
        "emitted_events_after_cooldown": emitted_count,
        "cooldown_suppressed_windows": cooldown_suppressed_count,
        "cooldown_sec": float(config.audio.cooldown_sec),
        "reasons": dict(reasons),
        "note": "Diagnostic only: no EventBus events, REST requests or config changes.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"CSV: {csv_path}")
    print(f"SUMMARY: {summary_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Проверка текущих аудиодетекторов на WAV")
    parser.add_argument("wav", help="путь к WAV-файлу")
    parser.add_argument("--config", default=None, help="путь к detector_config.json")
    parser.add_argument("--device-key", default="__default__", help="ключ профиля микрофона, например 12")
    parser.add_argument("--panns-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--window-sec", type=float, default=None, help="длина окна; по умолчанию из config")
    parser.add_argument("--step-sec", type=float, default=None, help="шаг анализа; по умолчанию из config")
    parser.add_argument("--output-dir", default=None, help="куда сохранить CSV и summary.json")
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(test_wav(build_parser().parse_args()))
