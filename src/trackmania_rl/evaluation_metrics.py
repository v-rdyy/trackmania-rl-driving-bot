"""Precision, orientation, and stuck metrics shared by policy evaluators."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

STEERING_HYSTERESIS = 0.10
STEERING_DELTA_DEADBAND = 0.05
OSCILLATION_WINDOW_SECONDS = 2.0
OSCILLATION_CROSSINGS_THRESHOLD = 4
LATERAL_THRESHOLDS = (5.0, 10.0, 20.0)
UPSIDE_DOWN_COSINE_THRESHOLD = 0.0
STUCK_WINDOW_SECONDS = 2.0
STUCK_PROGRESS_UNITS = 1.0
STUCK_WORLD_DISTANCE_UNITS = 2.0


def steering_precision_metrics(
    samples: list[tuple[float, float]],
) -> dict[str, Any]:
    if not samples:
        raise ValueError("steering metrics require at least one sample")
    times = np.asarray([sample[0] for sample in samples], dtype=np.float64)
    values = np.asarray([sample[1] for sample in samples], dtype=np.float64)
    if not np.isfinite(np.concatenate((times, values))).all():
        raise ValueError("steering samples contain nonfinite values")
    if np.any(np.diff(times) < 0.0):
        raise ValueError("steering sample times must be nondecreasing")
    if np.any(np.abs(values) > 1.0 + 1e-9):
        raise ValueError("normalized steering values must stay in [-1, 1]")

    crossing_times: list[float] = []
    last_side = 0
    for timestamp, value in zip(times, values, strict=True):
        side = 1 if value >= STEERING_HYSTERESIS else -1 if value <= -STEERING_HYSTERESIS else 0
        if side and last_side and side != last_side:
            crossing_times.append(float(timestamp))
        if side:
            last_side = side

    direction_reversals = 0
    last_delta_side = 0
    for delta in np.diff(values):
        delta_side = (
            1
            if delta >= STEERING_DELTA_DEADBAND
            else -1
            if delta <= -STEERING_DELTA_DEADBAND
            else 0
        )
        if delta_side and last_delta_side and delta_side != last_delta_side:
            direction_reversals += 1
        if delta_side:
            last_delta_side = delta_side

    peak_crossings = 0
    peak_window_end: float | None = None
    left = 0
    for right, timestamp in enumerate(crossing_times):
        while crossing_times[left] < timestamp - OSCILLATION_WINDOW_SECONDS:
            left += 1
        crossings = right - left + 1
        if crossings > peak_crossings:
            peak_crossings = crossings
            peak_window_end = timestamp

    peak_window = None
    if peak_window_end is not None:
        peak_window = {
            "start_seconds": max(0.0, peak_window_end - OSCILLATION_WINDOW_SECONDS),
            "end_seconds": peak_window_end,
        }
    return {
        "sample_count": len(samples),
        "mean_absolute_steering": float(np.abs(values).mean()),
        "rms_steering": float(np.sqrt(np.square(values).mean())),
        "maximum_absolute_steering": float(np.abs(values).max()),
        "total_variation": float(np.abs(np.diff(values)).sum()),
        "hysteresis_sign_crossings": len(crossing_times),
        "significant_direction_reversals": direction_reversals,
        "peak_sign_crossings_in_2_seconds": peak_crossings,
        "peak_oscillation_window": peak_window,
        "oscillation_detected": peak_crossings >= OSCILLATION_CROSSINGS_THRESHOLD,
    }


def _condition_periods(
    times_ms: np.ndarray,
    condition: np.ndarray,
) -> list[dict[str, float]]:
    periods: list[dict[str, float]] = []
    start_ms: float | None = None
    end_ms: float | None = None
    for index in range(1, len(times_ms)):
        if bool(condition[index]):
            if start_ms is None:
                start_ms = float(times_ms[index - 1])
            end_ms = float(times_ms[index])
        elif start_ms is not None and end_ms is not None:
            periods.append(
                {
                    "start_seconds": start_ms / 1000.0,
                    "end_seconds": end_ms / 1000.0,
                    "duration_seconds": (end_ms - start_ms) / 1000.0,
                }
            )
            start_ms = None
            end_ms = None
    if start_ms is not None and end_ms is not None:
        periods.append(
            {
                "start_seconds": start_ms / 1000.0,
                "end_seconds": end_ms / 1000.0,
                "duration_seconds": (end_ms - start_ms) / 1000.0,
            }
        )
    return periods


def _period_summary(periods: list[dict[str, float]]) -> dict[str, Any]:
    durations = [float(period["duration_seconds"]) for period in periods]
    return {
        "period_count": len(periods),
        "total_duration_seconds": float(sum(durations)),
        "maximum_duration_seconds": max(durations, default=0.0),
        "periods": periods,
    }


def trajectory_precision_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) < 2:
        raise ValueError("trajectory metrics require at least two records")
    times_ms = np.asarray(
        [record["race_time_ms"] for record in records],
        dtype=np.float64,
    )
    positions = np.asarray(
        [record["position"] for record in records],
        dtype=np.float64,
    )
    progress = np.asarray(
        [record["progress"] for record in records],
        dtype=np.float64,
    )
    lateral = np.asarray(
        [record["lateral_offset"] for record in records],
        dtype=np.float64,
    )
    upright = np.asarray(
        [record["upright_cosine"] for record in records],
        dtype=np.float64,
    )
    values = np.concatenate(
        (times_ms, positions.ravel(), progress, lateral, upright)
    )
    if not np.isfinite(values).all():
        raise ValueError("trajectory contains nonfinite precision telemetry")
    if np.any(np.diff(times_ms) <= 0.0):
        raise ValueError("trajectory race times must be strictly increasing")

    time_deltas_seconds = np.diff(times_ms) / 1000.0
    absolute_lateral = np.abs(lateral)
    lateral_metrics: dict[str, Any] = {
        "mean_absolute_offset": float(absolute_lateral.mean()),
        "rms_offset": float(np.sqrt(np.square(lateral).mean())),
        "p95_absolute_offset": float(np.percentile(absolute_lateral, 95)),
        "maximum_absolute_offset": float(absolute_lateral.max()),
    }
    for threshold in LATERAL_THRESHOLDS:
        mask = absolute_lateral[1:] > threshold
        lateral_metrics[f"seconds_above_{threshold:g}_units"] = float(
            time_deltas_seconds[mask].sum()
        )

    upside_down = upright < UPSIDE_DOWN_COSINE_THRESHOLD
    upside_periods = _condition_periods(times_ms, upside_down)
    upside_metrics = {
        "minimum_upright_cosine": float(upright.min()),
        "upside_down_detected": bool(upside_periods),
        **_period_summary(upside_periods),
    }

    segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    cumulative_distance = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    stuck_candidates = np.zeros(len(records), dtype=bool)
    stuck_window_ms = STUCK_WINDOW_SECONDS * 1000.0
    for index in range(1, len(records)):
        target_time = times_ms[index] - stuck_window_ms
        start = int(np.searchsorted(times_ms, target_time, side="left"))
        if times_ms[index] - times_ms[start] < stuck_window_ms:
            continue
        progress_gain = progress[index] - progress[start]
        world_distance = cumulative_distance[index] - cumulative_distance[start]
        stuck_candidates[index] = bool(
            progress_gain < STUCK_PROGRESS_UNITS
            and world_distance < STUCK_WORLD_DISTANCE_UNITS
        )

    candidate_periods = _condition_periods(times_ms, stuck_candidates)
    stuck_periods: list[dict[str, float]] = []
    for period in candidate_periods:
        start_seconds = max(
            float(times_ms[0]) / 1000.0,
            float(period["start_seconds"]) - STUCK_WINDOW_SECONDS,
        )
        end_seconds = float(period["end_seconds"])
        stuck_periods.append(
            {
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "duration_seconds": end_seconds - start_seconds,
            }
        )
    stuck_metrics = {
        "stuck_detected": bool(stuck_periods),
        "candidate_sample_count": int(stuck_candidates.sum()),
        **_period_summary(stuck_periods),
    }

    return {
        "duration_seconds": float((times_ms[-1] - times_ms[0]) / 1000.0),
        "lateral_deviation": lateral_metrics,
        "upside_down": upside_metrics,
        "stuck": stuck_metrics,
    }


def evaluation_metric_protocol() -> dict[str, Any]:
    return {
        "steering_hysteresis": STEERING_HYSTERESIS,
        "steering_delta_deadband": STEERING_DELTA_DEADBAND,
        "oscillation_window_seconds": OSCILLATION_WINDOW_SECONDS,
        "oscillation_crossings_threshold": OSCILLATION_CROSSINGS_THRESHOLD,
        "lateral_threshold_units": list(LATERAL_THRESHOLDS),
        "upside_down_upright_cosine_below": UPSIDE_DOWN_COSINE_THRESHOLD,
        "stuck_window_seconds": STUCK_WINDOW_SECONDS,
        "stuck_progress_gain_below_units": STUCK_PROGRESS_UNITS,
        "stuck_world_distance_below_units": STUCK_WORLD_DISTANCE_UNITS,
    }
