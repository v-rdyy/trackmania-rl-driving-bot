"""Locate and size V5 steering reversals relative to the V4 baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from trackmania_rl.evaluation_metrics import STEERING_DELTA_DEADBAND
from trackmania_rl.observations import ReferencePath


DEFAULT_V4_ACTIONS = WORKSPACE_ROOT / "runs" / "reward_v4" / "evaluation_actions.jsonl"
DEFAULT_V5_ACTIONS = WORKSPACE_ROOT / "runs" / "reward_v5" / "evaluation_actions.jsonl"
DEFAULT_V4_SUMMARY = WORKSPACE_ROOT / "runs" / "reward_v4" / "evaluation_summary.json"
DEFAULT_V5_SUMMARY = WORKSPACE_ROOT / "runs" / "reward_v5" / "evaluation_summary.json"
DEFAULT_REFERENCE = WORKSPACE_ROOT / "data" / "tracks" / "a01_reference_path.csv"
DEFAULT_OUTPUT = WORKSPACE_ROOT / "runs" / "reward_v5" / "reversal_diagnostic.json"
PROGRESS_BIN_COUNT = 10
SMALL_EXCURSION_MAXIMUM = 0.25
LARGE_EXCURSION_MINIMUM = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-actions", type=Path, default=DEFAULT_V4_ACTIONS)
    parser.add_argument("--v5-actions", type=Path, default=DEFAULT_V5_ACTIONS)
    parser.add_argument("--v4-summary", type=Path, default=DEFAULT_V4_SUMMARY)
    parser.add_argument("--v5-summary", type=Path, default=DEFAULT_V5_SUMMARY)
    parser.add_argument("--reference-path", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def active_race_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Mirror the evaluator's disclosed countdown-restart filtering."""
    resets = [
        index
        for index in range(1, len(records))
        if int(records[index]["race_time_ms"])
        < int(records[index - 1]["race_time_ms"])
    ]
    if not resets:
        return records, 0
    start = resets[-1]
    while start < len(records) and int(records[start]["race_time_ms"]) < 0:
        start += 1
    if start >= len(records):
        raise ValueError("race clock restarted without reaching active race time")
    return records[start:], start


def progress_bin_index(progress: float, track_length: float) -> int:
    if not math.isfinite(progress) or not math.isfinite(track_length) or track_length <= 0:
        raise ValueError("progress and track length must be finite with positive length")
    fraction = min(1.0, max(0.0, progress / track_length))
    return min(PROGRESS_BIN_COUNT - 1, int(fraction * PROGRESS_BIN_COUNT))


def reversal_events(
    records: list[dict[str, Any]],
    *,
    track_length: float,
) -> list[dict[str, Any]]:
    """Return local steering-slope reversals using Decision 0008's deadband."""
    if len(records) < 2:
        return []
    steering = [float(record["raw_action"][0]) for record in records]
    if not np.isfinite(steering).all() or any(abs(value) > 1.0 + 1e-9 for value in steering):
        raise ValueError("steering actions must be finite and normalized")

    events: list[dict[str, Any]] = []
    last_delta_side = 0
    run_anchor = steering[0]
    for index in range(1, len(records)):
        previous = steering[index - 1]
        current = steering[index]
        delta = current - previous
        delta_side = (
            1
            if delta >= STEERING_DELTA_DEADBAND
            else -1
            if delta <= -STEERING_DELTA_DEADBAND
            else 0
        )
        if delta_side and last_delta_side and delta_side != last_delta_side:
            progress = float(records[index]["progress"])
            events.append(
                {
                    "episode": int(records[index]["episode"]),
                    "step": int(records[index]["step"]),
                    "race_time_ms": int(records[index]["race_time_ms"]),
                    "progress": progress,
                    "progress_fraction": min(1.0, max(0.0, progress / track_length)),
                    "progress_bin": progress_bin_index(progress, track_length),
                    "flip_step_delta": abs(delta),
                    "incoming_excursion": abs(previous - run_anchor),
                    "previous_steer": previous,
                    "current_steer": current,
                }
            )
            run_anchor = previous
        if delta_side:
            last_delta_side = delta_side
    return events


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def summarize_events(events: list[dict[str, Any]], episode_count: int) -> dict[str, Any]:
    if episode_count <= 0:
        raise ValueError("episode count must be positive")
    flip_deltas = [float(event["flip_step_delta"]) for event in events]
    excursions = [float(event["incoming_excursion"]) for event in events]
    bins: list[dict[str, Any]] = []
    for index in range(PROGRESS_BIN_COUNT):
        selected = [event for event in events if int(event["progress_bin"]) == index]
        selected_excursions = [float(event["incoming_excursion"]) for event in selected]
        bins.append(
            {
                "index": index,
                "start_percent": index * 10,
                "end_percent": (index + 1) * 10,
                "count": len(selected),
                "per_episode": len(selected) / episode_count,
                "median_incoming_excursion": percentile(selected_excursions, 50),
            }
        )
    small_count = sum(value <= SMALL_EXCURSION_MAXIMUM for value in excursions)
    large_count = sum(value > LARGE_EXCURSION_MINIMUM for value in excursions)
    return {
        "event_count": len(events),
        "events_per_episode": len(events) / episode_count,
        "median_flip_step_delta": percentile(flip_deltas, 50),
        "p90_flip_step_delta": percentile(flip_deltas, 90),
        "median_incoming_excursion": percentile(excursions, 50),
        "p90_incoming_excursion": percentile(excursions, 90),
        "small_excursion_count": small_count,
        "small_excursion_fraction": small_count / len(events) if events else 0.0,
        "large_excursion_count": large_count,
        "large_excursion_fraction": large_count / len(events) if events else 0.0,
        "progress_bins": bins,
    }


def expected_reversal_count(summary: dict[str, Any]) -> int:
    return sum(
        int(episode["trajectory"]["precision"]["steering"]["significant_direction_reversals"])
        for episode in summary["episodes_detail"]
    )


def analyze_version(
    action_path: Path,
    summary_path: Path,
    *,
    track_length: float,
) -> dict[str, Any]:
    raw_records = read_jsonl(action_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in raw_records:
        grouped.setdefault(int(record["episode"]), []).append(record)

    events: list[dict[str, Any]] = []
    active_record_count = 0
    discarded_record_count = 0
    episode_event_counts: list[dict[str, int]] = []
    for episode, records in sorted(grouped.items()):
        ordered = sorted(records, key=lambda record: int(record["step"]))
        active, discarded = active_race_records(ordered)
        episode_events = reversal_events(active, track_length=track_length)
        events.extend(episode_events)
        active_record_count += len(active)
        discarded_record_count += discarded
        episode_event_counts.append({"episode": episode, "event_count": len(episode_events)})

    expected = expected_reversal_count(summary)
    if len(events) != expected:
        raise ValueError(
            f"detected {len(events)} reversals but evaluator summary records {expected}"
        )
    if discarded_record_count != int(summary["discarded_startup_action_records"]):
        raise ValueError("countdown filtering does not match evaluator summary")

    episode_count = len(grouped)
    return {
        "action_log": str(action_path.resolve().relative_to(WORKSPACE_ROOT)),
        "action_log_sha256": sha256(action_path),
        "evaluation_summary": str(summary_path.resolve().relative_to(WORKSPACE_ROOT)),
        "evaluation_summary_sha256": sha256(summary_path),
        "episode_count": episode_count,
        "raw_record_count": len(raw_records),
        "active_record_count": active_record_count,
        "discarded_startup_record_count": discarded_record_count,
        "episode_event_counts": episode_event_counts,
        **summarize_events(events, episode_count),
    }


def relative_change(current: float, baseline: float) -> float:
    if baseline == 0.0:
        raise ValueError("relative change requires a nonzero baseline")
    return (current - baseline) / baseline


def main() -> int:
    args = parse_args()
    reference = ReferencePath.from_csv(args.reference_path.resolve())
    v4 = analyze_version(
        args.v4_actions.resolve(),
        args.v4_summary.resolve(),
        track_length=reference.total_length,
    )
    v5 = analyze_version(
        args.v5_actions.resolve(),
        args.v5_summary.resolve(),
        track_length=reference.total_length,
    )
    bin_changes = []
    for baseline, current in zip(v4["progress_bins"], v5["progress_bins"], strict=True):
        bin_changes.append(
            {
                "start_percent": baseline["start_percent"],
                "end_percent": baseline["end_percent"],
                "v4_count": baseline["count"],
                "v5_count": current["count"],
                "count_change": current["count"] - baseline["count"],
                "per_episode_change": current["per_episode"] - baseline["per_episode"],
            }
        )

    result = {
        "status": "complete",
        "protocol": {
            "event": (
                "change in the sign of successive steering deltas after ignoring "
                f"per-step deltas below {STEERING_DELTA_DEADBAND}"
            ),
            "steering_delta_deadband": STEERING_DELTA_DEADBAND,
            "progress_bin_count": PROGRESS_BIN_COUNT,
            "track_length": reference.total_length,
            "small_excursion_maximum": SMALL_EXCURSION_MAXIMUM,
            "large_excursion_minimum": LARGE_EXCURSION_MINIMUM,
            "excursion_note": (
                "descriptive post-evaluation bands over normalized steering travel "
                "between local extrema; not reward thresholds"
            ),
        },
        "v4": v4,
        "v5": v5,
        "comparison": {
            "event_count_change": v5["event_count"] - v4["event_count"],
            "events_per_episode_change": (
                v5["events_per_episode"] - v4["events_per_episode"]
            ),
            "median_flip_step_delta_relative_change": relative_change(
                v5["median_flip_step_delta"], v4["median_flip_step_delta"]
            ),
            "p90_flip_step_delta_relative_change": relative_change(
                v5["p90_flip_step_delta"], v4["p90_flip_step_delta"]
            ),
            "median_incoming_excursion_relative_change": relative_change(
                v5["median_incoming_excursion"], v4["median_incoming_excursion"]
            ),
            "p90_incoming_excursion_relative_change": relative_change(
                v5["p90_incoming_excursion"], v4["p90_incoming_excursion"]
            ),
            "progress_bin_changes": bin_changes,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"V4={v4['event_count']} V5={v5['event_count']} "
        f"change={result['comparison']['event_count_change']:+d}",
        flush=True,
    )
    print(f"wrote {args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
