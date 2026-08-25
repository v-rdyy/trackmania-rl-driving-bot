"""Evaluate V6 through the fixed V4/V5-comparable A01 protocol."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))

import evaluate_reward_v3 as evaluator
from diagnose_reward_v5_reversals import (
    active_race_records,
    reversal_events,
)
from trackmania_rl.rewards import clustered_reversal_frequency_reward

MAX_OSCILLATION_EPISODES = 10
MIN_FINISHES = 19
MAX_MEAN_LAP_MS = 25_500.0
MAX_MEAN_REVERSALS = 28.0
MAX_MEAN_HOTSPOT_REVERSALS = 6.4
HOTSPOT_BINS = {1, 2, 9}


def configure_evaluator() -> None:
    run_dir = WORKSPACE_ROOT / "runs" / "reward_v6"
    evaluator.EXPERIMENT_LABEL = "reward-v6"
    evaluator.EXPERIMENT_SLUG = "reward_v6"
    evaluator.PROTOCOL_LABEL = "reward_v6.md"
    evaluator.REWARD_FUNCTION = clustered_reversal_frequency_reward
    evaluator.EXPECTED_EPISODES = 20
    evaluator.DEFAULT_RUN_TAG = None
    evaluator.RUN_DIR = run_dir
    evaluator.DEFAULT_CHECKPOINT = (
        WORKSPACE_ROOT / "checkpoints" / "reward_v6" / "final_model.zip"
    )
    evaluator.DEFAULT_ACTION_LOG = run_dir / "evaluation_actions.jsonl"
    evaluator.DEFAULT_SUMMARY = run_dir / "evaluation_summary.json"
    evaluator.DEFAULT_REPLAY_DIR = (
        WORKSPACE_ROOT / "artifacts" / "replays" / "reward_v6_evaluation"
    )


def location_metrics_from_events(
    events: list[dict[str, Any]],
    *,
    episode_count: int,
) -> dict[str, Any]:
    if episode_count <= 0:
        raise ValueError("episode count must be positive")
    bin_counts = [
        sum(int(event["progress_bin"]) == index for event in events)
        for index in range(10)
    ]
    hotspot_count = sum(
        int(event["progress_bin"]) in HOTSPOT_BINS for event in events
    )
    return {
        "event_count": len(events),
        "mean_events_per_episode": len(events) / episode_count,
        "progress_bins": [
            {
                "start_percent": index * 10,
                "end_percent": (index + 1) * 10,
                "count": count,
                "per_episode": count / episode_count,
            }
            for index, count in enumerate(bin_counts)
        ],
        "hotspots": {
            "progress_bins": ["10-20%", "20-30%", "90-100%"],
            "event_count": hotspot_count,
            "mean_events_per_episode": hotspot_count / episode_count,
            "outside_hotspot_event_count": len(events) - hotspot_count,
            "outside_hotspot_mean_per_episode": (
                len(events) - hotspot_count
            ) / episode_count,
            "v4_baseline_mean_per_episode": 8.0,
            "v5_baseline_mean_per_episode": 12.1,
        },
    }


def normalized_frequency_cost_metrics(
    events: list[dict[str, Any]],
    *,
    episode_count: int,
) -> dict[str, Any]:
    grouped: dict[int, list[int]] = {}
    for event in events:
        grouped.setdefault(int(event["episode"]), []).append(int(event["step"]))
    costs: list[float] = []
    for event_steps in grouped.values():
        history: list[int] = []
        for step in sorted(event_steps):
            history.append(step)
            history = [candidate for candidate in history if candidate >= step - 20]
            excess = max(0, len(history) - 3)
            if excess:
                costs.append(0.05 * excess)
    return {
        "penalized_event_count": len(costs),
        "total_frequency_cost": sum(costs),
        "mean_frequency_cost_per_episode": sum(costs) / episode_count,
        "maximum_single_event_cost": max(costs, default=0.0),
    }


def boundary_instrumentation_mismatches(
    active: list[dict[str, Any]],
    detected_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    detected_steps = {int(event["step"]) for event in detected_events}
    logged_by_step = {
        int(record["step"]): record
        for record in active
        if bool(record.get("steering_slope_reversal", False))
    }
    missing = sorted(detected_steps - set(logged_by_step))
    if missing:
        raise ValueError(f"live V6 instrumentation missed reversal steps: {missing}")
    first_active_step = int(active[0]["step"])
    mismatches: list[dict[str, Any]] = []
    for step in sorted(set(logged_by_step) - detected_steps):
        record = logged_by_step[step]
        window_count = int(record["steering_reversals_in_window"])
        if step - first_active_step > 20 or window_count > 3:
            raise ValueError(
                "unexpected V6 reversal instrumentation mismatch outside the "
                f"unpenalized startup boundary: step={step} count={window_count}"
            )
        mismatches.append(
            {
                "step": step,
                "race_time_ms": int(record["race_time_ms"]),
                "steering_reversals_in_window": window_count,
                "frequency_cost": 0.0,
            }
        )
    return mismatches


def reversal_location_metrics(
    action_log: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in action_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(int(record["episode"]), []).append(record)

    events: list[dict[str, Any]] = []
    logged_reversal_count = 0
    actual_logged_frequency_costs: list[float] = []
    boundary_mismatches: list[dict[str, Any]] = []
    track_length = float(summary["reference_path_total_length"])
    for episode, episode_records in sorted(grouped.items()):
        ordered = sorted(episode_records, key=lambda record: int(record["step"]))
        active, _ = active_race_records(ordered)
        episode_events = reversal_events(active, track_length=track_length)
        events.extend(episode_events)
        for mismatch in boundary_instrumentation_mismatches(active, episode_events):
            boundary_mismatches.append({"episode": episode, **mismatch})
        for record in active:
            if bool(record.get("steering_slope_reversal", False)):
                logged_reversal_count += 1
                window_count = int(record["steering_reversals_in_window"])
                excess = max(0, window_count - 3)
                if excess:
                    actual_logged_frequency_costs.append(0.05 * excess)

    expected = sum(
        int(episode["trajectory"]["precision"]["steering"]["significant_direction_reversals"])
        for episode in summary["episodes_detail"]
    )
    if len(events) != expected:
        raise ValueError(
            "V6 offline reversal reconstruction does not reproduce evaluator "
            f"metrics: detected={len(events)} expected={expected}"
        )

    metrics = location_metrics_from_events(events, episode_count=len(grouped))
    normalized_costs = normalized_frequency_cost_metrics(
        events,
        episode_count=len(grouped),
    )
    metrics.update(
        {
            "logged_reversal_count": logged_reversal_count,
            "normalized_reversal_count": len(events),
            "instrumentation_boundary_mismatches": boundary_mismatches,
            "normalization_note": (
                "The offline Decision 0008 reconstruction is authoritative after "
                "discarding countdown prefixes. Any listed live-only event was "
                "limited to the first unpenalized 2-second boundary window."
            ),
            "actual_logged_frequency_cost": sum(actual_logged_frequency_costs),
            **normalized_costs,
            "v4_baseline_mean_reversals": 35.0,
            "v5_baseline_mean_reversals": 37.1,
        }
    )
    return metrics


def success_gate(summary: dict[str, Any]) -> dict[str, Any]:
    oscillation_episodes = int(
        summary["precision_summary"]["oscillation_detected_episodes"]
    )
    finishes = int(summary["finishes"])
    mean_lap_ms = summary["average_finish_time_ms"]
    mean_reversals = float(
        summary["precision_summary"]["mean_significant_direction_reversals"]
    )
    mean_hotspot_reversals = float(
        summary["v6_reversal_frequency"]["hotspots"]["mean_events_per_episode"]
    )
    checks = {
        "oscillation_at_most_10_of_20": oscillation_episodes <= MAX_OSCILLATION_EPISODES,
        "finishes_at_least_19_of_20": finishes >= MIN_FINISHES,
        "mean_lap_at_most_25_500_ms": (
            mean_lap_ms is not None and float(mean_lap_ms) <= MAX_MEAN_LAP_MS
        ),
        "mean_reversals_at_most_28": mean_reversals <= MAX_MEAN_REVERSALS,
        "mean_hotspot_reversals_at_most_6_4": (
            mean_hotspot_reversals <= MAX_MEAN_HOTSPOT_REVERSALS
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "oscillation_detected_episodes": oscillation_episodes,
            "finishes": finishes,
            "average_finish_time_ms": mean_lap_ms,
            "mean_significant_direction_reversals": mean_reversals,
            "mean_hotspot_reversals": mean_hotspot_reversals,
        },
    }


def main() -> int:
    configure_evaluator()
    result = evaluator.main()
    summary = json.loads(evaluator.DEFAULT_SUMMARY.read_text(encoding="utf-8"))
    summary["v6_reversal_frequency"] = reversal_location_metrics(
        evaluator.DEFAULT_ACTION_LOG,
        summary,
    )
    summary["v6_success_gate"] = success_gate(summary)
    evaluator.DEFAULT_SUMMARY.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        "V6 frozen gate: "
        f"passed={summary['v6_success_gate']['passed']} "
        f"checks={summary['v6_success_gate']['checks']}",
        flush=True,
    )
    print(
        f"final summary SHA-256={evaluator.sha256(evaluator.DEFAULT_SUMMARY)}",
        flush=True,
    )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
