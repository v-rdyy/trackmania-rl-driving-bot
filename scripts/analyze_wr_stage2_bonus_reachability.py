"""Audit why the frozen Stage 2 localized drift bonus never became eligible."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from trackmania_rl.rewards import (
    WR_STAGE2_ASSISTED_ZONES,
    WR_STAGE2_MIN_ABS_BODY_UP_YAW_RATE,
    WR_STAGE2_MIN_ABS_SLIP_ANGLE_DEGREES,
    WR_STAGE2_MIN_DISPLAY_SPEED,
    WR_STAGE2_MIN_GROUND_CONTACTS,
    WR_STAGE2_MIN_SLIDING_WHEELS,
)

DEFAULT_OUTPUT = (
    WORKSPACE_ROOT
    / "runs"
    / "wr_chase_stage2_bonus_reachability"
    / "analysis.json"
)
DATASETS = {
    "target_zero_deterministic": (
        WORKSPACE_ROOT
        / "runs"
        / "wr_chase_stage2"
        / "gates"
        / "gate_00000000_evaluation_actions.jsonl",
        WORKSPACE_ROOT
        / "runs"
        / "wr_chase_stage2"
        / "gates"
        / "gate_00000000_evaluation.json",
    ),
    "gate_500k_deterministic": (
        WORKSPACE_ROOT
        / "runs"
        / "wr_chase_stage2"
        / "gates"
        / "gate_00500000_evaluation_actions.jsonl",
        WORKSPACE_ROOT
        / "runs"
        / "wr_chase_stage2"
        / "gates"
        / "gate_00500000_evaluation.json",
    ),
    "gate_750k_deterministic": (
        WORKSPACE_ROOT
        / "runs"
        / "wr_chase_stage2_intermediate"
        / "gate_00750000_deterministic_actions.jsonl",
        WORKSPACE_ROOT
        / "runs"
        / "wr_chase_stage2_intermediate"
        / "gate_00750000_deterministic_summary.json",
    ),
    "gate_1m_deterministic": (
        WORKSPACE_ROOT
        / "runs"
        / "wr_chase_stage2"
        / "gates"
        / "gate_01000000_evaluation_actions.jsonl",
        WORKSPACE_ROOT
        / "runs"
        / "wr_chase_stage2"
        / "gates"
        / "gate_01000000_evaluation.json",
    ),
    "gate_1m_stochastic_seed_20260828": (
        WORKSPACE_ROOT
        / "runs"
        / "wr_chase_stage2_policy_mode"
        / "gate_01000000_stochastic_actions.jsonl",
        WORKSPACE_ROOT
        / "runs"
        / "wr_chase_stage2_policy_mode"
        / "gate_01000000_stochastic_summary.json",
    ),
}
ZONE_NAMES = ("first_turn", "final_corner")
CONDITION_NAMES = ("ground", "speed", "sliding", "slip", "yaw")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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


def zone_name(progress: float) -> str | None:
    for name, (low, high) in zip(ZONE_NAMES, WR_STAGE2_ASSISTED_ZONES, strict=True):
        if low <= progress <= high:
            return name
    return None


def condition_flags(record: dict[str, Any]) -> dict[str, bool]:
    return {
        "ground": int(record["ground_contact_count"])
        >= WR_STAGE2_MIN_GROUND_CONTACTS,
        "speed": float(record["display_speed"]) >= WR_STAGE2_MIN_DISPLAY_SPEED,
        "sliding": int(record["sliding_wheel_count"])
        >= WR_STAGE2_MIN_SLIDING_WHEELS,
        "slip": abs(float(record["slip_angle_degrees"]))
        >= WR_STAGE2_MIN_ABS_SLIP_ANGLE_DEGREES,
        "yaw": abs(float(record["body_up_yaw_rate"]))
        >= WR_STAGE2_MIN_ABS_BODY_UP_YAW_RATE,
    }


def validate_record(record: dict[str, Any]) -> None:
    required = (
        "episode",
        "race_time_ms",
        "progress",
        "display_speed",
        "ground_contact_count",
        "sliding_wheel_count",
        "slip_angle_degrees",
        "body_up_yaw_rate",
        "new_high_water_progress_delta",
        "full_simstate_available",
        "reward_function",
    )
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(f"reachability record is missing {missing}")
    if record["full_simstate_available"] is not True:
        raise ValueError("reachability audit requires complete live SimState")
    if record["reward_function"] != "localized_drift_assistance_reward":
        raise ValueError("reachability record used the wrong reward function")
    values = (
        float(record["progress"]),
        float(record["display_speed"]),
        float(record["slip_angle_degrees"]),
        float(record["body_up_yaw_rate"]),
        float(record["new_high_water_progress_delta"]),
    )
    if not np.isfinite(values).all():
        raise ValueError("reachability record contains nonfinite telemetry")


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def closest_score(record: dict[str, Any]) -> float:
    ratios = (
        int(record["ground_contact_count"]) / WR_STAGE2_MIN_GROUND_CONTACTS,
        float(record["display_speed"]) / WR_STAGE2_MIN_DISPLAY_SPEED,
        int(record["sliding_wheel_count"]) / WR_STAGE2_MIN_SLIDING_WHEELS,
        abs(float(record["slip_angle_degrees"]))
        / WR_STAGE2_MIN_ABS_SLIP_ANGLE_DEGREES,
        abs(float(record["body_up_yaw_rate"]))
        / WR_STAGE2_MIN_ABS_BODY_UP_YAW_RATE,
    )
    return statistics.fmean(min(1.0, max(0.0, ratio)) for ratio in ratios)


def compact_sample(record: dict[str, Any]) -> dict[str, Any]:
    flags = condition_flags(record)
    return {
        "dataset": record.get("_dataset"),
        "episode": int(record["episode"]),
        "race_time_ms": int(record["race_time_ms"]),
        "progress": float(record["progress"]),
        "display_speed": float(record["display_speed"]),
        "ground_contact_count": int(record["ground_contact_count"]),
        "sliding_wheel_count": int(record["sliding_wheel_count"]),
        "abs_slip_angle_degrees": abs(float(record["slip_angle_degrees"])),
        "abs_body_up_yaw_rate": abs(float(record["body_up_yaw_rate"])),
        "new_high_water_progress_delta": float(
            record["new_high_water_progress_delta"]
        ),
        "conditions_met": [name for name in CONDITION_NAMES if flags[name]],
        "condition_count": sum(flags.values()),
        "closeness_score": closest_score(record),
    }


def count_joint(records: list[dict[str, Any]], names: Iterable[str]) -> int:
    selected = tuple(names)
    return sum(
        all(condition_flags(record)[name] for name in selected)
        for record in records
    )


def summarize_zone(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("reachability zone contains no samples")
    flags = [condition_flags(record) for record in records]
    values = {
        "ground": [float(record["ground_contact_count"]) for record in records],
        "speed": [float(record["display_speed"]) for record in records],
        "sliding": [float(record["sliding_wheel_count"]) for record in records],
        "slip": [abs(float(record["slip_angle_degrees"])) for record in records],
        "yaw": [abs(float(record["body_up_yaw_rate"])) for record in records],
    }
    thresholds = {
        "ground": float(WR_STAGE2_MIN_GROUND_CONTACTS),
        "speed": float(WR_STAGE2_MIN_DISPLAY_SPEED),
        "sliding": float(WR_STAGE2_MIN_SLIDING_WHEELS),
        "slip": float(WR_STAGE2_MIN_ABS_SLIP_ANGLE_DEGREES),
        "yaw": float(WR_STAGE2_MIN_ABS_BODY_UP_YAW_RATE),
    }
    condition_summaries = {}
    for name in CONDITION_NAMES:
        met = sum(row[name] for row in flags)
        condition_summaries[name] = {
            "threshold": thresholds[name],
            "samples_met": met,
            "fraction_met": met / len(records),
            "maximum": max(values[name]),
            "p95": percentile(values[name], 95),
            "median": percentile(values[name], 50),
            "maximum_margin": max(values[name]) - thresholds[name],
        }
    match_counts = [sum(row.values()) for row in flags]
    maximum_match = max(match_counts)
    ranked = sorted(
        records,
        key=lambda record: (
            sum(condition_flags(record).values()),
            closest_score(record),
        ),
        reverse=True,
    )
    speed_ground = [
        record
        for record in records
        if condition_flags(record)["speed"] and condition_flags(record)["ground"]
    ]
    return {
        "samples": len(records),
        "episodes_entered": len(
            {
                (str(record.get("_dataset")), int(record["episode"]))
                for record in records
            }
        ),
        "datasets_present": sorted(
            {str(record.get("_dataset")) for record in records}
        ),
        "progress_minimum": min(float(record["progress"]) for record in records),
        "progress_maximum": max(float(record["progress"]) for record in records),
        "conditions": condition_summaries,
        "positive_high_water_progress_samples": sum(
            float(record["new_high_water_progress_delta"]) > 0.0
            for record in records
        ),
        "joint_counts": {
            "ground_and_speed": count_joint(records, ("ground", "speed")),
            "ground_speed_yaw": count_joint(records, ("ground", "speed", "yaw")),
            "ground_speed_slip": count_joint(records, ("ground", "speed", "slip")),
            "ground_speed_slip_yaw": count_joint(
                records, ("ground", "speed", "slip", "yaw")
            ),
            "ground_speed_sliding": count_joint(
                records, ("ground", "speed", "sliding")
            ),
            "all_except_speed": count_joint(
                records, ("ground", "sliding", "slip", "yaw")
            ),
            "all_except_sliding": count_joint(
                records, ("ground", "speed", "slip", "yaw")
            ),
            "all_except_slip": count_joint(
                records, ("ground", "speed", "sliding", "yaw")
            ),
            "all_reward_conditions": count_joint(records, CONDITION_NAMES),
        },
        "maximum_simultaneously_satisfied": maximum_match,
        "match_count_histogram": {
            str(count): match_counts.count(count) for count in range(6)
        },
        "speed_ground_subset": {
            "samples": len(speed_ground),
            "maximum_sliding_wheels": max(
                (int(record["sliding_wheel_count"]) for record in speed_ground),
                default=0,
            ),
            "maximum_abs_slip_angle_degrees": max(
                (
                    abs(float(record["slip_angle_degrees"]))
                    for record in speed_ground
                ),
                default=0.0,
            ),
            "maximum_abs_body_up_yaw_rate": max(
                (
                    abs(float(record["body_up_yaw_rate"]))
                    for record in speed_ground
                ),
                default=0.0,
            ),
        },
        "closest_samples": [compact_sample(record) for record in ranked[:5]],
    }


def summarize_track_wide(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("track-wide reachability audit has no samples")
    all_dynamics = [
        record
        for record in records
        if all(condition_flags(record).values())
    ]
    outside = [
        record
        for record in all_dynamics
        if zone_name(float(record["progress"])) is None
    ]
    return {
        "samples": len(records),
        "sliding_samples": sum(
            condition_flags(record)["sliding"] for record in records
        ),
        "slip_samples": sum(condition_flags(record)["slip"] for record in records),
        "all_dynamics_samples": len(all_dynamics),
        "outside_assisted_zone_all_dynamics_samples": len(outside),
        "outside_assisted_zone_examples": [
            compact_sample(record) for record in outside[:10]
        ],
    }


def file_binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(WORKSPACE_ROOT)),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def main() -> int:
    args = parse_args()
    datasets: dict[str, Any] = {}
    aggregate: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ZONE_NAMES
    }
    aggregate_track_wide: list[dict[str, Any]] = []
    for dataset, (actions_path, summary_path) in DATASETS.items():
        actions_path = actions_path.resolve()
        summary_path = summary_path.resolve()
        if not actions_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"reachability evidence missing for {dataset}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        records = read_jsonl(actions_path)
        for record in records:
            validate_record(record)
            record["_dataset"] = dataset
        aggregate_track_wide.extend(records)
        zones: dict[str, Any] = {}
        for name in ZONE_NAMES:
            zone_records = [
                record
                for record in records
                if zone_name(float(record["progress"])) == name
            ]
            zones[name] = summarize_zone(zone_records)
            aggregate[name].extend(zone_records)
        datasets[dataset] = {
            "action_log": file_binding(actions_path),
            "evaluation_summary": file_binding(summary_path),
            "episodes": int(summary["episodes"]),
            "finishes": int(summary["finishes"]),
            "action_records": len(records),
            "track_wide": summarize_track_wide(records),
            "zones": zones,
        }
    result = {
        "status": "complete",
        "reward_contract": {
            "zones": {
                name: list(bounds)
                for name, bounds in zip(
                    ZONE_NAMES, WR_STAGE2_ASSISTED_ZONES, strict=True
                )
            },
            "minimum_ground_contacts": WR_STAGE2_MIN_GROUND_CONTACTS,
            "minimum_display_speed": WR_STAGE2_MIN_DISPLAY_SPEED,
            "minimum_sliding_wheels": WR_STAGE2_MIN_SLIDING_WHEELS,
            "minimum_abs_slip_angle_degrees": (
                WR_STAGE2_MIN_ABS_SLIP_ANGLE_DEGREES
            ),
            "minimum_abs_body_up_yaw_rate": (
                WR_STAGE2_MIN_ABS_BODY_UP_YAW_RATE
            ),
            "all_conditions_must_hold_on_the_same_100ms_sample": True,
        },
        "datasets": datasets,
        "aggregate": {
            **{
                name: summarize_zone(records) for name, records in aggregate.items()
            },
            "track_wide": summarize_track_wide(aggregate_track_wide),
        },
        "scope": (
            "All preserved direct-live Stage 2 evaluation paths available before "
            "Stage 2b; training interactions were not logged with full SimState."
        ),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    print(f"SHA-256={sha256(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
