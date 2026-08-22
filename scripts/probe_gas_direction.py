"""Resolve TMNF's signed analog Gas direction from an identical A01 snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from trackmania_rl.env import EnvironmentConfig, LiveTmiSession
from trackmania_rl.observations import ReferencePath, build_observation
from trackmania_rl.tmi_bridge import ProtocolError
from trackmania_rl.video_capture import restart_trackmania_race

REFERENCE_PATH = WORKSPACE_ROOT / "data" / "tracks" / "a01_reference_path.csv"
DEFAULT_SAMPLES = (
    WORKSPACE_ROOT / "artifacts" / "telemetry" / "gas_direction_probe.jsonl"
)
DEFAULT_SUMMARY = WORKSPACE_ROOT / "artifacts" / "smoke" / "gas_direction_probe.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare positive and negative analog Gas from one A01 snapshot."
    )
    parser.add_argument("--port", type=int, default=8478)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--simulation-speed", type=float, default=6.0)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def workspace_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return WORKSPACE_ROOT / path


def classify_direction(branch: dict[str, Any]) -> str:
    displacement = float(branch["signed_start_tangent_displacement"])
    progress_delta = float(branch["maximum_progress_delta"])
    maximum_forward_velocity = float(branch["maximum_car_forward_velocity"])
    minimum_forward_velocity = float(branch["minimum_car_forward_velocity"])
    if displacement >= 5.0 and progress_delta >= 5.0 and maximum_forward_velocity > 1.0:
        return "forward"
    if displacement <= -5.0 and minimum_forward_velocity < -1.0:
        return "backward"
    return "ambiguous"


def resolve_mapping(branches: list[dict[str, Any]]) -> str:
    directions: dict[str, set[str]] = {"throttle": set(), "brake": set()}
    counts = {"throttle": 0, "brake": 0}
    for branch in branches:
        requested_pedal = str(branch["requested_pedal"])
        if requested_pedal not in directions:
            raise ValueError(f"unknown requested pedal: {branch}")
        directions[requested_pedal].add(str(branch["direction"]))
        counts[requested_pedal] += 1
    if counts != {"throttle": 2, "brake": 2}:
        raise ValueError(f"expected two trials per pedal, got {counts}")
    if directions["throttle"] == {"forward"} and directions["brake"] == {"backward"}:
        return "current_throttle_brake_labels_correct"
    if directions["throttle"] == {"backward"} and directions["brake"] == {"forward"}:
        return "current_throttle_brake_labels_reversed"
    return "ambiguous"


def resolve_protocol_polarity(branches: list[dict[str, Any]]) -> str:
    directions: dict[int, set[str]] = {1: set(), -1: set()}
    for branch in branches:
        sign = int(np.sign(int(branch["applied_gas"])))
        if sign not in directions:
            raise ValueError(f"branch has zero applied gas: {branch}")
        directions[sign].add(str(branch["direction"]))
    if directions[1] == {"backward"} and directions[-1] == {"forward"}:
        return "negative_forward_positive_backward"
    if directions[1] == {"forward"} and directions[-1] == {"backward"}:
        return "positive_forward_negative_backward"
    return "ambiguous"


def sample_state(
    state: object,
    reference_path: ReferencePath,
    *,
    trial: int,
    branch_name: str,
    requested_pedal: str,
    step: int,
    applied_gas: int,
) -> dict[str, Any]:
    position = np.asarray(state.position, dtype=np.float64)
    velocity = np.asarray(state.velocity, dtype=np.float64)
    rotation = np.asarray(state.rotation_matrix, dtype=np.float64)
    values = np.concatenate((position, velocity, rotation.ravel()))
    if not np.isfinite(values).all():
        raise ProtocolError("gas-direction probe received nonfinite state data")
    projection = reference_path.project(position)
    local_velocity = rotation.T @ velocity
    return {
        "trial": trial,
        "branch": branch_name,
        "requested_pedal": requested_pedal,
        "step": step,
        "race_time_ms": int(state.race_time),
        "applied_gas": applied_gas,
        "display_speed": int(state.display_speed),
        "position": position.tolist(),
        "velocity": velocity.tolist(),
        "car_forward_velocity": float(local_velocity[2]),
        "progress": projection.progress,
        "lateral_offset": projection.lateral_offset,
        "vertical_offset": projection.vertical_offset,
        "path_tangent_xz": projection.tangent_xz.tolist(),
    }


def summarize_branch(records: list[dict[str, Any]]) -> dict[str, Any]:
    start = records[0]
    end = records[-1]
    start_position = np.asarray(start["position"], dtype=np.float64)
    end_position = np.asarray(end["position"], dtype=np.float64)
    start_tangent = np.asarray(start["path_tangent_xz"], dtype=np.float64)
    displacement_xz = end_position[[0, 2]] - start_position[[0, 2]]
    progresses = np.asarray([record["progress"] for record in records])
    forward_velocities = np.asarray(
        [record["car_forward_velocity"] for record in records]
    )
    summary = {
        "trial": int(start["trial"]),
        "branch": str(start["branch"]),
        "requested_pedal": str(start["requested_pedal"]),
        "applied_gas": int(records[1]["applied_gas"]),
        "steps": len(records) - 1,
        "start_position": start["position"],
        "end_position": end["position"],
        "world_displacement": float(np.linalg.norm(end_position - start_position)),
        "signed_start_tangent_displacement": float(
            np.dot(displacement_xz, start_tangent)
        ),
        "start_progress": float(progresses[0]),
        "end_progress": float(progresses[-1]),
        "maximum_progress_delta": float(progresses.max() - progresses[0]),
        "minimum_progress_delta": float(progresses.min() - progresses[0]),
        "maximum_car_forward_velocity": float(forward_velocities.max()),
        "minimum_car_forward_velocity": float(forward_velocities.min()),
        "maximum_display_speed": max(int(record["display_speed"]) for record in records),
        "minimum_vertical_offset": min(
            float(record["vertical_offset"]) for record in records
        ),
        "final_vertical_offset": float(end["vertical_offset"]),
    }
    summary["direction"] = classify_direction(summary)
    return summary


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    samples_path = workspace_path(args.samples)
    summary_path = workspace_path(args.summary)
    if args.steps < 10:
        raise SystemExit("--steps must be at least 10")
    if not 0.0 < args.simulation_speed <= 1000.0:
        raise SystemExit("--simulation-speed must be in (0, 1000]")

    restart_trackmania_race()
    reference_path = ReferencePath.from_csv(REFERENCE_PATH)
    session = LiveTmiSession(
        EnvironmentConfig(
            port=args.port,
            simulation_speed=args.simulation_speed,
            step_period_ms=100,
        )
    )
    branch_specs = [
        ("throttle_first", "throttle", np.asarray([0.0, 1.0, 0.0], dtype=np.float32)),
        ("brake_first", "brake", np.asarray([0.0, 0.0, 1.0], dtype=np.float32)),
        ("brake_repeat", "brake", np.asarray([0.0, 0.0, 1.0], dtype=np.float32)),
        ("throttle_repeat", "throttle", np.asarray([0.0, 1.0, 0.0], dtype=np.float32)),
    ]
    all_records: list[dict[str, Any]] = []
    branch_summaries: list[dict[str, Any]] = []
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        state = session.prepare()
        _, diagnostics = build_observation(state, reference_path)
        state = session.reset(diagnostics)
        for trial, (branch_name, requested_pedal, action) in enumerate(branch_specs):
            if trial:
                state = session.reset()
            records = [
                sample_state(
                    state,
                    reference_path,
                    trial=trial,
                    branch_name=branch_name,
                    requested_pedal=requested_pedal,
                    step=0,
                    applied_gas=0,
                )
            ]
            for step in range(1, args.steps + 1):
                result = session.advance(action)
                state = result.state
                records.append(
                    sample_state(
                        state,
                        reference_path,
                        trial=trial,
                        branch_name=branch_name,
                        requested_pedal=requested_pedal,
                        step=step,
                        applied_gas=result.applied_gas,
                    )
                )
            all_records.extend(records)
            branch_summary = summarize_branch(records)
            branch_summaries.append(branch_summary)
            print(
                f"trial={trial + 1}/4 branch={branch_name} "
                f"gas={branch_summary['applied_gas']} "
                f"direction={branch_summary['direction']} "
                f"signed_displacement="
                f"{branch_summary['signed_start_tangent_displacement']:.3f} "
                f"progress_delta={branch_summary['maximum_progress_delta']:.3f}",
                flush=True,
            )
    finally:
        session.close()

    with samples_path.open("w", encoding="utf-8", newline="\n") as output:
        for record in all_records:
            output.write(json.dumps(record, separators=(",", ":"), allow_nan=False))
            output.write("\n")

    resolution = resolve_mapping(branch_summaries)
    summary = {
        "status": "resolved" if resolution != "ambiguous" else "ambiguous",
        "resolution": resolution,
        "protocol_gas_polarity": resolve_protocol_polarity(branch_summaries),
        "simulation_speed": args.simulation_speed,
        "step_period_ms": 100,
        "steps_per_branch": args.steps,
        "identical_snapshot_trials": True,
        "branch_summaries": branch_summaries,
        "samples": str(samples_path.relative_to(WORKSPACE_ROOT)),
        "samples_sha256": sha256(samples_path),
    }
    write_json(summary_path, summary)
    print(f"gas-direction resolution={resolution}", flush=True)
    print(f"summary SHA-256={sha256(summary_path)}", flush=True)
    if resolution == "ambiguous":
        raise ProtocolError("signed Gas direction remained ambiguous")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
