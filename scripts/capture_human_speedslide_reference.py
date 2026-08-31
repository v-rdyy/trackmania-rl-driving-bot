"""Capture one owner-driven A01 speedslide attempt with live SimState dynamics.

This recorder never injects driving inputs during the active lap.  It samples
at 20 ms for diagnostic resolution and derives the exact 100 ms view used by
Stage 2's reward gate.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, TextIO

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))

from capture_wr_reference_speedslide import (
    ZONE_NAMES,
    annotate_high_water_progress,
    condition_flags,
    sha256,
    state_record,
    summarize_optional_zone,
    write_json,
    zone_name,
)
from trackmania_rl.env import EnvironmentConfig, LiveTmiSession
from trackmania_rl.game_launch import ensure_trackmania_running
from trackmania_rl.observations import ReferencePath
from trackmania_rl.rewards import (
    WR_STAGE2_ASSISTED_ZONES,
    WR_STAGE2_MIN_ABS_BODY_UP_YAW_RATE,
    WR_STAGE2_MIN_ABS_SLIP_ANGLE_DEGREES,
    WR_STAGE2_MIN_DISPLAY_SPEED,
    WR_STAGE2_MIN_GROUND_CONTACTS,
    WR_STAGE2_MIN_SLIDING_WHEELS,
)
from trackmania_rl.tmi_bridge import ProtocolError
from trackmania_rl.video_capture import focus_window, wait_for_foreground

RAW_SAMPLE_PERIOD_MS = 20
STAGE2_SAMPLE_PERIOD_MS = 100
CONFIRMED_MIN_SLIDING_WHEELS = 2
CONFIRMED_MIN_CONSECUTIVE_SAMPLES = 3
DEFAULT_REFERENCE_PATH = WORKSPACE_ROOT / "data" / "tracks" / "a01_reference_path.csv"
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT / "artifacts" / "analysis" / "human_speedslide_reference"
)
DEFAULT_REPLAY_ROOT = (
    WORKSPACE_ROOT / "artifacts" / "replays" / "human_speedslide_reference"
)
DEFAULT_TMI_SCRIPTS = Path.home() / "Documents" / "TMInterface" / "Scripts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--take-id", default="take_001")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--replay-root", type=Path, default=DEFAULT_REPLAY_ROOT)
    parser.add_argument("--reference-path", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--tmi-scripts-dir", type=Path, default=DEFAULT_TMI_SCRIPTS)
    parser.add_argument("--map", default="A01-Race.Challenge.Gbx")
    parser.add_argument("--port", type=int, default=8478)
    parser.add_argument("--max-race-ms", type=int, default=45_000)
    return parser.parse_args()


def validate_take_id(value: str) -> str:
    if not value or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in value):
        raise ValueError(
            "take id must contain only lowercase letters, digits, underscores, or hyphens"
        )
    return value


def human_input_fields(state: object) -> dict[str, bool | int | str]:
    try:
        values: dict[str, bool | int | str] = {
            "input_source": "human_live_simstate",
            "input_accelerate": bool(state.input_accelerate),
            "input_brake": bool(state.input_brake),
            "input_left": bool(state.input_left),
            "input_right": bool(state.input_right),
            "input_steer_raw": int(state.input_steer),
            "input_gas_raw": int(state.input_gas),
        }
    except AttributeError as error:
        raise ProtocolError("live SimState is missing raw human input fields") from error
    return values


def downsample_stage2(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sampled = [
        dict(record)
        for record in records
        if int(record["race_time_ms"]) % STAGE2_SAMPLE_PERIOD_MS == 0
    ]
    if not sampled or int(sampled[0]["race_time_ms"]) != 0:
        raise ProtocolError("human capture does not contain the Stage 2 time-zero sample")
    times = [int(record["race_time_ms"]) for record in sampled]
    if any(
        current - previous != STAGE2_SAMPLE_PERIOD_MS
        for previous, current in zip(times, times[1:])
    ):
        raise ProtocolError("human capture has gaps in the derived 100 ms view")
    # Stage 2 updates its progress high-water mark only once per action step.
    # Recompute this view instead of inheriting deltas from the 20 ms diagnostic
    # stream, where an intermediate peak could otherwise suppress later credit.
    annotate_high_water_progress(sampled)
    return sampled


def gate_windows(
    records: list[dict[str, Any]],
    *,
    minimum_sliding_wheels: int,
    minimum_samples: int,
    sample_period_ms: int,
) -> list[dict[str, Any]]:
    qualifying = [
        record
        for record in records
        if all(condition_flags(record).values())
        and int(record["sliding_wheel_count"]) >= minimum_sliding_wheels
    ]
    grouped: list[list[dict[str, Any]]] = []
    for record in qualifying:
        if (
            not grouped
            or int(record["race_time_ms"])
            != int(grouped[-1][-1]["race_time_ms"]) + sample_period_ms
        ):
            grouped.append([record])
        else:
            grouped[-1].append(record)
    return [
        {
            "start_race_time_ms": int(window[0]["race_time_ms"]),
            "end_race_time_ms": int(window[-1]["race_time_ms"]),
            "samples": len(window),
            "covered_step_time_ms": len(window) * sample_period_ms,
            "elapsed_between_samples_ms": (len(window) - 1) * sample_period_ms,
            "maximum_display_speed": max(int(row["display_speed"]) for row in window),
            "maximum_abs_slip_angle_degrees": max(
                abs(float(row["slip_angle_degrees"])) for row in window
            ),
            "maximum_abs_body_up_yaw_rate": max(
                abs(float(row["body_up_yaw_rate"])) for row in window
            ),
            "maximum_sliding_wheels": max(
                int(row["sliding_wheel_count"]) for row in window
            ),
        }
        for window in grouped
        if len(window) >= minimum_samples
    ]


def analyze_human_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ProtocolError("human speedslide capture is empty")
    if any(record.get("full_simstate_available") is not True for record in records):
        raise ProtocolError("human speedslide capture requires complete live SimState")
    if any(record.get("input_source") != "human_live_simstate" for record in records):
        raise ProtocolError("human speedslide capture has an invalid input source")
    raw_times = [int(record["race_time_ms"]) for record in records]
    raw_deltas = [
        current - previous
        for previous, current in zip(raw_times, raw_times[1:])
    ]
    terminal_partial_step = bool(
        raw_deltas
        and records[-1]["race_finished"]
        and 0 < raw_deltas[-1] <= RAW_SAMPLE_PERIOD_MS
    )
    regular_deltas = raw_deltas[:-1] if terminal_partial_step else raw_deltas
    if raw_times[0] != 0 or any(
        delta != RAW_SAMPLE_PERIOD_MS for delta in regular_deltas
    ):
        raise ProtocolError("human speedslide capture is not contiguous at 20 ms")
    stage2 = downsample_stage2(records)
    first_wall = float(records[0]["capture_wall_seconds"])
    last_wall = float(records[-1]["capture_wall_seconds"])
    wall_elapsed = last_wall - first_wall
    race_elapsed = int(records[-1]["race_time_ms"]) - int(
        records[0]["race_time_ms"]
    )
    zones: dict[str, Any] = {}
    for name in ZONE_NAMES:
        raw_zone = [
            record
            for record in records
            if zone_name(float(record["progress"])) == name
        ]
        stage2_zone = [
            record
            for record in stage2
            if zone_name(float(record["progress"])) == name
        ]
        raw_summary = summarize_optional_zone(raw_zone)
        raw_summary["all_condition_windows"] = gate_windows(
            raw_zone,
            minimum_sliding_wheels=1,
            minimum_samples=1,
            sample_period_ms=RAW_SAMPLE_PERIOD_MS,
        )
        stage2_summary = summarize_optional_zone(stage2_zone)
        confirmed = gate_windows(
            stage2_zone,
            minimum_sliding_wheels=CONFIRMED_MIN_SLIDING_WHEELS,
            minimum_samples=CONFIRMED_MIN_CONSECUTIVE_SAMPLES,
            sample_period_ms=STAGE2_SAMPLE_PERIOD_MS,
        )
        zones[name] = {
            "raw_20ms": raw_summary,
            "stage2_100ms": stage2_summary,
            "confirmed_stage2_windows": confirmed,
        }
    zones_entered = all(zones[name]["stage2_100ms"]["entered"] for name in ZONE_NAMES)
    full_dynamics_gate_both = all(
        int(
            zones[name]["stage2_100ms"]["joint_counts"][
                "all_dynamics_conditions"
            ]
        )
        > 0
        for name in ZONE_NAMES
    )
    positive_reward_eligibility_both = all(
        int(
            zones[name]["stage2_100ms"]["joint_counts"][
                "positive_bonus_eligible"
            ]
        )
        > 0
        for name in ZONE_NAMES
    )
    confirmed_both = all(zones[name]["confirmed_stage2_windows"] for name in ZONE_NAMES)
    return {
        "outcome": {
            "race_finished": bool(records[-1]["race_finished"]),
            "terminal_race_time_ms": int(records[-1]["race_time_ms"]),
            "maximum_progress": max(float(record["progress"]) for record in records),
        },
        "raw_sample_period_ms": RAW_SAMPLE_PERIOD_MS,
        "raw_samples": len(records),
        "stage2_sample_period_ms": STAGE2_SAMPLE_PERIOD_MS,
        "stage2_samples": len(stage2),
        "capture_timing": {
            "wall_elapsed_seconds": wall_elapsed,
            "race_elapsed_ms": race_elapsed,
            "race_clock_to_wall_clock_ratio": (
                race_elapsed / (wall_elapsed * 1000.0)
                if wall_elapsed > 0.0
                else None
            ),
        },
        "original_stage2_contract": {
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
            "requires_positive_100ms_high_water_progress_delta": True,
        },
        "separate_confirmed_slide_diagnostic": {
            "minimum_sliding_wheels": CONFIRMED_MIN_SLIDING_WHEELS,
            "minimum_consecutive_samples": CONFIRMED_MIN_CONSECUTIVE_SAMPLES,
            "sample_period_ms": STAGE2_SAMPLE_PERIOD_MS,
        },
        "both_zones_entered_at_100ms": zones_entered,
        "full_dynamics_gate_observed_in_both_zones_at_100ms": (
            full_dynamics_gate_both
        ),
        "positive_reward_eligibility_observed_in_both_zones_at_100ms": (
            positive_reward_eligibility_both
        ),
        "confirmed_three_sample_slide_in_both_zones_at_100ms": confirmed_both,
        "zones": zones,
    }


def wait_for_file(path: Path, timeout_seconds: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if path.is_file() and path.stat().st_size > 0:
            return
        time.sleep(0.05)
    raise ProtocolError(f"TMInterface did not recover human inputs: {path}")


def capture_human_lap(
    args: argparse.Namespace,
    replay_filename: str,
    telemetry_path: Path,
) -> tuple[list[dict[str, Any]], Path]:
    target, _ = ensure_trackmania_running(port=args.port, confirm_existing=False)
    session = LiveTmiSession(
        EnvironmentConfig(
            port=args.port,
            simulation_speed=1.0,
            step_period_ms=RAW_SAMPLE_PERIOD_MS,
            max_episode_ms=args.max_race_ms,
            map_to_load=args.map,
            auto_respawn_on_connect=False,
            wait_for_race_start_on_connect=False,
        )
    )
    reference = ReferencePath.from_csv(args.reference_path)
    external_replay = args.tmi_scripts_dir / replay_filename
    if external_replay.exists():
        raise ProtocolError(f"human input replay already exists: {external_replay}")
    records: list[dict[str, Any]] = []
    telemetry_output: TextIO | None = None
    high_water_progress: float | None = None
    previous_progress: float | None = None
    capture_started_wall: float | None = None
    try:
        session.prepare()
        session.client.execute_command("unload")
        focus_window(target)
        wait_for_foreground(target)
        session.client.give_up()
        countdown_seen = False
        while True:
            step = session.advance_playback()
            if step.race_time_ms < 0:
                countdown_seen = True
                continue
            if not countdown_seen:
                continue
            row = state_record(
                step.state,
                reference,
                race_time_ms=step.race_time_ms,
                race_finished=step.race_finished,
            )
            row.update(human_input_fields(step.state))
            if capture_started_wall is None:
                capture_started_wall = time.monotonic()
            row["capture_wall_seconds"] = round(
                time.monotonic() - capture_started_wall, 6
            )
            progress = float(row["progress"])
            if high_water_progress is None or previous_progress is None:
                high_water_progress = progress
                previous_progress = progress
                row["previous_progress"] = progress
                row["new_high_water_progress_delta"] = 0.0
            else:
                row["previous_progress"] = previous_progress
                row["new_high_water_progress_delta"] = max(
                    0.0, progress - high_water_progress
                )
                high_water_progress = max(high_water_progress, progress)
                previous_progress = progress
            records.append(row)
            if telemetry_output is None:
                telemetry_path.parent.mkdir(parents=True, exist_ok=True)
                telemetry_output = telemetry_path.open(
                    "x", encoding="utf-8", newline="\n"
                )
            telemetry_output.write(
                json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
            )
            telemetry_output.flush()
            if step.race_finished or step.race_time_ms >= args.max_race_ms:
                break
        if not countdown_seen:
            raise ProtocolError("human capture never entered a fresh countdown")
        session.recover_inputs(replay_filename)
        wait_for_file(external_replay)
        return records, external_replay
    finally:
        if telemetry_output is not None:
            telemetry_output.close()
        session.close()


def main() -> int:
    args = parse_args()
    args.take_id = validate_take_id(args.take_id)
    args.output_root = args.output_root.resolve()
    args.replay_root = args.replay_root.resolve()
    args.reference_path = args.reference_path.resolve()
    args.tmi_scripts_dir = args.tmi_scripts_dir.resolve()
    if args.max_race_ms <= 0 or args.max_race_ms % RAW_SAMPLE_PERIOD_MS:
        raise ValueError("maximum race time must be a positive multiple of 20 ms")
    if not args.reference_path.is_file():
        raise FileNotFoundError(args.reference_path)
    if not args.tmi_scripts_dir.is_dir():
        raise FileNotFoundError(args.tmi_scripts_dir)
    output_dir = args.output_root / args.take_id
    telemetry_path = output_dir / "telemetry_20ms.jsonl"
    analysis_path = output_dir / "analysis.json"
    local_replay = args.replay_root / f"{args.take_id}.txt"
    if telemetry_path.exists() or analysis_path.exists() or local_replay.exists():
        raise ProtocolError(f"human speedslide take already exists: {args.take_id}")

    replay_filename = f"human_speedslide_{args.take_id}.txt"
    records, external_replay = capture_human_lap(
        args, replay_filename, telemetry_path
    )
    local_replay.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(external_replay, local_replay)
    if sha256(local_replay) != sha256(external_replay):
        raise ProtocolError("recovered human input replay copy changed")
    analysis = analyze_human_records(records)
    gate_both = analysis[
        "positive_reward_eligibility_observed_in_both_zones_at_100ms"
    ]
    result = {
        "status": (
            "both_zone_gate_observed"
            if gate_both
            else "captured_without_both_zone_gate"
        ),
        "track": "A01-Race",
        "input_source": "owner_live_controls",
        "reward_or_policy_used": False,
        "map": args.map,
        "reference_path": str(args.reference_path.relative_to(WORKSPACE_ROOT)),
        "reference_path_sha256": sha256(args.reference_path),
        "telemetry": str(telemetry_path.relative_to(WORKSPACE_ROOT)),
        "telemetry_sha256": sha256(telemetry_path),
        "input_replay": str(local_replay.relative_to(WORKSPACE_ROOT)),
        "input_replay_sha256": sha256(local_replay),
        **analysis,
    }
    write_json(analysis_path, result)
    print(
        f"human reference {result['status']}; "
        f"finished={analysis['outcome']['race_finished']}; "
        f"time={analysis['outcome']['terminal_race_time_ms']}ms",
        flush=True,
    )
    print(f"analysis SHA-256={sha256(analysis_path)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
