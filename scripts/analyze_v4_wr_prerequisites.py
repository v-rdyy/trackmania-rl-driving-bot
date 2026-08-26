"""Measure V4's A01 downhill airtime and first-turn entry line."""

from __future__ import annotations

import argparse
from bisect import bisect_right
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psutil

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from trackmania_rl.env import EnvironmentConfig, LiveTmiSession
from trackmania_rl.game_launch import close_trackmania, ensure_trackmania_running
from trackmania_rl.observations import ReferencePath, signed_heading_error
from trackmania_rl.tmi_bridge import ProtocolError
from trackmania_rl.video_capture import restart_trackmania_race


DEFAULT_EVALUATION_SUMMARY = (
    WORKSPACE_ROOT / "runs" / "reward_v4" / "evaluation_100_summary.json"
)
DEFAULT_REFERENCE_PATH = WORKSPACE_ROOT / "data" / "tracks" / "a01_reference_path.csv"
DEFAULT_OUTPUT_DIR = (
    WORKSPACE_ROOT / "artifacts" / "analysis" / "v4_wr_prerequisites"
)
DEFAULT_TMI_SCRIPTS = Path.home() / "Documents" / "TMInterface" / "Scripts"
SELECTION_QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)
SELECTION_LABELS = ("best", "lower_quartile", "median", "upper_quartile", "worst")
DROP_PROGRESS_MIN = 300.0
DROP_PROGRESS_MAX = 650.0
TURN_ENTRY_PROGRESS = 560.0
TURN_ENTRY_WINDOW = (540.0, 600.0)
MAX_EXISTING_GAME_WINDOWS = 4
EXPECTED_TMFOREVER_EXECUTABLE = (
    Path.home()
    / "AppData"
    / "Local"
    / "TMLoader"
    / "database"
    / "TmForever"
    / "products"
    / "TmForever"
    / "2.12.0"
    / "TmForever.exe"
)


@dataclass(frozen=True)
class ReplayCase:
    label: str
    episode: int
    expected_finish_time_ms: int
    replay: Path
    replay_sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-summary", type=Path, default=DEFAULT_EVALUATION_SUMMARY)
    parser.add_argument("--reference-path", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tmi-scripts-dir", type=Path, default=DEFAULT_TMI_SCRIPTS)
    parser.add_argument("--map-to-load", default="A01-Race.Challenge.Gbx")
    parser.add_argument("--port", type=int, default=8478)
    parser.add_argument("--simulation-speed", type=float, default=6.0)
    parser.add_argument("--reuse-game", action="store_true")
    parser.add_argument("--postprocess-existing", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def close_all_trackmania_windows() -> int:
    closed = 0
    for _ in range(MAX_EXISTING_GAME_WINDOWS):
        if not close_trackmania():
            return closed
        closed += 1
    if close_trackmania():
        raise RuntimeError(
            "more TrackMania windows were open than the clean-start safety limit"
        )
    return closed


def is_verified_tmforever_executable(path: str | None) -> bool:
    if not path:
        return False
    return os.path.normcase(os.path.abspath(path)) == os.path.normcase(
        os.path.abspath(EXPECTED_TMFOREVER_EXECUTABLE)
    )


def terminate_verified_tmforever_processes() -> list[int]:
    targets: list[psutil.Process] = []
    for process in psutil.process_iter(["pid", "exe"]):
        try:
            if is_verified_tmforever_executable(process.info.get("exe")):
                targets.append(process)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    process_ids = sorted(process.pid for process in targets)
    for process in targets:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(targets, timeout=5.0)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    _, still_alive = psutil.wait_procs(alive, timeout=5.0)
    if still_alive:
        raise RuntimeError(
            "verified TmForever processes did not exit: "
            f"{[process.pid for process in still_alive]}"
        )
    return process_ids


def select_representative_cases(summary: dict[str, Any]) -> list[ReplayCase]:
    finishes = sorted(
        (episode for episode in summary["episodes_detail"] if episode["finished"]),
        key=lambda episode: (
            int(episode["terminal_race_time_ms"]),
            int(episode["episode"]),
        ),
    )
    if len(finishes) < len(SELECTION_QUANTILES):
        raise ValueError("evaluation summary has too few finishes for five quantiles")
    last_index = len(finishes) - 1
    indices = [math.floor(last_index * quantile) for quantile in SELECTION_QUANTILES]
    selected: list[ReplayCase] = []
    for label, index in zip(SELECTION_LABELS, indices, strict=True):
        episode = finishes[index]
        selected.append(
            ReplayCase(
                label=label,
                episode=int(episode["episode"]) + 1,
                expected_finish_time_ms=int(episode["terminal_race_time_ms"]),
                replay=WORKSPACE_ROOT / str(episode["input_replay"]),
                replay_sha256=str(episode["input_replay_sha256"]),
            )
        )
    if len({case.episode for case in selected}) != len(selected):
        raise ValueError("quantile selection produced duplicate replay episodes")
    return selected


def state_record(
    state: object,
    reference: ReferencePath,
    *,
    race_finished: bool,
) -> dict[str, Any]:
    position = np.asarray(state.position, dtype=np.float64)
    velocity = np.asarray(state.velocity, dtype=np.float64)
    rotation = np.asarray(state.rotation_matrix, dtype=np.float64)
    angular_velocity = np.asarray(
        state.dyna.current_state.angular_speed,
        dtype=np.float64,
    )
    values = np.concatenate(
        (position, velocity, rotation.ravel(), angular_velocity)
    )
    if not np.isfinite(values).all():
        raise ProtocolError("V4 replay produced nonfinite prerequisite telemetry")
    projection = reference.project(position)
    local_velocity = rotation.T @ velocity
    car_forward_xz = rotation[[0, 2], 2]
    heading_error = signed_heading_error(projection.tangent_xz, car_forward_xz)
    wheel_contacts = [
        bool(wheel.real_time_state.has_ground_contact)
        for wheel in state.simulation_wheels
    ]
    wheel_sliding = [
        bool(wheel.real_time_state.is_sliding)
        for wheel in state.simulation_wheels
    ]
    forward_speed = float(local_velocity[2])
    right_speed = float(local_velocity[0])
    slip_angle = math.degrees(
        math.atan2(right_speed, max(abs(forward_speed), 1e-9))
    )
    return {
        "race_time_ms": int(state.race_time),
        "race_finished": bool(race_finished),
        "display_speed": int(state.display_speed),
        "position": position.tolist(),
        "velocity": velocity.tolist(),
        "rotation_matrix": rotation.tolist(),
        "angular_velocity": angular_velocity.tolist(),
        "local_forward_velocity": forward_speed,
        "local_right_velocity": right_speed,
        "local_up_velocity": float(local_velocity[1]),
        "slip_angle_degrees": slip_angle,
        "body_up_yaw_rate": float(np.dot(angular_velocity, rotation[:, 1])),
        "progress": float(projection.progress),
        "lateral_offset_from_reference": float(projection.lateral_offset),
        "vertical_offset_from_reference": float(projection.vertical_offset),
        "heading_error_radians": float(heading_error),
        "wheel_ground_contacts": wheel_contacts,
        "wheel_sliding": wheel_sliding,
        "ground_contact_count": sum(wheel_contacts),
        "sliding_wheel_count": sum(wheel_sliding),
    }


def parse_replay_input_events(text: str) -> dict[str, list[tuple[int, int]]]:
    events: dict[str, list[tuple[int, int]]] = {"steer": [], "gas": []}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 3 or parts[1].lower() not in events:
            raise ValueError(f"unsupported replay command on line {line_number}: {line!r}")
        timestamp_ms = round(float(parts[0]) * 1000.0)
        value = int(parts[2])
        if timestamp_ms < 0 or not -65536 <= value <= 65536:
            raise ValueError(f"invalid replay command on line {line_number}: {line!r}")
        events[parts[1].lower()].append((timestamp_ms, value))
    for command, command_events in events.items():
        if not command_events:
            raise ValueError(f"replay contains no {command} commands")
        if any(
            later[0] < earlier[0]
            for earlier, later in zip(command_events, command_events[1:])
        ):
            raise ValueError(f"replay {command} commands are not time ordered")
    return events


def annotate_replay_inputs(
    records: list[dict[str, Any]],
    events: dict[str, list[tuple[int, int]]],
) -> None:
    command_times = {
        command: [event[0] for event in command_events]
        for command, command_events in events.items()
    }
    for record in records:
        race_time_ms = max(0, int(record["race_time_ms"]))
        values: dict[str, int] = {}
        for command, command_events in events.items():
            index = bisect_right(command_times[command], race_time_ms) - 1
            values[command] = command_events[index][1] if index >= 0 else 0
        record["input_source"] = "preserved_replay_commands"
        record["input_steer"] = float(values["steer"]) / 65536.0
        record["input_throttle"] = max(0.0, -float(values["gas"]) / 65536.0)
        record["input_brake"] = max(0.0, float(values["gas"]) / 65536.0)


def contiguous_true_runs(values: list[bool]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs


def interpolate_at_progress(
    records: list[dict[str, Any]],
    target_progress: float,
) -> dict[str, Any]:
    fields = (
        "race_time_ms",
        "display_speed",
        "lateral_offset_from_reference",
        "vertical_offset_from_reference",
        "heading_error_radians",
        "local_forward_velocity",
        "local_right_velocity",
        "slip_angle_degrees",
        "body_up_yaw_rate",
    )
    held_input_fields = ("input_steer", "input_throttle", "input_brake")
    for before, after in zip(records, records[1:]):
        first = float(before["progress"])
        second = float(after["progress"])
        if first <= target_progress <= second and second > first:
            fraction = (target_progress - first) / (second - first)
            interpolated = {
                field: float(before[field])
                + fraction * (float(after[field]) - float(before[field]))
                for field in fields
            }
            interpolated["progress"] = target_progress
            for field in held_input_fields:
                interpolated[field] = float(before[field])
            interpolated["position"] = (
                np.asarray(before["position"], dtype=np.float64)
                + fraction
                * (
                    np.asarray(after["position"], dtype=np.float64)
                    - np.asarray(before["position"], dtype=np.float64)
                )
            ).tolist()
            interpolated["bracketing_race_times_ms"] = [
                int(before["race_time_ms"]),
                int(after["race_time_ms"]),
            ]
            return interpolated
    raise ValueError(f"replay does not cross target progress {target_progress}")


def analyze_records(
    records: list[dict[str, Any]],
    reference: ReferencePath,
) -> dict[str, Any]:
    active = sorted(
        (record for record in records if int(record["race_time_ms"]) >= 0),
        key=lambda record: int(record["race_time_ms"]),
    )
    in_drop = [
        DROP_PROGRESS_MIN <= float(record["progress"]) <= DROP_PROGRESS_MAX
        for record in active
    ]
    fully_airborne = [
        in_region and int(record["ground_contact_count"]) == 0
        for record, in_region in zip(active, in_drop, strict=True)
    ]
    runs = contiguous_true_runs(fully_airborne)
    if not runs:
        raise ValueError("no all-four-wheels-airborne segment found on A01 dropdown")
    start, end = max(runs, key=lambda run: run[1] - run[0])
    if start == 0 or end + 1 >= len(active):
        raise ValueError("dropdown airborne segment lacks contact bracketing samples")
    takeoff = active[start - 1]
    landing = active[end + 1]
    airborne = active[start : end + 1]
    if int(takeoff["ground_contact_count"]) == 0:
        raise ValueError("dropdown takeoff bracket is not grounded")
    if int(landing["ground_contact_count"]) == 0:
        raise ValueError("dropdown landing bracket is not grounded")

    takeoff_position = np.asarray(takeoff["position"], dtype=np.float64)
    landing_position = np.asarray(landing["position"], dtype=np.float64)
    horizontal_chord = (landing_position - takeoff_position)[[0, 2]]
    horizontal_distance = float(np.linalg.norm(horizontal_chord))
    if horizontal_distance <= 1e-9:
        raise ValueError("dropdown trajectory has no horizontal displacement")
    takeoff_projection = reference.project(takeoff_position)
    diagonal_angle = signed_heading_error(
        takeoff_projection.tangent_xz,
        horizontal_chord,
    )

    entry = interpolate_at_progress(active, TURN_ENTRY_PROGRESS)
    entry_window = [
        record
        for record in active
        if TURN_ENTRY_WINDOW[0]
        <= float(record["progress"])
        <= TURN_ENTRY_WINDOW[1]
    ]
    if not entry_window:
        raise ValueError("replay has no samples inside the first-turn entry window")

    return {
        "dropdown": {
            "analysis_progress_range": [DROP_PROGRESS_MIN, DROP_PROGRESS_MAX],
            "all_wheels_airborne_sample_count": len(airborne),
            "airtime_estimate_ms": int(landing["race_time_ms"])
            - int(airborne[0]["race_time_ms"]),
            "sampling_uncertainty_ms": 100,
            "first_airborne_race_time_ms": int(airborne[0]["race_time_ms"]),
            "last_airborne_race_time_ms": int(airborne[-1]["race_time_ms"]),
            "takeoff": takeoff,
            "landing": landing,
            "progress_delta": float(landing["progress"]) - float(takeoff["progress"]),
            "horizontal_displacement": horizontal_distance,
            "world_elevation_change": float(landing_position[1] - takeoff_position[1]),
            "trajectory_diagonal_angle_degrees_from_reference": math.degrees(
                diagonal_angle
            ),
            "airborne_lateral_offset_minimum": min(
                float(record["lateral_offset_from_reference"])
                for record in airborne
            ),
            "airborne_lateral_offset_maximum": max(
                float(record["lateral_offset_from_reference"])
                for record in airborne
            ),
            "airborne_vertical_offset_minimum": min(
                float(record["vertical_offset_from_reference"])
                for record in airborne
            ),
            "airborne_vertical_offset_maximum": max(
                float(record["vertical_offset_from_reference"])
                for record in airborne
            ),
            "minimum_throttle_takeoff_through_landing": min(
                float(record["input_throttle"])
                for record in active[start - 1 : end + 2]
            ),
            "maximum_brake_takeoff_through_landing": max(
                float(record["input_brake"])
                for record in active[start - 1 : end + 2]
            ),
        },
        "first_turn_entry": {
            "fixed_progress": TURN_ENTRY_PROGRESS,
            "entry": entry,
            "window_progress_range": list(TURN_ENTRY_WINDOW),
            "window_sample_count": len(entry_window),
            "window_lateral_offset_minimum": min(
                float(record["lateral_offset_from_reference"])
                for record in entry_window
            ),
            "window_lateral_offset_maximum": max(
                float(record["lateral_offset_from_reference"])
                for record in entry_window
            ),
            "window_lateral_offset_mean": float(
                np.mean(
                    [
                        float(record["lateral_offset_from_reference"])
                        for record in entry_window
                    ]
                )
            ),
        },
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, separators=(",", ":"), allow_nan=False))
            output.write("\n")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def aggregate_results(cases: list[dict[str, Any]]) -> dict[str, Any]:
    def values(*keys: str) -> list[float]:
        rows: list[float] = []
        for case in cases:
            value: Any = case
            for key in keys:
                value = value[key]
            rows.append(float(value))
        return rows

    def distribution(rows: list[float]) -> dict[str, float]:
        return {
            "minimum": min(rows),
            "mean": float(np.mean(rows)),
            "maximum": max(rows),
            "population_stddev": float(np.std(rows)),
        }

    return {
        "airtime_estimate_ms": distribution(
            values("measurements", "dropdown", "airtime_estimate_ms")
        ),
        "trajectory_diagonal_angle_degrees_from_reference": distribution(
            values(
                "measurements",
                "dropdown",
                "trajectory_diagonal_angle_degrees_from_reference",
            )
        ),
        "takeoff_display_speed": distribution(
            values("measurements", "dropdown", "takeoff", "display_speed")
        ),
        "landing_display_speed": distribution(
            values("measurements", "dropdown", "landing", "display_speed")
        ),
        "turn_entry_lateral_offset_from_reference": distribution(
            values(
                "measurements",
                "first_turn_entry",
                "entry",
                "lateral_offset_from_reference",
            )
        ),
        "turn_entry_display_speed": distribution(
            values(
                "measurements",
                "first_turn_entry",
                "entry",
                "display_speed",
            )
        ),
        "turn_entry_slip_angle_degrees": distribution(
            values(
                "measurements",
                "first_turn_entry",
                "entry",
                "slip_angle_degrees",
            )
        ),
    }


def finalize_case_result(
    case: ReplayCase,
    records: list[dict[str, Any]],
    *,
    output_dir: Path,
    reference: ReferencePath,
) -> dict[str, Any]:
    if not case.replay.is_file():
        raise FileNotFoundError(case.replay)
    local_hash = sha256(case.replay)
    if local_hash != case.replay_sha256:
        raise ProtocolError(
            f"V4 replay hash mismatch for episode {case.episode}: {local_hash}"
        )
    observed_finish_time = int(records[-1]["race_time_ms"])
    if not bool(records[-1]["race_finished"]):
        raise ProtocolError(f"V4 episode {case.episode} telemetry did not finish")
    if abs(observed_finish_time - case.expected_finish_time_ms) > 100:
        raise ProtocolError(
            f"V4 episode {case.episode} replay time changed by "
            f"{observed_finish_time - case.expected_finish_time_ms}ms"
        )
    replay_events = parse_replay_input_events(case.replay.read_text(encoding="utf-8"))
    annotate_replay_inputs(records, replay_events)
    telemetry_path = output_dir / f"v4_scale100_ep_{case.episode:02d}.jsonl"
    write_jsonl(telemetry_path, records)
    return {
        "selection_label": case.label,
        "episode": case.episode,
        "input_replay": str(case.replay.relative_to(WORKSPACE_ROOT)),
        "input_replay_sha256": local_hash,
        "expected_finish_time_ms": case.expected_finish_time_ms,
        "observed_finish_time_ms": observed_finish_time,
        "telemetry": str(telemetry_path.relative_to(WORKSPACE_ROOT)),
        "telemetry_sha256": sha256(telemetry_path),
        "telemetry_records": len(records),
        "measurements": analyze_records(records, reference),
    }


def main() -> int:
    args = parse_args()
    if not 0.0 < args.simulation_speed <= 1000.0:
        raise SystemExit("--simulation-speed must be in (0, 1000]")
    summary_path = args.evaluation_summary.resolve()
    reference_path = args.reference_path.resolve()
    output_dir = args.output_dir.resolve()
    tmi_scripts_dir = args.tmi_scripts_dir.resolve()
    evaluation = json.loads(summary_path.read_text(encoding="utf-8"))
    cases = select_representative_cases(evaluation)
    reference = ReferencePath.from_csv(reference_path)

    results: list[dict[str, Any]] = []
    if args.postprocess_existing:
        for index, case in enumerate(cases, start=1):
            telemetry_path = output_dir / f"v4_scale100_ep_{case.episode:02d}.jsonl"
            if not telemetry_path.is_file():
                raise FileNotFoundError(telemetry_path)
            records = [
                json.loads(line)
                for line in telemetry_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            result = finalize_case_result(
                case,
                records,
                output_dir=output_dir,
                reference=reference,
            )
            results.append(result)
            measurements = result["measurements"]
            print(
                f"postprocessed {index}/{len(cases)} V4 episode {case.episode}: "
                f"airtime={measurements['dropdown']['airtime_estimate_ms']}ms "
                "entry_offset="
                f"{measurements['first_turn_entry']['entry']['lateral_offset_from_reference']:.3f}",
                flush=True,
            )
    else:
        if not args.reuse_game:
            closed_games = close_all_trackmania_windows()
            terminated_games = terminate_verified_tmforever_processes()
            print(
                f"closed {closed_games} pre-existing TrackMania window(s); "
                f"terminated verified residual processes={terminated_games}",
                flush=True,
            )
        ensure_trackmania_running(port=args.port, confirm_existing=True)
        restart_trackmania_race()
        session = LiveTmiSession(
            EnvironmentConfig(
                port=args.port,
                simulation_speed=args.simulation_speed,
                max_episode_ms=45_000,
                map_to_load=args.map_to_load,
                auto_respawn_on_connect=False,
            )
        )
        try:
            prepared = session.prepare()
            reset_projection = reference.project(prepared.position)
            session.reset(reset_projection)
            for index, case in enumerate(cases, start=1):
                if not case.replay.is_file():
                    raise FileNotFoundError(case.replay)
                local_hash = sha256(case.replay)
                if local_hash != case.replay_sha256:
                    raise ProtocolError(
                        f"V4 replay hash mismatch for episode {case.episode}: {local_hash}"
                    )
                external_replay = tmi_scripts_dir / case.replay.name
                if not external_replay.is_file():
                    raise FileNotFoundError(external_replay)
                if sha256(external_replay) != local_hash:
                    raise ProtocolError(
                        f"TMInterface replay copy differs for episode {case.episode}"
                    )

                session.client.execute_command("unload")
                session.client.execute_command(f"load {external_replay.name}")
                state = session.reset(reset_projection if index == 1 else None)
                records = [state_record(state, reference, race_finished=False)]
                while not records[-1]["race_finished"]:
                    step = session.advance_playback()
                    records.append(
                        state_record(
                            step.state,
                            reference,
                            race_finished=step.race_finished,
                        )
                    )
                    if int(records[-1]["race_time_ms"]) > 45_100:
                        raise ProtocolError(
                            f"V4 episode {case.episode} exceeded the safety limit"
                        )
                result = finalize_case_result(
                    case,
                    records,
                    output_dir=output_dir,
                    reference=reference,
                )
                results.append(result)
                measurements = result["measurements"]
                print(
                    f"analyzed {index}/{len(cases)} V4 episode {case.episode}: "
                    f"airtime={measurements['dropdown']['airtime_estimate_ms']}ms "
                    "entry_offset="
                    f"{measurements['first_turn_entry']['entry']['lateral_offset_from_reference']:.3f}",
                    flush=True,
                )
        finally:
            try:
                session.client.execute_command("unload")
            except (OSError, RuntimeError):
                pass
            session.close()
            if not args.reuse_game:
                terminated_games = terminate_verified_tmforever_processes()
                print(
                    "terminated analyzer-owned TrackMania process(es): "
                    f"{terminated_games}",
                    flush=True,
                )

    result = {
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_evaluation_summary": str(summary_path.relative_to(WORKSPACE_ROOT)),
        "source_evaluation_summary_sha256": sha256(summary_path),
        "reference_path": str(reference_path.relative_to(WORKSPACE_ROOT)),
        "reference_path_sha256": sha256(reference_path),
        "reference_line_limitation": (
            "Signed lateral offset is relative to the intentionally cautious "
            "33.280-second driven reference from Decision 0003, not surveyed road "
            "center geometry."
        ),
        "selection": {
            "method": "best, lower quartile, median, upper quartile, and worst among the 98 successful V4 scale-evaluation finishes",
            "quantiles": list(SELECTION_QUANTILES),
            "case_count": len(results),
        },
        "measurement_protocol": {
            "control_period_ms": 100,
            "simulation_speed": args.simulation_speed,
            "postprocessed_existing_telemetry": args.postprocess_existing,
            "dropdown_progress_range": [DROP_PROGRESS_MIN, DROP_PROGRESS_MAX],
            "airborne_definition": "all four SimulationWheel.has_ground_contact flags are false",
            "airtime_estimate": "time from first observed all-wheel-airborne sample to first subsequent grounded sample; +/- one 100ms sample",
            "turn_entry_progress": TURN_ENTRY_PROGRESS,
            "turn_entry_basis": "reference path reaches the lower platform and begins its first strong heading change near progress 560",
            "turn_entry_window": list(TURN_ENTRY_WINDOW),
            "lateral_sign": "positive is car-right of the driven reference line",
            "input_source": (
                "checksum-pinned replay commands joined by race timestamp; live "
                "SimState input accessors are excluded because they do not reproduce "
                "loaded playback commands; fixed-progress interpolation holds the "
                "last command rather than interpolating discrete inputs"
            ),
        },
        "cases": results,
        "aggregate": aggregate_results(results),
    }
    summary_output = output_dir / "analysis_summary.json"
    write_json(summary_output, result)
    print(f"analysis summary SHA-256={sha256(summary_output)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
