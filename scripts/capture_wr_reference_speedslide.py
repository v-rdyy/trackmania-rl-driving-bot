"""Capture Axell's A01 WR replay through live SimState and audit slide thresholds.

The downloaded replay is provenance, not telemetry.  Its inputs are replayed in
the local simulator, and the resulting dynamics are accepted as a WR reference
only when the local 100 ms trajectory stays within a frozen tolerance of the
positions embedded in the source ghost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pygbx import Gbx, GbxType

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))

from extract_replay_inputs import extract_fastest_ghost
from trackmania_rl.env import EnvironmentConfig, LiveTmiSession, TrackmaniaEnv
from trackmania_rl.game_launch import close_trackmania, ensure_trackmania_running
from trackmania_rl.observations import ReferencePath, signed_heading_error
from trackmania_rl.rewards import (
    WR_STAGE2_ASSISTED_ZONES,
    WR_STAGE2_MIN_ABS_BODY_UP_YAW_RATE,
    WR_STAGE2_MIN_ABS_SLIP_ANGLE_DEGREES,
    WR_STAGE2_MIN_DISPLAY_SPEED,
    WR_STAGE2_MIN_GROUND_CONTACTS,
    WR_STAGE2_MIN_SLIDING_WHEELS,
)
from trackmania_rl.tmi_bridge import ProtocolError

SOURCE_URL = "https://tmnf.exchange/recordgbx/12847157"
SOURCE_LEADERBOARD_URL = "https://tmnf.exchange/trackreplayshow/2233"
EXPECTED_SOURCE_SHA256 = (
    "1A253411D641D26CFC57F428DCD0F5F6EA8557FBEBC8AD30B363623EFA8F644F"
)
EXPECTED_GHOST_LOGIN = "fwo_axell"
EXPECTED_RACE_TIME_MS = 23_770
EXPECTED_GHOST_SAMPLE_PERIOD_MS = 100

# Frozen before the first local playback.  Do not relax these after a result.
FINISH_FIDELITY_TOLERANCE_MS = 100
POSITION_FIDELITY_TOLERANCE_UNITS = 1.0

DEFAULT_SOURCE_REPLAY = (
    WORKSPACE_ROOT
    / "artifacts"
    / "replays"
    / "wr_reference"
    / "a01_axell_23770.Replay.Gbx"
)
DEFAULT_EXTRACTED_INPUT = (
    WORKSPACE_ROOT
    / "artifacts"
    / "replays"
    / "wr_reference"
    / "a01_axell_23770_inputs.txt"
)
DEFAULT_OUTPUT_DIR = (
    WORKSPACE_ROOT / "artifacts" / "analysis" / "wr_reference_speedslide"
)
DEFAULT_REFERENCE_PATH = WORKSPACE_ROOT / "data" / "tracks" / "a01_reference_path.csv"
DEFAULT_TMI_SCRIPTS = Path.home() / "Documents" / "TMInterface" / "Scripts"
ZONE_NAMES = ("first_turn", "final_corner")


@dataclass(frozen=True)
class SourceGhost:
    input_script: str
    metadata: dict[str, object]
    positions_by_race_time: dict[int, list[float]]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-replay", type=Path, default=DEFAULT_SOURCE_REPLAY)
    parser.add_argument("--extracted-input", type=Path, default=DEFAULT_EXTRACTED_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reference-path", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--tmi-scripts-dir", type=Path, default=DEFAULT_TMI_SCRIPTS)
    parser.add_argument("--map", default="A01-Race.Challenge.Gbx")
    parser.add_argument("--port", type=int, default=8478)
    parser.add_argument("--simulation-speed", type=float, default=1.0)
    parser.add_argument("--max-race-ms", type=int, default=45_000)
    parser.add_argument("--reuse-game", action="store_true")
    parser.add_argument(
        "--postprocess-existing",
        action="store_true",
        help="analyze a preserved telemetry.jsonl after a live-run analysis failure",
    )
    return parser.parse_args()


def _vector3_values(vector: object) -> list[float]:
    return [float(vector.x), float(vector.y), float(vector.z)]


def load_source_ghost(path: Path) -> SourceGhost:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_hash = sha256(path)
    if actual_hash != EXPECTED_SOURCE_SHA256:
        raise ProtocolError(
            "WR source replay hash mismatch: "
            f"expected {EXPECTED_SOURCE_SHA256}, got {actual_hash}"
        )
    input_script, metadata = extract_fastest_ghost(path)
    if str(metadata["selected_ghost_login"]) != EXPECTED_GHOST_LOGIN:
        raise ProtocolError("WR source selected an unexpected ghost login")
    if int(metadata["selected_ghost_race_time_ms"]) != EXPECTED_RACE_TIME_MS:
        raise ProtocolError("WR source selected an unexpected finish time")

    ghosts = Gbx(str(path)).get_classes_by_ids(
        [GbxType.CTN_GHOST, GbxType.CTN_GHOST_OLD]
    )
    matches = [
        ghost
        for ghost in ghosts
        if str(getattr(ghost, "login", "")) == EXPECTED_GHOST_LOGIN
        and int(getattr(ghost, "race_time", -1)) == EXPECTED_RACE_TIME_MS
    ]
    if len(matches) != 1:
        raise ProtocolError("WR source does not contain exactly one expected ghost")
    ghost = matches[0]
    if int(ghost.sample_period) != EXPECTED_GHOST_SAMPLE_PERIOD_MS:
        raise ProtocolError("WR source ghost is not sampled at 100 ms")
    last_comparison_time = (
        EXPECTED_RACE_TIME_MS // EXPECTED_GHOST_SAMPLE_PERIOD_MS
    ) * EXPECTED_GHOST_SAMPLE_PERIOD_MS
    required_records = last_comparison_time // EXPECTED_GHOST_SAMPLE_PERIOD_MS + 1
    if len(ghost.records) < required_records:
        raise ProtocolError("WR source ghost has incomplete race-time positions")
    positions = {
        index * EXPECTED_GHOST_SAMPLE_PERIOD_MS: _vector3_values(record.position)
        for index, record in enumerate(ghost.records[:required_records])
    }
    if not np.isfinite(np.asarray(list(positions.values()), dtype=np.float64)).all():
        raise ProtocolError("WR source ghost positions are nonfinite")
    metadata = {
        **metadata,
        "ghost_sample_period_ms": int(ghost.sample_period),
        "ghost_position_records": len(ghost.records),
        "ghost_game_version": str(ghost.game_version),
        "ghost_checkpoint_times_ms": [int(value) for value in ghost.cp_times],
    }
    return SourceGhost(input_script, metadata, positions)


def write_or_verify_text(path: Path, text: str) -> str:
    payload = text.encode("utf-8")
    expected_hash = bytes_sha256(payload)
    if path.exists():
        if not path.is_file() or sha256(path) != expected_hash:
            raise ProtocolError(f"refusing to overwrite different artifact: {path}")
        return expected_hash
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    if sha256(path) != expected_hash:
        raise ProtocolError(f"artifact write failed checksum verification: {path}")
    return expected_hash


def copy_or_verify(source: Path, destination: Path) -> None:
    source_hash = sha256(source)
    if destination.exists():
        if not destination.is_file() or sha256(destination) != source_hash:
            raise ProtocolError(
                f"refusing to overwrite different TMInterface input: {destination}"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256(destination) != source_hash:
        raise ProtocolError("TMInterface input copy failed checksum verification")


def state_record(
    state: object,
    reference: ReferencePath,
    *,
    race_time_ms: int,
    race_finished: bool,
) -> dict[str, Any]:
    if int(state.race_time) != int(race_time_ms):
        raise ProtocolError("live state race time disagrees with callback race time")
    position = np.asarray(state.position, dtype=np.float64)
    rotation = np.asarray(state.rotation_matrix, dtype=np.float64)
    if position.shape != (3,) or rotation.shape != (3, 3):
        raise ProtocolError("live WR state has invalid position or rotation shape")
    dynamics = TrackmaniaEnv._full_simstate_log(state)
    if dynamics.get("full_simstate_available") is not True:
        raise ProtocolError("live WR state is missing complete SimState dynamics")
    projection = reference.project(position)
    car_forward_xz = rotation[[0, 2], 2]
    heading_error = signed_heading_error(projection.tangent_xz, car_forward_xz)
    return {
        "race_time_ms": int(race_time_ms),
        "race_finished": bool(race_finished),
        "display_speed": int(state.display_speed),
        "position": position.tolist(),
        "progress": float(projection.progress),
        "lateral_offset_from_reference": float(projection.lateral_offset),
        "vertical_offset_from_reference": float(projection.vertical_offset),
        "heading_error_radians": float(heading_error),
        **dynamics,
    }


def annotate_high_water_progress(records: list[dict[str, Any]]) -> None:
    if not records:
        raise ProtocolError("WR capture produced no active-race records")
    high_water = float(records[0]["progress"])
    previous = high_water
    for record in records:
        progress = float(record["progress"])
        record["previous_progress"] = previous
        record["new_high_water_progress_delta"] = max(0.0, progress - high_water)
        high_water = max(high_water, progress)
        previous = progress


def zone_name(progress: float) -> str | None:
    for name, bounds in zip(ZONE_NAMES, WR_STAGE2_ASSISTED_ZONES, strict=True):
        if bounds[0] <= progress <= bounds[1]:
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


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "samples": 0,
            "minimum": None,
            "median": None,
            "p95": None,
            "maximum": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "samples": len(values),
        "minimum": float(array.min()),
        "median": float(statistics.median(values)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def _contiguous_windows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    qualifying = [record for record in records if all(condition_flags(record).values())]
    windows: list[list[dict[str, Any]]] = []
    for record in qualifying:
        if (
            not windows
            or int(record["race_time_ms"])
            != int(windows[-1][-1]["race_time_ms"]) + EXPECTED_GHOST_SAMPLE_PERIOD_MS
        ):
            windows.append([record])
        else:
            windows[-1].append(record)
    return [
        {
            "start_race_time_ms": int(window[0]["race_time_ms"]),
            "end_race_time_ms": int(window[-1]["race_time_ms"]),
            "samples": len(window),
            "sample_span_ms": len(window) * EXPECTED_GHOST_SAMPLE_PERIOD_MS,
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
        for window in windows
    ]


def summarize_zone(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ProtocolError("WR trajectory did not enter an assisted zone")
    flags = [condition_flags(record) for record in records]
    full = [
        record
        for record, row_flags in zip(records, flags, strict=True)
        if all(row_flags.values())
    ]
    positive_bonus = [
        record
        for record in full
        if float(record["new_high_water_progress_delta"]) > 0.0
    ]
    sliding = [record for record in records if int(record["sliding_wheel_count"]) >= 1]
    speed_ground = [
        record
        for record, row_flags in zip(records, flags, strict=True)
        if row_flags["speed"] and row_flags["ground"]
    ]
    return {
        "samples": len(records),
        "race_time_min_ms": min(int(row["race_time_ms"]) for row in records),
        "race_time_max_ms": max(int(row["race_time_ms"]) for row in records),
        "progress_minimum": min(float(row["progress"]) for row in records),
        "progress_maximum": max(float(row["progress"]) for row in records),
        "condition_counts": {
            name: sum(row[name] for row in flags)
            for name in ("ground", "speed", "sliding", "slip", "yaw")
        },
        "joint_counts": {
            "ground_and_speed": sum(row["ground"] and row["speed"] for row in flags),
            "ground_speed_sliding": sum(
                row["ground"] and row["speed"] and row["sliding"] for row in flags
            ),
            "ground_speed_slip": sum(
                row["ground"] and row["speed"] and row["slip"] for row in flags
            ),
            "all_dynamics_conditions": len(full),
            "positive_bonus_eligible": len(positive_bonus),
        },
        "all_condition_windows": _contiguous_windows(records),
        "speed": _distribution([float(row["display_speed"]) for row in records]),
        "abs_slip_angle_degrees": _distribution(
            [abs(float(row["slip_angle_degrees"])) for row in records]
        ),
        "abs_body_up_yaw_rate": _distribution(
            [abs(float(row["body_up_yaw_rate"])) for row in records]
        ),
        "sliding_wheel_count": _distribution(
            [float(row["sliding_wheel_count"]) for row in records]
        ),
        "sliding_samples_signature": {
            "samples": len(sliding),
            "speed": _distribution([float(row["display_speed"]) for row in sliding]),
            "abs_slip_angle_degrees": _distribution(
                [abs(float(row["slip_angle_degrees"])) for row in sliding]
            ),
            "abs_body_up_yaw_rate": _distribution(
                [abs(float(row["body_up_yaw_rate"])) for row in sliding]
            ),
            "maximum_sliding_wheels": max(
                (int(row["sliding_wheel_count"]) for row in sliding), default=0
            ),
        },
        "speed_ground_subset": {
            "samples": len(speed_ground),
            "maximum_abs_slip_angle_degrees": max(
                (abs(float(row["slip_angle_degrees"])) for row in speed_ground),
                default=0.0,
            ),
            "maximum_sliding_wheels": max(
                (int(row["sliding_wheel_count"]) for row in speed_ground), default=0
            ),
        },
    }


def summarize_optional_zone(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "entered": False,
            "samples": 0,
            "all_condition_windows": [],
            "condition_counts": {
                name: 0 for name in ("ground", "speed", "sliding", "slip", "yaw")
            },
            "joint_counts": {
                "ground_and_speed": 0,
                "ground_speed_sliding": 0,
                "ground_speed_slip": 0,
                "all_dynamics_conditions": 0,
                "positive_bonus_eligible": 0,
            },
        }
    return {"entered": True, **summarize_zone(records)}


def fidelity_metrics(
    records: list[dict[str, Any]], source_positions: dict[int, list[float]]
) -> dict[str, Any]:
    if not records:
        raise ProtocolError("WR fidelity audit has no records")
    records_by_time: dict[int, dict[str, Any]] = {}
    for record in records:
        race_time = int(record["race_time_ms"])
        if race_time in records_by_time:
            raise ProtocolError("WR fidelity audit has duplicate race times")
        records_by_time[race_time] = record
    expected_times = sorted(source_positions)
    missing_times = [time for time in expected_times if time not in records_by_time]
    error_rows: list[tuple[dict[str, Any], float]] = []
    for race_time in expected_times:
        if race_time not in records_by_time:
            continue
        record = records_by_time[race_time]
        error = float(
            np.linalg.norm(
                np.asarray(record["position"], dtype=np.float64)
                - np.asarray(source_positions[race_time], dtype=np.float64)
            )
        )
        error_rows.append((record, error))
    all_errors = [error for _, error in error_rows]
    zone_errors = {
        name: [
            error
            for record, error in error_rows
            if zone_name(float(record["progress"])) == name
        ]
        for name in ZONE_NAMES
    }
    observed_finish = int(records[-1]["race_time_ms"])
    finish_error = abs(observed_finish - EXPECTED_RACE_TIME_MS)
    overall = _distribution(all_errors)
    zones = {name: _distribution(values) for name, values in zone_errors.items()}
    overall_maximum = overall["maximum"]
    accepted = bool(
        records[-1].get("race_finished") is True
        and not missing_times
        and finish_error <= FINISH_FIDELITY_TOLERANCE_MS
        and overall["samples"] == len(expected_times)
        and overall_maximum is not None
        and float(overall_maximum) <= POSITION_FIDELITY_TOLERANCE_UNITS
        and all(
            int(zones[name]["samples"] or 0) > 0
            and zones[name]["maximum"] is not None
            and float(zones[name]["maximum"]) <= POSITION_FIDELITY_TOLERANCE_UNITS
            for name in ZONE_NAMES
        )
    )
    return {
        "accepted_as_wr_reference": accepted,
        "finish_observed_ms": observed_finish,
        "finish_expected_ms": EXPECTED_RACE_TIME_MS,
        "finish_absolute_error_ms": finish_error,
        "finish_tolerance_ms": FINISH_FIDELITY_TOLERANCE_MS,
        "position_tolerance_units": POSITION_FIDELITY_TOLERANCE_UNITS,
        "expected_position_samples": len(expected_times),
        "missing_position_times_ms": missing_times,
        "overall_position_error_units": overall,
        "zone_position_error_units": zones,
    }


def analyze_records(
    records: list[dict[str, Any]], source_positions: dict[int, list[float]]
) -> dict[str, Any]:
    if any(record.get("full_simstate_available") is not True for record in records):
        raise ProtocolError("WR analysis requires complete live SimState")
    if any(
        not math.isfinite(float(record[field]))
        for record in records
        for field in (
            "progress",
            "display_speed",
            "slip_angle_degrees",
            "body_up_yaw_rate",
            "new_high_water_progress_delta",
        )
    ):
        raise ProtocolError("WR analysis contains nonfinite telemetry")
    zones = {
        name: summarize_optional_zone(
            [
                record
                for record in records
                if zone_name(float(record["progress"])) == name
            ]
        )
        for name in ZONE_NAMES
    }
    all_flags = [condition_flags(record) for record in records]
    return {
        "fidelity": fidelity_metrics(records, source_positions),
        "zones": zones,
        "track_wide": {
            "samples": len(records),
            "sliding_samples": sum(row["sliding"] for row in all_flags),
            "slip_samples": sum(row["slip"] for row in all_flags),
            "all_dynamics_condition_samples": sum(
                all(row.values()) for row in all_flags
            ),
        },
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    if path.exists():
        raise ProtocolError(f"refusing to overwrite telemetry: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, separators=(",", ":"), allow_nan=False))
            output.write("\n")


def write_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise ProtocolError(f"refusing to overwrite analysis: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def capture_live_records(args: argparse.Namespace, external_input: Path) -> list[dict[str, Any]]:
    if not math.isfinite(args.simulation_speed) or args.simulation_speed <= 0.0:
        raise ValueError("simulation speed must be positive and finite")
    if args.max_race_ms <= EXPECTED_RACE_TIME_MS:
        raise ValueError("maximum race time must exceed the source finish")
    launched = False
    session: LiveTmiSession | None = None
    if not args.reuse_game:
        close_trackmania()
    try:
        _, launched = ensure_trackmania_running(
            port=args.port,
            confirm_existing=not args.reuse_game,
        )
        session = LiveTmiSession(
            EnvironmentConfig(
                port=args.port,
                simulation_speed=args.simulation_speed,
                step_period_ms=EXPECTED_GHOST_SAMPLE_PERIOD_MS,
                max_episode_ms=args.max_race_ms,
                map_to_load=args.map,
                auto_respawn_on_connect=False,
                wait_for_race_start_on_connect=True,
            )
        )
        reference = ReferencePath.from_csv(args.reference_path)
        session.prepare()
        session.client.execute_command("unload")
        session.client.execute_command(f"load {external_input.name}")
        session.client.give_up()
        records: list[dict[str, Any]] = []
        countdown_seen = False
        while True:
            step = session.advance_playback()
            if step.race_time_ms < 0:
                countdown_seen = True
                continue
            if not countdown_seen:
                continue
            records.append(
                state_record(
                    step.state,
                    reference,
                    race_time_ms=step.race_time_ms,
                    race_finished=step.race_finished,
                )
            )
            if step.race_finished or step.race_time_ms >= args.max_race_ms:
                break
        if not countdown_seen:
            raise ProtocolError("WR playback never entered a fresh countdown")
        annotate_high_water_progress(records)
        return records
    finally:
        if session is not None:
            try:
                session.client.execute_command("unload")
            except (OSError, RuntimeError):
                pass
            session.close()
        if launched and not args.reuse_game:
            close_trackmania()


def main() -> int:
    args = parse_args()
    args.source_replay = args.source_replay.resolve()
    args.extracted_input = args.extracted_input.resolve()
    args.output_dir = args.output_dir.resolve()
    args.reference_path = args.reference_path.resolve()
    args.tmi_scripts_dir = args.tmi_scripts_dir.resolve()
    telemetry_path = args.output_dir / "telemetry.jsonl"
    analysis_path = args.output_dir / "analysis.json"
    if analysis_path.exists():
        raise ProtocolError("WR reference analysis already exists")
    if args.postprocess_existing:
        if not telemetry_path.is_file():
            raise ProtocolError("WR reference telemetry does not exist for postprocessing")
    elif telemetry_path.exists():
        raise ProtocolError("WR reference telemetry already exists")
    if not args.reference_path.is_file():
        raise FileNotFoundError(args.reference_path)
    if not args.postprocess_existing and not args.tmi_scripts_dir.is_dir():
        raise FileNotFoundError(args.tmi_scripts_dir)

    source = load_source_ghost(args.source_replay)
    input_hash = write_or_verify_text(args.extracted_input, source.input_script)
    if args.postprocess_existing:
        records = [
            json.loads(line)
            for line in telemetry_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        external_input = args.tmi_scripts_dir / args.extracted_input.name
        copy_or_verify(args.extracted_input, external_input)
        records = capture_live_records(args, external_input)
        write_jsonl(telemetry_path, records)
    analysis = analyze_records(records, source.positions_by_race_time)
    result = {
        "status": (
            "accepted_wr_reference"
            if analysis["fidelity"]["accepted_as_wr_reference"]
            else "rejected_wr_reference"
        ),
        "track": "A01-Race",
        "source": {
            "leaderboard_url": SOURCE_LEADERBOARD_URL,
            "replay_url": SOURCE_URL,
            "replay_path": str(args.source_replay.relative_to(WORKSPACE_ROOT)),
            "replay_sha256": sha256(args.source_replay),
            "replay_bytes": args.source_replay.stat().st_size,
            **source.metadata,
        },
        "extracted_input": {
            "path": str(args.extracted_input.relative_to(WORKSPACE_ROOT)),
            "sha256": input_hash,
            "bytes": args.extracted_input.stat().st_size,
        },
        "capture": {
            "source": "local_replay_resimulation_live_simstate",
            "map": args.map,
            "simulation_speed": args.simulation_speed,
            "step_period_ms": EXPECTED_GHOST_SAMPLE_PERIOD_MS,
            "telemetry": str(telemetry_path.relative_to(WORKSPACE_ROOT)),
            "telemetry_sha256": sha256(telemetry_path),
            "telemetry_records": len(records),
        },
        "threshold_contract": {
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
            "minimum_abs_body_up_yaw_rate": WR_STAGE2_MIN_ABS_BODY_UP_YAW_RATE,
        },
        **analysis,
        "publication_note": (
            "The third-party replay is retained locally for calibration only. "
            "Its download availability does not establish redistribution rights."
        ),
    }
    write_json(analysis_path, result)
    print(
        f"WR reference {result['status']}; "
        f"finish={analysis['fidelity']['finish_observed_ms']}ms; "
        "max_position_error="
        f"{analysis['fidelity']['overall_position_error_units']['maximum']:.6f}",
        flush=True,
    )
    print(f"analysis SHA-256={sha256(analysis_path)}", flush=True)
    if not analysis["fidelity"]["accepted_as_wr_reference"]:
        raise ProtocolError("local WR replay failed the frozen fidelity gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
