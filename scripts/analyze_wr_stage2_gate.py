"""Analyze one WR-chase Stage 2 gate from direct-live evaluation logs only."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from trackmania_rl.evaluation_metrics import (  # noqa: E402
    aggregate_precision_metrics,
    steering_precision_metrics,
    trajectory_precision_metrics,
)
from trackmania_rl.observations import ReferencePath, signed_heading_error  # noqa: E402

RUN_DIR = WORKSPACE_ROOT / "runs" / "wr_chase_stage2"
GATE_DIR = RUN_DIR / "gates"
MANIFEST_PATH = RUN_DIR / "run_manifest.json"
ANALYSIS_ROOT = WORKSPACE_ROOT / "artifacts" / "analysis" / "wr_chase_stage2"
REFERENCE_PATH = WORKSPACE_ROOT / "data" / "tracks" / "a01_reference_path.csv"

EXPECTED_EPISODES = 10
GATE_SIZE = 500_000
TIMEBOX = 2_000_000
STEP_PERIOD_MS = 100
TERMINAL_TIME_TOLERANCE_MS = 0
REWARD_TOLERANCE = 1e-6
HIGH_WATER_TOLERANCE = 1e-6
DERIVED_DYNAMICS_TOLERANCE = 1e-9
REFERENCE_DIAGNOSTIC_TOLERANCE = 1e-6

EXPECTED_INITIAL_CHECKPOINT_SHA256 = (
    "BA056E0B42D7CAEE4D02B6AB8E0D592BE6A363068E75AAEBC3ED8487791B4044"
)
EXPECTED_REWARD_FUNCTION = "localized_drift_assistance_reward"
EXPECTED_TELEMETRY_SOURCE = "direct_live_evaluation_simstate"

KNOWN_ZONES = {
    "first_turn": (680.0, 930.0),
    "final_corner": (1100.0, 1410.0),
}
CANDIDATE_MIN_SAMPLES = 2
CONFIRMED_MIN_SAMPLES = 3
MIN_DISPLAY_SPEED = 350
MIN_ABS_SLIP_ANGLE_DEGREES = 1.0
MIN_ABS_YAW_RATE = 0.25
MIN_GROUND_CONTACTS = 3
CANDIDATE_MIN_SLIDING_WHEELS = 1
CONFIRMED_MIN_SLIDING_WHEELS = 2
DISCOVERY_EPISODES = 3

BONUS_PROGRESS_CLAMP = 20.0
BONUS_COEFFICIENT = 0.50
V4_PROGRESS_CLAMP = 20.0
V4_PROGRESS_NORMALIZATION = 10.0
V4_TIME_COST = 0.10
V4_FINISH_BONUS = 50.0
V4_FAILURE_PENALTY = 250.0

DROP_PROGRESS_RANGE = (300.0, 650.0)
TURN_ENTRY_PROGRESS = 560.0
TURN_ENTRY_WINDOW = (540.0, 600.0)


class AnalysisError(ValueError):
    """The frozen Stage 2 evidence contract was not satisfied."""


def gate_slug(target: int) -> str:
    return f"gate_{target:08d}"


def validate_target(target: int) -> None:
    if target < 0:
        raise AnalysisError("Stage 2 target cannot be negative")
    if target % GATE_SIZE:
        raise AnalysisError(f"Stage 2 target must be a multiple of {GATE_SIZE}")
    if target > TIMEBOX:
        raise AnalysisError(f"Stage 2 target exceeds frozen {TIMEBOX}-step time box")


def evaluation_summary_path(target: int) -> Path:
    return GATE_DIR / f"{gate_slug(target)}_evaluation.json"


def evaluation_action_log_path(target: int) -> Path:
    return GATE_DIR / f"{gate_slug(target)}_evaluation_actions.jsonl"


def output_path(target: int) -> Path:
    return ANALYSIS_ROOT / gate_slug(target) / "analysis_summary.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(WORKSPACE_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def optional_sha256(path: Path) -> str | None:
    return sha256(path) if path.is_file() else None


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-target", type=int, required=True)
    parser.add_argument("--evaluation-summary", type=Path)
    parser.add_argument("--action-log", type=Path)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _require_finite(record: dict[str, Any], fields: tuple[str, ...]) -> None:
    for field in fields:
        if field not in record:
            raise AnalysisError(f"live action record is missing {field!r}")
        try:
            value = float(record[field])
        except (TypeError, ValueError) as exc:
            raise AnalysisError(f"live field {field!r} is not numeric") from exc
        if not math.isfinite(value):
            raise AnalysisError(f"live field {field!r} is nonfinite")


def _normalize_alias(
    record: dict[str, Any],
    first: str,
    second: str,
) -> None:
    if first not in record and second not in record:
        raise AnalysisError(f"live action record is missing {first!r}/{second!r}")
    if first in record and second in record:
        if not math.isclose(
            float(record[first]),
            float(record[second]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise AnalysisError(f"live aliases {first!r}/{second!r} disagree")
    elif first in record:
        record[second] = record[first]
    else:
        record[first] = record[second]


def _workspace_artifact(raw_path: object, *, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise AnalysisError(f"{label} path is missing")
    resolved = (WORKSPACE_ROOT / raw_path).resolve()
    try:
        resolved.relative_to(WORKSPACE_ROOT.resolve())
    except ValueError as exc:
        raise AnalysisError(f"{label} path escapes the workspace") from exc
    return resolved


def _require_bound_file(
    gate: dict[str, Any],
    *,
    path_field: str,
    hash_field: str,
    actual_path: Path | None = None,
) -> Path:
    recorded_path = _workspace_artifact(gate.get(path_field), label=path_field)
    if actual_path is not None and recorded_path != actual_path.resolve():
        raise AnalysisError(f"{path_field} does not match the requested evidence file")
    if not recorded_path.is_file():
        raise AnalysisError(f"{path_field} file is missing: {recorded_path}")
    recorded_hash = gate.get(hash_field)
    actual_hash = sha256(recorded_path)
    if recorded_hash != actual_hash:
        raise AnalysisError(
            f"{hash_field} does not match the current {path_field} file"
        )
    return recorded_path


def _manifest_targets(manifest: dict[str, Any]) -> list[int]:
    gates = manifest.get("gates")
    if not isinstance(gates, list) or not gates:
        raise AnalysisError("Stage 2 manifest has no gates")
    try:
        targets = [int(gate["target_additional_steps"]) for gate in gates]
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisError("Stage 2 manifest contains an invalid gate target") from exc
    expected = list(range(0, targets[-1] + GATE_SIZE, GATE_SIZE))
    if targets != expected:
        raise AnalysisError(
            "Stage 2 manifest gates are not the exact ordered 0/500k prefix"
        )
    if targets[-1] > TIMEBOX:
        raise AnalysisError("Stage 2 manifest exceeds the frozen time box")
    return targets


def validate_evidence_binding(
    manifest: dict[str, Any],
    target: int,
    evaluation_path: Path,
    action_path: Path,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    """Bind the requested gate, manifest, evaluation, log, and checkpoint."""
    targets = _manifest_targets(manifest)
    if targets[-1] != target:
        raise AnalysisError("requested analysis is not the manifest's latest gate")
    if manifest.get("reward_function") != EXPECTED_REWARD_FUNCTION:
        raise AnalysisError("Stage 2 manifest reward function changed")
    if manifest.get("telemetry_source_required") != EXPECTED_TELEMETRY_SOURCE:
        raise AnalysisError("Stage 2 manifest telemetry source changed")
    if manifest.get("replay_telemetry_fallback_allowed") is not False:
        raise AnalysisError("Stage 2 manifest does not forbid replay fallback")
    if int(manifest.get("gate_size_nominal", -1)) != GATE_SIZE:
        raise AnalysisError("Stage 2 manifest gate cadence changed")
    if int(manifest.get("timebox", -1)) != TIMEBOX:
        raise AnalysisError("Stage 2 manifest time box changed")
    if int(manifest.get("evaluation_episodes", -1)) != EXPECTED_EPISODES:
        raise AnalysisError("Stage 2 manifest evaluation episode count changed")
    if int(manifest.get("step_period_ms", -1)) != STEP_PERIOD_MS:
        raise AnalysisError("Stage 2 manifest step period changed")
    initialization = manifest.get("stage2_initialization", {})
    if initialization.get("sha256") != EXPECTED_INITIAL_CHECKPOINT_SHA256:
        raise AnalysisError("Stage 2 manifest does not pin the approved Gate 2 base")

    for prior in manifest["gates"][:-1]:
        if (
            prior.get("evaluation_status") != "complete"
            or prior.get("telemetry_status") != "complete"
            or prior.get("telemetry_conclusion_valid") is not True
            or prior.get("stage2_decision") != "continue"
        ):
            raise AnalysisError("an earlier Stage 2 gate was not validly continued")

    gate = manifest["gates"][-1]
    if gate.get("evaluation_status") != "complete":
        raise AnalysisError("requested Stage 2 gate evaluation is not complete")
    if target == 0 and (
        gate.get("checkpoint") != initialization.get("path")
        or gate.get("checkpoint_sha256") != initialization.get("sha256")
    ):
        raise AnalysisError("baseline gate does not use the approved initialization")
    if target > 0:
        if gate.get("status") != "complete":
            raise AnalysisError("trained Stage 2 gate is not marked complete")
        training_path = _require_bound_file(
            gate,
            path_field="training_summary",
            hash_field="training_summary_sha256",
        )
        try:
            training = json.loads(training_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise AnalysisError("Stage 2 training summary is malformed") from exc
        if (
            training.get("status") != "complete"
            or int(training.get("gate_target_additional_steps", -1)) != target
            or training.get("checkpoint") != gate.get("checkpoint")
            or training.get("checkpoint_sha256") != gate.get("checkpoint_sha256")
            or int(training.get("final_model_timesteps", -1))
            != int(gate.get("model_timesteps", -2))
        ):
            raise AnalysisError("training summary does not match the manifest gate")
    if int(manifest.get("latest_evaluated_gate", -1)) != target:
        raise AnalysisError("manifest latest evaluated gate does not match request")
    summary_file = _require_bound_file(
        gate,
        path_field="evaluation_summary",
        hash_field="evaluation_summary_sha256",
        actual_path=evaluation_path,
    )
    action_file = _require_bound_file(
        gate,
        path_field="evaluation_action_log",
        hash_field="evaluation_action_log_sha256",
        actual_path=action_path,
    )
    if summary_file != evaluation_path.resolve() or action_file != action_path.resolve():
        raise AnalysisError("requested Stage 2 evidence paths do not match manifest")
    if evaluation.get("action_log_sha256") != sha256(action_file):
        raise AnalysisError("evaluation summary is not bound to its action log")
    if int(evaluation.get("gate_target_additional_steps", -1)) != target:
        raise AnalysisError("evaluation summary gate target does not match request")
    if evaluation.get("measurement_source") != EXPECTED_TELEMETRY_SOURCE:
        raise AnalysisError("evaluation summary is not marked direct-live")
    if evaluation.get("replay_telemetry_fallback_used") is not False:
        raise AnalysisError("evaluation summary does not rule out replay fallback")
    expected_run_tag = f"{gate_slug(target)}_baseline" if target == 0 else gate_slug(target)
    if evaluation.get("run_tag") != expected_run_tag:
        raise AnalysisError("evaluation summary run tag does not match its gate")
    if evaluation.get("deterministic") is not True:
        raise AnalysisError("Stage 2 evaluation is not deterministic")
    if not math.isclose(
        float(evaluation.get("simulation_speed", math.nan)),
        6.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise AnalysisError("Stage 2 evaluation speed changed")
    if evaluation.get("checkpoint") != gate.get("checkpoint"):
        raise AnalysisError("evaluation checkpoint path does not match manifest gate")
    if evaluation.get("checkpoint_sha256") != gate.get("checkpoint_sha256"):
        raise AnalysisError("evaluation checkpoint hash does not match manifest gate")
    checkpoint = _require_bound_file(
        gate,
        path_field="checkpoint",
        hash_field="checkpoint_sha256",
    )
    if sha256(checkpoint) != evaluation.get("checkpoint_sha256"):
        raise AnalysisError("evaluation checkpoint does not match the current file")
    if evaluation.get("reference_path_sha256") != sha256(REFERENCE_PATH):
        raise AnalysisError("evaluation reference path does not match Stage 2 analysis")
    if int(gate.get("deterministic_finishes", -1)) != int(
        evaluation.get("finishes", -2)
    ):
        raise AnalysisError("manifest finish total disagrees with evaluation summary")
    if gate.get("deterministic_best_finish_time_ms") != evaluation.get(
        "best_finish_time_ms"
    ):
        raise AnalysisError("manifest best time disagrees with evaluation summary")
    gate_mean = gate.get("deterministic_mean_finish_time_ms")
    evaluation_mean = evaluation.get("average_finish_time_ms")
    if gate_mean is None or evaluation_mean is None:
        if gate_mean != evaluation_mean:
            raise AnalysisError("manifest mean time disagrees with evaluation summary")
    elif not math.isclose(
        float(gate_mean), float(evaluation_mean), rel_tol=0.0, abs_tol=1e-9
    ):
        raise AnalysisError("manifest mean time disagrees with evaluation summary")
    audit = gate.get("direct_live_simstate_audit", {})
    if not (
        audit.get("complete") is True
        and audit.get("source") == EXPECTED_TELEMETRY_SOURCE
        and audit.get("replay_fallback_used") is False
        and int(audit.get("episodes", -1)) == EXPECTED_EPISODES
        and audit.get("terminal_outcomes_matched") is True
    ):
        raise AnalysisError("manifest direct-live evaluator audit is incomplete")

    details = evaluation.get("episodes_detail")
    if not isinstance(details, list) or len(details) != EXPECTED_EPISODES:
        raise AnalysisError("evaluation summary does not bind ten replay records")
    if int(evaluation.get("input_replay_count", -1)) != EXPECTED_EPISODES:
        raise AnalysisError("evaluation summary did not preserve ten input replays")
    summary_replay_hashes = evaluation.get("input_replay_sha256")
    if not isinstance(summary_replay_hashes, list) or len(summary_replay_hashes) != EXPECTED_EPISODES:
        raise AnalysisError("evaluation replay hash list is incomplete")
    replay_dir = _workspace_artifact(
        gate.get("evaluation_replay_dir"), label="evaluation_replay_dir"
    )
    if not replay_dir.is_dir():
        raise AnalysisError("Stage 2 evaluation replay directory is missing")
    replay_paths: list[Path] = []
    replay_hashes: list[str] = []
    for detail in sorted(details, key=lambda item: int(item["episode"])):
        replay = _workspace_artifact(detail.get("input_replay"), label="input_replay")
        try:
            replay.relative_to(replay_dir)
        except ValueError as exc:
            raise AnalysisError("an input replay is outside the gate replay directory") from exc
        if not replay.is_file():
            raise AnalysisError(f"preserved input replay is missing: {replay}")
        if replay.suffix.lower() != ".txt" or replay.stat().st_size <= 0:
            raise AnalysisError("preserved input replay is not a nonempty text replay")
        if int(detail.get("input_replay_bytes", -1)) != replay.stat().st_size:
            raise AnalysisError("preserved input replay size changed")
        replay_hash = sha256(replay)
        if detail.get("input_replay_sha256") != replay_hash:
            raise AnalysisError("preserved input replay hash changed")
        replay_paths.append(replay)
        replay_hashes.append(replay_hash)
    if len(set(replay_paths)) != EXPECTED_EPISODES:
        raise AnalysisError("Stage 2 input replay paths are not unique")
    if summary_replay_hashes != replay_hashes:
        raise AnalysisError("evaluation replay hash list disagrees with episode records")
    return gate


def validate_pending_analysis_request(
    manifest: dict[str, Any], target: int, summary_path: Path
) -> dict[str, Any]:
    """Reject stale/repeated/noncanonical requests without mutating evidence."""
    targets = _manifest_targets(manifest)
    if targets[-1] != target:
        raise AnalysisError("only the latest Stage 2 gate may be analyzed")
    if summary_path.resolve() != output_path(target).resolve():
        raise AnalysisError("Stage 2 analysis must use its canonical gate output path")
    gate = manifest["gates"][-1]
    if gate.get("evaluation_status") != "complete":
        raise AnalysisError("latest Stage 2 gate evaluation is not complete")
    if (
        gate.get("telemetry_status") != "pending"
        or gate.get("stage2_decision") != "pending"
    ):
        raise AnalysisError("latest Stage 2 gate is not pending first analysis")
    return gate


def normalize_live_record(source: dict[str, Any]) -> dict[str, Any]:
    """Validate one direct-live row and expose both historical field aliases."""
    record = dict(source)
    if record.get("full_simstate_available") is not True:
        raise AnalysisError("every Stage 2 row requires complete live SimState")
    if record.get("reward_function") != EXPECTED_REWARD_FUNCTION:
        raise AnalysisError("every Stage 2 row must use the frozen Stage 2 reward")

    for field in ("wheel_ground_contacts", "wheel_sliding"):
        wheels = record.get(field)
        if not isinstance(wheels, list) or len(wheels) != 4:
            raise AnalysisError(f"{field} must contain exactly four booleans")
        if any(type(value) is not bool for value in wheels):
            raise AnalysisError(f"{field} must contain exactly four booleans")

    if int(record.get("ground_contact_count", -1)) != sum(
        record["wheel_ground_contacts"]
    ):
        raise AnalysisError("ground_contact_count disagrees with wheel booleans")
    if int(record.get("sliding_wheel_count", -1)) != sum(record["wheel_sliding"]):
        raise AnalysisError("sliding_wheel_count disagrees with wheel booleans")

    _normalize_alias(record, "lateral_offset", "lateral_offset_from_reference")
    _normalize_alias(record, "vertical_offset", "vertical_offset_from_reference")
    _normalize_alias(record, "heading_error", "heading_error_radians")

    raw_action = record.get("raw_action")
    if not isinstance(raw_action, list) or len(raw_action) != 3:
        raise AnalysisError("raw_action must contain steer, throttle, and brake")
    if any(not math.isfinite(float(value)) for value in raw_action):
        raise AnalysisError("raw_action contains nonfinite values")
    if any(abs(float(value)) > 1.0 + 1e-9 for value in raw_action):
        raise AnalysisError("raw_action is outside the normalized action space")
    for field, value in zip(
        ("input_steer", "input_throttle", "input_brake"),
        raw_action,
        strict=True,
    ):
        expected = float(value)
        if field in record and not math.isclose(
            float(record[field]), expected, rel_tol=0.0, abs_tol=1e-9
        ):
            raise AnalysisError(f"{field} disagrees with raw_action")
        record[field] = expected

    _require_finite(
        record,
        (
            "race_time_ms",
            "progress",
            "previous_progress",
            "new_high_water_progress_delta",
            "display_speed",
            "slip_angle_degrees",
            "body_up_yaw_rate",
            "reward",
            "lateral_offset",
            "vertical_offset",
            "heading_error",
            "upright_cosine",
        ),
    )
    for field in ("position", "velocity", "angular_velocity"):
        values = record.get(field)
        if not isinstance(values, list) or len(values) != 3:
            raise AnalysisError(f"{field} must contain three finite values")
        if any(not math.isfinite(float(value)) for value in values):
            raise AnalysisError(f"{field} contains nonfinite values")
    rotation = record.get("rotation_matrix")
    if (
        not isinstance(rotation, list)
        or len(rotation) != 3
        or any(not isinstance(row, list) or len(row) != 3 for row in rotation)
        or any(
            not math.isfinite(float(value))
            for row in rotation
            for value in row
        )
    ):
        raise AnalysisError("rotation_matrix must be a finite 3x3 matrix")

    velocity_array = np.asarray(record["velocity"], dtype=np.float64)
    rotation_array = np.asarray(rotation, dtype=np.float64)
    angular_velocity_array = np.asarray(
        record["angular_velocity"], dtype=np.float64
    )
    local_velocity = rotation_array.T @ velocity_array
    recomputed_slip = math.degrees(
        math.atan2(
            float(local_velocity[0]),
            max(abs(float(local_velocity[2])), 1e-9),
        )
    )
    recomputed_yaw = float(
        np.dot(angular_velocity_array, rotation_array[:, 1])
    )
    if not math.isclose(
        float(record["slip_angle_degrees"]),
        recomputed_slip,
        rel_tol=0.0,
        abs_tol=DERIVED_DYNAMICS_TOLERANCE,
    ):
        raise AnalysisError("logged slip angle disagrees with raw live SimState")
    if not math.isclose(
        float(record["body_up_yaw_rate"]),
        recomputed_yaw,
        rel_tol=0.0,
        abs_tol=DERIVED_DYNAMICS_TOLERANCE,
    ):
        raise AnalysisError("logged body-up yaw rate disagrees with raw live SimState")
    record["slip_angle_degrees"] = recomputed_slip
    record["body_up_yaw_rate"] = recomputed_yaw

    for field in ("terminated", "truncated", "race_finished"):
        if type(record.get(field)) is not bool:
            raise AnalysisError(f"{field} must be a boolean")
    if bool(record["race_finished"]) != bool(record["terminated"]):
        raise AnalysisError("race_finished and terminated disagree")
    if bool(record["terminated"]) and bool(record["truncated"]):
        raise AnalysisError("a live row cannot be both terminated and truncated")
    return record


def validate_reference_diagnostics(
    records: list[dict[str, Any]], reference: ReferencePath
) -> None:
    """Recompute reward/zone diagnostics from raw live pose and the pinned path."""
    for record in records:
        position = np.asarray(record["position"], dtype=np.float64)
        rotation = np.asarray(record["rotation_matrix"], dtype=np.float64)
        projection = reference.project(position)
        heading = signed_heading_error(
            projection.tangent_xz,
            rotation[[0, 2], 2],
        )
        recomputed = {
            "progress": float(projection.progress),
            "lateral_offset": float(projection.lateral_offset),
            "vertical_offset": float(projection.vertical_offset),
            "heading_error": float(heading),
            "upright_cosine": float(rotation[1, 1]),
        }
        for field, expected in recomputed.items():
            if not math.isclose(
                float(record[field]),
                expected,
                rel_tol=0.0,
                abs_tol=REFERENCE_DIAGNOSTIC_TOLERANCE,
            ):
                raise AnalysisError(
                    f"logged {field} disagrees with raw pose/reference path"
                )


def evaluated_race_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Drop a disclosed pre-race prefix after the final clock restart."""
    resets = [
        index
        for index in range(1, len(records))
        if int(records[index]["race_time_ms"])
        < int(records[index - 1]["race_time_ms"])
    ]
    if not resets:
        start = 0
    else:
        start = resets[-1]
    while start < len(records) and int(records[start]["race_time_ms"]) < 0:
        start += 1
    if start >= len(records):
        raise AnalysisError("race clock never reaches active race time")
    active = records[start:]
    if any(
        int(later["race_time_ms"]) < int(earlier["race_time_ms"])
        for earlier, later in zip(active, active[1:])
    ):
        raise AnalysisError("evaluated race timestamps are not monotonic")
    return active, start


def validate_progress_continuity(records: list[dict[str, Any]]) -> None:
    """Ensure every reward delta joins the preceding live progress exactly."""
    for previous, current in zip(records, records[1:]):
        if not math.isclose(
            float(current["previous_progress"]),
            float(previous["progress"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise AnalysisError(
                "previous_progress does not match the preceding live progress"
            )


def _episode_details(evaluation: dict[str, Any]) -> dict[int, dict[str, Any]]:
    if int(evaluation.get("episodes", -1)) != EXPECTED_EPISODES:
        raise AnalysisError(
            f"Stage 2 evaluation must declare exactly {EXPECTED_EPISODES} episodes"
        )
    details = evaluation.get("episodes_detail")
    if not isinstance(details, list) or len(details) != EXPECTED_EPISODES:
        count = len(details) if isinstance(details, list) else 0
        raise AnalysisError(
            f"Stage 2 evaluation has {count} episodes, expected {EXPECTED_EPISODES}"
        )
    by_episode: dict[int, dict[str, Any]] = {}
    for detail in details:
        episode = int(detail["episode"])
        if episode in by_episode:
            raise AnalysisError(f"duplicate evaluation episode {episode}")
        by_episode[episode] = detail
    expected = set(range(EXPECTED_EPISODES))
    if set(by_episode) != expected:
        raise AnalysisError("evaluation episodes must be numbered 0 through 9")
    return by_episode


def group_live_records(
    raw_records: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, list[dict[str, Any]]]]:
    details = _episode_details(evaluation)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for source in raw_records:
        if "episode" not in source:
            raise AnalysisError("live action record is missing episode")
        episode = int(source["episode"])
        if episode not in details:
            raise AnalysisError(f"unexpected action-log episode {episode}")
        grouped[episode].append(normalize_live_record(source))
    if set(grouped) != set(details) or any(not records for records in grouped.values()):
        raise AnalysisError("action log must contain exactly ten nonempty episode logs")

    active: dict[int, list[dict[str, Any]]] = {}
    for episode in range(EXPECTED_EPISODES):
        episode_records = grouped[episode]
        try:
            steps = [int(record["step"]) for record in episode_records]
        except (KeyError, TypeError, ValueError) as exc:
            raise AnalysisError(f"episode {episode} has an invalid step index") from exc
        if steps != list(range(len(episode_records))):
            raise AnalysisError(
                f"episode {episode} action steps are not the exact ordered sequence"
            )
        validate_progress_continuity(episode_records)
        terminal_indices = [
            index
            for index, record in enumerate(episode_records)
            if bool(record["terminated"]) or bool(record["truncated"])
        ]
        if terminal_indices != [len(episode_records) - 1]:
            raise AnalysisError(
                f"episode {episode} must contain one terminal row, and it must be last"
            )
        detail = details[episode]
        if "steps" in detail and int(detail["steps"]) != len(episode_records):
            raise AnalysisError(f"episode {episode} step count does not match action log")
        if "total_reward" in detail and not math.isclose(
            float(detail["total_reward"]),
            sum(float(record["reward"]) for record in episode_records),
            rel_tol=0.0,
            abs_tol=REWARD_TOLERANCE,
        ):
            raise AnalysisError(f"episode {episode} reward total does not match action log")
        evaluated, _ = evaluated_race_records(episode_records)
        expected_time = int(details[episode]["terminal_race_time_ms"])
        observed_time = int(evaluated[-1]["race_time_ms"])
        if abs(observed_time - expected_time) > TERMINAL_TIME_TOLERANCE_MS:
            raise AnalysisError(
                f"episode {episode} terminal time differs by "
                f"{observed_time - expected_time}ms"
            )
        expected_finished = bool(details[episode]["finished"])
        if bool(evaluated[-1]["race_finished"]) != expected_finished:
            raise AnalysisError(f"episode {episode} terminal outcome does not match")
        expected_truncated = bool(
            details[episode]["timeout"]
            or details[episode]["off_track"]
            or details[episode]["fallen"]
            or details[episode]["stuck"]
        )
        if bool(evaluated[-1]["truncated"]) != expected_truncated:
            raise AnalysisError(f"episode {episode} terminal truncation does not match")
        active[episode] = evaluated
    return dict(grouped), active


def zone_for_progress(progress: float) -> str | None:
    for name, (low, high) in KNOWN_ZONES.items():
        if low <= progress <= high:
            return name
    return None


def sample_meets_drift_gate(record: dict[str, Any], *, sliding_wheels: int) -> bool:
    return bool(
        int(record["ground_contact_count"]) >= MIN_GROUND_CONTACTS
        and int(record["display_speed"]) >= MIN_DISPLAY_SPEED
        and int(record["sliding_wheel_count"]) >= sliding_wheels
        and abs(float(record["slip_angle_degrees"]))
        >= MIN_ABS_SLIP_ANGLE_DEGREES
        and abs(float(record["body_up_yaw_rate"])) >= MIN_ABS_YAW_RATE
    )


def contiguous_sample_runs(
    records: list[dict[str, Any]],
    *,
    sliding_wheels: int,
    progress_range: tuple[float, float] | None = None,
) -> list[list[dict[str, Any]]]:
    runs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for record in records:
        in_range = bool(
            progress_range is None
            or progress_range[0] <= float(record["progress"]) <= progress_range[1]
        )
        meets = in_range and sample_meets_drift_gate(
            record,
            sliding_wheels=sliding_wheels,
        )
        consecutive = bool(
            current
            and int(record["race_time_ms"])
            - int(current[-1]["race_time_ms"])
            == STEP_PERIOD_MS
        )
        if meets:
            if current and not consecutive:
                runs.append(current)
                current = []
            current.append(record)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def summarize_run(run: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_count": len(run),
        "duration_ms": len(run) * STEP_PERIOD_MS,
        "start_race_time_ms": int(run[0]["race_time_ms"]),
        "end_race_time_ms": int(run[-1]["race_time_ms"]),
        "start_progress": float(run[0]["progress"]),
        "end_progress": float(run[-1]["progress"]),
        "maximum_display_speed": max(int(record["display_speed"]) for record in run),
        "maximum_abs_slip_angle_degrees": max(
            abs(float(record["slip_angle_degrees"])) for record in run
        ),
        "maximum_abs_body_up_yaw_rate": max(
            abs(float(record["body_up_yaw_rate"])) for record in run
        ),
        "minimum_sliding_wheels": min(
            int(record["sliding_wheel_count"]) for record in run
        ),
    }


def drift_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen Stage 1 window detector to direct-live rows."""
    by_zone: dict[str, Any] = {}
    for name, bounds in KNOWN_ZONES.items():
        zone_records = [
            record for record in records if bounds[0] <= float(record["progress"]) <= bounds[1]
        ]
        candidates = [
            run
            for run in contiguous_sample_runs(
                records,
                sliding_wheels=CANDIDATE_MIN_SLIDING_WHEELS,
                progress_range=bounds,
            )
            if len(run) >= CANDIDATE_MIN_SAMPLES
        ]
        confirmed = [
            run
            for run in contiguous_sample_runs(
                records,
                sliding_wheels=CONFIRMED_MIN_SLIDING_WHEELS,
                progress_range=bounds,
            )
            if len(run) >= CONFIRMED_MIN_SAMPLES
        ]
        by_zone[name] = {
            "progress_range": list(bounds),
            "samples": len(zone_records),
            "maximum_sliding_wheels": max(
                (int(record["sliding_wheel_count"]) for record in zone_records),
                default=0,
            ),
            "maximum_abs_slip_angle_degrees": max(
                (abs(float(record["slip_angle_degrees"])) for record in zone_records),
                default=0.0,
            ),
            "candidate_windows": [summarize_run(run) for run in candidates],
            "confirmed_windows": [summarize_run(run) for run in confirmed],
        }
    track_candidates = [
        run
        for run in contiguous_sample_runs(
            records,
            sliding_wheels=CANDIDATE_MIN_SLIDING_WHEELS,
        )
        if len(run) >= CANDIDATE_MIN_SAMPLES
    ]
    track_confirmed = [
        run
        for run in contiguous_sample_runs(
            records,
            sliding_wheels=CONFIRMED_MIN_SLIDING_WHEELS,
        )
        if len(run) >= CONFIRMED_MIN_SAMPLES
    ]
    return {
        "known_zones": by_zone,
        "track_wide_candidate_windows": [summarize_run(run) for run in track_candidates],
        "track_wide_confirmed_windows": [summarize_run(run) for run in track_confirmed],
    }


def recompute_reward_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Reconstruct episode high-water progress, V4, bonus, and total reward."""
    if not records:
        raise AnalysisError("reward audit requires a nonempty episode")
    high_water = float(records[0]["previous_progress"])
    rows: list[dict[str, Any]] = []
    zone_totals = {
        name: {
            "drift_attempt_steps": 0,
            "rewarded_steps": 0,
            "localized_bonus": 0.0,
        }
        for name in KNOWN_ZONES
    }
    maximum_reward_error = 0.0
    maximum_high_water_error = 0.0

    for index, record in enumerate(records):
        progress = float(record["progress"])
        previous_progress = float(record["previous_progress"])
        progress_delta = progress - previous_progress
        recomputed_new_high_water = max(0.0, progress - high_water)
        logged_new_high_water = float(record["new_high_water_progress_delta"])
        high_water_error = abs(recomputed_new_high_water - logged_new_high_water)
        maximum_high_water_error = max(maximum_high_water_error, high_water_error)

        v4_reward = (
            min(max(progress_delta, -V4_PROGRESS_CLAMP), V4_PROGRESS_CLAMP)
            / V4_PROGRESS_NORMALIZATION
            - V4_TIME_COST
        )
        if bool(record["terminated"]):
            v4_reward += V4_FINISH_BONUS
        elif bool(record["truncated"]):
            v4_reward -= V4_FAILURE_PENALTY

        zone = zone_for_progress(progress)
        drift_attempt = sample_meets_drift_gate(
            record,
            sliding_wheels=CANDIDATE_MIN_SLIDING_WHEELS,
        )
        localized_bonus = 0.0
        if zone is not None and drift_attempt:
            localized_bonus = BONUS_COEFFICIENT * (
                min(max(recomputed_new_high_water, 0.0), BONUS_PROGRESS_CLAMP)
                / BONUS_PROGRESS_CLAMP
            )
            zone_totals[zone]["drift_attempt_steps"] += 1
            if localized_bonus > 0.0:
                zone_totals[zone]["rewarded_steps"] += 1
                zone_totals[zone]["localized_bonus"] += localized_bonus

        recomputed_reward = v4_reward + localized_bonus
        logged_reward = float(record["reward"])
        reward_error = abs(recomputed_reward - logged_reward)
        maximum_reward_error = max(maximum_reward_error, reward_error)
        rows.append(
            {
                "record_index": index,
                "race_time_ms": int(record["race_time_ms"]),
                "progress": progress,
                "zone": zone,
                "drift_attempt": drift_attempt,
                "recomputed_new_high_water_progress_delta": recomputed_new_high_water,
                "logged_new_high_water_progress_delta": logged_new_high_water,
                "recomputed_v4_reward": v4_reward,
                "recomputed_localized_drift_bonus": localized_bonus,
                "recomputed_stage2_reward": recomputed_reward,
                "logged_reward": logged_reward,
                "absolute_reward_error": reward_error,
            }
        )
        high_water = max(high_water, progress)

    valid = bool(
        maximum_reward_error <= REWARD_TOLERANCE
        and maximum_high_water_error <= HIGH_WATER_TOLERANCE
    )
    return {
        "valid": valid,
        "row_count": len(rows),
        "reward_tolerance": REWARD_TOLERANCE,
        "high_water_tolerance": HIGH_WATER_TOLERANCE,
        "maximum_absolute_reward_error": maximum_reward_error,
        "maximum_absolute_high_water_delta_error": maximum_high_water_error,
        "logged_reward_total": sum(float(record["reward"]) for record in records),
        "recomputed_reward_total": sum(
            float(row["recomputed_stage2_reward"]) for row in rows
        ),
        "recomputed_localized_bonus_total": sum(
            float(row["recomputed_localized_drift_bonus"]) for row in rows
        ),
        "known_zones": zone_totals,
        "mismatch_rows": [
            row
            for row in rows
            if float(row["absolute_reward_error"]) > REWARD_TOLERANCE
            or abs(
                float(row["recomputed_new_high_water_progress_delta"])
                - float(row["logged_new_high_water_progress_delta"])
            )
            > HIGH_WATER_TOLERANCE
        ],
    }


def _contiguous_true_runs(values: list[bool]) -> list[tuple[int, int]]:
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


def _interpolate_entry(records: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "race_time_ms",
        "display_speed",
        "lateral_offset_from_reference",
        "vertical_offset_from_reference",
        "heading_error_radians",
        "slip_angle_degrees",
        "body_up_yaw_rate",
    )
    for before, after in zip(records, records[1:]):
        first = float(before["progress"])
        second = float(after["progress"])
        if first <= TURN_ENTRY_PROGRESS <= second and second > first:
            fraction = (TURN_ENTRY_PROGRESS - first) / (second - first)
            entry = {
                field: float(before[field])
                + fraction * (float(after[field]) - float(before[field]))
                for field in fields
            }
            entry["progress"] = TURN_ENTRY_PROGRESS
            return entry
    raise ValueError("trajectory does not cross the fixed first-turn entry sample")


def prerequisite_metrics(
    records: list[dict[str, Any]],
    reference: ReferencePath,
) -> dict[str, Any] | None:
    """Port the Stage 1 drop/entry math without importing game-facing tooling."""
    active = sorted(
        (record for record in records if int(record["race_time_ms"]) >= 0),
        key=lambda record: int(record["race_time_ms"]),
    )
    in_drop = [
        DROP_PROGRESS_RANGE[0] <= float(record["progress"]) <= DROP_PROGRESS_RANGE[1]
        for record in active
    ]
    airborne = [
        inside and int(record["ground_contact_count"]) == 0
        for record, inside in zip(active, in_drop, strict=True)
    ]
    runs = _contiguous_true_runs(airborne)
    if not runs:
        return None
    start, end = max(runs, key=lambda run: run[1] - run[0])
    if start == 0 or end + 1 >= len(active):
        return None
    takeoff = active[start - 1]
    landing = active[end + 1]
    if (
        int(takeoff["ground_contact_count"]) == 0
        or int(landing["ground_contact_count"]) == 0
    ):
        return None
    try:
        entry = _interpolate_entry(active)
    except ValueError:
        return None

    takeoff_position = np.asarray(takeoff["position"], dtype=np.float64)
    landing_position = np.asarray(landing["position"], dtype=np.float64)
    chord = (landing_position - takeoff_position)[[0, 2]]
    if float(np.linalg.norm(chord)) <= 1e-9:
        return None
    tangent = reference.project(takeoff_position).tangent_xz
    diagonal = math.degrees(signed_heading_error(tangent, chord))
    bracket = active[start - 1 : end + 2]
    return {
        "dropdown": {
            "analysis_progress_range": list(DROP_PROGRESS_RANGE),
            "all_wheels_airborne_sample_count": end - start + 1,
            "airtime_estimate_ms": int(landing["race_time_ms"])
            - int(active[start]["race_time_ms"]),
            "sampling_uncertainty_ms": STEP_PERIOD_MS,
            "trajectory_diagonal_angle_degrees_from_reference": diagonal,
            "minimum_throttle_takeoff_through_landing": min(
                float(record["input_throttle"]) for record in bracket
            ),
            "maximum_brake_takeoff_through_landing": max(
                float(record["input_brake"]) for record in bracket
            ),
            "landing_display_speed": int(landing["display_speed"]),
        },
        "first_turn_entry": entry,
    }


def precision_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    steering = [
        (float(record["race_time_ms"]) / 1000.0, float(record["input_steer"]))
        for record in records
    ]
    return {
        "steering": steering_precision_metrics(steering),
        **trajectory_precision_metrics(records),
    }


def aggregate_discovery(cases: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        zone: sum(
            bool(case["drift"]["known_zones"][zone]["confirmed_windows"])
            for case in cases
        )
        for zone in KNOWN_ZONES
    }
    triggers = [zone for zone, count in counts.items() if count >= DISCOVERY_EPISODES]
    return {
        "required_episodes": DISCOVERY_EPISODES,
        "direct_live_valid_episodes": len(cases),
        "known_zone_confirmed_episode_counts": counts,
        "telemetry_induction_triggered": bool(triggers),
        "trigger_locations": triggers,
        "visual_confirmation_required": bool(triggers),
    }


def lap_metrics(evaluation: dict[str, Any], details: dict[int, dict[str, Any]]) -> dict[str, Any]:
    finish_times = [
        int(detail["terminal_race_time_ms"])
        for detail in details.values()
        if bool(detail["finished"])
    ]
    finishes = len(finish_times)
    if "finishes" in evaluation and int(evaluation["finishes"]) != finishes:
        raise AnalysisError("evaluation finish total disagrees with episode details")
    return {
        "finishes": finishes,
        "finish_rate": finishes / EXPECTED_EPISODES,
        "best_finish_time_ms": min(finish_times) if finish_times else None,
        "average_finish_time_ms": statistics.fmean(finish_times) if finish_times else None,
        "worst_finish_time_ms": max(finish_times) if finish_times else None,
    }


def analyze_evaluation(
    evaluation: dict[str, Any],
    raw_records: list[dict[str, Any]],
    *,
    reference: ReferencePath,
) -> dict[str, Any]:
    if int(evaluation.get("raw_action_records", len(raw_records))) != len(raw_records):
        raise AnalysisError("evaluation raw action count disagrees with action log")
    details = _episode_details(evaluation)
    grouped, active = group_live_records(raw_records, evaluation)
    for episode in range(EXPECTED_EPISODES):
        validate_reference_diagnostics(grouped[episode], reference)
    evaluated_count = sum(len(records) for records in active.values())
    if int(evaluation.get("action_records", evaluated_count)) != evaluated_count:
        raise AnalysisError("evaluation active action count disagrees with action log")
    cases: list[dict[str, Any]] = []
    for episode in range(EXPECTED_EPISODES):
        audit = recompute_reward_audit(grouped[episode])
        cases.append(
            {
                "episode": episode,
                "expected_finished": bool(details[episode]["finished"]),
                "expected_terminal_race_time_ms": int(
                    details[episode]["terminal_race_time_ms"]
                ),
                "observed_finished": bool(active[episode][-1]["race_finished"]),
                "observed_terminal_race_time_ms": int(
                    active[episode][-1]["race_time_ms"]
                ),
                "raw_records": len(grouped[episode]),
                "evaluated_records": len(active[episode]),
                "measurement_source": "direct_live_evaluation_simstate",
                "reward_audit": audit,
                "drift": drift_metrics(active[episode]),
                "prerequisites": prerequisite_metrics(active[episode], reference),
                "precision": precision_metrics(active[episode]),
            }
        )

    reward_valid = all(bool(case["reward_audit"]["valid"]) for case in cases)
    zone_bonus = {
        zone: {
            "drift_attempt_steps": sum(
                int(case["reward_audit"]["known_zones"][zone]["drift_attempt_steps"])
                for case in cases
            ),
            "rewarded_steps": sum(
                int(case["reward_audit"]["known_zones"][zone]["rewarded_steps"])
                for case in cases
            ),
            "localized_bonus": sum(
                float(case["reward_audit"]["known_zones"][zone]["localized_bonus"])
                for case in cases
            ),
        }
        for zone in KNOWN_ZONES
    }
    discovery = aggregate_discovery(cases)
    zone_summary: dict[str, Any] = {}
    for zone in KNOWN_ZONES:
        candidate_windows = [
            {"episode": int(case["episode"]), **window}
            for case in cases
            for window in case["drift"]["known_zones"][zone]["candidate_windows"]
        ]
        confirmed_windows = [
            {"episode": int(case["episode"]), **window}
            for case in cases
            for window in case["drift"]["known_zones"][zone]["confirmed_windows"]
        ]
        zone_summary[zone] = {
            **zone_bonus[zone],
            "candidate_window_count": len(candidate_windows),
            "confirmed_window_count": len(confirmed_windows),
            "candidate_episode_count": sum(
                bool(case["drift"]["known_zones"][zone]["candidate_windows"])
                for case in cases
            ),
            "confirmed_episode_count": discovery[
                "known_zone_confirmed_episode_counts"
            ][zone],
            "candidate_windows": candidate_windows,
            "confirmed_windows": confirmed_windows,
        }
    return {
        "instrumentation_valid": reward_valid,
        "instrumentation_errors": (
            []
            if reward_valid
            else ["logged Stage 2 reward/high-water audit does not match formula"]
        ),
        "measurement_source": "direct_live_evaluation_simstate",
        "lap_metrics": lap_metrics(evaluation, details),
        "cases": cases,
        "reward_audit": {
            "valid": reward_valid,
            "known_zones": zone_bonus,
            "logged_reward_total": sum(
                float(case["reward_audit"]["logged_reward_total"]) for case in cases
            ),
            "recomputed_reward_total": sum(
                float(case["reward_audit"]["recomputed_reward_total"]) for case in cases
            ),
            "recomputed_localized_bonus_total": sum(
                float(case["reward_audit"]["recomputed_localized_bonus_total"])
                for case in cases
            ),
        },
        "zones": zone_summary,
        "discovery": discovery,
        "precision": aggregate_precision_metrics(
            [case["precision"] for case in cases]
        ),
    }


def assess_stage2(manifest: dict[str, Any]) -> dict[str, Any]:
    gates = manifest.get("gates", [])
    if not gates:
        return {"decision": "continue", "reason": "no analyzed Stage 2 gate"}
    try:
        _manifest_targets(manifest)
    except AnalysisError as exc:
        return {
            "decision": "instrumentation_required",
            "reason": str(exc),
        }
    analyzed = [
        gate
        for gate in gates
        if gate.get("live_analysis_status", gate.get("telemetry_status"))
        in {"complete", "invalid"}
    ]
    if len(analyzed) != len(gates):
        return {
            "decision": "instrumentation_required",
            "reason": "Stage 2 gate history contains an unanalyzed gate",
        }
    if any(not bool(gate.get("telemetry_conclusion_valid")) for gate in analyzed):
        return {
            "decision": "instrumentation_required",
            "reason": "a Stage 2 direct-live analysis or reward audit is invalid",
        }
    latest = analyzed[-1]

    latest_target = int(latest["target_additional_steps"])
    if latest_target == 0:
        if bool(latest.get("telemetry_induction_triggered")):
            return {
                "decision": "baseline_comparability_review",
                "reason": (
                    "the untouched initialization already shows a repeated "
                    "confirmed slide, so later induction cannot be attributed "
                    "to Stage 2 without review"
                ),
                "baseline_excluded_from_safety_sequence": True,
            }
        return {
            "decision": "continue",
            "reason": "valid target-zero comparability baseline completed",
            "baseline_excluded_from_safety_sequence": True,
        }

    trained = [
        gate
        for gate in analyzed
        if int(gate["target_additional_steps"]) > 0
        and bool(gate.get("telemetry_conclusion_valid"))
    ]
    if any(bool(gate.get("telemetry_induction_triggered")) for gate in trained):
        return {
            "decision": "review_induction",
            "reason": "confirmed localized slide repeated in at least 3/10 episodes",
        }

    latest_finishes = int(latest.get("deterministic_finishes", -1))
    if latest_finishes == 0:
        return {
            "decision": "safety_review",
            "reason": "latest trained gate finished 0/10 deterministic episodes",
        }
    recent = trained[-2:]
    if len(recent) == 2 and all(
        int(gate.get("deterministic_finishes", -1)) <= 5 for gate in recent
    ):
        return {
            "decision": "safety_review",
            "reason": "two consecutive trained gates finished at most 5/10 episodes",
        }
    if latest_target >= TIMEBOX:
        return {
            "decision": "ineffective_within_timebox",
            "reason": (
                "no repeated slide induction occurred within the frozen "
                "two-million-interaction Stage 2 budget"
            ),
        }
    return {
        "decision": "continue",
        "reason": "no induction, safety trigger, or time-box boundary yet",
    }


def _manifest_gate(manifest: dict[str, Any], target: int) -> dict[str, Any]:
    matching = [
        gate
        for gate in manifest.get("gates", [])
        if int(gate["target_additional_steps"]) == target
    ]
    if len(matching) != 1:
        raise AnalysisError("Stage 2 manifest must contain exactly one matching gate")
    return matching[0]


def update_manifest(
    manifest_path: Path,
    target: int,
    summary_path: Path,
    result: dict[str, Any],
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gate = _manifest_gate(manifest, target)
    if (
        gate.get("telemetry_status") != "pending"
        or gate.get("stage2_decision") != "pending"
    ):
        raise AnalysisError("Stage 2 gate is not pending first analysis")
    valid = bool(result.get("instrumentation_valid"))
    lap = result.get("lap_metrics", {})
    discovery = result.get("discovery", {})
    updates = {
        "telemetry_status": "complete" if valid else "invalid",
        "live_analysis_status": "complete" if valid else "invalid",
        "telemetry_summary": str(summary_path.relative_to(WORKSPACE_ROOT)),
        "telemetry_summary_sha256": sha256(summary_path),
        "telemetry_conclusion_valid": valid,
        "telemetry_induction_triggered": bool(
            discovery.get("telemetry_induction_triggered", False)
        ),
    }
    if valid:
        updates.update(
            {
                "deterministic_finishes": lap.get("finishes"),
                "deterministic_best_finish_time_ms": lap.get(
                    "best_finish_time_ms"
                ),
                "deterministic_mean_finish_time_ms": lap.get(
                    "average_finish_time_ms"
                ),
            }
        )
    gate.update(updates)
    decision = assess_stage2(manifest)
    gate["stage2_decision"] = decision["decision"]
    manifest["latest_stage2_assessment"] = decision
    manifest["status"] = {
        "continue": "ready_for_next_gate",
        "review_induction": "induction_requires_visual_review",
        "baseline_comparability_review": "baseline_comparability_review_required",
        "instrumentation_required": "instrumentation_required",
        "safety_review": "safety_review_required",
        "ineffective_within_timebox": "bonus_ineffective_within_timebox",
    }[decision["decision"]]
    write_json(manifest_path, manifest)


def main() -> int:
    args = parse_args()
    validate_target(args.gate_target)
    evaluation_path = (args.evaluation_summary or evaluation_summary_path(args.gate_target)).resolve()
    action_path = (args.action_log or evaluation_action_log_path(args.gate_target)).resolve()
    summary_path = (args.output or output_path(args.gate_target)).resolve()
    if summary_path.exists():
        raise SystemExit(f"Stage 2 analysis already exists: {summary_path}")
    manifest_path = args.manifest.resolve()
    if not manifest_path.is_file():
        raise SystemExit(f"Stage 2 manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Stage 2 manifest cannot be read: {exc}") from exc
    try:
        validate_pending_analysis_request(
            manifest,
            args.gate_target,
            summary_path,
        )
    except AnalysisError as exc:
        raise SystemExit(str(exc)) from exc
    base = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate_target_additional_steps": args.gate_target,
        "evaluation_summary": display_path(evaluation_path),
        "evaluation_summary_sha256": optional_sha256(evaluation_path),
        "evaluation_action_log": display_path(action_path),
        "evaluation_action_log_sha256": optional_sha256(action_path),
        "reference_path": str(REFERENCE_PATH.relative_to(WORKSPACE_ROOT)),
        "reference_path_sha256": sha256(REFERENCE_PATH),
        "measurement_protocol": {
            "source": "direct_live_evaluation_simstate_only",
            "expected_episodes": EXPECTED_EPISODES,
            "step_period_ms": STEP_PERIOD_MS,
            "terminal_time_tolerance_ms": TERMINAL_TIME_TOLERANCE_MS,
            "known_zones": {name: list(bounds) for name, bounds in KNOWN_ZONES.items()},
            "candidate_min_samples": CANDIDATE_MIN_SAMPLES,
            "confirmed_min_samples": CONFIRMED_MIN_SAMPLES,
            "discovery_minimum_episodes": DISCOVERY_EPISODES,
            "localized_bonus_coefficient": BONUS_COEFFICIENT,
            "localized_bonus_progress_clamp": BONUS_PROGRESS_CLAMP,
        },
    }
    exit_code = 0
    try:
        if not evaluation_path.is_file():
            raise AnalysisError(
                f"Stage 2 evaluation summary is missing: {evaluation_path}"
            )
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        validate_evidence_binding(
            manifest,
            args.gate_target,
            evaluation_path,
            action_path,
            evaluation,
        )
        if not action_path.is_file():
            raise AnalysisError(f"Stage 2 action log is missing: {action_path}")
        raw_records = [
            json.loads(line)
            for line in action_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        analyzed = analyze_evaluation(
            evaluation,
            raw_records,
            reference=ReferencePath.from_csv(REFERENCE_PATH),
        )
        valid = bool(analyzed["instrumentation_valid"])
        result = {"status": "complete" if valid else "invalid", **base, **analyzed}
        if not valid:
            exit_code = 2
    except (AnalysisError, KeyError, TypeError, ValueError) as exc:
        result = {
            "status": "invalid",
            **base,
            "instrumentation_valid": False,
            "instrumentation_error": str(exc),
        }
        exit_code = 2
    write_json(summary_path, result)
    update_manifest(manifest_path, args.gate_target, summary_path, result)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"Stage 2 analysis SHA-256={sha256(summary_path)}", flush=True)
    print(
        f"Stage 2 decision={manifest['latest_stage2_assessment']['decision']}",
        flush=True,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
