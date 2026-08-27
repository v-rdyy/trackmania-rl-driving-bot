"""Run one frozen direct-live evaluation for an A01 WR-chase Stage 2 gate."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))

import evaluate_reward_v3 as evaluator
from train_reward_v3 import sha256, write_json
from train_wr_stage2 import (
    EXPECTED_INITIAL_CHECKPOINT_SHA256,
    INITIAL_CHECKPOINT,
    INITIAL_MODEL_TIMESTEPS,
    MANIFEST_PATH,
    RUN_DIR,
    TIMEBOX,
    gate_checkpoint,
    gate_slug,
    gate_summary,
    load_manifest,
    new_manifest,
    verify_initial_checkpoint,
)
from trackmania_rl.rewards import localized_drift_assistance_reward
from trackmania_rl.tmi_bridge import ProtocolError

EXPECTED_EPISODES = 10
EVALUATION_SPEED = 6.0
REQUIRED_LIVE_FIELDS = (
    "full_simstate_available",
    "previous_progress",
    "new_high_water_progress_delta",
    "velocity",
    "rotation_matrix",
    "angular_velocity",
    "slip_angle_degrees",
    "body_up_yaw_rate",
    "wheel_ground_contacts",
    "wheel_sliding",
    "ground_contact_count",
    "sliding_wheel_count",
)


def parse_gate_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gate-target", type=int, required=True)
    return parser.parse_known_args()


def validate_evaluation_target(target: int) -> None:
    if target < 0:
        raise ValueError("Stage 2 evaluation target cannot be negative")
    if target % 500_000:
        raise ValueError("Stage 2 evaluation target must be 0 or a 500000-step gate")
    if target > TIMEBOX:
        raise ValueError(f"Stage 2 evaluation target exceeds the {TIMEBOX} time box")


def load_evaluation_gate(target: int) -> dict[str, Any]:
    """Resolve target 0 to Gate 2 and later targets to Stage 2 outputs."""
    validate_evaluation_target(target)
    if target == 0:
        initialization = verify_initial_checkpoint()
        return {
            "status": "baseline",
            "gate_target_additional_steps": 0,
            "checkpoint": initialization["path"],
            "checkpoint_sha256": initialization["sha256"],
            "final_model_timesteps": INITIAL_MODEL_TIMESTEPS,
        }

    path = gate_summary(target)
    if not path.is_file():
        raise ProtocolError(f"Stage 2 training summary is missing: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ProtocolError(f"Stage 2 training gate is not complete: {path}")
    if int(summary["gate_target_additional_steps"]) != target:
        raise ProtocolError("Stage 2 training summary gate does not match request")
    checkpoint = gate_checkpoint(target)
    if not checkpoint.is_file():
        raise ProtocolError(f"Stage 2 gate checkpoint is missing: {checkpoint}")
    if sha256(checkpoint) != summary["checkpoint_sha256"]:
        raise ProtocolError("Stage 2 gate checkpoint hash changed before evaluation")
    return summary


def configure_evaluator(target: int, gate: dict[str, Any]) -> dict[str, Path]:
    slug = gate_slug(target)
    action_log = RUN_DIR / "gates" / f"{slug}_evaluation_actions.jsonl"
    summary = RUN_DIR / "gates" / f"{slug}_evaluation.json"
    replay_dir = WORKSPACE_ROOT / "artifacts" / "replays" / "wr_chase_stage2" / slug
    evaluator.EXPERIMENT_LABEL = f"WR-chase Stage 2 {slug}"
    evaluator.EXPERIMENT_SLUG = "wr_chase_stage2"
    evaluator.PROTOCOL_LABEL = "wr-chase-plan.md Stage 2"
    evaluator.REWARD_FUNCTION = localized_drift_assistance_reward
    evaluator.RUN_DIR = RUN_DIR
    evaluator.DEFAULT_CHECKPOINT = gate_checkpoint(target)
    evaluator.DEFAULT_ACTION_LOG = action_log
    evaluator.DEFAULT_SUMMARY = summary
    evaluator.DEFAULT_REPLAY_DIR = replay_dir
    evaluator.EXPECTED_EPISODES = EXPECTED_EPISODES
    evaluator.DEFAULT_RUN_TAG = f"{slug}_baseline" if target == 0 else slug
    evaluator.EXPECTED_CHECKPOINT_SHA256 = str(gate["checkpoint_sha256"])
    return {
        "action_log": action_log,
        "summary": summary,
        "replay_dir": replay_dir,
    }


def _finite_scalars(record: dict[str, Any]) -> list[float]:
    return [
        float(record["previous_progress"]),
        float(record["progress"]),
        float(record["new_high_water_progress_delta"]),
        float(record["display_speed"]),
        float(record["slip_angle_degrees"]),
        float(record["body_up_yaw_rate"]),
        *[float(value) for value in record["velocity"]],
        *[
            float(value)
            for row in record["rotation_matrix"]
            for value in row
        ],
        *[float(value) for value in record["angular_velocity"]],
    ]


def validate_direct_live_records(
    records: list[dict[str, Any]], summary: dict[str, Any]
) -> dict[str, Any]:
    """Reject incomplete or terminal-mismatched telemetry without replay fallback."""
    if not records:
        raise ProtocolError("Stage 2 evaluation action log is empty")
    if int(summary.get("episodes", -1)) != EXPECTED_EPISODES:
        raise ProtocolError("Stage 2 evaluation summary does not contain 10 episodes")
    expected_episode_ids = set(range(EXPECTED_EPISODES))
    actual_episode_ids = {int(record["episode"]) for record in records}
    if actual_episode_ids != expected_episode_ids:
        raise ProtocolError(
            "Stage 2 direct-live records do not cover exactly episodes 0 through 9"
        )

    for index, record in enumerate(records):
        missing = [field for field in REQUIRED_LIVE_FIELDS if field not in record]
        if missing:
            raise ProtocolError(
                f"Stage 2 direct-live record {index} is missing {missing}"
            )
        if record["full_simstate_available"] is not True:
            raise ProtocolError(
                f"Stage 2 direct-live record {index} has incomplete SimState"
            )
        if record.get("reward_function") != "localized_drift_assistance_reward":
            raise ProtocolError(
                f"Stage 2 record {index} used the wrong reward function"
            )
        if len(record["velocity"]) != 3 or len(record["angular_velocity"]) != 3:
            raise ProtocolError(f"Stage 2 record {index} has invalid vector dynamics")
        rotation = record["rotation_matrix"]
        if len(rotation) != 3 or any(len(row) != 3 for row in rotation):
            raise ProtocolError(f"Stage 2 record {index} has invalid rotation dynamics")
        contact_values = record["wheel_ground_contacts"]
        sliding_values = record["wheel_sliding"]
        if (
            not isinstance(contact_values, list)
            or not isinstance(sliding_values, list)
            or len(contact_values) != 4
            or len(sliding_values) != 4
            or any(type(value) is not bool for value in contact_values)
            or any(type(value) is not bool for value in sliding_values)
        ):
            raise ProtocolError(f"Stage 2 record {index} does not contain four wheels")
        contacts = list(contact_values)
        sliding = list(sliding_values)
        if int(record["ground_contact_count"]) != sum(contacts):
            raise ProtocolError(f"Stage 2 record {index} has inconsistent contacts")
        if int(record["sliding_wheel_count"]) != sum(sliding):
            raise ProtocolError(f"Stage 2 record {index} has inconsistent slide flags")
        if not all(math.isfinite(value) for value in _finite_scalars(record)):
            raise ProtocolError(f"Stage 2 record {index} has nonfinite live dynamics")

    details = {
        int(episode["episode"]): episode
        for episode in summary.get("episodes_detail", [])
    }
    if set(details) != expected_episode_ids:
        raise ProtocolError("Stage 2 summary episode detail is incomplete")
    records_by_episode = {
        episode: [
            record for record in records if int(record["episode"]) == episode
        ]
        for episode in expected_episode_ids
    }
    for episode, episode_records in records_by_episode.items():
        final = episode_records[-1]
        detail = details[episode]
        terminal_truncated = bool(
            detail["timeout"]
            or detail["off_track"]
            or detail["fallen"]
            or detail["stuck"]
        )
        if bool(final["race_finished"]) != bool(detail["finished"]):
            raise ProtocolError(
                f"Stage 2 episode {episode} live finish does not match summary"
            )
        if bool(final["truncated"]) != terminal_truncated:
            raise ProtocolError(
                f"Stage 2 episode {episode} live truncation does not match summary"
            )
        if int(final["race_time_ms"]) != int(detail["terminal_race_time_ms"]):
            raise ProtocolError(
                f"Stage 2 episode {episode} terminal time does not match summary"
            )

    return {
        "source": "direct_live_evaluation_simstate",
        "complete": True,
        "records": len(records),
        "episodes": EXPECTED_EPISODES,
        "terminal_outcomes_matched": True,
        "replay_fallback_used": False,
        "required_fields": list(REQUIRED_LIVE_FIELDS),
    }


def validate_direct_live_action_log(
    action_log: Path, evaluation_summary: dict[str, Any]
) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in action_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return validate_direct_live_records(records, evaluation_summary)


def bind_stage2_summary(target: int, summary: dict[str, Any]) -> dict[str, Any]:
    """Stamp the wrapper-owned gate identity onto the generic evaluation."""
    bound = dict(summary)
    bound.update(
        {
            "gate_target_additional_steps": target,
            "measurement_source": "direct_live_evaluation_simstate",
            "replay_telemetry_fallback_used": False,
        }
    )
    return bound


def record_evaluation(target: int, paths: dict[str, Path]) -> None:
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    live_audit = validate_direct_live_action_log(paths["action_log"], summary)
    summary = bind_stage2_summary(target, summary)
    # The generic evaluator cannot know the Stage 2 gate number. Stamp it into
    # the immutable wrapper summary before its hash is recorded in the manifest
    # so the offline analyzer can prove that all three evidence files belong to
    # the same gate.
    write_json(paths["summary"], summary)
    manifest = load_manifest()
    if target == 0:
        if not manifest:
            manifest = new_manifest(verify_initial_checkpoint())
        if manifest.get("gates"):
            raise ProtocolError("Stage 2 baseline target 0 is already recorded")
        gate: dict[str, Any] = {
            "status": "baseline_complete",
            "target_additional_steps": 0,
            "actual_additional_steps": 0,
            "model_timesteps": INITIAL_MODEL_TIMESTEPS,
            "checkpoint": str(INITIAL_CHECKPOINT.relative_to(WORKSPACE_ROOT)),
            "checkpoint_sha256": EXPECTED_INITIAL_CHECKPOINT_SHA256,
        }
        manifest["gates"].append(gate)
    else:
        if not manifest:
            raise ProtocolError("Stage 2 manifest disappeared during evaluation")
        matching = [
            item
            for item in manifest.get("gates", [])
            if int(item["target_additional_steps"]) == target
        ]
        if len(matching) != 1:
            raise ProtocolError("Stage 2 manifest does not contain exactly one gate")
        gate = matching[0]

    gate.update(
        {
            "evaluation_status": "complete",
            "evaluation_summary": str(paths["summary"].relative_to(WORKSPACE_ROOT)),
            "evaluation_summary_sha256": sha256(paths["summary"]),
            "evaluation_action_log": str(
                paths["action_log"].relative_to(WORKSPACE_ROOT)
            ),
            "evaluation_action_log_sha256": sha256(paths["action_log"]),
            "evaluation_replay_dir": str(
                paths["replay_dir"].relative_to(WORKSPACE_ROOT)
            ),
            "deterministic_finishes": int(summary["finishes"]),
            "deterministic_best_finish_time_ms": summary["best_finish_time_ms"],
            "deterministic_mean_finish_time_ms": summary[
                "average_finish_time_ms"
            ],
            "direct_live_simstate_audit": live_audit,
            "telemetry_status": "pending",
            "stage2_decision": "pending",
        }
    )
    manifest.update(
        {
            "status": "awaiting_gate_live_analysis",
            "latest_evaluated_gate": target,
        }
    )
    write_json(MANIFEST_PATH, manifest)


def main() -> int:
    args, remaining = parse_gate_args()
    validate_evaluation_target(args.gate_target)
    gate = load_evaluation_gate(args.gate_target)
    paths = configure_evaluator(args.gate_target, gate)
    sys.argv = [sys.argv[0], *remaining]
    result = evaluator.main()
    if result == 0:
        record_evaluation(args.gate_target, paths)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
