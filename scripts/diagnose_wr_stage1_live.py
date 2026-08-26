"""Resolve a Stage 1 replay-fidelity block with direct live SimState logging."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))

import analyze_v4_wr_prerequisites as prerequisite_analysis
import analyze_wr_stage1_gate as stage_analysis
import evaluate_reward_v3 as evaluator
from evaluate_wr_stage1 import EXPECTED_EPISODES, load_training_gate
from train_reward_v3 import sha256, write_json
from train_wr_stage1 import MANIFEST_PATH, RUN_DIR, gate_checkpoint, gate_slug, validate_target
from trackmania_rl.observations import ReferencePath
from trackmania_rl.rewards import signed_progress_efficiency_reward
from trackmania_rl.tmi_bridge import ProtocolError

ANALYSIS_ROOT = WORKSPACE_ROOT / "artifacts" / "analysis" / "wr_chase_stage1"
REFERENCE_PATH = WORKSPACE_ROOT / "data" / "tracks" / "a01_reference_path.csv"


def parse_gate_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gate-target", type=int, required=True)
    return parser.parse_known_args()


def diagnostic_paths(target: int) -> dict[str, Path]:
    slug = gate_slug(target)
    return {
        "action_log": RUN_DIR / "gates" / f"{slug}_live_diagnostic_actions.jsonl",
        "evaluation_summary": RUN_DIR
        / "gates"
        / f"{slug}_live_diagnostic_evaluation.json",
        "replay_dir": WORKSPACE_ROOT
        / "artifacts"
        / "replays"
        / "wr_chase_stage1"
        / f"{slug}_live_diagnostic",
        "analysis_summary": ANALYSIS_ROOT
        / slug
        / "live_diagnostic_summary.json",
    }


def configure_evaluator(
    target: int,
    training: dict[str, Any],
) -> dict[str, Path]:
    paths = diagnostic_paths(target)
    evaluator.EXPERIMENT_LABEL = f"WR-chase Stage 1 {gate_slug(target)} live diagnostic"
    evaluator.EXPERIMENT_SLUG = "wr_chase_stage1_live_diagnostic"
    evaluator.PROTOCOL_LABEL = "wr-chase-plan.md live instrumentation diagnostic"
    evaluator.REWARD_FUNCTION = signed_progress_efficiency_reward
    evaluator.RUN_DIR = RUN_DIR
    evaluator.DEFAULT_CHECKPOINT = gate_checkpoint(target)
    evaluator.DEFAULT_ACTION_LOG = paths["action_log"]
    evaluator.DEFAULT_SUMMARY = paths["evaluation_summary"]
    evaluator.DEFAULT_REPLAY_DIR = paths["replay_dir"]
    evaluator.EXPECTED_EPISODES = EXPECTED_EPISODES
    evaluator.DEFAULT_RUN_TAG = f"{gate_slug(target)}_live_diag"
    evaluator.EXPECTED_CHECKPOINT_SHA256 = str(training["checkpoint_sha256"])
    return paths


def normalized_records(path: Path) -> dict[int, list[dict[str, Any]]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grouped: dict[int, list[dict[str, Any]]] = {
        episode: [] for episode in range(EXPECTED_EPISODES)
    }
    for record in records:
        grouped[int(record["episode"])].append(record)
    result: dict[int, list[dict[str, Any]]] = {}
    for episode, episode_records in grouped.items():
        result[episode], _ = evaluator.evaluated_race_records(episode_records)
        if not result[episode]:
            raise ProtocolError(f"live diagnostic episode {episode + 1} is empty")
        if not all(
            bool(record.get("full_simstate_available")) for record in result[episode]
        ):
            raise ProtocolError(
                f"live diagnostic episode {episode + 1} lacks full SimState"
            )
    return result


def analyze_live_diagnostic(
    target: int,
    paths: dict[str, Path],
) -> dict[str, Any]:
    evaluation = json.loads(
        paths["evaluation_summary"].read_text(encoding="utf-8")
    )
    records = normalized_records(paths["action_log"])
    reference = ReferencePath.from_csv(REFERENCE_PATH)
    cases: list[dict[str, Any]] = []
    for episode in range(EXPECTED_EPISODES):
        episode_records = records[episode]
        prerequisite: dict[str, Any] | None
        try:
            prerequisite = prerequisite_analysis.analyze_records(
                episode_records,
                reference,
            )
        except ValueError:
            prerequisite = None
        cases.append(
            {
                "episode": episode + 1,
                "records": len(episode_records),
                "finished": bool(episode_records[-1]["race_finished"]),
                "terminal_race_time_ms": int(episode_records[-1]["race_time_ms"]),
                "prerequisites": prerequisite,
                "replay_fidelity": {
                    "trusted": True,
                    "source": "direct_live_evaluation_simstate",
                },
                "drift": stage_analysis.drift_metrics(episode_records),
            }
        )
    result = {
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "instrumentation-only deterministic rerun after input replay diverged; "
            "does not replace the frozen gate reliability/lap-time evaluation"
        ),
        "gate_target_additional_steps": target,
        "checkpoint": str(gate_checkpoint(target).relative_to(WORKSPACE_ROOT)),
        "checkpoint_sha256": sha256(gate_checkpoint(target)),
        "formal_evaluation_summary": str(
            stage_analysis.evaluation_summary_path(target).relative_to(WORKSPACE_ROOT)
        ),
        "formal_evaluation_summary_sha256": sha256(
            stage_analysis.evaluation_summary_path(target)
        ),
        "diagnostic_evaluation_summary": str(
            paths["evaluation_summary"].relative_to(WORKSPACE_ROOT)
        ),
        "diagnostic_evaluation_summary_sha256": sha256(
            paths["evaluation_summary"]
        ),
        "diagnostic_action_log": str(paths["action_log"].relative_to(WORKSPACE_ROOT)),
        "diagnostic_action_log_sha256": sha256(paths["action_log"]),
        "diagnostic_lap_metrics_not_used_for_formal_gate": stage_analysis.lap_metrics(
            evaluation
        ),
        "measurement_protocol": {
            "source": "direct TrackmaniaEnv action-step SimState",
            "step_period_ms": 100,
            "known_zones": {
                name: list(bounds) for name, bounds in stage_analysis.KNOWN_ZONES.items()
            },
            "candidate_min_samples": stage_analysis.CANDIDATE_MIN_SAMPLES,
            "confirmed_min_samples": stage_analysis.CONFIRMED_MIN_SAMPLES,
            "minimum_display_speed": stage_analysis.MIN_DISPLAY_SPEED,
            "minimum_abs_slip_angle_degrees": (
                stage_analysis.MIN_ABS_SLIP_ANGLE_DEGREES
            ),
            "minimum_abs_body_up_yaw_rate": stage_analysis.MIN_ABS_YAW_RATE,
            "minimum_ground_contacts": stage_analysis.MIN_GROUND_CONTACTS,
            "candidate_minimum_sliding_wheels": (
                stage_analysis.CANDIDATE_MIN_SLIDING_WHEELS
            ),
            "confirmed_minimum_sliding_wheels": (
                stage_analysis.CONFIRMED_MIN_SLIDING_WHEELS
            ),
        },
        "cases": cases,
        "discovery": stage_analysis.aggregate_discovery(cases),
    }
    write_json(paths["analysis_summary"], result)
    return result


def record_resolution(
    target: int,
    paths: dict[str, Path],
    result: dict[str, Any],
) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    matching = [
        gate
        for gate in manifest.get("gates", [])
        if int(gate["target_additional_steps"]) == target
    ]
    if len(matching) != 1:
        raise ProtocolError("manifest does not contain exactly one diagnostic gate")
    gate = matching[0]
    gate.update(
        {
            "replay_telemetry_summary": gate.get("telemetry_summary"),
            "replay_telemetry_summary_sha256": gate.get("telemetry_summary_sha256"),
            "replay_telemetry_conclusion_valid": gate.get(
                "telemetry_conclusion_valid"
            ),
            "live_diagnostic_status": "complete",
            "live_diagnostic_evaluation_summary": str(
                paths["evaluation_summary"].relative_to(WORKSPACE_ROOT)
            ),
            "live_diagnostic_evaluation_summary_sha256": sha256(
                paths["evaluation_summary"]
            ),
            "live_diagnostic_summary": str(
                paths["analysis_summary"].relative_to(WORKSPACE_ROOT)
            ),
            "live_diagnostic_summary_sha256": sha256(paths["analysis_summary"]),
            "telemetry_conclusion_valid": bool(
                result["discovery"]["telemetry_conclusion_valid"]
            ),
            "telemetry_discovery_triggered": bool(
                result["discovery"]["telemetry_discovery_triggered"]
            ),
        }
    )
    decision = stage_analysis.assess_stage1(manifest)
    gate["stage1_decision"] = decision["decision"]
    manifest["latest_stage1_assessment"] = decision
    manifest["status"] = {
        "continue": "ready_for_next_gate",
        "review_discovery": "discovery_requires_visual_review",
        "instrumentation_required": "instrumentation_required",
        "plateau": "plateau_detected",
        "timebox_inconclusive": "timebox_complete_inconclusive",
    }[decision["decision"]]
    write_json(MANIFEST_PATH, manifest)


def main() -> int:
    args, remaining = parse_gate_args()
    validate_target(args.gate_target)
    training = load_training_gate(args.gate_target)
    paths = configure_evaluator(args.gate_target, training)
    sys.argv = [sys.argv[0], *remaining]
    outcome = evaluator.main()
    if outcome != 0:
        return outcome
    result = analyze_live_diagnostic(args.gate_target, paths)
    record_resolution(args.gate_target, paths, result)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    print(f"live diagnostic SHA-256={sha256(paths['analysis_summary'])}", flush=True)
    print(
        f"Stage 1 decision={manifest['latest_stage1_assessment']['decision']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
