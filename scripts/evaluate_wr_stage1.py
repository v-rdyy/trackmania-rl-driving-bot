"""Run the frozen ten-episode evaluation for one WR-chase Stage 1 gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))

import evaluate_reward_v3 as evaluator
from train_reward_v3 import sha256, write_json
from train_wr_stage1 import (
    MANIFEST_PATH,
    RUN_DIR,
    gate_checkpoint,
    gate_slug,
    gate_summary,
    validate_target,
)
from trackmania_rl.rewards import signed_progress_efficiency_reward
from trackmania_rl.tmi_bridge import ProtocolError

EXPECTED_EPISODES = 10


def parse_gate_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gate-target", type=int, required=True)
    return parser.parse_known_args()


def load_training_gate(target: int) -> dict[str, Any]:
    path = gate_summary(target)
    if not path.is_file():
        raise ProtocolError(f"Stage 1 training summary is missing: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ProtocolError(f"Stage 1 training gate is not complete: {path}")
    if int(summary["gate_target_additional_steps"]) != target:
        raise ProtocolError("Stage 1 training summary gate does not match request")
    checkpoint = gate_checkpoint(target)
    if not checkpoint.is_file():
        raise ProtocolError(f"Stage 1 gate checkpoint is missing: {checkpoint}")
    if sha256(checkpoint) != summary["checkpoint_sha256"]:
        raise ProtocolError("Stage 1 gate checkpoint hash changed before evaluation")
    return summary


def configure_evaluator(target: int, training: dict[str, Any]) -> dict[str, Path]:
    slug = gate_slug(target)
    action_log = RUN_DIR / "gates" / f"{slug}_evaluation_actions.jsonl"
    summary = RUN_DIR / "gates" / f"{slug}_evaluation.json"
    replay_dir = (
        WORKSPACE_ROOT / "artifacts" / "replays" / "wr_chase_stage1" / slug
    )
    evaluator.EXPERIMENT_LABEL = f"WR-chase Stage 1 {slug}"
    evaluator.EXPERIMENT_SLUG = "wr_chase_stage1"
    evaluator.PROTOCOL_LABEL = "wr-chase-plan.md"
    evaluator.REWARD_FUNCTION = signed_progress_efficiency_reward
    evaluator.RUN_DIR = RUN_DIR
    evaluator.DEFAULT_CHECKPOINT = gate_checkpoint(target)
    evaluator.DEFAULT_ACTION_LOG = action_log
    evaluator.DEFAULT_SUMMARY = summary
    evaluator.DEFAULT_REPLAY_DIR = replay_dir
    evaluator.EXPECTED_EPISODES = EXPECTED_EPISODES
    evaluator.DEFAULT_RUN_TAG = slug
    evaluator.EXPECTED_CHECKPOINT_SHA256 = str(training["checkpoint_sha256"])
    return {
        "action_log": action_log,
        "summary": summary,
        "replay_dir": replay_dir,
    }


def record_evaluation(target: int, paths: dict[str, Path]) -> None:
    if not MANIFEST_PATH.is_file():
        raise ProtocolError("Stage 1 manifest disappeared during evaluation")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    matching = [
        gate
        for gate in manifest.get("gates", [])
        if int(gate["target_additional_steps"]) == target
    ]
    if len(matching) != 1:
        raise ProtocolError("Stage 1 manifest does not contain exactly one gate")
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
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
        }
    )
    manifest["status"] = "awaiting_gate_telemetry"
    write_json(MANIFEST_PATH, manifest)


def main() -> int:
    args, remaining = parse_gate_args()
    validate_target(args.gate_target)
    training = load_training_gate(args.gate_target)
    paths = configure_evaluator(args.gate_target, training)
    sys.argv = [sys.argv[0], *remaining]
    result = evaluator.main()
    if result == 0:
        record_evaluation(args.gate_target, paths)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
