"""Evaluate V4 through the fixed replay-backed V3-comparable protocol."""

from __future__ import annotations

import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))

import evaluate_reward_v3 as evaluator
from trackmania_rl.rewards import signed_progress_efficiency_reward


def configure_evaluator() -> None:
    run_dir = WORKSPACE_ROOT / "runs" / "reward_v4"
    evaluator.EXPERIMENT_LABEL = "reward-v4"
    evaluator.EXPERIMENT_SLUG = "reward_v4"
    evaluator.PROTOCOL_LABEL = "reward_v4.md"
    evaluator.REWARD_FUNCTION = signed_progress_efficiency_reward
    evaluator.RUN_DIR = run_dir
    evaluator.DEFAULT_CHECKPOINT = (
        WORKSPACE_ROOT / "checkpoints" / "reward_v4" / "final_model.zip"
    )
    evaluator.DEFAULT_ACTION_LOG = run_dir / "evaluation_actions.jsonl"
    evaluator.DEFAULT_SUMMARY = run_dir / "evaluation_summary.json"
    evaluator.DEFAULT_REPLAY_DIR = (
        WORKSPACE_ROOT / "artifacts" / "replays" / "reward_v4_evaluation"
    )


def main() -> int:
    configure_evaluator()
    return evaluator.main()


if __name__ == "__main__":
    raise SystemExit(main())
