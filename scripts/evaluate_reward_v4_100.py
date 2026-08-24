"""Run the preregistered 100-episode V4 scale validation."""

from __future__ import annotations

import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))

import evaluate_reward_v4 as reward_v4_evaluation


def configure_evaluator() -> None:
    reward_v4_evaluation.configure_evaluator()
    evaluator = reward_v4_evaluation.evaluator
    run_dir = WORKSPACE_ROOT / "runs" / "reward_v4"
    evaluator.PROTOCOL_LABEL = "Decision 0010"
    evaluator.EXPECTED_EPISODES = 100
    evaluator.DEFAULT_RUN_TAG = "scale100"
    evaluator.DEFAULT_ACTION_LOG = run_dir / "evaluation_100_actions.jsonl"
    evaluator.DEFAULT_SUMMARY = run_dir / "evaluation_100_summary.json"
    evaluator.DEFAULT_REPLAY_DIR = (
        WORKSPACE_ROOT / "artifacts" / "replays" / "reward_v4_evaluation_100"
    )


def main() -> int:
    configure_evaluator()
    return reward_v4_evaluation.evaluator.main()


if __name__ == "__main__":
    raise SystemExit(main())
