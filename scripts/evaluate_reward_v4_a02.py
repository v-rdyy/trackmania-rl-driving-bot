"""Measure the frozen V4 policy on A02 without retraining or adaptation."""

from __future__ import annotations

import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))

import evaluate_reward_v3 as evaluator
from trackmania_rl.rewards import signed_progress_efficiency_reward

V4_CHECKPOINT_SHA256 = (
    "6DF90018CEC877796F6865BB6CB8D1A86929D84B6826642D26001DC5871C63F2"
)


def configure_evaluator() -> None:
    run_dir = WORKSPACE_ROOT / "runs" / "reward_v4_a02_zero_shot"
    evaluator.EXPERIMENT_LABEL = "reward-v4 A02 zero-shot"
    evaluator.EXPERIMENT_SLUG = "reward_v4_a02_zero_shot"
    evaluator.PROTOCOL_LABEL = "Decision 0011"
    evaluator.REWARD_FUNCTION = signed_progress_efficiency_reward
    evaluator.EXPECTED_EPISODES = 20
    evaluator.DEFAULT_RUN_TAG = None
    evaluator.RUN_DIR = run_dir
    evaluator.DEFAULT_CHECKPOINT = (
        WORKSPACE_ROOT / "checkpoints" / "reward_v4" / "final_model.zip"
    )
    evaluator.DEFAULT_ACTION_LOG = run_dir / "evaluation_actions.jsonl"
    evaluator.DEFAULT_SUMMARY = run_dir / "evaluation_summary.json"
    evaluator.DEFAULT_REPLAY_DIR = (
        WORKSPACE_ROOT / "artifacts" / "replays" / "reward_v4_a02_zero_shot"
    )
    evaluator.TRACK_LABEL = "A02-Race"
    evaluator.MAP_TO_LOAD = "A02-Race.Challenge.Gbx"
    evaluator.REFERENCE_PATH = (
        WORKSPACE_ROOT / "data" / "tracks" / "a02_reference_path.csv"
    )
    evaluator.HUMAN_PB_MS = None
    evaluator.FINAL_JUMP_PROGRESS = None
    evaluator.EXPECTED_CHECKPOINT_SHA256 = V4_CHECKPOINT_SHA256


def main() -> int:
    configure_evaluator()
    return evaluator.main()


if __name__ == "__main__":
    raise SystemExit(main())
