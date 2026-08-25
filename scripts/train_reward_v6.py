"""Train the pre-registered A01-only V6 policy from the pinned V4 checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))

import train_reward_v5 as trainer
from trackmania_rl.rewards import (
    V6_FREE_REVERSALS_PER_WINDOW,
    V6_REVERSAL_FREQUENCY_COEFFICIENT,
    V6_REVERSAL_WINDOW_SECONDS,
    V6_STEERING_DELTA_DEADBAND,
    clustered_reversal_frequency_reward,
)


REWARD_CONTRACT = (
    "clip(progress_delta, -20, 20) / 10 - 0.10 "
    "- 0.05 * reversal_t * max(0, reversals_in_2_seconds - 3) "
    "+ 50 on finish - 250 on verified failure; first three clustered "
    "reversals are free and quiet frames receive no reversal cost"
)
TENSORBOARD_RUN_NAME = "reward_v6_clustered_reversal_frequency"


def configure_trainer() -> None:
    run_dir = WORKSPACE_ROOT / "runs" / "reward_v6"
    checkpoint_dir = WORKSPACE_ROOT / "checkpoints" / "reward_v6"
    trainer.EXPERIMENT_LABEL = "reward-v6"
    trainer.RUN_DIR = run_dir
    trainer.CHECKPOINT_DIR = checkpoint_dir
    trainer.FINAL_CHECKPOINT = checkpoint_dir / "final_model"
    trainer.MANIFEST_PATH = run_dir / "run_manifest.json"
    trainer.SUMMARY_PATH = run_dir / "training_summary.json"
    trainer.MONITOR_PREFIX = run_dir / "training"
    trainer.REWARD_DOC = WORKSPACE_ROOT / "reward_v6.md"
    trainer.REWARD_FUNCTION = clustered_reversal_frequency_reward
    trainer.REWARD_FUNCTION_NAME = "clustered_reversal_frequency_reward"
    trainer.REWARD_CONTRACT = REWARD_CONTRACT
    trainer.REWARD_METADATA = {
        "steering_delta_deadband": V6_STEERING_DELTA_DEADBAND,
        "reversal_window_seconds": V6_REVERSAL_WINDOW_SECONDS,
        "free_reversals_per_window": V6_FREE_REVERSALS_PER_WINDOW,
        "reversal_frequency_coefficient": V6_REVERSAL_FREQUENCY_COEFFICIENT,
        "v5_magnitude_penalty_included": False,
        "training_track": "A01-Race",
    }
    trainer.TENSORBOARD_RUN_NAME = TENSORBOARD_RUN_NAME
    trainer.CHECKPOINT_NAME_PREFIX = "ppo_reward_v6"


def main() -> int:
    configure_trainer()
    return trainer.main()


if __name__ == "__main__":
    raise SystemExit(main())
