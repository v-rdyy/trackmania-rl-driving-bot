from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = WORKSPACE_ROOT / "scripts" / "train_reward_v6.py"
SPEC = importlib.util.spec_from_file_location("train_reward_v6", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RewardV6TrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        MODULE.configure_trainer()

    def test_runner_uses_pinned_v4_parent_and_distinct_v6_artifacts(self) -> None:
        self.assertEqual(MODULE.trainer.V4_STARTING_TIMESTEPS, 2_002_944)
        self.assertEqual(MODULE.trainer.ADDITIONAL_TIMESTEPS, 1_000_000)
        self.assertEqual(MODULE.trainer.TARGET_TOTAL_TIMESTEPS, 3_002_944)
        self.assertEqual(MODULE.trainer.RUN_DIR.name, "reward_v6")
        self.assertEqual(MODULE.trainer.CHECKPOINT_DIR.name, "reward_v6")
        self.assertEqual(
            MODULE.trainer.EXPECTED_V4_CHECKPOINT_SHA256,
            "6DF90018CEC877796F6865BB6CB8D1A86929D84B6826642D26001DC5871C63F2",
        )

    def test_reward_contract_freezes_frequency_term_without_v5_magnitude_cost(self) -> None:
        self.assertEqual(
            MODULE.trainer.REWARD_FUNCTION.__name__,
            "clustered_reversal_frequency_reward",
        )
        self.assertIn("reversal_t * max(0, reversals_in_2_seconds - 3)", MODULE.REWARD_CONTRACT)
        self.assertNotIn("abs(steer_t - steer_t_minus_1)", MODULE.REWARD_CONTRACT)
        self.assertEqual(MODULE.trainer.REWARD_METADATA["steering_delta_deadband"], 0.05)
        self.assertEqual(MODULE.trainer.REWARD_METADATA["reversal_window_seconds"], 2.0)
        self.assertEqual(MODULE.trainer.REWARD_METADATA["free_reversals_per_window"], 3)
        self.assertEqual(MODULE.trainer.REWARD_METADATA["reversal_frequency_coefficient"], 0.05)
        self.assertFalse(MODULE.trainer.REWARD_METADATA["v5_magnitude_penalty_included"])
        self.assertEqual(MODULE.trainer.REWARD_METADATA["training_track"], "A01-Race")

    def test_runner_uses_distinct_tensorboard_and_checkpoint_names(self) -> None:
        self.assertEqual(
            MODULE.trainer.TENSORBOARD_RUN_NAME,
            "reward_v6_clustered_reversal_frequency",
        )
        self.assertEqual(MODULE.trainer.CHECKPOINT_NAME_PREFIX, "ppo_reward_v6")


if __name__ == "__main__":
    unittest.main()
