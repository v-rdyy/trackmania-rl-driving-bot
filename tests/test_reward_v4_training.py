from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = WORKSPACE_ROOT / "scripts" / "train_reward_v4.py"
SPEC = importlib.util.spec_from_file_location("train_reward_v4", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RewardV4TrainingTests(unittest.TestCase):
    def test_runner_freezes_distinct_v4_artifacts_and_budget(self) -> None:
        self.assertEqual(MODULE.ADDITIONAL_TIMESTEPS, 1_000_000)
        self.assertEqual(MODULE.V3_STARTING_TIMESTEPS, 1_001_472)
        self.assertEqual(MODULE.TARGET_TOTAL_TIMESTEPS, 2_001_472)
        self.assertEqual(
            MODULE.TENSORBOARD_RUN_NAME,
            "reward_v4_signed_progress_efficiency",
        )
        self.assertEqual(MODULE.RUN_DIR.name, "reward_v4")

    def test_runner_freezes_corrected_failure_thresholds(self) -> None:
        self.assertEqual(MODULE.STUCK_WINDOW_MS, 2_000)
        self.assertEqual(MODULE.STUCK_PROGRESS_GAIN_UNITS, 1.0)
        self.assertEqual(MODULE.STUCK_WORLD_DISTANCE_UNITS, 2.0)

    def test_v3_checkpoint_gate_accepts_only_the_pinned_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "final_model.zip"
            checkpoint.write_bytes(b"pinned V3 checkpoint")
            expected_hash = MODULE.sha256(checkpoint)
            original = (
                MODULE.WORKSPACE_ROOT,
                MODULE.V3_CHECKPOINT,
                MODULE.EXPECTED_V3_CHECKPOINT_SHA256,
            )
            try:
                MODULE.WORKSPACE_ROOT = root
                MODULE.V3_CHECKPOINT = checkpoint
                MODULE.EXPECTED_V3_CHECKPOINT_SHA256 = expected_hash
                result = MODULE.verify_v3_checkpoint()
                MODULE.EXPECTED_V3_CHECKPOINT_SHA256 = "0" * 64
                with self.assertRaisesRegex(Exception, "hash changed"):
                    MODULE.verify_v3_checkpoint()
            finally:
                (
                    MODULE.WORKSPACE_ROOT,
                    MODULE.V3_CHECKPOINT,
                    MODULE.EXPECTED_V3_CHECKPOINT_SHA256,
                ) = original

            self.assertEqual(result["sha256"], expected_hash)
            self.assertEqual(result["num_timesteps"], 1_001_472)

    def test_reward_contract_records_every_approved_term(self) -> None:
        self.assertIn("clip(progress_delta, -20, 20) / 10", MODULE.REWARD_CONTRACT)
        self.assertIn("- 0.10", MODULE.REWARD_CONTRACT)
        self.assertIn("+ 50", MODULE.REWARD_CONTRACT)
        self.assertIn("- 250", MODULE.REWARD_CONTRACT)


if __name__ == "__main__":
    unittest.main()
