from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = WORKSPACE_ROOT / "scripts" / "train_wr_stage2b.py"
SPEC = importlib.util.spec_from_file_location("train_wr_stage2b", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class WrStage2bTrainingTests(unittest.TestCase):
    def test_frozen_gate_optimizer_and_initialization_contract(self) -> None:
        self.assertEqual(MODULE.INITIAL_MODEL_TIMESTEPS, 3_506_176)
        self.assertEqual(MODULE.GATE_SIZE, 50_000)
        self.assertEqual(MODULE.CHECKPOINT_INTERVAL, 50_000)
        self.assertEqual(MODULE.TIMEBOX, 500_000)
        self.assertEqual(MODULE.TARGET_KL, 0.01)
        self.assertIs(MODULE.PPO_CLASS, MODULE.AuditedKlPPO)
        self.assertEqual(
            MODULE.EXPECTED_INITIAL_CHECKPOINT_SHA256,
            "8A06E00055886E8E671E988D5D6948C6688B74780EDB32A87C6CC213870C8326",
        )

    def test_stage2b_uses_isolated_reward_and_artifact_roots(self) -> None:
        self.assertIs(
            MODULE.REWARD_FUNCTION,
            MODULE.graduated_final_corner_assistance_reward,
        )
        self.assertEqual(
            MODULE.REWARD_FUNCTION_NAME,
            "graduated_final_corner_assistance_reward",
        )
        self.assertEqual(MODULE.RUN_DIR.name, "wr_chase_stage2b")
        self.assertEqual(MODULE.CHECKPOINT_DIR.name, "wr_chase_stage2b")
        self.assertEqual(
            MODULE.TENSORBOARD_RUN_NAME,
            "wr_chase_stage2b_graduated_final_corner",
        )
        self.assertIn("progress 1100..1410", MODULE.REWARD_CONTRACT)
        self.assertIn("speed >=400", MODULE.REWARD_CONTRACT)

    def test_shared_gate_validation_uses_stage2b_cadence_after_configuration(self) -> None:
        names = (
            "RUN_DIR",
            "GATE_SUMMARY_DIR",
            "CHECKPOINT_DIR",
            "MANIFEST_PATH",
            "MONITOR_PREFIX",
            "INITIAL_CHECKPOINT",
            "EXPECTED_INITIAL_CHECKPOINT_SHA256",
            "INITIAL_MODEL_TIMESTEPS",
            "GATE_SIZE",
            "CHECKPOINT_INTERVAL",
            "TIMEBOX",
            "TENSORBOARD_RUN_NAME",
            "REWARD_FUNCTION",
            "REWARD_FUNCTION_NAME",
            "RUN_LABEL",
            "CHECKPOINT_NAME_PREFIX",
            "PPO_CLASS",
            "TARGET_KL",
            "INITIAL_SOURCE",
            "REWARD_CONTRACT",
        )
        original = {name: getattr(MODULE.trainer, name) for name in names}
        try:
            MODULE.configure_trainer()
            for valid in (50_000, 250_000, 500_000):
                MODULE.trainer.validate_training_target(valid)
            for invalid in (0, 25_000, 550_000):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    MODULE.trainer.validate_training_target(invalid)
        finally:
            for name, value in original.items():
                setattr(MODULE.trainer, name, value)


if __name__ == "__main__":
    unittest.main()
