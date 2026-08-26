from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = WORKSPACE_ROOT / "scripts" / "evaluate_wr_stage1.py"
SPEC = importlib.util.spec_from_file_location("evaluate_wr_stage1", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class WrStage1EvaluationTests(unittest.TestCase):
    def test_gate_configuration_uses_v4_reward_and_ten_episodes(self) -> None:
        paths = MODULE.configure_evaluator(
            500_000,
            {"checkpoint_sha256": "A" * 64},
        )
        self.assertEqual(MODULE.evaluator.EXPECTED_EPISODES, 10)
        self.assertIs(
            MODULE.evaluator.REWARD_FUNCTION,
            MODULE.signed_progress_efficiency_reward,
        )
        self.assertEqual(MODULE.evaluator.EXPECTED_CHECKPOINT_SHA256, "A" * 64)
        self.assertIn("gate_00500000", str(paths["summary"]))
        self.assertIn("gate_00500000", str(paths["replay_dir"]))


if __name__ == "__main__":
    unittest.main()
