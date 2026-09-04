from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluate_wr_continuous_checkpoints.py"
SPEC = importlib.util.spec_from_file_location("continuous_retrospective", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RetrospectiveTests(unittest.TestCase):
    def test_inventory_is_baseline_plus_seven_clean_checkpoints(self):
        inventory = MODULE.checkpoint_inventory()
        self.assertEqual(inventory[0]["label"], "base")
        self.assertEqual([x["additional_interactions"] for x in inventory[1:]],
                         [251904, 501760, 751616, 1001472, 1251328, 1501184, 1751040])

    def test_evaluator_configuration_is_deterministic_live_a01(self):
        item = MODULE.checkpoint_inventory()[0]
        MODULE.configure(item)
        self.assertTrue(MODULE.evaluator.POLICY_DETERMINISTIC)
        self.assertEqual(MODULE.evaluator.SIMULATION_SPEED, 100.0)
        self.assertIsNone(MODULE.evaluator.MAP_TO_LOAD)
        self.assertTrue(MODULE.evaluator.AUTO_RESPAWN_ON_CONNECT)
        self.assertFalse(MODULE.evaluator.WAIT_FOR_RACE_START_ON_CONNECT)
        self.assertIs(MODULE.evaluator.REWARD_FUNCTION, MODULE.signed_progress_efficiency_reward)

    def test_completed_base_evaluation_can_resume_without_overwrite(self):
        base = MODULE.checkpoint_inventory()[0]
        result = MODULE.evaluate_item(base)
        self.assertEqual(result["label"], "base")
        self.assertEqual(result["finishes"], 10)
        self.assertIn("oscillation_detected_episodes", result["precision"])


if __name__ == "__main__":
    unittest.main()
