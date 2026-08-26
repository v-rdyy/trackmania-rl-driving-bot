from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = WORKSPACE_ROOT / "scripts" / "diagnose_wr_stage1_live.py"
SPEC = importlib.util.spec_from_file_location("diagnose_wr_stage1_live", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class WrStage1LiveDiagnosticTests(unittest.TestCase):
    def test_direct_log_fields_are_normalized_for_prerequisite_analysis(self) -> None:
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn('record["lateral_offset_from_reference"]', source)
        self.assertIn('record["vertical_offset_from_reference"]', source)
        self.assertIn('record["heading_error_radians"]', source)

    def test_diagnostic_paths_never_replace_formal_gate_outputs(self) -> None:
        paths = MODULE.diagnostic_paths(1_500_000)
        self.assertIn("live_diagnostic", paths["action_log"].name)
        self.assertIn("live_diagnostic", paths["evaluation_summary"].name)
        self.assertEqual(paths["analysis_summary"].name, "live_diagnostic_summary.json")

    def test_evaluator_retains_v4_reward_and_ten_episodes(self) -> None:
        MODULE.configure_evaluator(
            1_500_000,
            {"checkpoint_sha256": "B" * 64},
        )
        self.assertEqual(MODULE.evaluator.EXPECTED_EPISODES, 10)
        self.assertIs(
            MODULE.evaluator.REWARD_FUNCTION,
            MODULE.signed_progress_efficiency_reward,
        )
        self.assertEqual(MODULE.evaluator.EXPECTED_CHECKPOINT_SHA256, "B" * 64)


if __name__ == "__main__":
    unittest.main()
