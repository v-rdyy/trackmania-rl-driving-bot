from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = WORKSPACE_ROOT / "scripts" / "diagnose_wr_stage2_policy_mode.py"
SPEC = importlib.util.spec_from_file_location(
    "diagnose_wr_stage2_policy_mode", SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class WrStage2PolicyModeTests(unittest.TestCase):
    def test_configuration_is_frozen_to_stochastic_gate1m(self) -> None:
        MODULE.configure_evaluator()
        self.assertFalse(MODULE.evaluator.POLICY_DETERMINISTIC)
        self.assertEqual(MODULE.evaluator.POLICY_RANDOM_SEED, 20_260_828)
        self.assertEqual(MODULE.evaluator.POLICY_SDE_SAMPLE_FREQ, 4)
        self.assertEqual(MODULE.evaluator.EXPECTED_EPISODES, 10)
        self.assertEqual(
            MODULE.evaluator.EXPECTED_CHECKPOINT_SHA256,
            MODULE.CHECKPOINT_SHA256,
        )
        self.assertIn("policy_mode", str(MODULE.evaluator.DEFAULT_SUMMARY))

    def test_summary_binding_rejects_deterministic_result(self) -> None:
        summary = {
            "deterministic": True,
            "policy_sampling": {
                "mode": "deterministic",
                "random_seed": None,
                "use_sde": False,
                "sde_sample_freq": None,
            },
        }
        with self.assertRaisesRegex(Exception, "was not stochastic"):
            MODULE.bind_diagnostic_summary(
                summary,
                source_evidence={},
                direct_live_audit={},
            )

    def test_summary_binding_records_zero_training_interactions(self) -> None:
        summary = {
            "deterministic": False,
            "policy_sampling": {
                "mode": "stochastic",
                "random_seed": 20_260_828,
                "use_sde": True,
                "sde_sample_freq": 4,
            },
        }
        bound = MODULE.bind_diagnostic_summary(
            summary,
            source_evidence={"checkpoint_sha256": "A" * 64},
            direct_live_audit={"complete": True},
        )
        diagnostic = bound["policy_mode_diagnostic"]
        self.assertEqual(diagnostic["training_interactions"], 0)
        self.assertTrue(diagnostic["direct_live_simstate_audit"]["complete"])


if __name__ == "__main__":
    unittest.main()
