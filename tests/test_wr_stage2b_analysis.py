from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = WORKSPACE_ROOT / "scripts" / "analyze_wr_stage2b_gate.py"
SPEC = importlib.util.spec_from_file_location("analyze_wr_stage2b_gate", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def rewarded_record() -> dict[str, object]:
    return {
        "previous_progress": 1_100.0,
        "progress": 1_120.0,
        "lateral_offset": 0.0,
        "heading_error": 0.0,
        "vertical_offset": 0.0,
        "previous_display_speed": 450,
        "display_speed": 450,
        "terminated": False,
        "truncated": False,
        "timeout": False,
        "off_track": False,
        "fallen": False,
        "stuck": False,
        "upright_cosine": 1.0,
        "new_high_water_progress_delta": 20.0,
        "full_simstate_available": True,
        "ground_contact_count": 4,
        "sliding_wheel_count": 0,
        "slip_angle_degrees": 0.5,
        "body_up_yaw_rate": 0.2,
        "reward": 2.025,
    }


class WrStage2bAnalysisTests(unittest.TestCase):
    def test_reward_audit_reconstructs_graduated_precursor_credit(self) -> None:
        audit = MODULE.reward_and_precursor_audit({0: [rewarded_record()]})
        self.assertTrue(audit["valid"])
        self.assertAlmostEqual(audit["graduated_bonus_total"], 0.125)
        self.assertEqual(audit["rewarded_step_count"], 1)
        self.assertAlmostEqual(audit["precursor_score_p50"], 0.25)

    def test_finish_and_onset_gates_pause_immediately(self) -> None:
        no_failures = {"repeated_fixed_location_failures": []}
        safety = MODULE.decision_for(50_000, 7, 0, no_failures, True)
        onset = MODULE.decision_for(50_000, 10, 3, no_failures, True)
        self.assertEqual(safety["decision"], "safety_review")
        self.assertEqual(onset["decision"], "review_induction")

    def test_baseline_does_not_apply_trained_gate_safety_pause(self) -> None:
        failures = {
            "repeated_fixed_location_failures": [
                {"signature": "stuck@progress_2150_2175", "episodes": 3}
            ]
        }
        decision = MODULE.decision_for(0, 7, 0, failures, True)
        self.assertEqual(decision["decision"], "continue")

    def test_onset_trend_is_separate_from_precursor_trend(self) -> None:
        manifest = {
            "gates": [
                {
                    "target_additional_steps": 0,
                    "accepted_onset_episode_count": 0,
                    "precursor_score_p95": 0.2,
                },
                {"target_additional_steps": 50_000},
            ]
        }
        trend = MODULE.trend_against_prior(manifest, 50_000, 0, 0.3)
        self.assertEqual(trend["onset"], "flat")
        self.assertEqual(trend["precursor"], "up")


if __name__ == "__main__":
    unittest.main()
