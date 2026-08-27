from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = WORKSPACE_ROOT / "scripts" / "evaluate_wr_stage2.py"
SPEC = importlib.util.spec_from_file_location("evaluate_wr_stage2", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def live_record(episode: int) -> dict[str, object]:
    return {
        "episode": episode,
        "step": 0,
        "race_time_ms": 24_900 + episode,
        "reward_function": "localized_drift_assistance_reward",
        "reward": 50.0,
        "race_finished": True,
        "truncated": False,
        "previous_progress": 2_100.0,
        "progress": 2_110.0,
        "new_high_water_progress_delta": 10.0,
        "display_speed": 500,
        "full_simstate_available": True,
        "velocity": [0.0, 0.0, 100.0],
        "rotation_matrix": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "angular_velocity": [0.0, 0.25, 0.0],
        "slip_angle_degrees": 1.0,
        "body_up_yaw_rate": 0.25,
        "wheel_ground_contacts": [True, True, True, True],
        "wheel_sliding": [True, True, False, False],
        "ground_contact_count": 4,
        "sliding_wheel_count": 2,
    }


def evaluation_summary() -> dict[str, object]:
    return {
        "episodes": 10,
        "episodes_detail": [
            {
                "episode": episode,
                "terminal_race_time_ms": 24_900 + episode,
                "finished": True,
                "timeout": False,
                "off_track": False,
                "fallen": False,
                "stuck": False,
            }
            for episode in range(10)
        ],
    }


class WrStage2EvaluationTests(unittest.TestCase):
    def test_baseline_and_gate_targets_are_frozen(self) -> None:
        for valid in (0, 500_000, 1_000_000, 2_000_000):
            MODULE.validate_evaluation_target(valid)
        for invalid in (-1, 250_000, 2_500_000):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                MODULE.validate_evaluation_target(invalid)

    def test_target_zero_configures_pinned_base_without_training(self) -> None:
        gate = {
            "checkpoint_sha256": MODULE.EXPECTED_INITIAL_CHECKPOINT_SHA256,
        }
        paths = MODULE.configure_evaluator(0, gate)
        self.assertEqual(MODULE.evaluator.DEFAULT_CHECKPOINT, MODULE.INITIAL_CHECKPOINT)
        self.assertEqual(MODULE.evaluator.EXPECTED_EPISODES, 10)
        self.assertIs(
            MODULE.evaluator.REWARD_FUNCTION,
            MODULE.localized_drift_assistance_reward,
        )
        self.assertEqual(
            MODULE.evaluator.EXPECTED_CHECKPOINT_SHA256,
            MODULE.EXPECTED_INITIAL_CHECKPOINT_SHA256,
        )
        self.assertIn("gate_00000000", str(paths["summary"]))
        self.assertIn("gate_00000000", str(paths["replay_dir"]))
        self.assertIn("baseline", str(MODULE.evaluator.DEFAULT_RUN_TAG))

    def test_trained_gate_configuration_stays_in_stage2_paths(self) -> None:
        paths = MODULE.configure_evaluator(
            500_000,
            {"checkpoint_sha256": "A" * 64},
        )
        self.assertEqual(MODULE.evaluator.EXPECTED_CHECKPOINT_SHA256, "A" * 64)
        self.assertIn("wr_chase_stage2", str(paths["action_log"]))
        self.assertIn("gate_00500000", str(paths["summary"]))
        self.assertIn("wr_chase_stage2", str(paths["replay_dir"]))

    def test_complete_live_records_pass_without_replay_fallback(self) -> None:
        records = [live_record(episode) for episode in range(10)]
        audit = MODULE.validate_direct_live_records(records, evaluation_summary())
        self.assertTrue(audit["complete"])
        self.assertEqual(audit["source"], "direct_live_evaluation_simstate")
        self.assertEqual(audit["episodes"], 10)
        self.assertTrue(audit["terminal_outcomes_matched"])
        self.assertFalse(audit["replay_fallback_used"])

    def test_incomplete_live_simstate_is_a_hard_failure(self) -> None:
        records = [live_record(episode) for episode in range(10)]
        records[3]["full_simstate_available"] = False
        with self.assertRaisesRegex(Exception, "incomplete SimState"):
            MODULE.validate_direct_live_records(records, evaluation_summary())

    def test_terminal_outcome_must_match_live_record(self) -> None:
        records = [live_record(episode) for episode in range(10)]
        records[7]["race_finished"] = False
        with self.assertRaisesRegex(Exception, "live finish does not match"):
            MODULE.validate_direct_live_records(records, evaluation_summary())

    def test_stage2_log_requires_bonus_reconstruction_fields(self) -> None:
        records = [live_record(episode) for episode in range(10)]
        del records[0]["new_high_water_progress_delta"]
        with self.assertRaisesRegex(Exception, "new_high_water_progress_delta"):
            MODULE.validate_direct_live_records(records, evaluation_summary())

    def test_wrapper_binds_generic_summary_to_direct_live_gate(self) -> None:
        original = {"episodes": 10}
        bound = MODULE.bind_stage2_summary(500_000, original)
        self.assertEqual(bound["gate_target_additional_steps"], 500_000)
        self.assertEqual(
            bound["measurement_source"], "direct_live_evaluation_simstate"
        )
        self.assertFalse(bound["replay_telemetry_fallback_used"])
        self.assertNotIn("gate_target_additional_steps", original)


if __name__ == "__main__":
    unittest.main()
