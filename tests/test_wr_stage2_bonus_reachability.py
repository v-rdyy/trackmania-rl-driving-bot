from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = WORKSPACE_ROOT / "scripts" / "analyze_wr_stage2_bonus_reachability.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_wr_stage2_bonus_reachability", SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def record(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "episode": 0,
        "race_time_ms": 1000,
        "progress": 700.0,
        "display_speed": 350,
        "ground_contact_count": 3,
        "sliding_wheel_count": 1,
        "slip_angle_degrees": 1.0,
        "body_up_yaw_rate": 0.25,
        "new_high_water_progress_delta": 10.0,
        "full_simstate_available": True,
        "reward_function": "localized_drift_assistance_reward",
    }
    return base | overrides


class WrStage2BonusReachabilityTests(unittest.TestCase):
    def test_zone_boundaries_match_frozen_reward(self) -> None:
        self.assertEqual(MODULE.zone_name(680.0), "first_turn")
        self.assertEqual(MODULE.zone_name(930.0), "first_turn")
        self.assertIsNone(MODULE.zone_name(1000.0))
        self.assertEqual(MODULE.zone_name(1100.0), "final_corner")
        self.assertEqual(MODULE.zone_name(1410.0), "final_corner")

    def test_every_condition_must_be_true_on_same_sample(self) -> None:
        qualifying = record()
        self.assertTrue(all(MODULE.condition_flags(qualifying).values()))
        for field, value in (
            ("ground_contact_count", 2),
            ("display_speed", 349),
            ("sliding_wheel_count", 0),
            ("slip_angle_degrees", 0.999),
            ("body_up_yaw_rate", 0.249),
        ):
            with self.subTest(field=field):
                flags = MODULE.condition_flags(record(**{field: value}))
                self.assertFalse(all(flags.values()))

    def test_summary_exposes_joint_near_miss(self) -> None:
        records = [
            record(sliding_wheel_count=0),
            record(slip_angle_degrees=0.5, body_up_yaw_rate=0.1),
        ]
        summary = MODULE.summarize_zone(records)
        self.assertEqual(summary["samples"], 2)
        self.assertEqual(summary["joint_counts"]["all_reward_conditions"], 0)
        self.assertEqual(summary["joint_counts"]["ground_and_speed"], 2)
        self.assertEqual(summary["joint_counts"]["all_except_sliding"], 1)
        self.assertEqual(summary["maximum_simultaneously_satisfied"], 4)

    def test_live_simstate_and_reward_identity_are_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "complete live SimState"):
            MODULE.validate_record(record(full_simstate_available=False))
        with self.assertRaisesRegex(ValueError, "wrong reward function"):
            MODULE.validate_record(record(reward_function="wrong"))

    def test_trackwide_activity_distinguishes_live_sensor_from_zone_gate(self) -> None:
        records = [
            record(progress=700.0, sliding_wheel_count=0, slip_angle_degrees=0.2),
            record(progress=2000.0),
        ]
        summary = MODULE.summarize_track_wide(records)
        self.assertEqual(summary["sliding_samples"], 1)
        self.assertEqual(summary["slip_samples"], 1)
        self.assertEqual(summary["all_dynamics_samples"], 1)
        self.assertEqual(summary["outside_assisted_zone_all_dynamics_samples"], 1)


if __name__ == "__main__":
    unittest.main()
