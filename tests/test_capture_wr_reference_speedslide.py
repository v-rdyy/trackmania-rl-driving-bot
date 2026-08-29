from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))

import capture_wr_reference_speedslide as module


def record(
    *,
    race_time_ms: int,
    progress: float,
    position: list[float] | None = None,
    speed: int = 400,
    contacts: int = 4,
    sliding: int = 2,
    slip: float = 2.0,
    yaw: float = 0.5,
    finished: bool = False,
) -> dict[str, object]:
    return {
        "race_time_ms": race_time_ms,
        "race_finished": finished,
        "position": position or [float(race_time_ms) / 100.0, 0.0, 0.0],
        "progress": progress,
        "display_speed": speed,
        "ground_contact_count": contacts,
        "sliding_wheel_count": sliding,
        "slip_angle_degrees": slip,
        "body_up_yaw_rate": yaw,
        "new_high_water_progress_delta": 10.0,
        "full_simstate_available": True,
    }


class WrReferenceSpeedslideTests(unittest.TestCase):
    def test_frozen_source_identity(self) -> None:
        self.assertEqual(module.EXPECTED_GHOST_LOGIN, "fwo_axell")
        self.assertEqual(module.EXPECTED_RACE_TIME_MS, 23_770)
        self.assertEqual(module.FINISH_FIDELITY_TOLERANCE_MS, 100)
        self.assertEqual(module.POSITION_FIDELITY_TOLERANCE_UNITS, 1.0)

    def test_zone_summary_requires_joint_conditions_on_same_sample(self) -> None:
        rows = [
            record(race_time_ms=1000, progress=700.0, sliding=0),
            record(race_time_ms=1100, progress=710.0, speed=349),
            record(race_time_ms=1200, progress=720.0),
            record(race_time_ms=1300, progress=730.0),
        ]
        summary = module.summarize_zone(rows)
        self.assertEqual(summary["condition_counts"]["speed"], 3)
        self.assertEqual(summary["condition_counts"]["sliding"], 3)
        self.assertEqual(summary["joint_counts"]["all_dynamics_conditions"], 2)
        self.assertEqual(summary["joint_counts"]["positive_bonus_eligible"], 2)
        self.assertEqual(len(summary["all_condition_windows"]), 1)
        self.assertEqual(summary["all_condition_windows"][0]["samples"], 2)

    def test_fidelity_accepts_exact_trajectory_and_rejects_large_error(self) -> None:
        rows = [
            record(race_time_ms=0, progress=700.0, position=[0.0, 0.0, 0.0]),
            record(race_time_ms=100, progress=1200.0, position=[1.0, 0.0, 0.0]),
            record(
                race_time_ms=23_800,
                progress=2200.0,
                position=[2.0, 0.0, 0.0],
                finished=True,
            ),
        ]
        source = {0: [0.0, 0.0, 0.0], 100: [1.0, 0.0, 0.0]}
        accepted = module.fidelity_metrics(rows, source)
        self.assertTrue(accepted["accepted_as_wr_reference"])

        rows[1]["position"] = [2.01, 0.0, 0.0]
        rejected = module.fidelity_metrics(rows, source)
        self.assertFalse(rejected["accepted_as_wr_reference"])
        self.assertGreater(
            rejected["overall_position_error_units"]["maximum"], 1.0
        )

    def test_fidelity_rejects_finish_outside_frozen_tolerance(self) -> None:
        rows = [
            record(race_time_ms=0, progress=700.0, position=[0.0, 0.0, 0.0]),
            record(
                race_time_ms=24_000,
                progress=1200.0,
                position=[1.0, 0.0, 0.0],
                finished=True,
            ),
        ]
        result = module.fidelity_metrics(rows, {0: [0.0, 0.0, 0.0]})
        self.assertFalse(result["accepted_as_wr_reference"])
        self.assertEqual(result["finish_absolute_error_ms"], 230)

    def test_analysis_preserves_an_unentered_zone_as_rejection_evidence(self) -> None:
        rows = [
            record(
                race_time_ms=0,
                progress=10.0,
                position=[0.0, 0.0, 0.0],
                finished=True,
            )
        ]
        result = module.analyze_records(rows, {0: [0.0, 0.0, 0.0]})
        self.assertFalse(result["zones"]["first_turn"]["entered"])
        self.assertFalse(result["zones"]["final_corner"]["entered"])
        self.assertFalse(result["fidelity"]["accepted_as_wr_reference"])

    def test_state_record_rejects_incomplete_simstate(self) -> None:
        reference = SimpleNamespace(
            project=lambda position: SimpleNamespace(
                progress=700.0,
                lateral_offset=0.0,
                vertical_offset=0.0,
                tangent_xz=np.array([0.0, 1.0]),
            )
        )
        state = SimpleNamespace(
            race_time=100,
            display_speed=400,
            position=np.zeros(3),
            rotation_matrix=np.eye(3),
        )
        original = module.TrackmaniaEnv.__dict__["_full_simstate_log"]
        module.TrackmaniaEnv._full_simstate_log = staticmethod(
            lambda state: {"full_simstate_available": False}
        )
        try:
            with self.assertRaisesRegex(module.ProtocolError, "complete SimState"):
                module.state_record(
                    state,
                    reference,
                    race_time_ms=100,
                    race_finished=False,
                )
        finally:
            module.TrackmaniaEnv._full_simstate_log = original


if __name__ == "__main__":
    unittest.main()
