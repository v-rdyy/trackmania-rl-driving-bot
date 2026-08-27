from __future__ import annotations

import sys
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from trackmania_rl.observations import ObservationDiagnostics
from trackmania_rl.rewards import (
    RewardTransition,
    clamped_forward_progress_reward,
    clustered_reversal_frequency_reward,
    dense_speed_reward,
    localized_drift_assistance_reward,
    localized_drift_bonus,
    phase1_smoke_reward,
    signed_progress_efficiency_reward,
    sparse_finish_reward,
    steering_rate_smoothness_reward,
)


def transition(
    *,
    display_speed: int = 100,
    previous_progress: float = 10.0,
    progress: float = 10.0,
    terminated: bool = False,
    truncated: bool = False,
    timed_out: bool | None = None,
    off_track: bool = False,
    fallen: bool = False,
    stuck: bool = False,
    steering_rate_change: float = 0.0,
    steering_slope_reversal: bool = False,
    steering_reversals_in_window: int = 0,
    new_high_water_progress_delta: float = 0.0,
    full_simstate_available: bool = False,
    ground_contact_count: int = 0,
    sliding_wheel_count: int = 0,
    slip_angle_degrees: float = 0.0,
    body_up_yaw_rate: float = 0.0,
) -> RewardTransition:
    previous_diagnostics = ObservationDiagnostics(
        progress=previous_progress,
        lateral_offset=0.0,
        heading_error=0.0,
        segment_index=1,
    )
    diagnostics = ObservationDiagnostics(
        progress=progress,
        lateral_offset=0.0,
        heading_error=0.0,
        segment_index=1,
    )
    return RewardTransition(
        previous_diagnostics=previous_diagnostics,
        diagnostics=diagnostics,
        display_speed=display_speed,
        elapsed_ms=100,
        terminated=terminated,
        truncated=truncated,
        timed_out=truncated if timed_out is None else timed_out,
        off_track=off_track,
        fallen=fallen,
        stuck=stuck,
        steering_rate_change=steering_rate_change,
        steering_slope_reversal=steering_slope_reversal,
        steering_reversals_in_window=steering_reversals_in_window,
        new_high_water_progress_delta=new_high_water_progress_delta,
        full_simstate_available=full_simstate_available,
        ground_contact_count=ground_contact_count,
        sliding_wheel_count=sliding_wheel_count,
        slip_angle_degrees=slip_angle_degrees,
        body_up_yaw_rate=body_up_yaw_rate,
    )


class RewardTests(unittest.TestCase):
    def test_phase1_reward_preserves_speed_and_truncation_contract(self) -> None:
        self.assertAlmostEqual(phase1_smoke_reward(transition()), 0.1)
        self.assertAlmostEqual(
            phase1_smoke_reward(transition(truncated=True)),
            -0.9,
        )

    def test_sparse_finish_reward_only_fires_on_termination(self) -> None:
        self.assertEqual(sparse_finish_reward(transition()), 0.0)
        self.assertEqual(
            sparse_finish_reward(transition(truncated=True)),
            0.0,
        )
        self.assertEqual(
            sparse_finish_reward(transition(terminated=True)),
            1.0,
        )

    def test_dense_speed_reward_has_no_terminal_shaping(self) -> None:
        self.assertEqual(dense_speed_reward(transition(display_speed=123)), 0.123)
        self.assertEqual(
            dense_speed_reward(transition(display_speed=123, truncated=True)),
            0.123,
        )
        self.assertEqual(
            dense_speed_reward(transition(display_speed=123, terminated=True)),
            0.123,
        )

    def test_v3_reward_normalizes_positive_forward_progress(self) -> None:
        self.assertAlmostEqual(
            clamped_forward_progress_reward(
                transition(previous_progress=10.0, progress=14.0)
            ),
            0.4,
        )

    def test_v3_reward_ignores_backward_progress(self) -> None:
        self.assertEqual(
            clamped_forward_progress_reward(
                transition(previous_progress=10.0, progress=7.0)
            ),
            0.0,
        )

    def test_v3_reward_clamps_projection_jumps_without_terminal_shaping(self) -> None:
        self.assertEqual(
            clamped_forward_progress_reward(
                transition(
                    previous_progress=10.0,
                    progress=25.0,
                    terminated=True,
                )
            ),
            1.0,
        )
        self.assertAlmostEqual(
            clamped_forward_progress_reward(
                transition(
                    previous_progress=10.0,
                    progress=14.0,
                    truncated=True,
                )
            ),
            0.4,
        )

    def test_v4_reward_uses_signed_progress_and_time_cost(self) -> None:
        self.assertAlmostEqual(
            signed_progress_efficiency_reward(
                transition(previous_progress=10.0, progress=14.0)
            ),
            0.3,
        )
        self.assertAlmostEqual(
            signed_progress_efficiency_reward(
                transition(previous_progress=10.0, progress=7.0)
            ),
            -0.4,
        )

    def test_v4_reward_clips_signed_progress_to_twenty_units(self) -> None:
        self.assertAlmostEqual(
            signed_progress_efficiency_reward(
                transition(previous_progress=10.0, progress=50.0)
            ),
            1.9,
        )
        self.assertAlmostEqual(
            signed_progress_efficiency_reward(
                transition(previous_progress=50.0, progress=10.0)
            ),
            -2.1,
        )

    def test_v4_reward_adds_finish_bonus_on_terminal_step(self) -> None:
        self.assertAlmostEqual(
            signed_progress_efficiency_reward(
                transition(
                    previous_progress=10.0,
                    progress=14.0,
                    terminated=True,
                )
            ),
            50.3,
        )

    def test_v4_reward_applies_failure_penalty_to_every_verified_truncation(self) -> None:
        failure_transitions = (
            transition(truncated=True, timed_out=True),
            transition(truncated=True, timed_out=False, off_track=True),
            transition(truncated=True, timed_out=False, fallen=True),
            transition(truncated=True, timed_out=False, stuck=True),
        )

        for failure in failure_transitions:
            with self.subTest(failure=failure):
                self.assertAlmostEqual(
                    signed_progress_efficiency_reward(failure),
                    -250.1,
                )

    def test_v4_finish_takes_precedence_over_truncation_defensively(self) -> None:
        self.assertAlmostEqual(
            signed_progress_efficiency_reward(
                transition(terminated=True, truncated=True)
            ),
            49.9,
        )

    def test_v5_matches_v4_when_steering_does_not_change(self) -> None:
        unchanged = transition(previous_progress=10.0, progress=14.0)

        self.assertEqual(
            steering_rate_smoothness_reward(unchanged),
            signed_progress_efficiency_reward(unchanged),
        )

    def test_v5_subtracts_only_absolute_steering_rate_change(self) -> None:
        changed = transition(
            previous_progress=10.0,
            progress=14.0,
            steering_rate_change=0.4,
        )

        self.assertAlmostEqual(
            steering_rate_smoothness_reward(changed),
            0.28,
        )

    def test_v5_preserves_v4_terminal_terms(self) -> None:
        self.assertAlmostEqual(
            steering_rate_smoothness_reward(
                transition(terminated=True, steering_rate_change=2.0)
            ),
            49.8,
        )
        self.assertAlmostEqual(
            steering_rate_smoothness_reward(
                transition(truncated=True, stuck=True, steering_rate_change=2.0)
            ),
            -250.2,
        )

    def test_v6_matches_v4_before_the_fourth_clustered_reversal(self) -> None:
        for reversal_count in range(4):
            with self.subTest(reversal_count=reversal_count):
                candidate = transition(
                    previous_progress=10.0,
                    progress=14.0,
                    steering_slope_reversal=True,
                    steering_reversals_in_window=reversal_count,
                )
                self.assertEqual(
                    clustered_reversal_frequency_reward(candidate),
                    signed_progress_efficiency_reward(candidate),
                )

    def test_v6_penalizes_frequency_only_when_a_new_reversal_occurs(self) -> None:
        fourth_reversal = transition(
            previous_progress=10.0,
            progress=14.0,
            steering_rate_change=2.0,
            steering_slope_reversal=True,
            steering_reversals_in_window=4,
        )
        quiet_frame_in_busy_window = transition(
            previous_progress=10.0,
            progress=14.0,
            steering_rate_change=2.0,
            steering_slope_reversal=False,
            steering_reversals_in_window=6,
        )

        self.assertAlmostEqual(
            clustered_reversal_frequency_reward(fourth_reversal),
            0.25,
        )
        self.assertAlmostEqual(
            clustered_reversal_frequency_reward(quiet_frame_in_busy_window),
            0.30,
        )

    def test_v6_preserves_v4_terminal_terms(self) -> None:
        self.assertAlmostEqual(
            clustered_reversal_frequency_reward(
                transition(
                    terminated=True,
                    steering_slope_reversal=True,
                    steering_reversals_in_window=5,
                )
            ),
            49.8,
        )
        self.assertAlmostEqual(
            clustered_reversal_frequency_reward(
                transition(
                    truncated=True,
                    stuck=True,
                    steering_slope_reversal=True,
                    steering_reversals_in_window=5,
                )
            ),
            -250.2,
        )

    def test_stage2_requires_live_simstate_even_outside_assisted_zones(self) -> None:
        with self.assertRaisesRegex(ValueError, "complete live SimState"):
            localized_drift_assistance_reward(
                transition(previous_progress=10.0, progress=14.0)
            )

    def test_stage2_adds_exact_bounded_bonus_at_every_frozen_threshold(self) -> None:
        candidate = transition(
            display_speed=350,
            previous_progress=680.0,
            progress=700.0,
            new_high_water_progress_delta=20.0,
            full_simstate_available=True,
            ground_contact_count=3,
            sliding_wheel_count=1,
            slip_angle_degrees=-1.0,
            body_up_yaw_rate=-0.25,
        )

        self.assertAlmostEqual(localized_drift_bonus(candidate), 0.5)
        self.assertAlmostEqual(
            localized_drift_assistance_reward(candidate),
            signed_progress_efficiency_reward(candidate) + 0.5,
        )

        stronger = transition(
            display_speed=500,
            previous_progress=680.0,
            progress=720.0,
            new_high_water_progress_delta=40.0,
            full_simstate_available=True,
            ground_contact_count=4,
            sliding_wheel_count=4,
            slip_angle_degrees=20.0,
            body_up_yaw_rate=10.0,
        )
        self.assertAlmostEqual(localized_drift_bonus(stronger), 0.5)

    def test_stage2_bonus_scales_only_with_new_high_water_progress(self) -> None:
        candidate = transition(
            display_speed=400,
            previous_progress=700.0,
            progress=710.0,
            new_high_water_progress_delta=5.0,
            full_simstate_available=True,
            ground_contact_count=4,
            sliding_wheel_count=2,
            slip_angle_degrees=2.0,
            body_up_yaw_rate=0.5,
        )

        self.assertAlmostEqual(localized_drift_bonus(candidate), 0.125)
        repeated_progress = transition(
            display_speed=400,
            previous_progress=700.0,
            progress=710.0,
            new_high_water_progress_delta=0.0,
            full_simstate_available=True,
            ground_contact_count=4,
            sliding_wheel_count=2,
            slip_angle_degrees=2.0,
            body_up_yaw_rate=0.5,
        )
        self.assertEqual(localized_drift_bonus(repeated_progress), 0.0)

    def test_stage2_bonus_is_restricted_to_both_inclusive_zones(self) -> None:
        for progress in (680.0, 930.0, 1100.0, 1410.0):
            with self.subTest(progress=progress):
                candidate = transition(
                    display_speed=400,
                    previous_progress=progress - 20.0,
                    progress=progress,
                    new_high_water_progress_delta=20.0,
                    full_simstate_available=True,
                    ground_contact_count=4,
                    sliding_wheel_count=2,
                    slip_angle_degrees=2.0,
                    body_up_yaw_rate=0.5,
                )
                self.assertAlmostEqual(localized_drift_bonus(candidate), 0.5)

        for progress in (679.999, 930.001, 1099.999, 1410.001):
            with self.subTest(progress=progress):
                candidate = transition(
                    display_speed=400,
                    previous_progress=progress - 20.0,
                    progress=progress,
                    new_high_water_progress_delta=20.0,
                    full_simstate_available=True,
                    ground_contact_count=4,
                    sliding_wheel_count=2,
                    slip_angle_degrees=2.0,
                    body_up_yaw_rate=0.5,
                )
                self.assertEqual(localized_drift_bonus(candidate), 0.0)

    def test_stage2_every_drift_attempt_gate_is_required(self) -> None:
        qualifying = {
            "display_speed": 400,
            "previous_progress": 700.0,
            "progress": 720.0,
            "new_high_water_progress_delta": 20.0,
            "full_simstate_available": True,
            "ground_contact_count": 4,
            "sliding_wheel_count": 2,
            "slip_angle_degrees": 2.0,
            "body_up_yaw_rate": 0.5,
        }
        failing_values = {
            "display_speed": 349,
            "ground_contact_count": 2,
            "sliding_wheel_count": 0,
            "slip_angle_degrees": 0.999,
            "body_up_yaw_rate": 0.249,
        }

        for field, value in failing_values.items():
            with self.subTest(field=field):
                candidate = transition(**(qualifying | {field: value}))
                self.assertEqual(localized_drift_bonus(candidate), 0.0)
                self.assertEqual(
                    localized_drift_assistance_reward(candidate),
                    signed_progress_efficiency_reward(candidate),
                )

    def test_stage2_preserves_v4_terminal_terms(self) -> None:
        qualifying = {
            "display_speed": 400,
            "previous_progress": 700.0,
            "progress": 720.0,
            "new_high_water_progress_delta": 20.0,
            "full_simstate_available": True,
            "ground_contact_count": 4,
            "sliding_wheel_count": 2,
            "slip_angle_degrees": 2.0,
            "body_up_yaw_rate": 0.5,
        }
        finished = transition(**qualifying, terminated=True)
        failed = transition(**qualifying, truncated=True, stuck=True)

        self.assertAlmostEqual(
            localized_drift_assistance_reward(finished),
            signed_progress_efficiency_reward(finished) + 0.5,
        )
        self.assertAlmostEqual(
            localized_drift_assistance_reward(failed),
            signed_progress_efficiency_reward(failed) + 0.5,
        )

    def test_stage2_rejects_invalid_wheel_counts_and_nonfinite_dynamics(self) -> None:
        base = {
            "display_speed": 400,
            "previous_progress": 700.0,
            "progress": 720.0,
            "new_high_water_progress_delta": 20.0,
            "full_simstate_available": True,
            "ground_contact_count": 4,
            "sliding_wheel_count": 2,
            "slip_angle_degrees": 2.0,
            "body_up_yaw_rate": 0.5,
        }
        for field, value in (
            ("ground_contact_count", 5),
            ("sliding_wheel_count", -1),
            ("slip_angle_degrees", float("nan")),
            ("body_up_yaw_rate", float("inf")),
            ("new_high_water_progress_delta", float("nan")),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                localized_drift_bonus(transition(**(base | {field: value})))


if __name__ == "__main__":
    unittest.main()
