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
    dense_speed_reward,
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


if __name__ == "__main__":
    unittest.main()
