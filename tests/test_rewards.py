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
    sparse_finish_reward,
)


def transition(
    *,
    display_speed: int = 100,
    previous_progress: float = 10.0,
    progress: float = 10.0,
    terminated: bool = False,
    truncated: bool = False,
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
        timed_out=truncated,
        off_track=False,
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


if __name__ == "__main__":
    unittest.main()
