from __future__ import annotations

import sys
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from trackmania_rl.observations import ObservationDiagnostics
from trackmania_rl.rewards import (
    RewardTransition,
    phase1_smoke_reward,
    sparse_finish_reward,
)


def transition(
    *,
    display_speed: int = 100,
    terminated: bool = False,
    truncated: bool = False,
) -> RewardTransition:
    diagnostics = ObservationDiagnostics(
        progress=10.0,
        lateral_offset=0.0,
        heading_error=0.0,
        segment_index=1,
    )
    return RewardTransition(
        previous_diagnostics=diagnostics,
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


if __name__ == "__main__":
    unittest.main()
