from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from trackmania_rl.observations import (
    OBSERVATION_SIZE,
    ReferencePath,
    build_observation,
)


class ObservationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = ReferencePath(
            distances=np.asarray([0.0, 10.0, 20.0]),
            points=np.asarray(
                [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]
            ),
        )
        self.rotation_forward_x = np.asarray(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        )

    def test_projection_reports_progress_and_signed_right_offset(self) -> None:
        projection = self.path.project(np.asarray([5.0, 0.0, -2.0]))

        self.assertAlmostEqual(projection.progress, 5.0)
        self.assertAlmostEqual(projection.lateral_offset, 2.0)
        self.assertAlmostEqual(projection.vertical_offset, 0.0)
        np.testing.assert_allclose(projection.point, [5.0, 0.0, 0.0])

    def test_observation_has_documented_shape_order_and_normalization(self) -> None:
        state = SimpleNamespace(
            position=np.asarray([5.0, 0.0, -2.0]),
            velocity=np.asarray([10.0, 0.0, 0.0]),
            rotation_matrix=self.rotation_forward_x,
            display_speed=100,
        )

        observation, diagnostics = build_observation(state, self.path)

        self.assertEqual(observation.shape, (OBSERVATION_SIZE,))
        self.assertEqual(observation.dtype, np.float32)
        np.testing.assert_allclose(observation[:6], [0.1, 0.1, 0.0, 0.0, 0.0, 0.04])
        np.testing.assert_allclose(observation[6:10], [0.1, -0.02, 0.15, -0.02])
        self.assertAlmostEqual(diagnostics.progress, 5.0)

    def test_observation_rejects_nonfinite_state(self) -> None:
        state = SimpleNamespace(
            position=np.asarray([np.nan, 0.0, 0.0]),
            velocity=np.zeros(3),
            rotation_matrix=self.rotation_forward_x,
            display_speed=0,
        )

        with self.assertRaisesRegex(ValueError, "nonfinite"):
            build_observation(state, self.path)


if __name__ == "__main__":
    unittest.main()
