from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT))

from scripts.build_reference_path import remove_stationary_points, resample_path


class BuildReferencePathTests(unittest.TestCase):
    def test_remove_stationary_points_keeps_real_motion(self) -> None:
        positions = np.asarray(
            [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [1.0, 0.0, 0.0]]
        )

        filtered = remove_stationary_points(positions, minimum_distance=0.05)

        np.testing.assert_allclose(filtered, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    def test_resample_path_uses_fixed_arc_length_spacing_and_endpoint(self) -> None:
        positions = np.asarray([[0.0, 0.0, 0.0], [12.0, 0.0, 0.0]])

        points, total_length = resample_path(
            positions, spacing=5.0, max_segment=20.0
        )

        self.assertEqual(total_length, 12.0)
        np.testing.assert_allclose(points[:, 0], [0.0, 5.0, 10.0, 12.0])

    def test_resample_path_rejects_teleport_like_segment(self) -> None:
        positions = np.asarray([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])

        with self.assertRaisesRegex(ValueError, "teleport-like"):
            resample_path(positions, spacing=5.0, max_segment=50.0)


if __name__ == "__main__":
    unittest.main()
