from __future__ import annotations

import unittest

from scripts.retrospect_reward_v2 import retention_accounting


def row(
    length: int,
    cause: str,
) -> dict[str, str]:
    return {
        "l": str(length),
        "race_finished": str(cause == "finish"),
        "timeout": str(cause == "timeout"),
        "fallen": str(cause == "fall"),
        "off_track": str(cause == "off_track"),
    }


class RewardV2RetrospectiveTests(unittest.TestCase):
    def test_retention_partition_excludes_checkpoint_crossing_episode(self) -> None:
        interrupted = [
            row(40, "fall"),
            row(70, "timeout"),
            row(20, "finish"),
        ]
        observed = interrupted + [row(30, "finish")]

        result = retention_accounting(observed, interrupted, checkpoint_steps=100)

        self.assertEqual(result["retained"]["episodes"], 2)
        self.assertEqual(result["retained"]["finishes"], 1)
        self.assertEqual(
            result["discarded_or_boundary_crossing"]["episodes"],
            2,
        )
        self.assertEqual(
            result["checkpoint_boundary"]["episode_start_step"],
            40,
        )
        self.assertEqual(
            result["checkpoint_boundary"]["episode_end_step"],
            110,
        )

    def test_retention_partition_rejects_nonprefix_snapshot(self) -> None:
        with self.assertRaisesRegex(ValueError, "not an exact prefix"):
            retention_accounting(
                [row(10, "fall")],
                [row(10, "timeout")],
                checkpoint_steps=5,
            )


if __name__ == "__main__":
    unittest.main()
