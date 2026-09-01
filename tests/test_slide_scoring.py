from __future__ import annotations

import sys
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from trackmania_rl.slide_scoring import (
    FINAL_CORNER_ZONE,
    paired_usefulness,
    score_episode,
    valid_slide_onset_sample,
)


def record(step: int, **overrides):
    value = {
        "episode": 0,
        "race_time_ms": 10_000 + step * 100,
        "progress": 1_120.0 + step * 12.0,
        "display_speed": 450.0,
        "new_high_water_progress_delta": 12.0,
        "upright_cosine": 0.99,
        "heading_error": 0.1,
        "lateral_offset": 4.0,
        "ground_contact_count": 4,
        "sliding_wheel_count": 0,
        "slip_angle_degrees": 0.3,
        "body_up_yaw_rate": 0.8,
        "timeout": False,
        "off_track": False,
        "fallen": False,
        "stuck": False,
        "truncated": False,
        "race_finished": False,
    }
    value.update(overrides)
    return value


class SlideScoringTests(unittest.TestCase):
    def test_valid_sample_uses_frozen_grounded_forward_envelope(self) -> None:
        self.assertTrue(valid_slide_onset_sample(record(0)))
        for field, invalid in (
            ("progress", FINAL_CORNER_ZONE[0] - 0.01),
            ("display_speed", 399.0),
            ("new_high_water_progress_delta", 0.0),
            ("upright_cosine", 0.79),
            ("heading_error", 0.8),
            ("lateral_offset", 20.01),
            ("ground_contact_count", 2),
        ):
            with self.subTest(field=field):
                self.assertFalse(valid_slide_onset_sample(record(0, **{field: invalid})))

    def test_primary_onset_requires_two_samples_and_two_wheels_once(self) -> None:
        records = [
            record(-1, progress=1_105.0),
            record(0, sliding_wheel_count=1, slip_angle_degrees=0.2),
            record(1, sliding_wheel_count=2, slip_angle_degrees=0.2),
            record(2),
        ]
        score = score_episode(records)["final_corner"]
        self.assertEqual(score["raw_slide_onset_count"], 1)
        self.assertEqual(score["accepted_slide_onset_count"], 1)
        # Neither slip nor yaw is a primary threshold.
        self.assertEqual(score["legacy_candidate_count"], 0)

        records[2]["sliding_wheel_count"] = 1
        score = score_episode(records)["final_corner"]
        self.assertEqual(score["raw_slide_onset_count"], 0)

    def test_safety_context_rejects_airborne_and_large_speed_loss(self) -> None:
        records = [
            record(-1, progress=1_105.0),
            record(0, sliding_wheel_count=1),
            record(1, sliding_wheel_count=2),
            record(2, ground_contact_count=2),
        ]
        score = score_episode(records)["final_corner"]
        self.assertEqual(score["raw_slide_onset_count"], 1)
        self.assertEqual(score["accepted_slide_onset_count"], 0)

        records[-1] = record(2, display_speed=424.0)
        score = score_episode(records)["final_corner"]
        self.assertEqual(score["accepted_slide_onset_count"], 0)

    def test_above_envelope_near_miss_cannot_become_onset(self) -> None:
        records = [
            record(-1, progress=1_105.0),
            record(0, slip_angle_degrees=0.51),
            record(1, slip_angle_degrees=0.75),
            record(2),
        ]
        score = score_episode(records)["final_corner"]
        self.assertEqual(score["above_envelope_near_miss_count"], 1)
        self.assertEqual(score["accepted_slide_onset_count"], 0)

    def test_legacy_confirmed_slide_remains_three_two_wheel_samples(self) -> None:
        records = [
            record(-1, progress=1_105.0),
            *[
                record(
                    step,
                    sliding_wheel_count=2,
                    slip_angle_degrees=1.2,
                    body_up_yaw_rate=0.3,
                )
                for step in range(3)
            ],
            record(3),
        ]
        score = score_episode(records)["final_corner"]
        self.assertEqual(score["legacy_candidate_count"], 1)
        self.assertEqual(score["legacy_confirmed_count"], 1)

    def test_usefulness_is_separate_from_onset(self) -> None:
        control = {
            "safe_finish": True,
            "final_corner": {"accepted_slide_onset_count": 0},
            "final_zone_traversal": {
                "exited": True,
                "traversal_time_ms": 2_000,
                "exit_speed": 460.0,
            },
        }
        treatment = {
            "safe_finish": True,
            "final_corner": {"accepted_slide_onset_count": 1},
            "final_zone_traversal": {
                "exited": True,
                "traversal_time_ms": 1_900,
                "exit_speed": 460.0,
            },
        }
        result = paired_usefulness(control, treatment)
        self.assertTrue(result["useful_candidate_before_visual_review"])

        treatment["final_zone_traversal"]["exit_speed"] = 459.0
        result = paired_usefulness(control, treatment)
        self.assertFalse(result["useful_candidate_before_visual_review"])


if __name__ == "__main__":
    unittest.main()
