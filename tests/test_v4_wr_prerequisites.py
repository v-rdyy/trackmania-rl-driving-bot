from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_v4_wr_prerequisites.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_v4_wr_prerequisites", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load V4 WR prerequisite analyzer")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeReference:
    def project(self, position: np.ndarray) -> object:
        del position
        return type("Projection", (), {"tangent_xz": np.asarray([1.0, 0.0])})()


def record(
    time_ms: int,
    progress: float,
    contacts: int,
    *,
    lateral: float = 0.0,
) -> dict[str, object]:
    return {
        "race_time_ms": time_ms,
        "progress": progress,
        "ground_contact_count": contacts,
        "display_speed": progress,
        "position": [progress, 100.0 - progress / 10.0, 0.0],
        "local_forward_velocity": progress / 10.0,
        "local_right_velocity": 0.0,
        "slip_angle_degrees": 0.0,
        "body_up_yaw_rate": 0.0,
        "lateral_offset_from_reference": lateral,
        "vertical_offset_from_reference": 1.0,
        "heading_error_radians": 0.0,
        "input_steer": 0.0,
        "input_throttle": 1.0,
        "input_brake": 0.0,
    }


class V4WrPrerequisiteTests(unittest.TestCase):
    def test_replay_commands_are_joined_by_race_timestamp(self) -> None:
        events = MODULE.parse_replay_input_events(
            "0.10 steer -32768\n0.10 gas -65536\n0.20 steer 16384\n0.30 gas 32768\n"
        )
        rows = [{"race_time_ms": 150}, {"race_time_ms": 350}]
        MODULE.annotate_replay_inputs(rows, events)
        self.assertEqual(rows[0]["input_steer"], -0.5)
        self.assertEqual(rows[0]["input_throttle"], 1.0)
        self.assertEqual(rows[0]["input_brake"], 0.0)
        self.assertEqual(rows[1]["input_steer"], 0.25)
        self.assertEqual(rows[1]["input_throttle"], 0.0)
        self.assertEqual(rows[1]["input_brake"], 0.5)

    def test_clean_start_closes_every_visible_game_window(self) -> None:
        with mock.patch.object(
            MODULE,
            "close_trackmania",
            side_effect=[True, True, False],
        ) as close:
            self.assertEqual(MODULE.close_all_trackmania_windows(), 2)
        self.assertEqual(close.call_count, 3)

    def test_residual_cleanup_only_matches_the_pinned_game_executable(self) -> None:
        expected = str(MODULE.EXPECTED_TMFOREVER_EXECUTABLE)
        self.assertTrue(MODULE.is_verified_tmforever_executable(expected.upper()))
        self.assertFalse(MODULE.is_verified_tmforever_executable(None))
        self.assertFalse(
            MODULE.is_verified_tmforever_executable(
                str(Path(expected).with_name("DifferentGame.exe"))
            )
        )

    def test_selects_five_successful_time_quantiles(self) -> None:
        episodes = [
            {
                "episode": index,
                "finished": index != 5,
                "terminal_race_time_ms": 25_000 + index * 10,
                "input_replay": f"episode_{index}.txt",
                "input_replay_sha256": str(index),
            }
            for index in range(9)
        ]
        cases = MODULE.select_representative_cases({"episodes_detail": episodes})
        self.assertEqual([case.label for case in cases], list(MODULE.SELECTION_LABELS))
        self.assertEqual([case.episode for case in cases], [1, 2, 4, 7, 9])

    def test_airtime_uses_four_wheel_contact_brackets(self) -> None:
        records = [
            record(0, 250.0, 4),
            record(100, 320.0, 4, lateral=1.0),
            record(200, 400.0, 0, lateral=2.0),
            record(300, 480.0, 0, lateral=3.0),
            record(400, 540.0, 0, lateral=4.0),
            record(500, 570.0, 2, lateral=5.0),
            record(600, 610.0, 4, lateral=6.0),
            record(700, 700.0, 4),
        ]
        result = MODULE.analyze_records(records, FakeReference())
        dropdown = result["dropdown"]
        self.assertEqual(dropdown["all_wheels_airborne_sample_count"], 3)
        self.assertEqual(dropdown["airtime_estimate_ms"], 300)
        self.assertEqual(dropdown["takeoff"]["race_time_ms"], 100)
        self.assertEqual(dropdown["landing"]["race_time_ms"], 500)

    def test_turn_entry_is_interpolated_at_fixed_progress(self) -> None:
        rows = [record(100, 550.0, 4, lateral=8.0), record(200, 570.0, 4, lateral=4.0)]
        rows[0]["input_steer"] = -0.5
        rows[1]["input_steer"] = 0.5
        entry = MODULE.interpolate_at_progress(rows, 560.0)
        self.assertEqual(entry["race_time_ms"], 150.0)
        self.assertEqual(entry["lateral_offset_from_reference"], 6.0)
        self.assertEqual(entry["input_steer"], -0.5)


if __name__ == "__main__":
    unittest.main()
