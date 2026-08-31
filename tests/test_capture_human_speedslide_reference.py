from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))

import capture_human_speedslide_reference as module


def row(
    race_time_ms: int,
    progress: float,
    *,
    sliding: int = 2,
    finished: bool = False,
) -> dict[str, object]:
    return {
        "race_time_ms": race_time_ms,
        "race_finished": finished,
        "progress": progress,
        "position": [float(race_time_ms), 0.0, 0.0],
        "display_speed": 400,
        "ground_contact_count": 4,
        "sliding_wheel_count": sliding,
        "slip_angle_degrees": 2.0,
        "body_up_yaw_rate": 0.5,
        "new_high_water_progress_delta": 1.0,
        "full_simstate_available": True,
        "input_source": "human_live_simstate",
        "capture_wall_seconds": race_time_ms / 1000.0,
    }


class HumanSpeedslideReferenceTests(unittest.TestCase):
    def test_take_id_is_path_safe(self) -> None:
        self.assertEqual(module.validate_take_id("take_001"), "take_001")
        for value in ("", "Take1", "../take", "take 1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                module.validate_take_id(value)

    def test_downsample_derives_exact_100ms_view(self) -> None:
        records = [row(time, 700.0) for time in range(0, 201, 20)]
        sampled = module.downsample_stage2(records)
        self.assertEqual([item["race_time_ms"] for item in sampled], [0, 100, 200])

    def test_downsample_recomputes_stage2_progress_high_water(self) -> None:
        records = [row(time, float(time)) for time in range(0, 101, 20)]
        records[2]["progress"] = 120.0
        module.annotate_high_water_progress(records)
        self.assertEqual(records[-1]["new_high_water_progress_delta"], 0.0)
        sampled = module.downsample_stage2(records)
        self.assertEqual(sampled[-1]["new_high_water_progress_delta"], 100.0)

    def test_confirmed_window_requires_three_samples_and_two_sliding_wheels(self) -> None:
        records = [row(time, 700.0) for time in (1000, 1100, 1200)]
        self.assertEqual(
            len(
                module.gate_windows(
                    records,
                    minimum_sliding_wheels=2,
                    minimum_samples=3,
                    sample_period_ms=100,
                )
            ),
            1,
        )
        records[1]["sliding_wheel_count"] = 1
        self.assertEqual(
            module.gate_windows(
                records,
                minimum_sliding_wheels=2,
                minimum_samples=3,
                sample_period_ms=100,
            ),
            [],
        )

    def test_analysis_compares_both_zones_without_reward(self) -> None:
        records: list[dict[str, object]] = []
        for time in range(0, 501, 20):
            progress = (
                700.0 + time / 2.0
                if time < 300
                else 1200.0 + (time - 300) / 2.0
            )
            records.append(row(time, progress, finished=time == 500))
        result = module.analyze_human_records(records)
        self.assertTrue(result["both_zones_entered_at_100ms"])
        self.assertTrue(result["full_dynamics_gate_observed_in_both_zones_at_100ms"])
        self.assertTrue(
            result[
                "positive_reward_eligibility_observed_in_both_zones_at_100ms"
            ]
        )
        self.assertTrue(
            result["confirmed_three_sample_slide_in_both_zones_at_100ms"]
        )

    def test_analysis_rejects_noncontiguous_raw_capture(self) -> None:
        records = [row(0, 700.0), row(40, 700.0)]
        with self.assertRaisesRegex(module.ProtocolError, "contiguous at 20 ms"):
            module.analyze_human_records(records)

    def test_analysis_accepts_off_grid_terminal_finish_callback(self) -> None:
        records = [row(time, 700.0) for time in range(0, 201, 20)]
        records.append(row(210, 700.0, finished=True))
        result = module.analyze_human_records(records)
        self.assertEqual(result["outcome"]["terminal_race_time_ms"], 210)

    def test_capture_flow_observes_without_advancing_bot_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scripts_dir = root / "tmi_scripts"
            scripts_dir.mkdir()
            telemetry_path = root / "take" / "telemetry.jsonl"

            def state(progress: float) -> SimpleNamespace:
                return SimpleNamespace(
                    progress=progress,
                    input_accelerate=True,
                    input_brake=False,
                    input_left=False,
                    input_right=True,
                    input_steer=123,
                    input_gas=-456,
                )

            class FakeClient:
                def __init__(self) -> None:
                    self.calls: list[object] = []

                def execute_command(self, command: str) -> None:
                    self.calls.append(("command", command))

                def give_up(self) -> None:
                    self.calls.append("give_up")

            class FakeSession:
                instance: "FakeSession | None" = None

                def __init__(self, config: object) -> None:
                    FakeSession.instance = self
                    self.config = config
                    self.client = FakeClient()
                    self.calls: list[object] = []
                    self.steps = iter(
                        (
                            SimpleNamespace(
                                race_time_ms=-20,
                                race_finished=False,
                                state=state(0.0),
                            ),
                            SimpleNamespace(
                                race_time_ms=0,
                                race_finished=False,
                                state=state(700.0),
                            ),
                            SimpleNamespace(
                                race_time_ms=20,
                                race_finished=True,
                                state=state(710.0),
                            ),
                        )
                    )

                def prepare(self) -> None:
                    self.calls.append("prepare")

                def advance_playback(self) -> object:
                    self.calls.append("advance_playback")
                    return next(self.steps)

                def recover_inputs(self, filename: str) -> None:
                    self.calls.append(("recover_inputs", filename))
                    (scripts_dir / filename).write_text("0.00 press up\n", encoding="utf-8")

                def close(self) -> None:
                    self.calls.append("close")

            args = SimpleNamespace(
                port=8478,
                map="A01-Race.Challenge.Gbx",
                max_race_ms=45_000,
                reference_path=root / "reference.csv",
                tmi_scripts_dir=scripts_dir,
            )

            def fake_state_record(
                current_state: object,
                reference: object,
                *,
                race_time_ms: int,
                race_finished: bool,
            ) -> dict[str, object]:
                del reference
                return {
                    "race_time_ms": race_time_ms,
                    "race_finished": race_finished,
                    "progress": float(current_state.progress),
                }

            with (
                mock.patch.object(
                    module,
                    "ensure_trackmania_running",
                    return_value=(object(), False),
                ) as ensure_game,
                mock.patch.object(module, "LiveTmiSession", FakeSession),
                mock.patch.object(
                    module.ReferencePath, "from_csv", return_value=object()
                ),
                mock.patch.object(module, "state_record", fake_state_record),
                mock.patch.object(module, "focus_window"),
                mock.patch.object(module, "wait_for_foreground"),
            ):
                records, recovered = module.capture_human_lap(
                    args, "human_take.txt", telemetry_path
                )

            ensure_game.assert_called_once_with(port=8478, confirm_existing=False)
            session = FakeSession.instance
            self.assertIsNotNone(session)
            assert session is not None
            self.assertFalse(session.config.wait_for_race_start_on_connect)
            self.assertEqual(
                session.client.calls,
                [("command", "unload"), "give_up"],
            )
            self.assertEqual(
                session.calls,
                [
                    "prepare",
                    "advance_playback",
                    "advance_playback",
                    "advance_playback",
                    ("recover_inputs", "human_take.txt"),
                    "close",
                ],
            )
            self.assertEqual(len(records), 2)
            self.assertEqual(recovered, scripts_dir / "human_take.txt")
            saved = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
            self.assertEqual([item["race_time_ms"] for item in saved], [0, 20])
            self.assertEqual(saved[-1]["new_high_water_progress_delta"], 10.0)


if __name__ == "__main__":
    unittest.main()
