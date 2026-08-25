from __future__ import annotations

import unittest

from scripts.extract_replay_inputs import ReplayInputEvent, replay_events_to_script


class ReplayConversionTests(unittest.TestCase):
    def test_converts_digital_ranges(self) -> None:
        events = [
            ReplayInputEvent(0, "_FakeIsRaceRunning", 1, 0),
            ReplayInputEvent(10, "Accelerate", 1, 0),
            ReplayInputEvent(1178, "SteerRight", 1, 0),
            ReplayInputEvent(1232, "SteerRight", 0, 0),
            ReplayInputEvent(2000, "_FakeFinishLine", 1, 0),
        ]

        self.assertEqual(
            replay_events_to_script(events, 2000),
            "0-2000 press up\n1160-1220 press right\n",
        )

    def test_discards_negative_preamble(self) -> None:
        events = [
            ReplayInputEvent(-2276, "Accelerate", 1, 0),
            ReplayInputEvent(-2270, "Accelerate", 0, 0),
            ReplayInputEvent(0, "_FakeIsRaceRunning", 1, 0),
            ReplayInputEvent(10, "Accelerate", 1, 0),
        ]

        self.assertEqual(replay_events_to_script(events, 1000), "0-1000 press up\n")


if __name__ == "__main__":
    unittest.main()
