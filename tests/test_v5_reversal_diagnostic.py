from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = WORKSPACE_ROOT / "scripts" / "diagnose_reward_v5_reversals.py"
SPEC = importlib.util.spec_from_file_location("diagnose_reward_v5_reversals", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def record(episode: int, step: int, race_time_ms: int, steer: float, progress: float) -> dict:
    return {
        "episode": episode,
        "step": step,
        "race_time_ms": race_time_ms,
        "raw_action": [steer, 1.0, 0.0],
        "progress": progress,
    }


class V5ReversalDiagnosticTests(unittest.TestCase):
    def test_countdown_restart_filter_matches_evaluator_semantics(self) -> None:
        records = [
            record(0, 0, 200, 0.0, 0.0),
            record(0, 1, 300, 0.1, 1.0),
            record(0, 2, -100, 0.2, 2.0),
            record(0, 3, 0, 0.3, 3.0),
        ]
        active, discarded = MODULE.active_race_records(records)
        self.assertEqual(discarded, 3)
        self.assertEqual([row["race_time_ms"] for row in active], [0])

    def test_reversal_events_preserve_location_and_excursion(self) -> None:
        values = [0.0, 0.10, 0.20, 0.15, 0.05, 0.0, 0.06, 0.12]
        records = [
            record(2, index, index * 100, value, index * 10.0)
            for index, value in enumerate(values)
        ]
        events = MODULE.reversal_events(records, track_length=100.0)
        self.assertEqual(len(events), 2)
        self.assertEqual([event["step"] for event in events], [3, 6])
        self.assertAlmostEqual(events[0]["incoming_excursion"], 0.20)
        self.assertAlmostEqual(events[1]["incoming_excursion"], 0.20)
        self.assertEqual([event["progress_bin"] for event in events], [3, 6])

    def test_finish_progress_stays_in_final_bin(self) -> None:
        self.assertEqual(MODULE.progress_bin_index(100.0, 100.0), 9)
        self.assertEqual(MODULE.progress_bin_index(150.0, 100.0), 9)


if __name__ == "__main__":
    unittest.main()
