from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "diagnose_simulation_speed_fidelity",
    ROOT / "scripts/diagnose_simulation_speed_fidelity.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SimulationSpeedFidelityTests(unittest.TestCase):
    def test_trajectory_comparison_is_zero_for_identical_logs(self) -> None:
        records = []
        for step, race_time in enumerate((100, 5_000, 10_000, 15_000, 20_000)):
            records.append(
                {
                    "episode": 0,
                    "step": step,
                    "race_time_ms": race_time,
                    "position": [step, 0.0, 0.0],
                    "progress": float(step),
                    "display_speed": 300 + step,
                    "raw_action": [0.1, 1.0, 0.0],
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actions.jsonl"
            import json

            path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            signature = MODULE.trajectory_signature(path)
        comparison = MODULE.compare_to_one_x(signature, signature)
        self.assertEqual(comparison["maximum_median_position_distance"], 0.0)
        self.assertEqual(comparison["maximum_absolute_median_progress_delta"], 0.0)
        self.assertEqual(comparison["maximum_median_action_l2_distance"], 0.0)

    def test_classification_requires_finish_and_lap_fidelity(self) -> None:
        signature = {
            "anchors_ms": [5_000, 10_000, 15_000, 20_000],
            "episodes": [
                {
                    "episode": 0,
                    "anchors": {
                        str(anchor): {
                            "position": [0.0, 0.0, 0.0],
                            "progress": 0.0,
                            "display_speed": 300,
                            "action": [0.0, 1.0, 0.0],
                        }
                        for anchor in (5_000, 10_000, 15_000, 20_000)
                    },
                }
            ],
        }
        common = {
            "finish_rate": 1.0,
            "best_finish_time_ms": 25_000,
            "worst_finish_time_ms": 25_000,
            "falls": 0,
            "stuck": 0,
            "signature": signature,
            "action_log": "actions.jsonl",
            "action_log_sha256": "A",
            "summary": "evaluation.json",
            "summary_sha256": "B",
        }
        results = MODULE.classify(
            [
                {**common, "speed": 1.0, "finishes": 5, "mean_finish_time_ms": 25_000},
                {**common, "speed": 6.0, "finishes": 5, "mean_finish_time_ms": 25_050},
                {**common, "speed": 100.0, "finishes": 4, "mean_finish_time_ms": 25_000},
            ]
        )
        self.assertTrue(results[1]["outcome_fidelity_gate"])
        self.assertFalse(results[2]["outcome_fidelity_gate"])


if __name__ == "__main__":
    unittest.main()
