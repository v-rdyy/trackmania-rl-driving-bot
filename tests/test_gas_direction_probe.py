from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = WORKSPACE_ROOT / "scripts" / "probe_gas_direction.py"
SPEC = importlib.util.spec_from_file_location("probe_gas_direction", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def branch(pedal: str, gas: int, direction: str) -> dict[str, object]:
    if direction == "forward":
        return {
            "applied_gas": gas,
            "requested_pedal": pedal,
            "signed_start_tangent_displacement": 20.0,
            "maximum_progress_delta": 20.0,
            "maximum_car_forward_velocity": 15.0,
            "minimum_car_forward_velocity": 0.0,
            "direction": direction,
        }
    return {
        "applied_gas": gas,
        "requested_pedal": pedal,
        "signed_start_tangent_displacement": -20.0,
        "maximum_progress_delta": 0.0,
        "maximum_car_forward_velocity": 0.0,
        "minimum_car_forward_velocity": -15.0,
        "direction": direction,
    }


class GasDirectionProbeTests(unittest.TestCase):
    def test_relative_artifact_path_is_resolved_from_workspace(self) -> None:
        relative = Path("artifacts") / "smoke" / "probe.json"
        self.assertEqual(MODULE.workspace_path(relative), WORKSPACE_ROOT / relative)
        with tempfile.TemporaryDirectory() as directory:
            absolute = Path(directory) / "probe.json"
            self.assertEqual(MODULE.workspace_path(absolute), absolute)

    def test_direction_classifier_uses_signed_motion_and_velocity(self) -> None:
        self.assertEqual(
            MODULE.classify_direction(branch("throttle", -65536, "forward")),
            "forward",
        )
        self.assertEqual(
            MODULE.classify_direction(branch("brake", 65536, "backward")),
            "backward",
        )

    def test_mapping_resolves_correct_labels(self) -> None:
        branches = [
            branch("throttle", -65536, "forward"),
            branch("brake", 65536, "backward"),
            branch("brake", 65536, "backward"),
            branch("throttle", -65536, "forward"),
        ]
        self.assertEqual(
            MODULE.resolve_mapping(branches),
            "current_throttle_brake_labels_correct",
        )

    def test_mapping_resolves_reversed_labels(self) -> None:
        branches = [
            branch("throttle", 65536, "backward"),
            branch("brake", -65536, "forward"),
            branch("brake", -65536, "forward"),
            branch("throttle", 65536, "backward"),
        ]
        self.assertEqual(
            MODULE.resolve_mapping(branches),
            "current_throttle_brake_labels_reversed",
        )

    def test_mapping_stays_ambiguous_on_inconsistent_trials(self) -> None:
        branches = [
            branch("throttle", -65536, "forward"),
            branch("brake", 65536, "forward"),
            branch("brake", 65536, "backward"),
            branch("throttle", -65536, "forward"),
        ]
        self.assertEqual(MODULE.resolve_mapping(branches), "ambiguous")

    def test_protocol_polarity_is_reported_separately_from_pedal_labels(self) -> None:
        branches = [
            branch("throttle", -65536, "forward"),
            branch("brake", 65536, "backward"),
        ]
        self.assertEqual(
            MODULE.resolve_protocol_polarity(branches),
            "negative_forward_positive_backward",
        )


if __name__ == "__main__":
    unittest.main()
