from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "train_wr_continuous",
    ROOT / "scripts/train_wr_continuous.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ContinuousRunnerTests(unittest.TestCase):
    def _fixture(self, directory: Path, *, speed: float = 2.0) -> Path:
        summary = directory / "summary.json"
        summary.write_text(
            json.dumps({"verified_simulation_speed": speed}),
            encoding="utf-8",
        )
        digest = hashlib.sha256(summary.read_bytes()).hexdigest().upper()
        config = directory / "speed.json"
        config.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "verified_simulation_speed": speed,
                    "study_summary": "summary.json",
                    "study_summary_sha256": digest,
                }
            ),
            encoding="utf-8",
        )
        return config

    def test_verified_speed_requires_pinned_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = self._fixture(directory)
            observed = MODULE.load_verified_speed_config(
                config,
                evidence_root=directory,
            )
            self.assertEqual(observed["verified_simulation_speed"], 2.0)
            (directory / "summary.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash changed"):
                MODULE.load_verified_speed_config(config, evidence_root=directory)

    def test_runner_rejects_speed_above_verified_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = self._fixture(directory, speed=6.0)
            with self.assertRaisesRegex(ValueError, "pinned to verified 2x"):
                MODULE.load_verified_speed_config(config, evidence_root=directory)

    def test_warmup_timeout_scales_for_verified_speed(self) -> None:
        self.assertEqual(
            MODULE.warmup_timeout_budget(60.0, 2.0),
            510.0,
        )
        self.assertEqual(
            MODULE.warmup_timeout_budget(60.0, 100.0),
            180.0,
        )


if __name__ == "__main__":
    unittest.main()
