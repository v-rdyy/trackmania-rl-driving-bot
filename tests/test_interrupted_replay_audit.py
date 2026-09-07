from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_interrupted_pure_discovery_replays.py"
SPEC = importlib.util.spec_from_file_location("audit_interrupted_replays", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class InterruptedReplayAuditTests(unittest.TestCase):
    def test_annotate_progress_uses_only_new_high_water_gain(self) -> None:
        records = [
            {"progress": 10.0},
            {"progress": 14.0},
            {"progress": 12.0},
            {"progress": 15.5},
        ]

        MODULE.annotate_progress(records)

        self.assertEqual(
            [row["previous_progress"] for row in records],
            [10.0, 10.0, 14.0, 12.0],
        )
        self.assertEqual(
            [row["new_high_water_progress_delta"] for row in records],
            [0.0, 4.0, 0.0, 1.5],
        )

    def test_load_and_verify_checksum_pinned_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            scripts_dir = root / "Scripts"
            replay_dir = root / "replays"
            run_dir.mkdir()
            scripts_dir.mkdir()
            replay_dir.mkdir()
            local = replay_dir / "episode.txt"
            local.write_text("0.000 gas 65536\n", encoding="utf-8")
            external = scripts_dir / local.name
            external.write_bytes(local.read_bytes())
            replay_hash = MODULE.sha256(local)
            row = {
                "episode": 1,
                "replay": str(local),
                "sha256": replay_hash,
                "finished": True,
                "race_time_ms": 25_000,
                "additional_interactions": 250,
            }
            (run_dir / "replays.jsonl").write_text(
                json.dumps(row) + "\n", encoding="utf-8"
            )

            self.assertEqual(MODULE.load_replays(run_dir), [row])
            self.assertEqual(
                MODULE.verify_replay(row, scripts_dir),
                (local.resolve(), external, replay_hash),
            )

    def test_load_replays_rejects_duplicate_episode_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = {"episode": 7}
            (root / "replays.jsonl").write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(MODULE.ProtocolError, "duplicate episodes"):
                MODULE.load_replays(root)


if __name__ == "__main__":
    unittest.main()
