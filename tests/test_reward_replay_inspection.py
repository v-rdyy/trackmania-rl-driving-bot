from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = WORKSPACE_ROOT / "scripts" / "inspect_reward_v3_replays.py"
SPEC = importlib.util.spec_from_file_location("inspect_reward_replays", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RewardReplayInspectionTests(unittest.TestCase):
    def test_video_label_follows_replay_version(self) -> None:
        self.assertEqual(
            MODULE.replay_experiment_label("reward_v6_final_deadbeef_ep_01"),
            "V6",
        )
        self.assertEqual(
            MODULE.replay_experiment_label("reward_v5_final_6324dfc2_ep_04"),
            "V5",
        )
        self.assertEqual(
            MODULE.replay_experiment_label("reward_v4_final_6df90018_ep_12"),
            "V4",
        )
        self.assertEqual(
            MODULE.replay_experiment_label("reward_v3_final_c9791b6e_ep_01"),
            "V3",
        )

    def test_unknown_replay_uses_neutral_label(self) -> None:
        self.assertEqual(MODULE.replay_experiment_label("manual_run"), "TrackMania")

    def test_workspace_relative_replay_is_not_forced_into_v3_directory(self) -> None:
        relative = Path("scripts") / "inspect_reward_v3_replays.py"
        self.assertEqual(
            MODULE.resolve_replay_path(relative),
            WORKSPACE_ROOT / relative,
        )

    def test_bare_replay_name_keeps_legacy_v3_default(self) -> None:
        replay = Path("preserved.txt")
        self.assertEqual(
            MODULE.resolve_replay_path(replay),
            MODULE.DEFAULT_REPLAY_DIR / replay,
        )


if __name__ == "__main__":
    unittest.main()
