from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path

import torch as th

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))

SPEC = importlib.util.spec_from_file_location(
    "evaluate_wr_stage2b_zone_exploration",
    WORKSPACE_ROOT / "scripts" / "evaluate_wr_stage2b_zone_exploration.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def trace(episode: int, step: int, *, zone_flag: float = 0.0, action=None):
    return {
        "episode": episode,
        "seed": MODULE.SEEDS[episode],
        "step": step,
        "zone_flag": zone_flag,
        "raw_action": action or [0.1, 0.9, 0.0],
        "post_action_progress": float(step * 10),
        "post_action_position": [float(step), 0.0, 0.0],
    }


class Stage2bZoneExplorationTests(unittest.TestCase):
    def test_protocol_constants_match_preregistration(self) -> None:
        self.assertEqual(MODULE.EPISODES, 20)
        self.assertEqual(MODULE.SEEDS[0], 20_260_830)
        self.assertEqual(MODULE.SEEDS[-1], 20_260_849)
        self.assertEqual(MODULE.SDE_SAMPLE_FREQUENCY, 4)
        self.assertEqual(MODULE.DEFAULT_ZONE_STD_MULTIPLIER, 2.0)
        self.assertEqual(
            MODULE.EXPECTED_CHECKPOINT_SHA256,
            "8A06E00055886E8E671E988D5D6948C6688B74780EDB32A87C6CC213870C8326",
        )

    def test_state_fingerprint_is_stable_and_value_sensitive(self) -> None:
        state = {"weight": th.tensor([[1.0, 2.0]]), "nested": [True, 3]}
        same = copy.deepcopy(state)
        self.assertEqual(
            MODULE.state_fingerprint(state), MODULE.state_fingerprint(same)
        )
        same["weight"][0, 1] = 2.5
        self.assertNotEqual(
            MODULE.state_fingerprint(state), MODULE.state_fingerprint(same)
        )

    def test_prezone_audit_accepts_matched_draws_and_activation_boundary(self) -> None:
        control = {episode: [] for episode in range(MODULE.EPISODES)}
        boosted = {episode: [] for episode in range(MODULE.EPISODES)}
        for episode in range(MODULE.EPISODES):
            control[episode] = [trace(episode, step) for step in range(4)]
            boosted[episode] = [
                trace(episode, step, zone_flag=1.0 if step >= 3 else 0.0)
                for step in range(4)
            ]
            boosted[episode][3]["raw_action"] = [-0.8, 0.7, 0.1]
        audit = MODULE.paired_prezone_audit(control, boosted)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["comparisons"], MODULE.EPISODES * 3)

    def test_prezone_audit_rejects_action_or_trajectory_mismatch(self) -> None:
        control = {episode: [trace(episode, 0)] for episode in range(MODULE.EPISODES)}
        boosted = copy.deepcopy(control)
        boosted[4][0]["raw_action"][0] += 0.01
        audit = MODULE.paired_prezone_audit(control, boosted)
        self.assertFalse(audit["passed"])
        self.assertEqual(audit["failures"][0]["reason"], "prezone_mismatch")


if __name__ == "__main__":
    unittest.main()
