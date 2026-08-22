from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from trackmania_rl.ppo_audit import (
    BoundedPpoActionStatsCallback,
    RawPpoActionAuditCallback,
    audit_action_pair,
)


class TinyContinuousEnv(gym.Env[np.ndarray, np.ndarray]):
    def __init__(self) -> None:
        self.action_space = gym.spaces.Box(
            low=np.asarray([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        del options
        self.steps = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        del action
        self.steps += 1
        return (
            np.zeros(2, dtype=np.float32),
            0.0,
            False,
            self.steps >= 2,
            {},
        )


class PpoActionAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.action_space = gym.spaces.Box(
            low=np.asarray([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def test_accepts_bounded_action_and_exact_affine_unscale(self) -> None:
        record = audit_action_pair(
            np.asarray([-0.5, 0.0, 0.5]),
            np.asarray([-0.5, 0.5, 0.75]),
            self.action_space,
        )

        self.assertTrue(record["valid"])
        self.assertTrue(record["normalized_in_range"])
        self.assertTrue(record["environment_in_range"])
        self.assertTrue(record["affine_unscale_match"])
        self.assertFalse(record["hidden_clipping"])

    def test_rejects_environment_action_that_hides_clipping(self) -> None:
        record = audit_action_pair(
            np.asarray([0.25, -0.5, 0.5]),
            np.asarray([0.25, 0.0, 0.75]),
            self.action_space,
        )

        self.assertFalse(record["valid"])
        self.assertFalse(record["affine_unscale_match"])
        self.assertIsNone(record["hidden_clipping"])

    def test_rejects_nonfinite_normalized_action(self) -> None:
        record = audit_action_pair(
            np.asarray([np.nan, 0.0, 0.0]),
            np.asarray([0.0, 0.5, 0.5]),
            self.action_space,
        )

        self.assertFalse(record["valid"])
        self.assertFalse(record["finite"])
        self.assertEqual(record["policy_normalized_action"][0], "nan")

    def test_callback_audits_every_squashed_ppo_step_without_clipping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actions.jsonl"
            callback = RawPpoActionAuditCallback(path)
            model = PPO(
                "MlpPolicy",
                TinyContinuousEnv(),
                n_steps=2,
                batch_size=2,
                n_epochs=1,
                use_sde=True,
                policy_kwargs={"squash_output": True},
                seed=1,
                device="cpu",
                verbose=0,
            )
            model.learn(total_timesteps=4, callback=callback)

            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(records), 4)
            self.assertTrue(all(record["valid"] for record in records))
            self.assertTrue(
                all(record["hidden_clipping"] is False for record in records)
            )

    def test_stats_callback_validates_without_per_step_files(self) -> None:
        callback = BoundedPpoActionStatsCallback()
        model = PPO(
            "MlpPolicy",
            TinyContinuousEnv(),
            n_steps=2,
            batch_size=2,
            n_epochs=1,
            use_sde=True,
            policy_kwargs={"squash_output": True},
            seed=1,
            device="cpu",
            verbose=0,
        )
        model.learn(total_timesteps=4, callback=callback)

        summary = callback.summary()
        self.assertEqual(summary["records_checked"], 4)
        self.assertTrue(summary["all_finite_in_range_and_affine"])
        self.assertFalse(summary["hidden_clipping"])
        self.assertEqual(len(summary["environment_action_minimum"]), 3)


if __name__ == "__main__":
    unittest.main()
