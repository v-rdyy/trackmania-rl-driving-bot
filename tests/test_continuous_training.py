from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trackmania_rl.continuous_training import (
    ContinuousControl, atomic_json, file_sha256, learn_until_stopped,
    require_constant_schedules, save_checkpoint,
)


class SmallEnv(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        return np.array([float(action[0]), 0], dtype=np.float32), float(action[0]), self.steps >= 8, False, {"race_time_ms": 800, "race_finished": True}


class KeepGoing(BaseCallback):
    def _on_step(self):
        return True


def make_model():
    return PPO("MlpPolicy", SmallEnv(), n_steps=32, batch_size=16, n_epochs=2,
               use_sde=True, sde_sample_freq=4,
               policy_kwargs={"squash_output": True}, seed=42, device="cpu", verbose=0)


class ContinuousTests(unittest.TestCase):
    def test_continuous_updates_match_stock_ppo(self):
        stock = make_model()
        stock.learn(64)
        continuous = make_model()
        updates = []
        learn_until_stopped(continuous, KeepGoing(), lambda: len(updates) == 2,
                            lambda: updates.append(continuous.num_timesteps), run_name="test")
        self.assertEqual(updates, [32, 64])
        self.assertEqual(stock.num_timesteps, continuous.num_timesteps)
        self.assertEqual(stock._n_updates, continuous._n_updates)
        for name, expected in stock.policy.state_dict().items():
            torch.testing.assert_close(continuous.policy.state_dict()[name], expected, rtol=0, atol=0)

    def test_stop_file_saves_loadable_model_and_marks_partial_rollout(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            control = ContinuousControl(directory, directory / "models", interval=30, minimum_free_bytes=0)
            model = make_model()
            class StopAfterUpdate(BaseCallback):
                def _on_step(self):
                    if self.num_timesteps == 35:
                        atomic_json(control.stop_path, {"reason": "test"})
                        control.last_stop_check = 0
                    return True
            with patch("builtins.print"):
                learn_until_stopped(model, CallbackList([StopAfterUpdate(), control]),
                                    control.should_stop, control.after_update, run_name="test")
            self.assertEqual(control.updates, 1)
            self.assertEqual(model.num_timesteps, 35)
            checkpoint = directory / "models/final_model.zip"
            record = save_checkpoint(model, checkpoint, learned_through=control.learned_through)
            self.assertEqual(record["unlearned_interactions"], 3)
            self.assertEqual(record["sha256"], file_sha256(checkpoint))
            reloaded = PPO.load(checkpoint, device="cpu")
            self.assertEqual(reloaded.num_timesteps, 35)
            self.assertTrue(reloaded.policy.optimizer.state_dict()["state"])
            periodic = json.loads(next((directory / "models").glob("step_*.json")).read_text())
            self.assertEqual(periodic["unlearned_interactions"], 0)
            with self.assertRaises(FileExistsError):
                save_checkpoint(model, checkpoint, learned_through=32)

    def test_nonconstant_schedule_is_rejected(self):
        model = make_model()
        model.lr_schedule = lambda remaining: 0.001 * remaining
        with self.assertRaises(ValueError):
            require_constant_schedules(model)

    def test_low_disk_fails_before_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            control = ContinuousControl(Path(tmp), Path(tmp), minimum_free_bytes=2**100)
            with self.assertRaises(OSError):
                control.check_disk()

    def test_atomic_json_replaces_complete_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "status.json"
            atomic_json(path, {"status": "training"})
            atomic_json(path, {"status": "stopped"})
            self.assertEqual(json.loads(path.read_text())["status"], "stopped")
            self.assertFalse(path.with_name("status.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
