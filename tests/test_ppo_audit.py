from __future__ import annotations

import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from trackmania_rl.ppo_audit import (
    AuditedKlPPO,
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

    def test_audited_ppo_logs_completed_epochs_and_distribution_range(self) -> None:
        model = AuditedKlPPO(
            "MlpPolicy",
            TinyContinuousEnv(),
            n_steps=2,
            batch_size=2,
            n_epochs=3,
            target_kl=None,
            use_sde=True,
            policy_kwargs={"squash_output": True},
            seed=7,
            device="cpu",
            verbose=0,
        )
        model.learn(total_timesteps=2)

        logged = model.logger.name_to_value
        self.assertEqual(logged["train/epochs_started"], 3)
        self.assertEqual(logged["train/epochs_completed"], 3)
        self.assertEqual(logged["train/kl_early_stop"], 0.0)
        self.assertIn("train/approx_kl", logged)
        self.assertLessEqual(logged["train/std_min"], logged["train/std"])
        self.assertGreaterEqual(logged["train/std_max"], logged["train/std"])
        self.assertEqual(len(model.kl_update_audit), 1)
        self.assertEqual(model.kl_update_audit[0]["epochs_completed"], 3)
        self.assertEqual(model.kl_update_audit[0]["model_timesteps"], 2)

    def test_kl_stop_matches_sb3_skips_triggering_batch_without_rollback(self) -> None:
        """A controlled second-batch overshoot retains batch one's Adam step.

        This is an offline toy-environment guard test, not Trackmania training.
        Compare all resulting weights against the installed SB3 implementation.
        """
        results = []
        for model_class in (PPO, AuditedKlPPO):
            model = model_class(
                "MlpPolicy", TinyContinuousEnv(), n_steps=4, batch_size=2,
                n_epochs=3, target_kl=.01, seed=91, device="cpu", verbose=0,
                ent_coef=.01,
            )
            before = {k: v.clone() for k, v in model.policy.state_dict().items()}
            original_evaluate = model.policy.evaluate_actions
            calls = 0

            def controlled_evaluate(*args, **kwargs):
                nonlocal calls
                calls += 1
                values, log_prob, entropy = original_evaluate(*args, **kwargs)
                if calls == 2:
                    log_prob = log_prob + 1.0  # Known above-threshold KL.
                return values, log_prob, entropy

            with patch.object(model.policy, "evaluate_actions", side_effect=controlled_evaluate), patch.object(
                model.policy.optimizer, "step", wraps=model.policy.optimizer.step
            ) as optimizer_step:
                model.learn(total_timesteps=4)
                self.assertEqual(calls, 2)
                self.assertEqual(optimizer_step.call_count, 1)
            after = {k: v.clone() for k, v in model.policy.state_dict().items()}
            self.assertTrue(any(not torch.equal(before[k], after[k]) for k in before))
            if isinstance(model, AuditedKlPPO):
                self.assertTrue(model.kl_update_audit[0]["kl_early_stop"])
                self.assertEqual(model.kl_update_audit[0]["epochs_completed"], 0)
            results.append(after)
        for key in results[0]:
            torch.testing.assert_close(results[0][key], results[1][key], rtol=0, atol=0)

    def test_nonfinite_gradient_is_rejected_before_optimizer_step(self) -> None:
        model = AuditedKlPPO(
            "MlpPolicy",
            TinyContinuousEnv(),
            n_steps=2,
            batch_size=2,
            n_epochs=1,
            target_kl=0.2,
            seed=19,
            device="cpu",
            verbose=0,
        )
        with patch(
            "trackmania_rl.ppo_audit.th.nn.utils.clip_grad_norm_",
            return_value=torch.tensor(float("nan")),
        ), patch.object(
            model.policy.optimizer,
            "step",
            wraps=model.policy.optimizer.step,
        ) as optimizer_step:
            with self.assertRaisesRegex(FloatingPointError, "gradient norm"):
                model.learn(total_timesteps=2)
            self.assertEqual(optimizer_step.call_count, 0)


if __name__ == "__main__":
    unittest.main()
