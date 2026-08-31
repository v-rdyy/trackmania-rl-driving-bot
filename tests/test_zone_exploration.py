from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import PPO

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from trackmania_rl.observations import OBSERVATION_SIZE
from trackmania_rl.zone_exploration import (
    FINAL_CORNER_PROGRESS_MAX,
    FINAL_CORNER_PROGRESS_MIN,
    FinalCornerZoneGsdPolicy,
    FinalCornerZoneObservationWrapper,
    final_corner_zone_flag,
)


class ProgressInfoEnv(gym.Env[np.ndarray, np.ndarray]):
    def __init__(self) -> None:
        self.action_space = gym.spaces.Box(
            low=np.asarray([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.ones(3, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self.progress = 0.0
        self.steps = 0
        self.last_action: np.ndarray | None = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        del options
        self.progress = 0.0
        self.steps = 0
        return np.zeros(OBSERVATION_SIZE, dtype=np.float32), {
            "progress": self.progress
        }

    def step(self, action):
        self.last_action = np.asarray(action, dtype=np.float32).copy()
        if not self.action_space.contains(self.last_action):
            raise ValueError("received an out-of-bounds action")
        self.steps += 1
        self.progress = (
            FINAL_CORNER_PROGRESS_MIN
            if self.steps % 2
            else FINAL_CORNER_PROGRESS_MAX + 1.0
        )
        observation = np.full(
            OBSERVATION_SIZE,
            min(self.steps / 10.0, 1.0),
            dtype=np.float32,
        )
        return observation, 3.25, False, self.steps >= 4, {
            "progress": self.progress,
            "sentinel": "unchanged",
        }


def make_model(
    env: gym.Env,
    *,
    policy=FinalCornerZoneGsdPolicy,
    zone_std_multiplier: float | None = None,
) -> PPO:
    policy_kwargs = {
        "squash_output": True,
        "log_std_init": -1.0,
        "net_arch": {"pi": [8], "vf": [8]},
    }
    if zone_std_multiplier is not None:
        policy_kwargs["zone_std_multiplier"] = zone_std_multiplier
    return PPO(
        policy,
        env,
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        use_sde=True,
        sde_sample_freq=4,
        policy_kwargs=policy_kwargs,
        seed=7,
        device="cpu",
        verbose=0,
    )


class ZoneExplorationTests(unittest.TestCase):
    def test_zone_bounds_are_inclusive_and_nonfinite_progress_is_rejected(self) -> None:
        self.assertEqual(float(final_corner_zone_flag(1099.999)), 0.0)
        self.assertEqual(float(final_corner_zone_flag(1100.0)), 1.0)
        self.assertEqual(float(final_corner_zone_flag(1410.0)), 1.0)
        self.assertEqual(float(final_corner_zone_flag(1410.001)), 0.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            final_corner_zone_flag(float("nan"))

    def test_wrapper_changes_only_observation_and_passes_step_results_through(self) -> None:
        base = ProgressInfoEnv()
        env = FinalCornerZoneObservationWrapper(base)
        observation, info = env.reset()
        self.assertEqual(observation.shape, (OBSERVATION_SIZE + 1,))
        np.testing.assert_array_equal(
            observation[:OBSERVATION_SIZE],
            np.zeros(OBSERVATION_SIZE, dtype=np.float32),
        )
        self.assertEqual(float(observation[-1]), 0.0)
        self.assertEqual(info, {"progress": 0.0})

        action = np.asarray([0.25, 0.5, 0.75], dtype=np.float32)
        observation, reward, terminated, truncated, info = env.step(action)
        np.testing.assert_array_equal(base.last_action, action)
        self.assertEqual(reward, 3.25)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["sentinel"], "unchanged")
        self.assertEqual(float(observation[-1]), 1.0)

    def test_zone_boost_changes_std_not_deterministic_action_or_value(self) -> None:
        model = make_model(FinalCornerZoneObservationWrapper(ProgressInfoEnv()))
        base = np.linspace(-0.5, 0.5, OBSERVATION_SIZE, dtype=np.float32)
        outside = th.as_tensor(np.concatenate((base, [0.0]))[None, :]).float()
        inside = th.as_tensor(np.concatenate((base, [1.0]))[None, :]).float()

        with th.no_grad():
            outside_action, outside_value, _ = model.policy(
                outside,
                deterministic=True,
            )
            inside_action, inside_value, _ = model.policy(
                inside,
                deterministic=True,
            )
            outside_std = (
                model.policy.get_distribution(outside).distribution.stddev.clone()
            )
            inside_std = (
                model.policy.get_distribution(inside).distribution.stddev.clone()
            )

        th.testing.assert_close(outside_action, inside_action)
        th.testing.assert_close(outside_value, inside_value)
        th.testing.assert_close(
            inside_std,
            outside_std * 2.0,
            atol=2e-5,
            rtol=2e-5,
        )

    def test_zone_boost_is_configurable_and_cannot_reduce_exploration(self) -> None:
        model = make_model(
            FinalCornerZoneObservationWrapper(ProgressInfoEnv()),
            zone_std_multiplier=1.5,
        )
        observation = th.ones((1, OBSERVATION_SIZE + 1), dtype=th.float32)
        observation[0, -1] = 0.0
        with th.no_grad():
            outside_std = (
                model.policy.get_distribution(observation).distribution.stddev.clone()
            )
            observation[0, -1] = 1.0
            inside_std = (
                model.policy.get_distribution(observation).distribution.stddev.clone()
            )
        th.testing.assert_close(
            inside_std,
            outside_std * 1.5,
            atol=2e-5,
            rtol=2e-5,
        )

        with self.assertRaisesRegex(ValueError, "at least 1.0"):
            make_model(
                FinalCornerZoneObservationWrapper(ProgressInfoEnv()),
                zone_std_multiplier=0.5,
            )

    def test_collection_and_update_log_probs_match_for_mixed_zone_batch(self) -> None:
        model = make_model(FinalCornerZoneObservationWrapper(ProgressInfoEnv()))
        observations = th.zeros((2, OBSERVATION_SIZE + 1), dtype=th.float32)
        observations[0, :OBSERVATION_SIZE] = th.linspace(-0.3, 0.3, OBSERVATION_SIZE)
        observations[1, :OBSERVATION_SIZE] = th.linspace(0.3, -0.3, OBSERVATION_SIZE)
        observations[1, -1] = 1.0
        model.policy.reset_noise(2)

        with th.no_grad():
            actions, _, collection_log_prob = model.policy(
                observations,
                deterministic=False,
            )
            _, update_log_prob, _ = model.policy.evaluate_actions(
                observations,
                actions,
            )

        th.testing.assert_close(collection_log_prob, update_log_prob)

    def test_standard_checkpoint_loads_exactly_into_zone_policy_and_trains(self) -> None:
        base_env = ProgressInfoEnv()
        standard = make_model(base_env, policy="MlpPolicy")
        standard.learn(total_timesteps=4)
        self.assertTrue(standard.policy.optimizer.state_dict()["state"])

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "standard_model"
            standard.save(checkpoint)
            wrapped = FinalCornerZoneObservationWrapper(ProgressInfoEnv())
            converted = PPO.load(
                checkpoint,
                env=wrapped,
                device="cpu",
                custom_objects={
                    "policy_class": FinalCornerZoneGsdPolicy,
                    "observation_space": wrapped.observation_space,
                },
            )

            self.assertEqual(converted.num_timesteps, standard.num_timesteps)
            self.assertTrue(converted.policy.optimizer.state_dict()["state"])
            standard_state = standard.policy.state_dict()
            converted_state = converted.policy.state_dict()
            self.assertEqual(standard_state.keys(), converted_state.keys())
            for name, expected in standard_state.items():
                th.testing.assert_close(converted_state[name], expected)

            base_observation = np.linspace(
                -0.25,
                0.25,
                OBSERVATION_SIZE,
                dtype=np.float32,
            )
            standard_action, _ = standard.predict(
                base_observation,
                deterministic=True,
            )
            converted_inside_observation = np.concatenate(
                (base_observation, np.asarray([1.0], np.float32))
            )
            converted_action, _ = converted.predict(
                converted_inside_observation,
                deterministic=True,
            )
            np.testing.assert_allclose(converted_action, standard_action, atol=1e-7)

            with th.no_grad():
                standard_std = (
                    standard.policy.get_distribution(
                        th.as_tensor(base_observation[None, :])
                    ).distribution.stddev.clone()
                )
                converted_outside_observation = np.concatenate(
                    (base_observation, np.asarray([0.0], np.float32))
                )
                converted_outside_std = (
                    converted.policy.get_distribution(
                        th.as_tensor(converted_outside_observation[None, :])
                    ).distribution.stddev.clone()
                )
            th.testing.assert_close(converted_outside_std, standard_std)

            converted.learn(total_timesteps=4, reset_num_timesteps=False)
            self.assertEqual(converted.num_timesteps, standard.num_timesteps + 4)


if __name__ == "__main__":
    unittest.main()
