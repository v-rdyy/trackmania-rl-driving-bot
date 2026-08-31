"""On-policy gSDE exploration that is conditional on a track-progress zone.

The wrapper appends a final-corner indicator to the existing engineered
observation.  The custom policy hides that indicator from the inherited actor
and critic networks, then uses it only to scale gSDE's state-dependent noise.
Because the indicator is retained in PPO's rollout buffer, both action
collection and policy optimization reconstruct the same conditional
distribution and therefore the same log-probability convention.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.distributions import (
    Distribution,
    StateDependentNoiseDistribution,
)
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs

from trackmania_rl.observations import OBSERVATION_SIZE

FINAL_CORNER_PROGRESS_MIN = 1_100.0
FINAL_CORNER_PROGRESS_MAX = 1_410.0
DEFAULT_ZONE_STD_MULTIPLIER = 2.0
ZONE_OBSERVATION_INDEX = OBSERVATION_SIZE
ZONE_OBSERVATION_SIZE = OBSERVATION_SIZE + 1


def final_corner_zone_flag(progress: float) -> np.float32:
    """Return the inclusive final-corner zone indicator used by the policy."""

    value = float(progress)
    if not math.isfinite(value):
        raise ValueError("progress must be finite")
    return np.float32(
        FINAL_CORNER_PROGRESS_MIN <= value <= FINAL_CORNER_PROGRESS_MAX
    )


class FinalCornerZoneObservationWrapper(gym.Wrapper):
    """Append a final-corner flag without changing actions, rewards, or info."""

    def __init__(self, env: gym.Env, *, enabled: bool = True) -> None:
        super().__init__(env)
        if not isinstance(enabled, bool):
            raise TypeError("zone exploration enabled flag must be boolean")
        self.enabled = enabled
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("zone exploration requires a Box observation space")
        if env.observation_space.shape != (OBSERVATION_SIZE,):
            raise ValueError(
                "zone exploration requires the documented "
                f"{OBSERVATION_SIZE}-value observation"
            )
        self.observation_space = gym.spaces.Box(
            low=np.concatenate(
                (
                    np.asarray(env.observation_space.low, dtype=np.float32),
                    np.asarray([0.0], dtype=np.float32),
                )
            ),
            high=np.concatenate(
                (
                    np.asarray(env.observation_space.high, dtype=np.float32),
                    np.asarray([1.0], dtype=np.float32),
                )
            ),
            dtype=np.float32,
        )

    def _with_zone_flag(
        self,
        observation: np.ndarray,
        info: dict[str, Any],
    ) -> np.ndarray:
        values = np.asarray(observation, dtype=np.float32)
        if values.shape != (OBSERVATION_SIZE,):
            raise ValueError(
                "underlying environment returned an unexpected observation shape"
            )
        if "progress" not in info:
            raise KeyError("zone exploration requires progress in environment info")
        return np.concatenate(
            (
                values,
                np.asarray(
                    [
                        final_corner_zone_flag(info["progress"])
                        if self.enabled
                        else np.float32(0.0)
                    ],
                    dtype=np.float32,
                ),
            )
        )

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return self._with_zone_flag(observation, info), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return (
            self._with_zone_flag(observation, info),
            reward,
            terminated,
            truncated,
            info,
        )


class StripZoneFlagExtractor(BaseFeaturesExtractor):
    """Expose only the original 26 values to the inherited actor and critic."""

    def __init__(self, observation_space: gym.spaces.Box) -> None:
        if observation_space.shape != (ZONE_OBSERVATION_SIZE,):
            raise ValueError(
                "zone flag extractor requires a "
                f"{ZONE_OBSERVATION_SIZE}-value observation"
            )
        super().__init__(observation_space, features_dim=OBSERVATION_SIZE)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return observations[..., :OBSERVATION_SIZE]


class FinalCornerZoneGsdPolicy(ActorCriticPolicy):
    """Scale gSDE standard deviation only when the buffered zone flag is set."""

    def __init__(
        self,
        *args,
        zone_std_multiplier: float = DEFAULT_ZONE_STD_MULTIPLIER,
        **kwargs,
    ) -> None:
        multiplier = float(zone_std_multiplier)
        if not math.isfinite(multiplier) or multiplier < 1.0:
            raise ValueError("zone_std_multiplier must be finite and at least 1.0")
        kwargs["features_extractor_class"] = StripZoneFlagExtractor
        kwargs.pop("features_extractor_kwargs", None)
        super().__init__(*args, **kwargs)
        if not self.use_sde or not isinstance(
            self.action_dist,
            StateDependentNoiseDistribution,
        ):
            raise ValueError("final-corner exploration requires a gSDE policy")
        self.zone_std_multiplier = multiplier

    def _actor_critic_latents(
        self,
        obs: PyTorchObs,
    ) -> tuple[th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            return self.mlp_extractor(features)
        policy_features, value_features = features
        return (
            self.mlp_extractor.forward_actor(policy_features),
            self.mlp_extractor.forward_critic(value_features),
        )

    def _zone_distribution(
        self,
        obs: PyTorchObs,
        latent_pi: th.Tensor,
    ) -> StateDependentNoiseDistribution:
        if not isinstance(obs, th.Tensor):
            raise TypeError("final-corner exploration requires tensor observations")
        if obs.shape[-1] != ZONE_OBSERVATION_SIZE:
            raise ValueError("final-corner zone flag is missing from observation")
        if not isinstance(self.action_dist, StateDependentNoiseDistribution):
            raise RuntimeError("final-corner exploration lost its gSDE distribution")

        zone_flag = obs[..., ZONE_OBSERVATION_INDEX].reshape(-1, 1)
        scale = 1.0 + (self.zone_std_multiplier - 1.0) * zone_flag.to(
            dtype=latent_pi.dtype
        )
        mean_actions = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(
            mean_actions,
            self.log_std,
            latent_pi * scale,
        )

    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        latent_pi, latent_vf = self._actor_critic_latents(obs)
        values = self.value_net(latent_vf)
        distribution = self._zone_distribution(obs, latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def evaluate_actions(
        self,
        obs: PyTorchObs,
        actions: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor | None]:
        latent_pi, latent_vf = self._actor_critic_latents(obs)
        distribution = self._zone_distribution(obs, latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._zone_distribution(obs, latent_pi)
