"""Per-step audit for bounded Stable-Baselines3 PPO policy actions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


def _json_values(values: np.ndarray) -> list[float | str]:
    return [float(value) if np.isfinite(value) else str(value) for value in values]


def audit_action_pair(
    normalized_action: np.ndarray,
    environment_action: np.ndarray,
    action_space: gym.spaces.Box,
    *,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Validate tanh output and its affine conversion to environment units."""
    normalized = np.asarray(normalized_action, dtype=np.float64)
    environment = np.asarray(environment_action, dtype=np.float64)
    expected_shape = action_space.shape
    finite = bool(np.isfinite(normalized).all() and np.isfinite(environment).all())
    shapes_valid = normalized.shape == expected_shape and environment.shape == expected_shape
    normalized_in_range = bool(
        normalized.shape == expected_shape
        and np.all(normalized >= -1.0 - tolerance)
        and np.all(normalized <= 1.0 + tolerance)
    )
    environment_in_range = bool(
        environment.shape == expected_shape
        and np.all(environment >= action_space.low - tolerance)
        and np.all(environment <= action_space.high + tolerance)
    )
    affine_match = False
    if shapes_valid and finite:
        expected_environment = action_space.low + 0.5 * (
            normalized + 1.0
        ) * (action_space.high - action_space.low)
        affine_match = bool(
            np.allclose(
                environment,
                expected_environment,
                atol=tolerance,
                rtol=0.0,
            )
        )

    valid = bool(
        finite
        and shapes_valid
        and normalized_in_range
        and environment_in_range
        and affine_match
    )
    return {
        "policy_normalized_action": _json_values(normalized.ravel()),
        "raw_environment_action": _json_values(environment.ravel()),
        "finite": finite,
        "normalized_in_range": normalized_in_range,
        "environment_in_range": environment_in_range,
        "affine_unscale_match": affine_match,
        "hidden_clipping": False if affine_match else None,
        "valid": valid,
    }


class RawPpoActionAuditCallback(BaseCallback):
    """Flush normalized and environment-unit PPO actions for every live step."""

    def __init__(self, path: Path) -> None:
        super().__init__(verbose=0)
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8", newline="\n")
        self.records_written = 0

    def _on_training_start(self) -> None:
        if not self.model.use_sde or not self.model.policy.squash_output:
            raise RuntimeError(
                "raw action audit requires PPO with use_sde=True and "
                "policy squash_output=True"
            )
        if not isinstance(self.training_env.action_space, gym.spaces.Box):
            raise RuntimeError("raw action audit requires a continuous Box action space")

    def _on_step(self) -> bool:
        normalized_batch = np.asarray(self.locals["actions"])
        environment_batch = np.asarray(self.locals["clipped_actions"])
        if normalized_batch.ndim != 2 or environment_batch.ndim != 2:
            raise RuntimeError("unexpected PPO action batch shape")
        if normalized_batch.shape != environment_batch.shape:
            raise RuntimeError("PPO normalized/environment action batches differ")

        for index, (normalized, environment) in enumerate(
            zip(normalized_batch, environment_batch, strict=True)
        ):
            record = audit_action_pair(
                normalized,
                environment,
                self.training_env.action_space,
            )
            record.update(
                {
                    "timestep": int(self.num_timesteps - len(normalized_batch) + index + 1),
                    "environment_index": index,
                }
            )
            self._file.write(
                json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
            )
            self._file.flush()
            self.records_written += 1
            if not record["valid"]:
                raise RuntimeError(
                    f"invalid raw PPO action at timestep {record['timestep']}"
                )
        return True

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def _on_training_end(self) -> None:
        self.close()
