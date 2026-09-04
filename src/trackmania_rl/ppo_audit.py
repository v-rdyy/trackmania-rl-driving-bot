"""Per-step audit for bounded Stable-Baselines3 PPO policy actions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance


class AuditedKlPPO(PPO):
    """SB3 2.9 PPO with per-update KL-stop and epoch audit logging.

    The optimization math intentionally matches ``PPO.train``.  The only
    additions are counters describing how much of each scheduled update ran
    and min/max action-distribution statistics alongside SB3's mean standard
    deviation.
    """

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses: list[float] = []
        pg_losses: list[float] = []
        value_losses: list[float] = []
        clip_fractions: list[float] = []
        approx_kl_divs: list[np.ndarray] = []
        continue_training = True
        epochs_started = 0
        epochs_completed = 0
        early_stop_epoch: int | None = None
        early_stop_approx_kl: float | None = None

        for epoch in range(self.n_epochs):
            epochs_started += 1
            approx_kl_divs = []
            epoch_completed = True
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                )
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio,
                    1 - clip_range,
                    1 + clip_range,
                )
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())
                clip_fractions.append(
                    th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                )

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())
                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )
                if not th.isfinite(loss):
                    raise FloatingPointError(
                        "nonfinite PPO loss before optimizer step"
                    )

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                if (
                    self.target_kl is not None
                    and approx_kl_div > 1.5 * self.target_kl
                ):
                    continue_training = False
                    epoch_completed = False
                    early_stop_epoch = epoch
                    early_stop_approx_kl = float(approx_kl_div)
                    if self.verbose >= 1:
                        print(
                            "Early stopping at step "
                            f"{epoch} due to reaching max kl: {approx_kl_div:.2f}"
                        )
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                gradient_norm = th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.max_grad_norm,
                )
                if not th.isfinite(gradient_norm):
                    raise FloatingPointError(
                        "nonfinite PPO gradient norm before optimizer step"
                    )
                self.policy.optimizer.step()

            self._n_updates += 1
            if epoch_completed:
                epochs_completed += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )
        entropy_loss_mean = float(np.mean(entropy_losses))
        policy_loss_mean = float(np.mean(pg_losses))
        value_loss_mean = float(np.mean(value_losses))
        approx_kl_mean = float(np.mean(approx_kl_divs))
        clip_fraction_mean = float(np.mean(clip_fractions))
        self.logger.record("train/entropy_loss", entropy_loss_mean)
        self.logger.record("train/policy_gradient_loss", policy_loss_mean)
        self.logger.record("train/value_loss", value_loss_mean)
        self.logger.record("train/approx_kl", approx_kl_mean)
        self.logger.record("train/clip_fraction", clip_fraction_mean)
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/epochs_started", epochs_started)
        self.logger.record("train/epochs_completed", epochs_completed)
        self.logger.record("train/kl_early_stop", float(not continue_training))
        self.logger.record(
            "train/kl_early_stop_epoch",
            -1 if early_stop_epoch is None else early_stop_epoch,
        )
        self.logger.record(
            "train/kl_early_stop_approx_kl",
            0.0 if early_stop_approx_kl is None else early_stop_approx_kl,
        )
        standard_deviation_mean: float | None = None
        standard_deviation_minimum: float | None = None
        standard_deviation_maximum: float | None = None
        if hasattr(self.policy, "log_std"):
            standard_deviation = th.exp(self.policy.log_std)
            standard_deviation_mean = standard_deviation.mean().item()
            standard_deviation_minimum = standard_deviation.min().item()
            standard_deviation_maximum = standard_deviation.max().item()
            self.logger.record("train/std", standard_deviation_mean)
            self.logger.record("train/std_min", standard_deviation_minimum)
            self.logger.record("train/std_max", standard_deviation_maximum)

        if not hasattr(self, "kl_update_audit"):
            self.kl_update_audit: list[dict[str, Any]] = []
        self.kl_update_audit.append(
            {
                "rollout_update": len(self.kl_update_audit) + 1,
                "model_timesteps": int(self.num_timesteps),
                "epochs_scheduled": int(self.n_epochs),
                "epochs_started": epochs_started,
                "epochs_completed": epochs_completed,
                "kl_early_stop": not continue_training,
                "kl_early_stop_epoch": early_stop_epoch,
                "kl_early_stop_approx_kl": early_stop_approx_kl,
                "approx_kl": approx_kl_mean,
                "clip_fraction": clip_fraction_mean,
                "policy_gradient_loss": policy_loss_mean,
                "value_loss": value_loss_mean,
                "entropy_loss": entropy_loss_mean,
                "action_distribution_std_mean": standard_deviation_mean,
                "action_distribution_std_minimum": standard_deviation_minimum,
                "action_distribution_std_maximum": standard_deviation_maximum,
            }
        )

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


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


class BoundedPpoActionStatsCallback(BaseCallback):
    """Validate every PPO action while retaining only aggregate range statistics."""

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self.records_checked = 0
        self._normalized_minimum: np.ndarray | None = None
        self._normalized_maximum: np.ndarray | None = None
        self._environment_minimum: np.ndarray | None = None
        self._environment_maximum: np.ndarray | None = None

    def _on_training_start(self) -> None:
        if not self.model.use_sde or not self.model.policy.squash_output:
            raise RuntimeError(
                "bounded action statistics require PPO with use_sde=True and "
                "policy squash_output=True"
            )
        if not isinstance(self.training_env.action_space, gym.spaces.Box):
            raise RuntimeError(
                "bounded action statistics require a continuous Box action space"
            )

    @staticmethod
    def _updated_extrema(
        current_minimum: np.ndarray | None,
        current_maximum: np.ndarray | None,
        value: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if current_minimum is None or current_maximum is None:
            return value.copy(), value.copy()
        return np.minimum(current_minimum, value), np.maximum(current_maximum, value)

    def _on_step(self) -> bool:
        normalized_batch = np.asarray(self.locals["actions"], dtype=np.float64)
        environment_batch = np.asarray(
            self.locals["clipped_actions"],
            dtype=np.float64,
        )
        if normalized_batch.ndim != 2 or normalized_batch.shape != environment_batch.shape:
            raise RuntimeError("unexpected PPO action batch shape")

        for normalized, environment in zip(
            normalized_batch,
            environment_batch,
            strict=True,
        ):
            record = audit_action_pair(
                normalized,
                environment,
                self.training_env.action_space,
            )
            if not record["valid"]:
                raise RuntimeError(
                    f"invalid bounded PPO action at timestep {self.num_timesteps}"
                )
            (
                self._normalized_minimum,
                self._normalized_maximum,
            ) = self._updated_extrema(
                self._normalized_minimum,
                self._normalized_maximum,
                normalized,
            )
            (
                self._environment_minimum,
                self._environment_maximum,
            ) = self._updated_extrema(
                self._environment_minimum,
                self._environment_maximum,
                environment,
            )
            self.records_checked += 1
        return True

    def summary(self) -> dict[str, Any]:
        if (
            self.records_checked == 0
            or self._normalized_minimum is None
            or self._normalized_maximum is None
            or self._environment_minimum is None
            or self._environment_maximum is None
        ):
            raise RuntimeError("no bounded PPO actions were checked")
        return {
            "records_checked": self.records_checked,
            "all_finite_in_range_and_affine": True,
            "hidden_clipping": False,
            "policy_normalized_minimum": self._normalized_minimum.tolist(),
            "policy_normalized_maximum": self._normalized_maximum.tolist(),
            "environment_action_minimum": self._environment_minimum.tolist(),
            "environment_action_maximum": self._environment_maximum.tolist(),
        }
