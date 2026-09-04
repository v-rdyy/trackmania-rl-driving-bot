"""Local-only controls for continuous PPO training; no model-service calls."""

from __future__ import annotations

import ctypes
import copy
import hashlib
import json
import os
import shutil
import time
import zipfile
from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


class OptimizerDivergenceError(RuntimeError):
    """An optimizer update was rolled back after violating a safety invariant."""

    def __init__(self, message: str, event: dict[str, Any]) -> None:
        super().__init__(message)
        self.event = event


def _nonfinite_tensor_paths(value: Any, prefix: str) -> list[str]:
    paths: list[str] = []
    if isinstance(value, torch.Tensor):
        if not value.detach().isfinite().all():
            paths.append(prefix)
    elif isinstance(value, dict):
        for key, child in value.items():
            paths.extend(_nonfinite_tensor_paths(child, f"{prefix}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            paths.extend(_nonfinite_tensor_paths(child, f"{prefix}[{index}]"))
    return paths


class OptimizerUpdateGuard:
    """Rollback one PPO update if state, metrics, or policy output diverges.

    The snapshot is held in memory for only the current update. A violating
    update is restored exactly, verified again, and handed to the runner for a
    unique recovery checkpoint and a controlled stop.
    """

    FINITE_METRICS = (
        "train/approx_kl",
        "train/clip_fraction",
        "train/entropy_loss",
        "train/explained_variance",
        "train/loss",
        "train/policy_gradient_loss",
        "train/value_loss",
    )

    def __init__(self, *, maximum_mean_kl: float) -> None:
        if not np.isfinite(maximum_mean_kl) or maximum_mean_kl <= 0:
            raise ValueError("maximum_mean_kl must be positive and finite")
        self.maximum_mean_kl = float(maximum_mean_kl)
        self.last_event: dict[str, Any] | None = None

    @staticmethod
    def _snapshot(model: Any) -> dict[str, Any]:
        return {
            "policy": copy.deepcopy(model.policy.state_dict()),
            "optimizer": copy.deepcopy(model.policy.optimizer.state_dict()),
            "n_updates": int(model._n_updates),
        }

    @staticmethod
    def _restore(model: Any, snapshot: dict[str, Any]) -> None:
        model.policy.load_state_dict(snapshot["policy"])
        model.policy.optimizer.load_state_dict(snapshot["optimizer"])
        model._n_updates = int(snapshot["n_updates"])

    @staticmethod
    def _rollout_nonfinite(model: Any) -> list[str]:
        paths: list[str] = []
        for name in (
            "observations",
            "actions",
            "rewards",
            "returns",
            "advantages",
            "values",
            "log_probs",
        ):
            value = getattr(model.rollout_buffer, name, None)
            if value is not None and not np.isfinite(value).all():
                paths.append(f"rollout_buffer.{name}")
        return paths

    @staticmethod
    def _state_nonfinite(model: Any) -> list[str]:
        paths = _nonfinite_tensor_paths(model.policy.state_dict(), "policy")
        paths.extend(
            _nonfinite_tensor_paths(
                model.policy.optimizer.state_dict(),
                "policy.optimizer",
            )
        )
        return paths

    @staticmethod
    def _policy_output_nonfinite(model: Any) -> list[str]:
        observations = np.asarray(model.rollout_buffer.observations)
        if observations.size == 0:
            return ["policy.output.missing_probe_observation"]
        probe = observations.reshape((-1, *model.observation_space.shape))[:64]
        actions, _ = model.predict(probe, deterministic=True)
        return [] if np.isfinite(actions).all() else ["policy.output.action"]

    def _post_update_violations(self, model: Any) -> tuple[list[str], float | None]:
        paths = self._state_nonfinite(model)
        try:
            paths.extend(self._policy_output_nonfinite(model))
        except Exception as error:
            paths.append(f"policy.output.exception:{type(error).__name__}")
        logger_values = model.logger.name_to_value
        approx_kl: float | None = None
        for name in self.FINITE_METRICS:
            if name not in logger_values:
                paths.append(f"logger.missing:{name}")
                continue
            value = float(logger_values[name])
            if not np.isfinite(value):
                paths.append(f"logger.nonfinite:{name}")
            if name == "train/approx_kl":
                approx_kl = value
        if approx_kl is not None and approx_kl > self.maximum_mean_kl:
            paths.append(
                f"logger.kl_spike:{approx_kl:.9g}>{self.maximum_mean_kl:.9g}"
            )
        return paths, approx_kl

    def validate_initial_state(self, model: Any) -> None:
        violations = self._state_nonfinite(model)
        if violations:
            raise ValueError(
                "refusing training with nonfinite initial model state: "
                + ", ".join(violations[:10])
            )

    def run_update(self, model: Any) -> None:
        rollout_violations = self._rollout_nonfinite(model)
        if rollout_violations:
            raise ValueError(
                "refusing optimizer update with nonfinite rollout: "
                + ", ".join(rollout_violations)
            )
        self.validate_initial_state(model)
        snapshot = self._snapshot(model)
        attempted_update_count = int(model._n_updates)
        try:
            model.train()
            violations, approx_kl = self._post_update_violations(model)
        except Exception as error:
            violations = [f"optimizer.exception:{type(error).__name__}:{error}"]
            approx_kl = None
        if not violations:
            return

        self._restore(model, snapshot)
        rollback_violations = self._state_nonfinite(model)
        try:
            rollback_violations.extend(self._policy_output_nonfinite(model))
        except Exception as error:
            rollback_violations.append(
                f"policy.output.exception:{type(error).__name__}:{error}"
            )
        event = {
            "reason": "optimizer_update_rejected",
            "model_timesteps_after_rollout": int(model.num_timesteps),
            "n_updates_before_attempt": attempted_update_count,
            "n_updates_after_rollback": int(model._n_updates),
            "target_kl": model.target_kl,
            "maximum_mean_kl": self.maximum_mean_kl,
            "observed_mean_kl": approx_kl,
            "violations": violations,
            "rollback_verified": not rollback_violations,
            "rollback_violations": rollback_violations,
            "detected_at_unix": time.time(),
        }
        self.last_event = event
        if rollback_violations:
            raise RuntimeError(
                "optimizer divergence rollback failed verification: "
                + ", ".join(rollback_violations[:10])
            )
        raise OptimizerDivergenceError(
            "optimizer update violated safety guard and was rolled back",
            event,
        )


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def save_checkpoint(model: Any, path: Path, *, learned_through: int) -> dict[str, Any]:
    """Publish a complete, checksum-pinned archive without replacing evidence."""
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".partial.zip")
    model.save(temporary)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise OSError("checkpoint archive failed CRC verification")
        if "policy.optimizer.pth" not in archive.namelist():
            raise OSError("checkpoint is missing optimizer state")
    with temporary.open("r+b") as source:
        os.fsync(source.fileno())
    digest = file_sha256(temporary)
    temporary.rename(path)
    record = {
        "path": str(path), "sha256": digest,
        "num_timesteps": int(model.num_timesteps),
        "learned_through_timesteps": learned_through,
        "unlearned_interactions": int(model.num_timesteps) - learned_through,
        "optimizer_epoch_counter": int(model._n_updates),
        "saved_at_unix": time.time(),
    }
    atomic_json(path.with_suffix(".json"), record)
    return record


class KeepAwake:
    """Release this thread's Windows idle-sleep request even on errors."""

    def __enter__(self) -> "KeepAwake":
        if os.name != "nt":
            return self
        self.api = ctypes.windll.kernel32.SetThreadExecutionState
        self.api.argtypes = [ctypes.c_uint]
        self.api.restype = ctypes.c_uint
        self.previous = self.api(0x80000003)  # CONTINUOUS | SYSTEM | DISPLAY
        if not self.previous:
            raise OSError("Windows rejected the keep-awake request")
        return self

    def __exit__(self, *_: Any) -> None:
        if os.name == "nt":
            self.api(self.previous | 0x80000000)


def require_constant_schedules(model: Any) -> None:
    """An unbounded run cannot use an end-time-dependent annealing schedule."""
    for name in ("lr_schedule", "clip_range", "clip_range_vf"):
        schedule = getattr(model, name, None)
        if schedule is not None:
            values = [float(schedule(p)) for p in (0.0, 0.25, 0.5, 0.75, 1.0)]
            if not all(np.isfinite(values)) or not all(v == values[0] for v in values):
                raise ValueError(f"continuous training requires constant {name}")


def learn_until_stopped(
    model: Any,
    callback: BaseCallback,
    should_stop: Callable[[], bool],
    after_update: Callable[[], None],
    *,
    run_name: str,
    optimizer_guard: OptimizerUpdateGuard | None = None,
) -> None:
    """SB3 2.9's rollout/update loop, with a stop request instead of a step cap.

    PPO's collect_rollouts and train are unchanged. Only fixed schedules are
    accepted; progress remains 1 because there is no predetermined end time.
    A stop during collection discards the partial rollout, not learned weights.
    """
    require_constant_schedules(model)
    if optimizer_guard is not None:
        optimizer_guard.validate_initial_state(model)
    _, callback = model._setup_learn(
        model.n_steps, callback, False, run_name, False,
    )
    callback.on_training_start(locals(), globals())
    iteration = 0
    try:
        while not should_stop():
            if not model.collect_rollouts(
                model.env, callback, model.rollout_buffer,
                n_rollout_steps=model.n_steps,
            ):
                break
            iteration += 1
            model._current_progress_remaining = 1.0
            if optimizer_guard is None:
                model.train()
            else:
                optimizer_guard.run_update(model)
            after_update()
            # Log after optimization so the latest update is not omitted at stop.
            model.dump_logs(iteration)
    finally:
        callback.on_training_end()


class ContinuousControl(BaseCallback):
    def __init__(
        self, run_dir: Path, checkpoint_dir: Path, *,
        interval: int = 250_000, minimum_free_bytes: int = 5 * 1024**3,
    ) -> None:
        super().__init__(verbose=0)
        self.run_dir, self.checkpoint_dir = run_dir, checkpoint_dir
        self.interval, self.minimum_free_bytes = interval, minimum_free_bytes
        self.stop_path = run_dir / "STOP.request"
        self.started = time.monotonic()
        self.last_status = 0.0
        self.stop_reason: str | None = None
        self.episodes = self.finishes = self.steps = 0
        self.recent: deque[dict[str, Any]] = deque(maxlen=500)
        self.best_ms: int | None = None
        self.latest_checkpoint: dict[str, Any] | None = None
        self.updates = 0
        self.starting_timesteps = self.learned_through = 0
        self.last_stop_check = 0.0

    def _on_training_start(self) -> None:
        self.started = time.monotonic()
        self.starting_timesteps = self.learned_through = int(self.model.num_timesteps)
        self.next_checkpoint = self.starting_timesteps + self.interval
        self.write_status("training")

    def should_stop(self) -> bool:
        now = time.monotonic()
        if now - self.last_stop_check >= 0.5:
            self.last_stop_check = now
            if self.stop_path.exists():
                self.stop_reason = "owner_requested_stop"
        return self.stop_reason is not None

    def _on_step(self) -> bool:
        for name in ("new_obs", "rewards"):
            if not np.isfinite(self.locals[name]).all():
                raise ValueError(f"nonfinite {name} during training")
        for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True):
            if done:
                self.episodes += 1
                finished = bool(info.get("race_finished", False))
                self.finishes += int(finished)
                lap = int(info["race_time_ms"])
                if finished:
                    self.best_ms = lap if self.best_ms is None else min(self.best_ms, lap)
                self.recent.append({"finished": finished, "lap_ms": lap})
        self.steps = int(self.model.num_timesteps) - self.starting_timesteps
        if time.monotonic() - self.last_status >= 30:
            self.check_disk()
            self.write_status("training")
        return not self.should_stop()

    def check_disk(self) -> None:
        if shutil.disk_usage(self.run_dir).free < self.minimum_free_bytes:
            raise OSError("less than 5 GiB free; stopping before disk exhaustion")

    def after_update(self) -> None:
        self.learned_through = int(self.model.num_timesteps)
        self.updates += 1
        for value in self.model.policy.parameters():
            if not value.detach().isfinite().all():
                raise ValueError("nonfinite policy parameter after optimization")
        if self.learned_through >= self.next_checkpoint:
            self.check_disk()
            additional = self.learned_through - self.starting_timesteps
            self.latest_checkpoint = save_checkpoint(
                self.model, self.checkpoint_dir / f"step_{additional:012d}.zip",
                learned_through=self.learned_through,
            )
            while self.next_checkpoint <= self.learned_through:
                self.next_checkpoint += self.interval
            self.write_status("training")

    def write_status(self, state: str, **extra: Any) -> dict[str, Any]:
        elapsed = time.monotonic() - self.started
        finishes = [x["lap_ms"] for x in self.recent if x["finished"]]
        value = {
            "status": state, "pid": os.getpid(), "updated_at_unix": time.time(),
            "additional_interactions": self.steps,
            "completed_updates": self.updates,
            "learned_through_timesteps": self.learned_through,
            "wall_seconds": elapsed,
            "interactions_per_second": self.steps / max(elapsed, 0.001),
            "episodes": self.episodes, "finishes": self.finishes,
            "finish_rate": self.finishes / self.episodes if self.episodes else None,
            "recent_500_finish_rate": (len(finishes) / len(self.recent)) if self.recent else None,
            "stochastic_best_ms": self.best_ms,
            "recent_500_stochastic_mean_ms": sum(finishes) / len(finishes) if finishes else None,
            "latest_checkpoint": self.latest_checkpoint,
            "stop_reason": self.stop_reason, **extra,
        }
        atomic_json(self.run_dir / "status.json", value)
        self.last_status = time.monotonic()
        print(json.dumps(value, separators=(",", ":")), flush=True)
        return value
