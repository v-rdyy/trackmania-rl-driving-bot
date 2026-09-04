"""Local-only controls for continuous PPO training; no model-service calls."""

from __future__ import annotations

import ctypes
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
from stable_baselines3.common.callbacks import BaseCallback


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
) -> None:
    """SB3 2.9's rollout/update loop, with a stop request instead of a step cap.

    PPO's collect_rollouts and train are unchanged. Only fixed schedules are
    accepted; progress remains 1 because there is no predetermined end time.
    A stop during collection discards the partial rollout, not learned weights.
    """
    require_constant_schedules(model)
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
            model.train()
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
