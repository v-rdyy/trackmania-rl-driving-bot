"""Run unchanged V4 PPO locally until STOP.request, with no evaluation gates."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_reward_v3 import close_model_logger, parse_power_setting_indices
from trackmania_rl.continuous_training import (
    ContinuousControl, KeepAwake, atomic_json, file_sha256,
    learn_until_stopped, require_constant_schedules, save_checkpoint,
)
from trackmania_rl.env import EnvironmentConfig, TrackmaniaEnv
from trackmania_rl.ppo_audit import BoundedPpoActionStatsCallback
from trackmania_rl.rewards import signed_progress_efficiency_reward

BASE = ROOT / "checkpoints/wr_chase_stage1/gate_01000000_model.zip"
BASE_HASH = "BA056E0B42D7CAEE4D02B6AB8E0D592BE6A363068E75AAEBC3ED8487791B4044"
BASE_STEPS = 3_004_416
REWARD = "clip(progress_delta, -20, 20) / 10 - 0.10 + 50 on finish - 250 on verified failure"


def verify_power() -> dict:
    settings = {}
    for group, alias in (
        ("SUB_SLEEP", "STANDBYIDLE"), ("SUB_SLEEP", "HIBERNATEIDLE"),
        ("SUB_SLEEP", "HYBRIDSLEEP"), ("SUB_VIDEO", "VIDEOIDLE"),
        ("SUB_DISK", "DISKIDLE"),
    ):
        output = subprocess.run(
            ["powercfg", "/query", "SCHEME_CURRENT", group, alias],
            check=True, capture_output=True, text=True,
        ).stdout
        settings[alias] = parse_power_setting_indices(output)
        if any(settings[alias].values()):
            raise RuntimeError(f"disable {alias} on AC/DC before launch")
    return settings


class ReplayEvidence(gym.Wrapper):
    """Save selected real game input replays before Monitor's auto-reset.

    Retain improving finishes per 250k block, every 1000th episode, and the
    first failure in each block. Replays support videos, NOT slide telemetry.
    """

    def __init__(self, env, run_name: str, run_dir: Path):
        super().__init__(env)
        self.run_name, self.run_dir = run_name, run_dir
        self.directory = ROOT / "artifacts/replays" / run_name
        self.directory.mkdir(parents=True, exist_ok=True)
        self.external = Path.home() / "Documents/TMInterface/Scripts"
        self.steps = self.episode = 0
        self.block = -1
        self.best = None
        self.failure_saved = False
        self.enabled = False

    def step(self, action):
        result = self.env.step(action)
        if not self.enabled:
            return result
        self.steps += 1
        block = self.steps // 250_000
        if block != self.block:
            self.block, self.best, self.failure_saved = block, None, False
        _, _, terminated, truncated, info = result
        if terminated or truncated:
            self.episode += 1
            lap = int(info["race_time_ms"])
            improved = terminated and (self.best is None or lap < self.best)
            first_failure = truncated and not self.failure_saved
            if improved or first_failure or self.episode % 1000 == 0:
                name = f"{self.run_name}_ep_{self.episode:08d}.txt"
                source, destination = self.external / name, self.directory / name
                if source.exists() or destination.exists():
                    raise FileExistsError(name)
                self.env.unwrapped.session.recover_inputs(name)
                # execute_command sends asynchronously. Wait for the game write
                # while it remains in the pending step, before any rewind.
                deadline = time.monotonic() + 5.0
                while not source.is_file() and time.monotonic() < deadline:
                    time.sleep(0.05)
                if not source.is_file() or source.stat().st_size == 0:
                    raise OSError(f"input recovery did not create {source}")
                shutil.copy2(source, destination)
                with (self.run_dir / "replays.jsonl").open("a", encoding="utf-8") as out:
                    out.write(json.dumps({
                        "episode": self.episode, "additional_interactions": self.steps,
                        "finished": bool(terminated), "race_time_ms": lap,
                        "replay": str(destination), "sha256": file_sha256(destination),
                    }) + "\n")
                if improved:
                    self.best = lap
                if first_failure:
                    self.failure_saved = True
        return result


def warmup(model, env, control, seconds: float) -> dict:
    obs, _ = env.reset()
    start = time.monotonic()
    steps = episodes = finishes = 0
    while time.monotonic() - start < seconds or episodes < 20:
        if control.should_stop():
            raise InterruptedError("owner stopped during preflight")
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if not np.isfinite(obs).all() or not np.isfinite(reward):
            raise ValueError("nonfinite warm-up data")
        steps += 1
        if terminated or truncated:
            episodes += 1
            finishes += int(terminated)
            obs, _ = env.reset()
        if time.monotonic() - start > seconds + 120:
            raise TimeoutError("warm-up could not complete 20 episodes")
    if not finishes:
        raise RuntimeError("warm-up produced no finishes; refusing unattended launch")
    elapsed = time.monotonic() - start
    return {"steps": steps, "episodes": episodes, "finishes": finishes,
            "wall_seconds": elapsed, "inference_steps_per_second": steps / elapsed,
            "optimizer_updates": 0, "simulation_speed": 100,
            "scope": "short current-host validation, not proof of overnight uptime"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--port", type=int, default=8478)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9_]+", args.run_name):
        raise ValueError("use lowercase letters, digits, and underscores in run name")
    run_dir = ROOT / "runs" / args.run_name
    checkpoints = ROOT / "checkpoints" / args.run_name
    # Exclusive creation prevents duplicate trainers or accidental overwritten runs.
    run_dir.mkdir(parents=True, exist_ok=False)
    control = ContinuousControl(run_dir, checkpoints)
    control.write_status("preflight")
    manifest = {"run_name": args.run_name, "pid": os.getpid(), "status": "preflight",
                "started_at_unix": time.time(), "reward": REWARD,
                "base_checkpoint": str(BASE), "base_sha256": BASE_HASH,
                "checkpoint_interval_nominal": 250_000,
                "duration": "until owner stop or health failure", "evaluation_gates": False,
                "no_api_or_codex_calls": True,
                "protocol_sha256": file_sha256(ROOT / "docs/overnight-pure-discovery-preflight.md"),
                "script_sha256": file_sha256(Path(__file__)),
                "core_sha256": file_sha256(ROOT / "src/trackmania_rl/continuous_training.py")}
    atomic_json(run_dir / "run_manifest.json", manifest)
    env = model = None
    audit = BoundedPpoActionStatsCallback()
    def interrupt(*_):
        control.stop_reason = "interrupt_requested"
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    try:
        manifest["power_settings"] = verify_power()
        control.check_disk()
        manifest["initial_free_bytes"] = shutil.disk_usage(run_dir).free
        if file_sha256(BASE) != BASE_HASH:
            raise ValueError("starting checkpoint hash changed")
        with KeepAwake():
            base_env = TrackmaniaEnv(config=EnvironmentConfig(
                port=args.port, simulation_speed=100.0,
                stuck_window_ms=2000, stuck_progress_gain_units=1.0,
                stuck_world_distance_units=2.0, legacy_reversed_pedal_mapping=False,
                # A01 is opened during preflight; no blind UI automation in trainer.
                map_to_load=None, auto_respawn_on_connect=True,
            ), reward_function=signed_progress_efficiency_reward)
            env = base_env
            model = PPO.load(BASE, device="cpu")
            require_constant_schedules(model)
            if model.num_timesteps != BASE_STEPS or model.target_kl is not None:
                raise ValueError("unexpected base timestep or KL setting")
            if not model.use_sde or not model.policy.squash_output:
                raise ValueError("expected bounded gSDE PPO")
            manifest["warmup"] = warmup(model, env, control, 60.0)
            if file_sha256(BASE) != BASE_HASH or model.num_timesteps != BASE_STEPS:
                raise ValueError("warm-up unexpectedly changed the base")
            manifest["status"] = "preflight_passed"
            atomic_json(run_dir / "run_manifest.json", manifest)
            print("LIVE_PREFLIGHT_PASSED " + json.dumps(manifest["warmup"]), flush=True)
            if args.preflight_only:
                control.write_status("preflight_passed", warmup=manifest["warmup"])
                return 0
            evidence = ReplayEvidence(base_env, args.run_name, run_dir)
            evidence.enabled = True
            env = Monitor(evidence, str(run_dir / "training"), info_keywords=(
                "race_time_ms", "timeout", "off_track", "fallen", "stuck",
                "race_finished", "progress", "display_speed",
            ))
            model.set_env(env)
            model.tensorboard_log = str(ROOT / "tensorboard")
            model.verbose = 0
            manifest["status"] = "training"
            manifest["training_started_at_unix"] = time.time()
            manifest["ppo"] = {k: getattr(model, k) for k in (
                "n_steps", "n_epochs", "batch_size", "learning_rate", "gamma",
                "gae_lambda", "ent_coef", "vf_coef", "target_kl", "use_sde",
                "sde_sample_freq", "max_grad_norm",
            )}
            manifest["ppo"]["clip_range"] = float(model.clip_range(1))
            atomic_json(run_dir / "run_manifest.json", manifest)
            learn_until_stopped(model, CallbackList([audit, control]),
                                control.should_stop, control.after_update,
                                run_name=args.run_name)
            control.latest_checkpoint = save_checkpoint(
                model, checkpoints / "final_model.zip", learned_through=control.learned_through,
            )
            control.write_status("stopped", action_validation=audit.summary() if audit.records_checked else None)
            manifest["status"] = "stopped"
    except BaseException as error:
        extra = {"error": repr(error)}
        if model is not None and control.updates > 0:
            try:
                extra["emergency_checkpoint"] = save_checkpoint(
                    model, checkpoints / "emergency_model.zip", learned_through=control.learned_through,
                )
            except Exception as save_error:
                extra["emergency_save_error"] = repr(save_error)
        control.write_status("failed", **extra)
        manifest.update(status="failed", **extra)
        raise
    finally:
        manifest["last_updated_at_unix"] = time.time()
        atomic_json(run_dir / "run_manifest.json", manifest)
        if env is not None:
            env.close()
        if model is not None:
            close_model_logger(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
