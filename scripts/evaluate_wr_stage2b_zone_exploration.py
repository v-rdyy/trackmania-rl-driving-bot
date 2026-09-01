"""Run and compare the frozen 20v20 final-zone exploration ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from stable_baselines3 import PPO

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))

import evaluate_reward_v3 as base_evaluator
from trackmania_rl.env import EnvironmentConfig, TrackmaniaEnv
from trackmania_rl.evaluation_metrics import aggregate_precision_metrics
from trackmania_rl.game_launch import close_trackmania, ensure_trackmania_running
from trackmania_rl.observations import ReferencePath
from trackmania_rl.rewards import signed_progress_efficiency_reward
from trackmania_rl.slide_scoring import (
    FINAL_CORNER_ZONE,
    paired_usefulness,
    score_episode,
    valid_slide_onset_sample,
)
from trackmania_rl.tmi_bridge import ProtocolError
from trackmania_rl.zone_exploration import (
    DEFAULT_ZONE_STD_MULTIPLIER,
    FinalCornerZoneGsdPolicy,
    FinalCornerZoneObservationWrapper,
)

RUN_ROOT = WORKSPACE_ROOT / "runs" / "wr_chase_stage2b_zone_exploration"
REPLAY_ROOT = (
    WORKSPACE_ROOT / "artifacts" / "replays" / "wr_chase_stage2b_zone_exploration"
)
ANALYSIS_PATH = RUN_ROOT / "analysis.json"
SELECTION_PATH = RUN_ROOT / "representative_selection.json"
CHECKPOINT = (
    WORKSPACE_ROOT / "checkpoints" / "wr_chase_stage2" / "gate_00500000_model.zip"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "8A06E00055886E8E671E988D5D6948C6688B74780EDB32A87C6CC213870C8326"
)
REFERENCE_PATH = WORKSPACE_ROOT / "data" / "tracks" / "a01_reference_path.csv"
DEFAULT_TMI_SCRIPTS = Path.home() / "Documents" / "TMInterface" / "Scripts"
ARMS = ("control", "boosted")
EPISODES = 20
SEED_START = 20_260_830
SEEDS = tuple(range(SEED_START, SEED_START + EPISODES))
SIMULATION_SPEED = 6.0
SDE_SAMPLE_FREQUENCY = 4
MAP_TO_LOAD = "A01-Race.Challenge.Gbx"
ACTION_TOLERANCE = 1e-6
PROGRESS_TOLERANCE = 2.0
POSITION_TOLERANCE = 2.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(WORKSPACE_ROOT))


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, separators=(",", ":"), allow_nan=False))
            output.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _update_fingerprint(digest: Any, value: Any) -> None:
    if isinstance(value, th.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor")
        digest.update(str(tensor.dtype).encode())
        digest.update(repr(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"ndarray")
        digest.update(str(array.dtype).encode())
        digest.update(repr(tuple(array.shape)).encode())
        digest.update(array.tobytes())
    elif isinstance(value, Mapping):
        digest.update(b"mapping")
        for key in sorted(value, key=lambda item: repr(item)):
            _update_fingerprint(digest, key)
            _update_fingerprint(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode())
        for item in value:
            _update_fingerprint(digest, item)
    elif value is None or isinstance(value, (str, int, float, bool)):
        digest.update(type(value).__name__.encode())
        digest.update(repr(value).encode())
    else:
        raise TypeError(f"unsupported fingerprint value: {type(value).__name__}")


def state_fingerprint(value: Any) -> str:
    digest = hashlib.sha256()
    _update_fingerprint(digest, value)
    return digest.hexdigest().upper()


def arm_paths(arm: str) -> dict[str, Path]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    run_dir = RUN_ROOT / arm
    return {
        "run_dir": run_dir,
        "action_log": run_dir / "actions.jsonl",
        "policy_trace": run_dir / "policy_trace.jsonl",
        "summary": run_dir / "summary.json",
        "failure": run_dir / "failure.json",
        "replay_dir": REPLAY_ROOT / arm,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--arm", choices=ARMS)
    operation.add_argument("--compare", action="store_true")
    parser.add_argument("--port", type=int, default=8478)
    parser.add_argument("--tmi-scripts-dir", type=Path, default=DEFAULT_TMI_SCRIPTS)
    parser.add_argument("--reuse-game", action="store_true")
    return parser.parse_args()


def _wait_for_replay(path: Path, timeout_seconds: float = 5.0) -> None:
    deadline = time.perf_counter() + timeout_seconds
    while time.perf_counter() < deadline:
        if path.is_file() and path.stat().st_size > 0:
            return
        time.sleep(0.05)
    raise ProtocolError(f"TMInterface did not create input replay: {path}")


def _load_zone_model(env: FinalCornerZoneObservationWrapper) -> PPO:
    model = PPO.load(
        CHECKPOINT,
        env=env,
        device="cpu",
        custom_objects={
            "policy_class": FinalCornerZoneGsdPolicy,
            "observation_space": env.observation_space,
        },
    )
    if not isinstance(model.policy, FinalCornerZoneGsdPolicy):
        raise ProtocolError("checkpoint did not load with the conditional gSDE policy")
    if not math.isclose(
        float(model.policy.zone_std_multiplier),
        DEFAULT_ZONE_STD_MULTIPLIER,
    ):
        raise ProtocolError("conditional gSDE multiplier is not the frozen 2x")
    if not model.use_sde or not model.policy.use_sde:
        raise ProtocolError("zone exploration checkpoint must use gSDE")
    return model


def _distribution_audit(model: PPO, observation: np.ndarray) -> dict[str, Any]:
    base = np.asarray(observation, dtype=np.float32).copy()
    if base.shape != (27,):
        raise ProtocolError("zone model observation does not contain 27 values")
    outside = base.copy()
    inside = base.copy()
    outside[-1] = 0.0
    inside[-1] = 1.0
    outside_tensor = th.as_tensor(outside[None, :])
    inside_tensor = th.as_tensor(inside[None, :])
    with th.no_grad():
        outside_action, outside_value, _ = model.policy(
            outside_tensor, deterministic=True
        )
        inside_action, inside_value, _ = model.policy(inside_tensor, deterministic=True)
        outside_std = model.policy.get_distribution(outside_tensor).distribution.stddev
        inside_std = model.policy.get_distribution(inside_tensor).distribution.stddev
    action_error = float(th.max(th.abs(inside_action - outside_action)).item())
    value_error = float(th.max(th.abs(inside_value - outside_value)).item())
    std_error = float(
        th.max(
            th.abs(
                inside_std
                - outside_std * float(DEFAULT_ZONE_STD_MULTIPLIER)
            )
        ).item()
    )
    passed = action_error <= 1e-7 and value_error <= 1e-7 and std_error <= 2e-5
    if not passed:
        raise ProtocolError("conditional gSDE distribution failed its 2x audit")
    return {
        "passed": True,
        "deterministic_action_max_abs_error": action_error,
        "value_max_abs_error": value_error,
        "two_x_std_max_abs_error": std_error,
    }


def _process_live_records(
    action_log: Path, episode_details: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]], dict[str, Any]]:
    raw_records = read_jsonl(action_log)
    if not all(
        record.get("finite") and record.get("within_range") and record.get("valid")
        for record in raw_records
    ):
        raise ProtocolError("zone ablation action log contains an invalid action")
    grouped: dict[int, list[dict[str, Any]]] = {
        episode: [] for episode in range(EPISODES)
    }
    for record in raw_records:
        episode = int(record["episode"])
        if episode not in grouped:
            raise ProtocolError("zone ablation action log has an unexpected episode")
        grouped[episode].append(record)
    evaluated: dict[int, list[dict[str, Any]]] = {}
    discarded = 0
    for episode, records in grouped.items():
        active, startup = base_evaluator.evaluated_race_records(records)
        if not active:
            raise ProtocolError(f"zone ablation episode {episode} has no active rows")
        evaluated[episode] = active
        discarded += startup
        episode_details[episode]["trajectory"] = base_evaluator.trajectory_metrics(active)
    base_evaluator.validate_action_record_count(
        raw_records=len(raw_records),
        evaluated_records=sum(len(rows) for rows in evaluated.values()),
        discarded_startup_records=discarded,
    )
    precision = aggregate_precision_metrics(
        [detail["trajectory"]["precision"] for detail in episode_details]
    )
    return raw_records, evaluated, {
        "discarded_startup_records": discarded,
        "precision_summary": precision,
    }


def run_arm(
    arm: str,
    *,
    port: int,
    tmi_scripts_dir: Path,
    reuse_game: bool,
) -> dict[str, Any]:
    paths = arm_paths(arm)
    for key in ("action_log", "policy_trace", "summary", "failure"):
        if paths[key].exists():
            raise FileExistsError(f"refusing to overwrite {paths[key]}")
    if paths["replay_dir"].exists() and any(paths["replay_dir"].iterdir()):
        raise FileExistsError(f"refusing to overwrite {paths['replay_dir']}")
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    paths["replay_dir"].mkdir(parents=True, exist_ok=True)
    if not tmi_scripts_dir.is_dir():
        raise FileNotFoundError(tmi_scripts_dir)
    if not CHECKPOINT.is_file() or sha256(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        raise ProtocolError("Gate 500,000 checkpoint is missing or changed")
    if not REFERENCE_PATH.is_file():
        raise FileNotFoundError(REFERENCE_PATH)

    if not reuse_game:
        close_trackmania()
    _, launched = ensure_trackmania_running(port=port, confirm_existing=True)
    print(f"TrackMania ready (launched={launched})", flush=True)

    reference = ReferencePath.from_csv(REFERENCE_PATH)
    base_env = TrackmaniaEnv(
        config=EnvironmentConfig(
            port=port,
            simulation_speed=SIMULATION_SPEED,
            stuck_window_ms=2_000,
            stuck_progress_gain_units=1.0,
            stuck_world_distance_units=2.0,
            legacy_reversed_pedal_mapping=False,
            map_to_load=MAP_TO_LOAD,
            auto_respawn_on_connect=False,
            wait_for_race_start_on_connect=True,
            bridge_response_timeout_ms=90_000,
        ),
        reward_function=signed_progress_efficiency_reward,
        action_log_path=paths["action_log"],
        reference_path=reference,
    )
    env = FinalCornerZoneObservationWrapper(base_env, enabled=arm == "boosted")
    model = _load_zone_model(env)
    model_timesteps_before = int(model.num_timesteps)
    policy_fingerprint_before = state_fingerprint(model.policy.state_dict())
    optimizer_fingerprint_before = state_fingerprint(
        model.policy.optimizer.state_dict()
    )
    checkpoint_hash_before = sha256(CHECKPOINT)
    trace: list[dict[str, Any]] = []
    episode_details: list[dict[str, Any]] = []
    distribution_audit: dict[str, Any] | None = None
    try:
        for episode, seed in enumerate(SEEDS):
            model.set_random_seed(seed)
            observation, current_info = env.reset(seed=seed)
            if distribution_audit is None:
                distribution_audit = _distribution_audit(model, observation)
            terminated = False
            truncated = False
            steps = 0
            total_reward = 0.0
            final_info = current_info
            while not (terminated or truncated):
                if steps % SDE_SAMPLE_FREQUENCY == 0:
                    model.policy.reset_noise(1)
                pre_action_progress = float(current_info["progress"])
                zone_flag = float(observation[-1])
                expected_flag = float(
                    arm == "boosted"
                    and FINAL_CORNER_ZONE[0]
                    <= pre_action_progress
                    <= FINAL_CORNER_ZONE[1]
                )
                if zone_flag != expected_flag:
                    raise ProtocolError("zone flag disagrees with pre-action progress")
                action, _ = model.predict(observation, deterministic=False)
                action_values = np.asarray(action, dtype=np.float64)
                if action_values.shape != (3,) or not np.isfinite(action_values).all():
                    raise ProtocolError("zone ablation policy emitted an invalid action")
                next_observation, reward, terminated, truncated, final_info = env.step(
                    action
                )
                if not np.isfinite(next_observation).all() or not math.isfinite(
                    float(reward)
                ):
                    raise ProtocolError("zone ablation received nonfinite live data")
                trace.append(
                    {
                        "arm": arm,
                        "episode": episode,
                        "seed": seed,
                        "step": steps,
                        "pre_action_progress": pre_action_progress,
                        "zone_flag": zone_flag,
                        "zone_noise_scale": 1.0
                        + zone_flag * (DEFAULT_ZONE_STD_MULTIPLIER - 1.0),
                        "noise_refreshed": steps % SDE_SAMPLE_FREQUENCY == 0,
                        "raw_action": action_values.tolist(),
                        "post_action_progress": float(final_info["progress"]),
                        "post_action_position": [
                            float(value) for value in final_info["position"]
                        ],
                        "race_time_ms": int(final_info["race_time_ms"]),
                    }
                )
                observation = next_observation
                current_info = final_info
                total_reward += float(reward)
                steps += 1
                if steps > 451:
                    raise ProtocolError("zone ablation episode exceeded safety limit")

            filename = f"wr_stage2b_zone_{arm}_seed_{seed}.txt"
            external_replay = tmi_scripts_dir / filename
            local_replay = paths["replay_dir"] / filename
            if external_replay.exists() or local_replay.exists():
                raise FileExistsError(f"refusing to replace replay {filename}")
            base_env.session.recover_inputs(filename)
            _wait_for_replay(external_replay)
            shutil.copy2(external_replay, local_replay)
            episode_details.append(
                {
                    "episode": episode,
                    "seed": seed,
                    "steps": steps,
                    "elapsed_ms": int(final_info["elapsed_ms"]),
                    "terminal_race_time_ms": int(final_info["race_time_ms"]),
                    "total_reward": total_reward,
                    "finished": bool(terminated),
                    "timeout": bool(final_info["timeout"]),
                    "off_track": bool(final_info["off_track"]),
                    "fallen": bool(final_info["fallen"]),
                    "stuck": bool(final_info["stuck"]),
                    "final_progress": float(final_info["progress"]),
                    "input_replay": _relative(local_replay),
                    "input_replay_bytes": local_replay.stat().st_size,
                    "input_replay_sha256": sha256(local_replay),
                }
            )
            print(
                f"{arm} episode={episode + 1}/{EPISODES} seed={seed} "
                f"finished={terminated} time={int(final_info['race_time_ms'])}",
                flush=True,
            )
    except Exception as error:
        write_jsonl(paths["policy_trace"], trace)
        write_json(
            paths["failure"],
            {
                "status": "failed",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "arm": arm,
                "error": repr(error),
                "completed_episodes": episode_details,
                "checkpoint_sha256": checkpoint_hash_before,
                "diagnostic_only": True,
                "training_interactions": 0,
            },
        )
        raise
    finally:
        env.close()

    write_jsonl(paths["policy_trace"], trace)
    raw_records, evaluated, processing = _process_live_records(
        paths["action_log"], episode_details
    )
    evaluated_records = [
        row for episode in range(EPISODES) for row in evaluated[episode]
    ]
    actions = np.asarray(
        [row["raw_action"] for row in evaluated_records], dtype=np.float64
    )
    finishes = sum(bool(detail["finished"]) for detail in episode_details)
    time_metrics = base_evaluator.lap_time_metrics(episode_details)
    model_timesteps_after = int(model.num_timesteps)
    policy_fingerprint_after = state_fingerprint(model.policy.state_dict())
    optimizer_fingerprint_after = state_fingerprint(
        model.policy.optimizer.state_dict()
    )
    checkpoint_hash_after = sha256(CHECKPOINT)
    integrity = {
        "checkpoint_sha256_before": checkpoint_hash_before,
        "checkpoint_sha256_after": checkpoint_hash_after,
        "model_timesteps_before": model_timesteps_before,
        "model_timesteps_after": model_timesteps_after,
        "policy_state_fingerprint_before": policy_fingerprint_before,
        "policy_state_fingerprint_after": policy_fingerprint_after,
        "optimizer_state_fingerprint_before": optimizer_fingerprint_before,
        "optimizer_state_fingerprint_after": optimizer_fingerprint_after,
    }
    integrity["unchanged"] = bool(
        checkpoint_hash_before == checkpoint_hash_after
        and model_timesteps_before == model_timesteps_after
        and policy_fingerprint_before == policy_fingerprint_after
        and optimizer_fingerprint_before == optimizer_fingerprint_after
    )
    if not integrity["unchanged"]:
        raise ProtocolError("fixed-policy ablation changed model or checkpoint state")
    summary = {
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "wr-chase-plan.md Stage 2b pre-design exploration ablation",
        "arm": arm,
        "diagnostic_only": True,
        "training_interactions": 0,
        "learn_called": False,
        "checkpoint_written": False,
        "reward_function": "signed_progress_efficiency_reward",
        "episodes": EPISODES,
        "seeds": list(SEEDS),
        "matched_seed_protocol": True,
        "simulation_speed": SIMULATION_SPEED,
        "policy_sampling": {
            "deterministic": False,
            "use_sde": True,
            "sde_sample_frequency": SDE_SAMPLE_FREQUENCY,
            "zone_flag_enabled": arm == "boosted",
            "outside_zone_std_multiplier": 1.0,
            "inside_zone_std_multiplier": (
                DEFAULT_ZONE_STD_MULTIPLIER if arm == "boosted" else 1.0
            ),
        },
        "distribution_audit": distribution_audit,
        "model_integrity": integrity,
        "checkpoint": _relative(CHECKPOINT),
        "checkpoint_sha256": checkpoint_hash_before,
        "reference_path": _relative(REFERENCE_PATH),
        "reference_path_sha256": sha256(REFERENCE_PATH),
        "action_log": _relative(paths["action_log"]),
        "action_log_sha256": sha256(paths["action_log"]),
        "policy_trace": _relative(paths["policy_trace"]),
        "policy_trace_sha256": sha256(paths["policy_trace"]),
        "raw_action_records": len(raw_records),
        "evaluated_action_records": len(evaluated_records),
        "discarded_startup_action_records": processing[
            "discarded_startup_records"
        ],
        "zone_flagged_action_count": sum(
            float(row["zone_flag"]) == 1.0 for row in trace
        ),
        "finishes": finishes,
        "finish_rate": finishes / EPISODES,
        "falls": sum(bool(detail["fallen"]) for detail in episode_details),
        "off_tracks": sum(bool(detail["off_track"]) for detail in episode_details),
        "stuck_truncations": sum(
            bool(detail["stuck"]) for detail in episode_details
        ),
        "timeouts": sum(bool(detail["timeout"]) for detail in episode_details),
        "precision_summary": processing["precision_summary"],
        "action_minimum": actions.min(axis=0).tolist(),
        "action_maximum": actions.max(axis=0).tolist(),
        "action_mean": actions.mean(axis=0).tolist(),
        **time_metrics,
        "input_replay_count": len(episode_details),
        "episodes_detail": episode_details,
    }
    write_json(paths["summary"], summary)
    print(
        f"{arm} complete: finishes={finishes}/{EPISODES} "
        f"summary_sha256={sha256(paths['summary'])}",
        flush=True,
    )
    return summary


def _records_by_episode(path: Path) -> dict[int, list[dict[str, Any]]]:
    grouped = {episode: [] for episode in range(EPISODES)}
    for row in read_jsonl(path):
        grouped[int(row["episode"])].append(row)
    return {
        episode: base_evaluator.evaluated_race_records(rows)[0]
        for episode, rows in grouped.items()
    }


def _trace_by_episode(path: Path) -> dict[int, list[dict[str, Any]]]:
    grouped = {episode: [] for episode in range(EPISODES)}
    for row in read_jsonl(path):
        grouped[int(row["episode"])].append(row)
    return grouped


def paired_prezone_audit(
    control_trace: Mapping[int, Sequence[Mapping[str, Any]]],
    boosted_trace: Mapping[int, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Audit matched RNG protocol and quantify live pre-zone trajectory jitter."""

    comparisons = 0
    maximum_action_error = 0.0
    maximum_progress_error = 0.0
    maximum_position_error = 0.0
    failures: list[dict[str, Any]] = []
    episode_details: list[dict[str, Any]] = []
    for episode, seed in enumerate(SEEDS):
        control = list(control_trace[episode])
        boosted = list(boosted_trace[episode])
        first_active = next(
            (
                index
                for index, row in enumerate(boosted)
                if float(row["zone_flag"]) == 1.0
            ),
            min(len(control), len(boosted)),
        )
        treatment_activated = any(
            float(row["zone_flag"]) == 1.0 for row in boosted
        )
        if not treatment_activated and len(control) != len(boosted):
            failures.append(
                {
                    "episode": episode,
                    "reason": "preactivation_trace_length",
                    "control_steps": len(control),
                    "boosted_steps": len(boosted),
                }
            )
        if any(float(row["zone_flag"]) != 0.0 for row in control):
            failures.append({"episode": episode, "reason": "control_zone_flag"})
        episode_action_error = 0.0
        episode_progress_error = 0.0
        episode_position_error = 0.0
        for index in range(first_active):
            if index >= len(control) or index >= len(boosted):
                failures.append({"episode": episode, "reason": "trace_length"})
                break
            left = control[index]
            right = boosted[index]
            if int(left["seed"]) != seed or int(right["seed"]) != seed:
                failures.append({"episode": episode, "reason": "seed"})
                break
            action_error = float(
                np.max(
                    np.abs(
                        np.asarray(left["raw_action"], dtype=np.float64)
                        - np.asarray(right["raw_action"], dtype=np.float64)
                    )
                )
            )
            progress_error = abs(
                float(left["post_action_progress"])
                - float(right["post_action_progress"])
            )
            position_error = float(
                np.linalg.norm(
                    np.asarray(left["post_action_position"], dtype=np.float64)
                    - np.asarray(right["post_action_position"], dtype=np.float64)
                )
            )
            maximum_action_error = max(maximum_action_error, action_error)
            maximum_progress_error = max(maximum_progress_error, progress_error)
            maximum_position_error = max(maximum_position_error, position_error)
            episode_action_error = max(episode_action_error, action_error)
            episode_progress_error = max(episode_progress_error, progress_error)
            episode_position_error = max(episode_position_error, position_error)
            comparisons += 1
            if int(left.get("step", -1)) != index or int(right.get("step", -1)) != index:
                failures.append(
                    {
                        "episode": episode,
                        "step": index,
                        "reason": "step_cadence",
                    }
                )
                break
            if bool(left.get("noise_refreshed")) != bool(
                right.get("noise_refreshed")
            ):
                failures.append(
                    {
                        "episode": episode,
                        "step": index,
                        "reason": "noise_refresh_cadence",
                    }
                )
                break
        if first_active > 0:
            entry_control = control[first_active - 1]
            entry_boosted = boosted[first_active - 1]
            entry_progress_error = abs(
                float(entry_control["post_action_progress"])
                - float(entry_boosted["post_action_progress"])
            )
            entry_position_error = float(
                np.linalg.norm(
                    np.asarray(entry_control["post_action_position"], dtype=np.float64)
                    - np.asarray(entry_boosted["post_action_position"], dtype=np.float64)
                )
            )
        else:
            entry_progress_error = 0.0
            entry_position_error = 0.0
        episode_details.append(
            {
                "episode": episode,
                "seed": seed,
                "treatment_activated": treatment_activated,
                "preactivation_comparisons": first_active,
                "maximum_action_abs_error": episode_action_error,
                "maximum_progress_error": episode_progress_error,
                "maximum_position_error": episode_position_error,
                "zone_entry_progress_error": entry_progress_error,
                "zone_entry_position_error": entry_position_error,
                "zone_entry_comparable": bool(
                    treatment_activated
                    and entry_progress_error <= PROGRESS_TOLERANCE
                    and entry_position_error <= POSITION_TOLERANCE
                ),
            }
        )
    return {
        "passed": not failures,
        "comparisons": comparisons,
        "maximum_action_abs_error": maximum_action_error,
        "maximum_progress_error": maximum_progress_error,
        "maximum_position_error": maximum_position_error,
        "tolerances": {
            "action_descriptive_only": ACTION_TOLERANCE,
            "progress": PROGRESS_TOLERANCE,
            "position": POSITION_TOLERANCE,
        },
        "zone_entry_comparable_episodes": sum(
            bool(detail["zone_entry_comparable"]) for detail in episode_details
        ),
        "episode_details": episode_details,
        "failures": failures,
    }


def _metric_distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "p50": None, "p95": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def _aggregate_arm(
    summary: Mapping[str, Any],
    records: Mapping[int, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    episode_scores: list[dict[str, Any]] = []
    slips: list[float] = []
    yaws: list[float] = []
    for episode, seed in enumerate(SEEDS):
        score = score_episode(records[episode])
        score["episode"] = episode
        score["seed"] = seed
        episode_scores.append(score)
        for row in records[episode]:
            if valid_slide_onset_sample(row):
                slips.append(abs(float(row["slip_angle_degrees"])))
                yaws.append(abs(float(row["body_up_yaw_rate"])))
    final_scores = [score["final_corner"] for score in episode_scores]
    first_scores = [score["first_turn_negative_control"] for score in episode_scores]
    aggregate = {
        "episodes": EPISODES,
        "finishes": int(summary["finishes"]),
        "finish_rate": float(summary["finish_rate"]),
        "best_finish_time_ms": summary.get("best_finish_time_ms"),
        "average_finish_time_ms": summary.get("average_finish_time_ms"),
        "worst_finish_time_ms": summary.get("worst_finish_time_ms"),
        "raw_slide_onset_episodes": sum(
            int(score["raw_slide_onset_count"]) > 0 for score in final_scores
        ),
        "raw_slide_onset_windows": sum(
            int(score["raw_slide_onset_count"]) for score in final_scores
        ),
        "accepted_slide_onset_episodes_before_visual": sum(
            int(score["accepted_slide_onset_count"]) > 0 for score in final_scores
        ),
        "accepted_slide_onset_windows_before_visual": sum(
            int(score["accepted_slide_onset_count"]) for score in final_scores
        ),
        "above_envelope_near_miss_episodes": sum(
            int(score["above_envelope_near_miss_count"]) > 0
            for score in final_scores
        ),
        "above_envelope_near_miss_windows": sum(
            int(score["above_envelope_near_miss_count"]) for score in final_scores
        ),
        "legacy_candidate_episodes": sum(
            int(score["legacy_candidate_count"]) > 0 for score in final_scores
        ),
        "legacy_confirmed_episodes": sum(
            int(score["legacy_confirmed_count"]) > 0 for score in final_scores
        ),
        "final_zone_slip_angle_degrees": _metric_distribution(slips),
        "final_zone_abs_body_up_yaw_rate": _metric_distribution(yaws),
        "first_turn_negative_control": {
            "raw_slide_onset_episodes": sum(
                int(score["raw_slide_onset_count"]) > 0 for score in first_scores
            ),
            "accepted_slide_onset_episodes_before_visual": sum(
                int(score["accepted_slide_onset_count"]) > 0
                for score in first_scores
            ),
        },
    }
    return aggregate, episode_scores


def _representatives(
    arm: str, summary: Mapping[str, Any], scores: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    details = list(summary["episodes_detail"])
    finished = [detail for detail in details if bool(detail["finished"])]
    if finished:
        times = [int(detail["terminal_race_time_ms"]) for detail in finished]
        mean_time = statistics.fmean(times)
        selected = {
            "best": min(finished, key=lambda item: int(item["terminal_race_time_ms"])),
            "average": min(
                finished,
                key=lambda item: abs(int(item["terminal_race_time_ms"]) - mean_time),
            ),
            "worst": max(finished, key=lambda item: int(item["terminal_race_time_ms"])),
        }
    else:
        selected = {
            "best": max(details, key=lambda item: float(item["final_progress"])),
            "average": sorted(details, key=lambda item: float(item["final_progress"]))[
                len(details) // 2
            ],
            "worst": min(details, key=lambda item: float(item["final_progress"])),
        }
    by_episode = {int(detail["episode"]): detail for detail in details}
    onset_episodes = [
        int(score["episode"])
        for score in scores
        if int(score["final_corner"]["accepted_slide_onset_count"]) > 0
    ]
    return {
        "arm": arm,
        "representatives": {
            label: {
                "episode": int(detail["episode"]),
                "seed": int(detail["seed"]),
                "finished": bool(detail["finished"]),
                "terminal_race_time_ms": int(detail["terminal_race_time_ms"]),
                "input_replay": detail["input_replay"],
                "input_replay_sha256": detail["input_replay_sha256"],
            }
            for label, detail in selected.items()
        },
        "accepted_onset_episodes_before_visual": [
            {
                "episode": episode,
                "seed": int(by_episode[episode]["seed"]),
                "input_replay": by_episode[episode]["input_replay"],
                "input_replay_sha256": by_episode[episode]["input_replay_sha256"],
            }
            for episode in onset_episodes
        ],
    }


def compare_arms() -> dict[str, Any]:
    summaries = {
        arm: json.loads(arm_paths(arm)["summary"].read_text(encoding="utf-8"))
        for arm in ARMS
    }
    for arm, summary in summaries.items():
        if (
            summary.get("status") != "complete"
            or summary.get("diagnostic_only") is not True
            or int(summary.get("training_interactions", -1)) != 0
            or summary.get("learn_called") is not False
            or summary.get("checkpoint_written") is not False
            or summary.get("reward_function") != "signed_progress_efficiency_reward"
            or summary.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256
            or summary.get("model_integrity", {}).get("unchanged") is not True
        ):
            raise ProtocolError(f"{arm} summary violates the frozen protocol")
    records = {
        arm: _records_by_episode(arm_paths(arm)["action_log"]) for arm in ARMS
    }
    traces = {arm: _trace_by_episode(arm_paths(arm)["policy_trace"]) for arm in ARMS}
    pairing = paired_prezone_audit(traces["control"], traces["boosted"])
    if not pairing["passed"]:
        raise ProtocolError("control/treatment traces are not matched before the zone")
    aggregates: dict[str, Any] = {}
    scores: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        aggregates[arm], scores[arm] = _aggregate_arm(summaries[arm], records[arm])
    usefulness = []
    pairing_by_episode = {
        int(detail["episode"]): detail
        for detail in pairing["episode_details"]
    }
    for episode, seed in enumerate(SEEDS):
        result = paired_usefulness(scores["control"][episode], scores["boosted"][episode])
        entry_comparable = bool(
            pairing_by_episode[episode]["zone_entry_comparable"]
        )
        if not entry_comparable:
            result["useful_candidate_before_visual_review"] = False
        result.update(
            {
                "episode": episode,
                "seed": seed,
                "zone_entry_comparable": entry_comparable,
                "excluded_for_entry_mismatch": not entry_comparable,
            }
        )
        usefulness.append(result)
    control_onsets = int(aggregates["control"]["accepted_slide_onset_episodes_before_visual"])
    boosted_onsets = int(aggregates["boosted"]["accepted_slide_onset_episodes_before_visual"])
    gate_before_visual = bool(boosted_onsets >= 3 and boosted_onsets - control_onsets >= 2)
    analysis = {
        "status": "awaiting_visual_review" if gate_before_visual else "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "wr-chase-plan.md frozen non-human scoring contract",
        "diagnostic_only": True,
        "training_interactions": 0,
        "raw_onset_comparison": {
            "control_episodes": aggregates["control"]["raw_slide_onset_episodes"],
            "control_windows": aggregates["control"]["raw_slide_onset_windows"],
            "boosted_episodes": aggregates["boosted"]["raw_slide_onset_episodes"],
            "boosted_windows": aggregates["boosted"]["raw_slide_onset_windows"],
            "episode_delta_boosted_minus_control": int(
                aggregates["boosted"]["raw_slide_onset_episodes"]
            )
            - int(aggregates["control"]["raw_slide_onset_episodes"]),
        },
        "accepted_onset_gate_before_visual": {
            "control_episodes": control_onsets,
            "boosted_episodes": boosted_onsets,
            "episode_delta_boosted_minus_control": boosted_onsets - control_onsets,
            "requires_boosted_episodes": 3,
            "requires_episode_delta": 2,
            "passes_before_visual_review": gate_before_visual,
            "final_decision": "pending_visual_review" if gate_before_visual else "no_effect",
        },
        "usefulness_before_visual": {
            "candidate_episodes": sum(
                bool(result["useful_candidate_before_visual_review"])
                for result in usefulness
            ),
            "matched_episode_results": usefulness,
            "final_genuine_useful_speedslide_count": None
            if gate_before_visual
            else 0,
        },
        "paired_prezone_audit": pairing,
        "arms": aggregates,
        "episode_scores": scores,
        "source_summaries": {
            arm: {
                "path": _relative(arm_paths(arm)["summary"]),
                "sha256": sha256(arm_paths(arm)["summary"]),
            }
            for arm in ARMS
        },
    }
    selections = {
        "status": "complete",
        "overlay": False,
        "arms": {
            arm: _representatives(arm, summaries[arm], scores[arm]) for arm in ARMS
        },
    }
    write_json(ANALYSIS_PATH, analysis)
    write_json(SELECTION_PATH, selections)
    print(
        "comparison complete: "
        f"raw_onsets={analysis['raw_onset_comparison']} "
        f"gate_before_visual={gate_before_visual}",
        flush=True,
    )
    return analysis


def main() -> int:
    args = parse_args()
    if args.compare:
        compare_arms()
        return 0
    run_arm(
        str(args.arm),
        port=int(args.port),
        tmi_scripts_dir=args.tmi_scripts_dir.resolve(),
        reuse_game=bool(args.reuse_game),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
