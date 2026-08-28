"""Reproduce the Gate 500k/Gate 1M policy-mode safety diagnosis offline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
from stable_baselines3 import PPO

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from trackmania_rl.observations import ReferencePath, build_observation

DEFAULT_GATE500_CHECKPOINT = (
    WORKSPACE_ROOT / "checkpoints" / "wr_chase_stage2" / "gate_00500000_model.zip"
)
DEFAULT_GATE1M_CHECKPOINT = (
    WORKSPACE_ROOT / "checkpoints" / "wr_chase_stage2" / "gate_01000000_model.zip"
)
DEFAULT_INTERMEDIATE_CHECKPOINT = (
    WORKSPACE_ROOT
    / "checkpoints"
    / "wr_chase_stage2"
    / "ppo_wr_stage2_3756176_steps.zip"
)
DEFAULT_GATE500_ACTIONS = (
    WORKSPACE_ROOT
    / "runs"
    / "wr_chase_stage2"
    / "gates"
    / "gate_00500000_evaluation_actions.jsonl"
)
DEFAULT_GATE1M_ACTIONS = (
    WORKSPACE_ROOT
    / "runs"
    / "wr_chase_stage2"
    / "gates"
    / "gate_01000000_evaluation_actions.jsonl"
)
DEFAULT_INTERMEDIATE_ACTIONS = (
    WORKSPACE_ROOT
    / "runs"
    / "wr_chase_stage2_intermediate"
    / "gate_00750000_deterministic_actions.jsonl"
)
DEFAULT_GATE500_SUMMARY = (
    WORKSPACE_ROOT
    / "runs"
    / "wr_chase_stage2"
    / "gates"
    / "gate_00500000_evaluation.json"
)
DEFAULT_GATE1M_SUMMARY = (
    WORKSPACE_ROOT
    / "runs"
    / "wr_chase_stage2"
    / "gates"
    / "gate_01000000_evaluation.json"
)
DEFAULT_INTERMEDIATE_SUMMARY = (
    WORKSPACE_ROOT
    / "runs"
    / "wr_chase_stage2_intermediate"
    / "gate_00750000_deterministic_summary.json"
)
DEFAULT_STOCHASTIC_ACTIONS = (
    WORKSPACE_ROOT
    / "runs"
    / "wr_chase_stage2_policy_mode"
    / "gate_01000000_stochastic_actions.jsonl"
)
DEFAULT_STOCHASTIC_SUMMARY = (
    WORKSPACE_ROOT
    / "runs"
    / "wr_chase_stage2_policy_mode"
    / "gate_01000000_stochastic_summary.json"
)
DEFAULT_MONITOR = WORKSPACE_ROOT / "runs" / "wr_chase_stage2" / "training.monitor.csv"
DEFAULT_REFERENCE = WORKSPACE_ROOT / "data" / "tracks" / "a01_reference_path.csv"
DEFAULT_OUTPUT = (
    WORKSPACE_ROOT
    / "runs"
    / "wr_chase_stage2_policy_mode"
    / "offline_diagnostic.json"
)
PROGRESS_LANDMARKS = (2000.0, 2100.0, 2140.0, 2160.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate500-checkpoint", type=Path, default=DEFAULT_GATE500_CHECKPOINT)
    parser.add_argument("--gate1m-checkpoint", type=Path, default=DEFAULT_GATE1M_CHECKPOINT)
    parser.add_argument(
        "--intermediate-checkpoint",
        type=Path,
        default=DEFAULT_INTERMEDIATE_CHECKPOINT,
    )
    parser.add_argument("--gate500-actions", type=Path, default=DEFAULT_GATE500_ACTIONS)
    parser.add_argument("--gate1m-actions", type=Path, default=DEFAULT_GATE1M_ACTIONS)
    parser.add_argument(
        "--intermediate-actions", type=Path, default=DEFAULT_INTERMEDIATE_ACTIONS
    )
    parser.add_argument("--gate500-summary", type=Path, default=DEFAULT_GATE500_SUMMARY)
    parser.add_argument("--gate1m-summary", type=Path, default=DEFAULT_GATE1M_SUMMARY)
    parser.add_argument(
        "--intermediate-summary", type=Path, default=DEFAULT_INTERMEDIATE_SUMMARY
    )
    parser.add_argument("--stochastic-actions", type=Path, default=DEFAULT_STOCHASTIC_ACTIONS)
    parser.add_argument("--stochastic-summary", type=Path, default=DEFAULT_STOCHASTIC_SUMMARY)
    parser.add_argument("--monitor", type=Path, default=DEFAULT_MONITOR)
    parser.add_argument("--reference-path", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def group_records(records: Iterable[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(int(record["episode"]), []).append(record)
    for episode in grouped:
        grouped[episode].sort(key=lambda row: int(row["step"]))
    return grouped


def state_from_record(record: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        display_speed=float(record["display_speed"]),
        position=np.asarray(record["position"], dtype=np.float64),
        velocity=np.asarray(record["velocity"], dtype=np.float64),
        rotation_matrix=np.asarray(record["rotation_matrix"], dtype=np.float64),
    )


def aligned_observations_and_actions(
    records: list[dict[str, Any]],
    reference: ReferencePath,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    progress: list[float] = []
    for episode_records in group_records(records).values():
        for current, following in zip(episode_records, episode_records[1:]):
            observation, _ = build_observation(state_from_record(current), reference)
            observations.append(observation)
            actions.append(np.asarray(following["raw_action"], dtype=np.float64))
            progress.append(float(current["progress"]))
    if not observations:
        raise ValueError("action log has no aligned observation/action pairs")
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.float64),
        np.asarray(progress, dtype=np.float64),
    )


def deterministic_predictions(model: PPO, observations: np.ndarray) -> np.ndarray:
    predictions, _ = model.predict(observations, deterministic=True)
    return np.asarray(predictions, dtype=np.float64)


def action_reproduction(
    model: PPO,
    observations: np.ndarray,
    logged_actions: np.ndarray,
) -> dict[str, Any]:
    predicted = deterministic_predictions(model, observations)
    errors = np.abs(predicted - logged_actions)
    return {
        "pairs": len(observations),
        "mean_absolute_error": float(errors.mean()),
        "maximum_absolute_error": float(errors.max()),
        "per_action_mean_absolute_error": errors.mean(axis=0).tolist(),
        "per_action_maximum_absolute_error": errors.max(axis=0).tolist(),
    }


def action_drift(
    first: PPO,
    second: PPO,
    observations: np.ndarray,
    progress: np.ndarray,
    *,
    minimum_progress: float,
) -> dict[str, Any]:
    selected = observations[progress >= minimum_progress]
    if not len(selected):
        raise ValueError("no observations remain for action-drift comparison")
    differences = np.abs(
        deterministic_predictions(second, selected)
        - deterministic_predictions(first, selected)
    )
    return {
        "minimum_progress": minimum_progress,
        "observation_count": len(selected),
        "mean_absolute_change": differences.mean(axis=0).tolist(),
        "p95_absolute_change": np.percentile(differences, 95, axis=0).tolist(),
        "maximum_absolute_change": differences.max(axis=0).tolist(),
    }


def first_at_progress(records: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    for record in records:
        if float(record["progress"]) >= threshold:
            return record
    raise ValueError(f"episode never reached progress {threshold}")


def landmark_lateral_offsets(
    records: list[dict[str, Any]],
    thresholds: Iterable[float] = PROGRESS_LANDMARKS,
) -> dict[str, Any]:
    grouped = group_records(records)
    result: dict[str, Any] = {}
    for threshold in thresholds:
        values = [
            float(first_at_progress(episode, threshold)["lateral_offset"])
            for episode in grouped.values()
        ]
        result[f"{threshold:.0f}"] = {
            "mean": statistics.fmean(values),
            "minimum": min(values),
            "maximum": max(values),
            "values": values,
        }
    return result


def largest_speed_loss(records: list[dict[str, Any]], minimum_progress: float) -> dict[str, Any]:
    candidates = [
        (previous, current, float(previous["display_speed"]) - float(current["display_speed"]))
        for previous, current in zip(records, records[1:])
        if float(previous["progress"]) >= minimum_progress
    ]
    if not candidates:
        raise ValueError("episode has no speed-loss candidates")
    previous, current, loss = max(candidates, key=lambda candidate: candidate[2])
    return {
        "race_time_ms": int(current["race_time_ms"]),
        "progress_before": float(previous["progress"]),
        "progress_after": float(current["progress"]),
        "lateral_offset": float(current["lateral_offset"]),
        "speed_before": float(previous["display_speed"]),
        "speed_after": float(current["display_speed"]),
        "speed_loss": loss,
        "throttle": float(current["raw_action"][1]),
        "brake": float(current["raw_action"][2]),
        "upright_cosine": float(current["upright_cosine"]),
        "position": [float(value) for value in current["position"]],
    }


def collision_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    events = [
        {"episode": episode, **largest_speed_loss(rows, 2000.0)}
        for episode, rows in group_records(records).items()
    ]
    return {
        "episode_count": len(events),
        "events": events,
        "mean_progress_after": statistics.fmean(event["progress_after"] for event in events),
        "mean_lateral_offset": statistics.fmean(event["lateral_offset"] for event in events),
        "mean_speed_before": statistics.fmean(event["speed_before"] for event in events),
        "mean_speed_after": statistics.fmean(event["speed_after"] for event in events),
        "mean_speed_loss": statistics.fmean(event["speed_loss"] for event in events),
        "mean_throttle": statistics.fmean(event["throttle"] for event in events),
        "mean_brake": statistics.fmean(event["brake"] for event in events),
    }


def read_monitor(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        source.readline()
        return list(csv.DictReader(source))


def rolling_finish_rates(rows: list[dict[str, str]]) -> dict[str, Any]:
    result: dict[str, Any] = {"episodes": len(rows)}
    for window in (10, 20, 50, 100, 500):
        selected = rows[-window:]
        finishes = sum(row["race_finished"] == "True" for row in selected)
        result[f"last_{window}"] = {
            "episodes": len(selected),
            "finishes": finishes,
            "finish_rate": finishes / len(selected),
        }
    return result


def policy_metadata(model: PPO) -> dict[str, Any]:
    latent_std = np.exp(model.policy.log_std.detach().cpu().numpy())
    return {
        "num_timesteps": int(model.num_timesteps),
        "use_sde": bool(model.use_sde),
        "sde_sample_freq": int(model.sde_sample_freq),
        "squash_output": bool(model.policy.squash_output),
        "ent_coef": float(model.ent_coef),
        "target_kl": model.target_kl,
        "latent_exploration_std_mean_by_action": latent_std.mean(axis=0).tolist(),
    }


def file_binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(WORKSPACE_ROOT)),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def main() -> int:
    args = parse_args()
    paths = {
        name: getattr(args, name).resolve()
        for name in (
            "gate500_checkpoint",
            "gate1m_checkpoint",
            "intermediate_checkpoint",
            "gate500_actions",
            "gate1m_actions",
            "intermediate_actions",
            "gate500_summary",
            "gate1m_summary",
            "intermediate_summary",
            "stochastic_actions",
            "stochastic_summary",
            "monitor",
            "reference_path",
        )
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    reference = ReferencePath.from_csv(paths["reference_path"])
    gate500_records = read_jsonl(paths["gate500_actions"])
    gate1m_records = read_jsonl(paths["gate1m_actions"])
    intermediate_records = read_jsonl(paths["intermediate_actions"])
    gate500_summary = read_json(paths["gate500_summary"])
    gate1m_summary = read_json(paths["gate1m_summary"])
    intermediate_summary = read_json(paths["intermediate_summary"])
    stochastic_summary = read_json(paths["stochastic_summary"])
    gate500_model = PPO.load(paths["gate500_checkpoint"], device="cpu")
    gate1m_model = PPO.load(paths["gate1m_checkpoint"], device="cpu")
    intermediate_model = PPO.load(paths["intermediate_checkpoint"], device="cpu")
    obs500, actions500, _ = aligned_observations_and_actions(gate500_records, reference)
    obs1m, actions1m, progress1m = aligned_observations_and_actions(
        gate1m_records, reference
    )
    obs_intermediate, actions_intermediate, progress_intermediate = (
        aligned_observations_and_actions(intermediate_records, reference)
    )
    result = {
        "status": "complete",
        "inputs": {name: file_binding(path) for name, path in paths.items()},
        "evaluation_outcomes": {
            "gate500_deterministic": {
                "finishes": gate500_summary["finishes"],
                "episodes": gate500_summary["episodes"],
            },
            "gate1m_deterministic": {
                "finishes": gate1m_summary["finishes"],
                "episodes": gate1m_summary["episodes"],
            },
            "intermediate_deterministic": {
                "finishes": intermediate_summary["finishes"],
                "episodes": intermediate_summary["episodes"],
                "model_timesteps": intermediate_summary[
                    "intermediate_checkpoint_diagnostic"
                ]["source_evidence"]["model_timesteps"],
            },
            "gate1m_stochastic": {
                "finishes": stochastic_summary["finishes"],
                "episodes": stochastic_summary["episodes"],
                "random_seed": stochastic_summary["policy_sampling"]["random_seed"],
                "sde_sample_freq": stochastic_summary["policy_sampling"]["sde_sample_freq"],
            },
        },
        "policy": {
            "gate500": policy_metadata(gate500_model),
            "gate1m": policy_metadata(gate1m_model),
            "intermediate": policy_metadata(intermediate_model),
            "gate500_action_reproduction": action_reproduction(
                gate500_model, obs500, actions500
            ),
            "gate1m_action_reproduction": action_reproduction(
                gate1m_model, obs1m, actions1m
            ),
            "intermediate_action_reproduction": action_reproduction(
                intermediate_model, obs_intermediate, actions_intermediate
            ),
            "gate500_to_intermediate_action_drift_on_intermediate_states": action_drift(
                gate500_model,
                intermediate_model,
                obs_intermediate,
                progress_intermediate,
                minimum_progress=2100.0,
            ),
            "gate500_to_gate1m_action_drift_on_gate1m_states": action_drift(
                gate500_model,
                gate1m_model,
                obs1m,
                progress1m,
                minimum_progress=2100.0,
            ),
        },
        "trajectory": {
            "gate500_lateral_landmarks": landmark_lateral_offsets(gate500_records),
            "gate1m_lateral_landmarks": landmark_lateral_offsets(gate1m_records),
            "intermediate_lateral_landmarks": landmark_lateral_offsets(
                intermediate_records
            ),
            "gate500_late_speed_loss": collision_summary(gate500_records),
            "gate1m_late_speed_loss": collision_summary(gate1m_records),
            "intermediate_late_speed_loss": collision_summary(
                intermediate_records
            ),
        },
        "training_monitor": rolling_finish_rates(read_monitor(paths["monitor"])),
        "interpretation_scope": (
            "Offline diagnosis binds existing evidence only; it performs no game "
            "interaction, training, checkpoint selection, or reward change."
        ),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    print(f"SHA-256={sha256(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
