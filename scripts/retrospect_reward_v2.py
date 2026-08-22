"""Reconcile reward-v2 training, evaluation, and replay evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = WORKSPACE_ROOT / "runs" / "reward_v2"
ROUND_TRIP_DIR = WORKSPACE_ROOT / "artifacts" / "replays" / "progression" / "round_trip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an auditable V2 retrospective from preserved evidence."
    )
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--round-trip-dir", type=Path, default=ROUND_TRIP_DIR)
    parser.add_argument(
        "--summary",
        type=Path,
        default=RUN_DIR / "retrospective_summary.json",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_monitor(path: Path) -> list[dict[str, str]]:
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    return list(csv.DictReader(lines))


def terminal_cause(row: dict[str, str]) -> str:
    flags = {
        "finish": row["race_finished"] == "True",
        "timeout": row["timeout"] == "True",
        "fall": row["fallen"] == "True",
        "off_track": row["off_track"] == "True",
    }
    causes = [cause for cause, active in flags.items() if active]
    if len(causes) != 1:
        raise ValueError(f"monitor row has {len(causes)} terminal causes: {row}")
    return causes[0]


def episode_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter(terminal_cause(row) for row in rows)
    return {
        "episodes": len(rows),
        "finishes": counts["finish"],
        "timeouts": counts["timeout"],
        "falls": counts["fall"],
        "off_tracks": counts["off_track"],
        "completed_episode_steps": sum(int(row["l"]) for row in rows),
    }


def retention_accounting(
    observed_rows: list[dict[str, str]],
    interrupted_rows: list[dict[str, str]],
    checkpoint_steps: int,
) -> dict[str, Any]:
    if checkpoint_steps <= 0:
        raise ValueError("checkpoint_steps must be positive")
    if observed_rows[: len(interrupted_rows)] != interrupted_rows:
        raise ValueError("interrupted monitor is not an exact prefix of final monitor")

    cumulative_steps = 0
    retained_prefix_count = 0
    boundary_index: int | None = None
    boundary_start_steps: int | None = None
    boundary_end_steps: int | None = None
    for index, row in enumerate(interrupted_rows):
        previous_steps = cumulative_steps
        cumulative_steps += int(row["l"])
        if cumulative_steps <= checkpoint_steps:
            retained_prefix_count = index + 1
            continue
        if boundary_index is None:
            boundary_index = index
            boundary_start_steps = previous_steps
            boundary_end_steps = cumulative_steps
        break

    retained_rows = (
        interrupted_rows[:retained_prefix_count]
        + observed_rows[len(interrupted_rows) :]
    )
    discarded_rows = interrupted_rows[retained_prefix_count:]
    return {
        "retained": episode_counts(retained_rows),
        "discarded_or_boundary_crossing": episode_counts(discarded_rows),
        "observed": episode_counts(observed_rows),
        "checkpoint_boundary": {
            "row_index_zero_based": boundary_index,
            "episode_start_step": boundary_start_steps,
            "episode_end_step": boundary_end_steps,
            "classification": "discarded_or_boundary_crossing",
        },
        "retained_prefix_episode_count": retained_prefix_count,
        "final_attempt_episode_count": len(observed_rows) - len(interrupted_rows),
    }


def main() -> int:
    args = parse_args()
    training = read_json(args.run_dir / "training_summary.json")
    original_evaluation_path = args.run_dir / "evaluation_summary.json"
    retrospective_evaluation_path = (
        args.run_dir / "retrospective_evaluation_summary.json"
    )
    replay_precision_path = args.round_trip_dir / "v2_precision_metrics.json"
    round_trip_path = args.round_trip_dir / "round_trip_summary.json"
    original_evaluation = read_json(original_evaluation_path)
    retrospective_evaluation = read_json(retrospective_evaluation_path)
    replay_precision = read_json(replay_precision_path)
    round_trip = read_json(round_trip_path)

    interrupted_monitor_path = args.run_dir / "attempt_9_cumulative_monitor.csv"
    observed_monitor_path = args.run_dir / "training.monitor.csv"
    interrupted_rows = read_monitor(interrupted_monitor_path)
    observed_rows = read_monitor(observed_monitor_path)
    accounting = retention_accounting(
        observed_rows,
        interrupted_rows,
        int(training["starting_timesteps"]),
    )
    observed = accounting["observed"]
    expected_observed = {
        "episodes": int(training["episodes"]),
        "finishes": int(training["finishes"]),
        "timeouts": int(training["timeouts"]),
        "falls": int(training["falls"]),
        "off_tracks": int(training["off_tracks"]),
    }
    if any(observed[key] != value for key, value in expected_observed.items()):
        raise ValueError(
            f"training summary does not match canonical monitor: "
            f"expected={expected_observed}, observed={observed}"
        )
    retained = accounting["retained"]
    discarded = accounting["discarded_or_boundary_crossing"]
    if retained["episodes"] + discarded["episodes"] != observed["episodes"]:
        raise ValueError("retained/discarded episode partition is incomplete")

    retained_interactions = int(training["actual_timesteps"])
    discarded_interactions = int(training["discarded_steps_after_checkpoint"])
    observed_interactions = int(training["environment_interactions_observed"])
    if retained_interactions + discarded_interactions != observed_interactions:
        raise ValueError("training interaction accounting does not reconcile")

    evaluation_finishes = int(original_evaluation["finishes"]) + int(
        retrospective_evaluation["finishes"]
    )
    evaluation_episodes = int(original_evaluation["episodes"]) + int(
        retrospective_evaluation["episodes"]
    )
    summary = {
        "status": "complete",
        "training_accounting": {
            **accounting,
            "retained_model_path_interactions": retained_interactions,
            "discarded_replayed_interactions": discarded_interactions,
            "all_observed_interactions": observed_interactions,
            "interactions_not_in_completed_monitor_episodes": (
                observed_interactions - observed["completed_episode_steps"]
            ),
        },
        "evaluations": {
            "original": {
                "episodes": int(original_evaluation["episodes"]),
                "finishes": int(original_evaluation["finishes"]),
                "finish_rate": float(original_evaluation["finish_rate"]),
                "summary_sha256": sha256(original_evaluation_path),
            },
            "precision_retrospective": {
                "episodes": int(retrospective_evaluation["episodes"]),
                "finishes": int(retrospective_evaluation["finishes"]),
                "finish_rate": float(retrospective_evaluation["finish_rate"]),
                "speed_farming_candidate_episodes": int(
                    retrospective_evaluation["speed_farming_candidate_episodes"]
                ),
                "precision": retrospective_evaluation["precision_summary"],
                "summary_sha256": sha256(retrospective_evaluation_path),
            },
            "descriptive_combined": {
                "episodes": evaluation_episodes,
                "finishes": evaluation_finishes,
                "finish_rate": evaluation_finishes / evaluation_episodes,
            },
        },
        "replay_archive": {
            "round_trip_status": round_trip["status"],
            "round_trip_case_count": int(round_trip["case_count"]),
            "shared_snapshot_rewind": bool(round_trip["shared_snapshot_rewind"]),
            "round_trip_summary_sha256": sha256(round_trip_path),
            "precision": replay_precision["all_v2"],
            "precision_summary_sha256": sha256(replay_precision_path),
        },
        "hypothesis_verdict": {
            "in_place_speed_farming_prediction_supported": False,
            "broader_speed_reward_alignment_concern_supported": True,
            "reason": (
                "V2 learned near-complete forward driving and sometimes finished, "
                "but repeated steering oscillation and last-checkpoint contact "
                "caused inversion and long stuck time."
            ),
        },
        "source_hashes": {
            "training_summary": sha256(args.run_dir / "training_summary.json"),
            "interrupted_monitor": sha256(interrupted_monitor_path),
            "canonical_monitor": sha256(observed_monitor_path),
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        "V2 retrospective complete: "
        f"retained_episodes={retained['episodes']} "
        f"discarded_episodes={discarded['episodes']} "
        f"precision_finishes={retrospective_evaluation['finishes']}/"
        f"{retrospective_evaluation['episodes']}",
        flush=True,
    )
    print(f"summary SHA-256={sha256(args.summary)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
