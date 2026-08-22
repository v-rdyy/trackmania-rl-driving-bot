"""Apply the fixed precision metrics to round-tripped V2 input replays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))

from trackmania_rl.evaluation_metrics import (
    aggregate_precision_metrics,
    evaluation_metric_protocol,
    steering_precision_metrics,
    trajectory_precision_metrics,
)
from validate_progress_replays import DEFAULT_CATALOG, DEFAULT_OUTPUT, load_cases, sha256

DEFAULT_SUMMARY = DEFAULT_OUTPUT / "v2_precision_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure oscillation, line precision, orientation, and stuck time."
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--telemetry-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def steering_samples(path: Path) -> list[tuple[float, float]]:
    samples: list[tuple[float, float]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        parts = raw_line.strip().split()
        if not parts or len(parts) >= 2 and parts[0].startswith("#"):
            continue
        if len(parts) != 3:
            raise ValueError(f"invalid input line {line_number} in {path}")
        if parts[1].lower() != "steer":
            continue
        samples.append((float(parts[0]), int(parts[2]) / 65536.0))
    if not samples:
        raise ValueError(f"input replay has no steer commands: {path}")
    return samples


def load_telemetry(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"round-trip telemetry is missing: {path}; run "
            "validate_progress_replays.py --all-v2 first"
        )
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    args = parse_args()
    cases = [
        case
        for case in load_cases(args.catalog).values()
        if case.stage_id.startswith("v2_")
    ]
    cases.sort(key=lambda case: case.case_id)
    episodes: list[dict[str, Any]] = []
    for case in cases:
        telemetry_path = (
            args.telemetry_dir / f"{case.stage_id}_ep_{case.episode:02d}.jsonl"
        )
        records = load_telemetry(telemetry_path)
        metrics = {
            "steering": steering_precision_metrics(
                steering_samples(case.input_replay)
            ),
            **trajectory_precision_metrics(records),
        }
        episodes.append(
            {
                "case_id": case.case_id,
                "stage_id": case.stage_id,
                "episode": case.episode,
                "expected_outcome": case.expected_outcome,
                "input_replay": str(case.input_replay.relative_to(WORKSPACE_ROOT)),
                "input_replay_sha256": sha256(case.input_replay),
                "telemetry": str(telemetry_path.relative_to(WORKSPACE_ROOT)),
                "telemetry_sha256": sha256(telemetry_path),
                "metrics": metrics,
            }
        )

    stages: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        stages.setdefault(str(episode["stage_id"]), []).append(episode)
    summary = {
        "status": "complete",
        "metric_protocol": evaluation_metric_protocol(),
        "all_v2": aggregate_precision_metrics(
            [episode["metrics"] for episode in episodes]
        ),
        "stages": {
            stage_id: aggregate_precision_metrics(
                [episode["metrics"] for episode in stage_episodes]
            )
            for stage_id, stage_episodes in sorted(stages.items())
        },
        "episodes": episodes,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        "V2 precision metrics complete: "
        f"episodes={summary['all_v2']['episode_count']} "
        f"oscillation={summary['all_v2']['oscillation_detected_episodes']} "
        f"upside_down={summary['all_v2']['upside_down_detected_episodes']} "
        f"stuck={summary['all_v2']['stuck_detected_episodes']}",
        flush=True,
    )
    print(f"summary SHA-256={sha256(args.summary)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
