"""Build a committed reference path from verified lap telemetry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
VERIFIED_SOURCE_SHA256 = (
    "C8469209C0A91A1421F052DBF055A900C9A092E27AA8561C230D2A650CBDB0CC"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resample verified track telemetry into a fixed-spacing path."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=WORKSPACE_ROOT / "artifacts" / "telemetry" / "phase0_manual_lap.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE_ROOT / "data" / "tracks" / "a01_reference_path.csv",
    )
    parser.add_argument("--spacing", type=float, default=5.0)
    parser.add_argument("--stationary-threshold", type=float, default=0.05)
    parser.add_argument("--max-segment", type=float, default=50.0)
    parser.add_argument(
        "--expected-source-sha256",
        default=VERIFIED_SOURCE_SHA256,
    )
    parser.add_argument("--track", default="A01-Race")
    parser.add_argument(
        "--source-kind",
        default="resampled_manual_driving_reference",
    )
    parser.add_argument(
        "--decision",
        default="docs/decisions/0003-a01-reference-path.md",
    )
    return parser.parse_args()


def load_telemetry(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) < 2:
        raise ValueError("reference telemetry needs at least two records")
    positions = np.asarray([row["position"] for row in rows], dtype=np.float64)
    race_times = np.asarray([row["race_time_ms"] for row in rows], dtype=np.int64)
    if positions.shape != (len(rows), 3) or not np.isfinite(positions).all():
        raise ValueError("reference telemetry positions must be finite XYZ triples")
    if np.any(np.diff(race_times) < 0):
        raise ValueError("reference telemetry race time must be monotonic")
    return positions, race_times


def remove_stationary_points(
    positions: np.ndarray, minimum_distance: float
) -> np.ndarray:
    kept = [positions[0]]
    for position in positions[1:]:
        if np.linalg.norm(position - kept[-1]) >= minimum_distance:
            kept.append(position)
    return np.asarray(kept, dtype=np.float64)


def resample_path(
    positions: np.ndarray, spacing: float, max_segment: float
) -> tuple[np.ndarray, float]:
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be positive and finite")
    if len(positions) < 2:
        raise ValueError("path needs at least two moving positions")
    segments = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    if not np.isfinite(segments).all() or np.any(segments <= 0):
        raise ValueError("path contains invalid or duplicate moving segments")
    if float(segments.max()) > max_segment:
        raise ValueError(
            f"path contains a {segments.max():.3f}-unit teleport-like segment"
        )

    cumulative = np.concatenate(([0.0], np.cumsum(segments)))
    total_length = float(cumulative[-1])
    targets = np.arange(0.0, total_length, spacing, dtype=np.float64)
    if targets.size == 0 or targets[-1] < total_length:
        targets = np.append(targets, total_length)
    resampled = np.column_stack(
        [np.interp(targets, cumulative, positions[:, axis]) for axis in range(3)]
    )
    return resampled, total_length


def write_outputs(
    output: Path,
    points: np.ndarray,
    *,
    source: Path,
    source_sha256: str,
    source_samples: int,
    moving_samples: int,
    spacing: float,
    total_length: float,
    track: str,
    source_kind: str,
    decision: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow(("index", "distance", "x", "y", "z"))
        for index, point in enumerate(points):
            distance = min(index * spacing, total_length)
            writer.writerow(
                (
                    index,
                    f"{distance:.6f}",
                    f"{point[0]:.9f}",
                    f"{point[1]:.9f}",
                    f"{point[2]:.9f}",
                )
            )

    try:
        source_label = source.resolve().relative_to(WORKSPACE_ROOT).as_posix()
    except ValueError:
        source_label = str(source)
    metadata = {
        "track": track,
        "kind": source_kind,
        "source": source_label,
        "source_sha256": source_sha256,
        "source_samples": source_samples,
        "moving_samples": moving_samples,
        "output_points": len(points),
        "spacing": spacing,
        "total_length": total_length,
        "horizontal_axes": ["x", "z"],
        "elevation_axis": "y",
        "decision": decision,
    }
    metadata_path = output.with_suffix(".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    source_bytes = args.input.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest().upper()
    if source_hash != args.expected_source_sha256.upper():
        raise SystemExit(
            f"source telemetry hash mismatch: expected "
            f"{args.expected_source_sha256.upper()}, got {source_hash}"
        )

    positions, _ = load_telemetry(args.input)
    moving = remove_stationary_points(positions, args.stationary_threshold)
    points, total_length = resample_path(moving, args.spacing, args.max_segment)
    write_outputs(
        args.output,
        points,
        source=args.input,
        source_sha256=source_hash,
        source_samples=len(positions),
        moving_samples=len(moving),
        spacing=args.spacing,
        total_length=total_length,
        track=args.track,
        source_kind=args.source_kind,
        decision=args.decision,
    )
    print(
        f"wrote {len(points)} {args.track} reference points at "
        f"{args.spacing:.3f}-unit "
        f"spacing over {total_length:.3f} units to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
