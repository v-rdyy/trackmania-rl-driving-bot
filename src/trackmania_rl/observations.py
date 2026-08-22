"""Phase 1 engineered observations for the A01 reference path.

Observation layout (26 float32 values):

* 0: displayed speed / 1000
* 1..3: car-frame velocity [forward, right, up] / 100
* 4: signed heading error / pi
* 5: signed lateral offset / 50 (positive is car-right of the path)
* 6..25: ten car-frame look-ahead pairs [forward, right] / 100,
  sampled 10, 20, ..., 100 path units ahead

TMInterface's rotation matrix columns are car [right, up, forward], so its
transpose converts world vectors into the car frame. Path projection and
heading use world X/Z as the horizontal plane; Y is elevation.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

OBSERVATION_SIZE = 26
LOOKAHEAD_DISTANCES = np.arange(10.0, 101.0, 10.0, dtype=np.float64)


@dataclass(frozen=True)
class PathProjection:
    segment_index: int
    segment_fraction: float
    progress: float
    point: np.ndarray
    tangent_xz: np.ndarray
    lateral_offset: float
    vertical_offset: float


@dataclass(frozen=True)
class ObservationDiagnostics:
    progress: float
    lateral_offset: float
    heading_error: float
    segment_index: int
    vertical_offset: float = 0.0


class ReferencePath:
    """Fixed-spacing open reference path with horizontal projection helpers."""

    def __init__(self, distances: np.ndarray, points: np.ndarray) -> None:
        self.distances = np.asarray(distances, dtype=np.float64)
        self.points = np.asarray(points, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("reference points must have shape (N, 3)")
        if self.distances.shape != (len(self.points),) or len(self.points) < 2:
            raise ValueError("reference distances must match at least two points")
        if not np.isfinite(self.points).all() or not np.isfinite(self.distances).all():
            raise ValueError("reference path must contain only finite values")
        if self.distances[0] != 0.0 or np.any(np.diff(self.distances) <= 0):
            raise ValueError("reference distances must start at zero and increase")

        self._segments = np.diff(self.points, axis=0)
        self._segments_xz = self._segments[:, [0, 2]]
        self._segment_xz_length_squared = np.sum(self._segments_xz**2, axis=1)
        if np.any(self._segment_xz_length_squared <= 0):
            raise ValueError("reference path contains a zero-length horizontal segment")

    @classmethod
    def from_csv(cls, path: Path) -> ReferencePath:
        distances: list[float] = []
        points: list[tuple[float, float, float]] = []
        with path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames != ["index", "distance", "x", "y", "z"]:
                raise ValueError("unexpected reference-path CSV columns")
            for expected_index, row in enumerate(reader):
                if int(row["index"]) != expected_index:
                    raise ValueError("reference-path indices must be contiguous")
                distances.append(float(row["distance"]))
                points.append((float(row["x"]), float(row["y"]), float(row["z"])))
        return cls(np.asarray(distances), np.asarray(points))

    @property
    def total_length(self) -> float:
        return float(self.distances[-1])

    def project(self, position: np.ndarray) -> PathProjection:
        position = np.asarray(position, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("position must be a finite XYZ vector")

        offsets_xz = position[[0, 2]] - self.points[:-1, [0, 2]]
        fractions = (
            np.sum(offsets_xz * self._segments_xz, axis=1)
            / self._segment_xz_length_squared
        )
        fractions = np.clip(fractions, 0.0, 1.0)
        projected_xz = self.points[:-1, [0, 2]] + (
            fractions[:, None] * self._segments_xz
        )
        deltas_xz = position[[0, 2]] - projected_xz
        segment_index = int(np.argmin(np.sum(deltas_xz**2, axis=1)))
        fraction = float(fractions[segment_index])
        point = self.points[segment_index] + fraction * self._segments[segment_index]
        tangent_xz = self._segments_xz[segment_index]
        tangent_xz = tangent_xz / np.linalg.norm(tangent_xz)
        offset_xz = position[[0, 2]] - point[[0, 2]]
        lateral_offset = float(
            tangent_xz[1] * offset_xz[0] - tangent_xz[0] * offset_xz[1]
        )
        vertical_offset = float(position[1] - point[1])
        progress = float(
            self.distances[segment_index]
            + fraction
            * (self.distances[segment_index + 1] - self.distances[segment_index])
        )
        return PathProjection(
            segment_index=segment_index,
            segment_fraction=fraction,
            progress=progress,
            point=point,
            tangent_xz=tangent_xz,
            lateral_offset=lateral_offset,
            vertical_offset=vertical_offset,
        )

    def points_at_distances(self, distances: np.ndarray) -> np.ndarray:
        distances = np.clip(
            np.asarray(distances, dtype=np.float64), 0.0, self.total_length
        )
        return np.column_stack(
            [
                np.interp(distances, self.distances, self.points[:, axis])
                for axis in range(3)
            ]
        )


def signed_heading_error(path_tangent_xz: np.ndarray, car_forward_xz: np.ndarray) -> float:
    path_tangent_xz = np.asarray(path_tangent_xz, dtype=np.float64)
    car_forward_xz = np.asarray(car_forward_xz, dtype=np.float64)
    car_norm = float(np.linalg.norm(car_forward_xz))
    if car_norm <= 1e-9:
        raise ValueError("car forward vector has no horizontal component")
    car_forward_xz = car_forward_xz / car_norm
    cross = (
        path_tangent_xz[0] * car_forward_xz[1]
        - path_tangent_xz[1] * car_forward_xz[0]
    )
    dot = float(np.dot(path_tangent_xz, car_forward_xz))
    return math.atan2(float(cross), dot)


def build_observation(
    state: object, reference_path: ReferencePath
) -> tuple[np.ndarray, ObservationDiagnostics]:
    """Convert one TMInterface simulation state to the documented 26-vector."""

    position = np.asarray(state.position, dtype=np.float64)
    velocity = np.asarray(state.velocity, dtype=np.float64)
    rotation = np.asarray(state.rotation_matrix, dtype=np.float64)
    if position.shape != (3,) or velocity.shape != (3,) or rotation.shape != (3, 3):
        raise ValueError("simulation state has unexpected vector/matrix shapes")
    if not np.isfinite(np.concatenate((position, velocity, rotation.ravel()))).all():
        raise ValueError("simulation state contains nonfinite observation inputs")

    projection = reference_path.project(position)
    local_velocity = rotation.T @ velocity
    car_forward_xz = rotation[[0, 2], 2]
    heading_error = signed_heading_error(projection.tangent_xz, car_forward_xz)

    lookahead_world = reference_path.points_at_distances(
        projection.progress + LOOKAHEAD_DISTANCES
    )
    lookahead_local = (rotation.T @ (lookahead_world - position).T).T

    observation = np.empty(OBSERVATION_SIZE, dtype=np.float32)
    observation[0] = float(state.display_speed) / 1000.0
    observation[1:4] = local_velocity[[2, 0, 1]] / 100.0
    observation[4] = heading_error / math.pi
    observation[5] = projection.lateral_offset / 50.0
    observation[6:] = (lookahead_local[:, [2, 0]] / 100.0).reshape(-1)
    if not np.isfinite(observation).all():
        raise ValueError("engineered observation contains nonfinite values")

    diagnostics = ObservationDiagnostics(
        progress=projection.progress,
        lateral_offset=projection.lateral_offset,
        heading_error=heading_error,
        segment_index=projection.segment_index,
        vertical_offset=projection.vertical_offset,
    )
    return observation, diagnostics
