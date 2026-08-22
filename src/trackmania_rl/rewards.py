"""Isolated, swappable reward functions for TrackMania experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from trackmania_rl.observations import ObservationDiagnostics


@dataclass(frozen=True)
class RewardTransition:
    """Environment transition data available to every reward version."""

    previous_diagnostics: ObservationDiagnostics
    diagnostics: ObservationDiagnostics
    display_speed: int
    elapsed_ms: int
    terminated: bool
    truncated: bool
    timed_out: bool
    off_track: bool
    fallen: bool = False


RewardFunction = Callable[[RewardTransition], float]

V3_PROGRESS_CLAMP_UNITS = 10.0


def phase1_smoke_reward(transition: RewardTransition) -> float:
    """Owner-approved disposable Phase 1 integration reward."""
    reward = transition.display_speed / 1000.0
    if transition.truncated:
        reward -= 1.0
    return reward


def sparse_finish_reward(transition: RewardTransition) -> float:
    """Reward v1: one on a finished race and zero for every other outcome."""
    return 1.0 if transition.terminated else 0.0


def dense_speed_reward(transition: RewardTransition) -> float:
    """Reward v2: normalized displayed speed with no terminal shaping."""
    return transition.display_speed / 1000.0


def clamped_forward_progress_reward(transition: RewardTransition) -> float:
    """Reward v3: positive centerline progress, capped per decision step."""
    forward_progress = max(
        0.0,
        transition.diagnostics.progress
        - transition.previous_diagnostics.progress,
    )
    return min(forward_progress, V3_PROGRESS_CLAMP_UNITS) / V3_PROGRESS_CLAMP_UNITS
