"""Isolated, swappable reward functions for TrackMania experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
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
    stuck: bool = False
    steering_rate_change: float = 0.0
    steering_slope_reversal: bool = False
    steering_reversals_in_window: int = 0
    new_high_water_progress_delta: float = 0.0
    full_simstate_available: bool = False
    ground_contact_count: int = 0
    sliding_wheel_count: int = 0
    slip_angle_degrees: float = 0.0
    body_up_yaw_rate: float = 0.0


RewardFunction = Callable[[RewardTransition], float]

V3_PROGRESS_CLAMP_UNITS = 10.0
V4_PROGRESS_CLAMP_UNITS = 20.0
V4_PROGRESS_NORMALIZATION_UNITS = 10.0
V4_TIME_COST = 0.10
V4_FINISH_BONUS = 50.0
V4_FAILURE_PENALTY = 250.0
V5_STEERING_RATE_COEFFICIENT = 0.05
V6_STEERING_DELTA_DEADBAND = 0.05
V6_REVERSAL_WINDOW_SECONDS = 2.0
V6_FREE_REVERSALS_PER_WINDOW = 3
V6_REVERSAL_FREQUENCY_COEFFICIENT = 0.05
WR_STAGE2_ASSISTED_ZONES = ((680.0, 930.0), (1100.0, 1410.0))
WR_STAGE2_PROGRESS_CLAMP_UNITS = 20.0
WR_STAGE2_MIN_GROUND_CONTACTS = 3
WR_STAGE2_MIN_DISPLAY_SPEED = 350
WR_STAGE2_MIN_SLIDING_WHEELS = 1
WR_STAGE2_MIN_ABS_SLIP_ANGLE_DEGREES = 1.0
WR_STAGE2_MIN_ABS_BODY_UP_YAW_RATE = 0.25
WR_STAGE2_BONUS_COEFFICIENT = 0.50


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


def signed_progress_efficiency_reward(transition: RewardTransition) -> float:
    """Reward v4: signed progress, elapsed-step cost, and terminal outcomes."""
    progress_delta = (
        transition.diagnostics.progress
        - transition.previous_diagnostics.progress
    )
    signed_progress = min(
        max(progress_delta, -V4_PROGRESS_CLAMP_UNITS),
        V4_PROGRESS_CLAMP_UNITS,
    ) / V4_PROGRESS_NORMALIZATION_UNITS
    reward = signed_progress - V4_TIME_COST
    if transition.terminated:
        reward += V4_FINISH_BONUS
    elif transition.truncated:
        reward -= V4_FAILURE_PENALTY
    return reward


def steering_rate_smoothness_reward(transition: RewardTransition) -> float:
    """Reward v5: V4 minus one normalized steering-rate-of-change term."""
    return signed_progress_efficiency_reward(transition) - (
        V5_STEERING_RATE_COEFFICIENT * transition.steering_rate_change
    )


def clustered_reversal_frequency_reward(transition: RewardTransition) -> float:
    """Reward v6: V4 minus an event-triggered clustered-reversal cost."""
    excess_reversals = max(
        0,
        transition.steering_reversals_in_window
        - V6_FREE_REVERSALS_PER_WINDOW,
    )
    frequency_cost = (
        V6_REVERSAL_FREQUENCY_COEFFICIENT * excess_reversals
        if transition.steering_slope_reversal
        else 0.0
    )
    return signed_progress_efficiency_reward(transition) - frequency_cost


def localized_drift_bonus(transition: RewardTransition) -> float:
    """Return the frozen WR-chase Stage 2 live-telemetry bonus."""
    if not transition.full_simstate_available:
        raise ValueError(
            "WR-chase Stage 2 requires complete live SimState dynamics"
        )
    if not (
        0 <= transition.ground_contact_count <= 4
        and 0 <= transition.sliding_wheel_count <= 4
    ):
        raise ValueError("live SimState wheel counts must be between zero and four")
    dynamics = (
        transition.new_high_water_progress_delta,
        transition.slip_angle_degrees,
        transition.body_up_yaw_rate,
    )
    if not all(math.isfinite(value) for value in dynamics):
        raise ValueError("WR-chase Stage 2 dynamics must be finite")

    current_progress = transition.diagnostics.progress
    in_assisted_zone = any(
        low <= current_progress <= high
        for low, high in WR_STAGE2_ASSISTED_ZONES
    )
    drift_attempt = bool(
        transition.ground_contact_count >= WR_STAGE2_MIN_GROUND_CONTACTS
        and transition.display_speed >= WR_STAGE2_MIN_DISPLAY_SPEED
        and transition.sliding_wheel_count >= WR_STAGE2_MIN_SLIDING_WHEELS
        and abs(transition.slip_angle_degrees)
        >= WR_STAGE2_MIN_ABS_SLIP_ANGLE_DEGREES
        and abs(transition.body_up_yaw_rate)
        >= WR_STAGE2_MIN_ABS_BODY_UP_YAW_RATE
    )
    if not (in_assisted_zone and drift_attempt):
        return 0.0

    new_progress_fraction = min(
        max(transition.new_high_water_progress_delta, 0.0),
        WR_STAGE2_PROGRESS_CLAMP_UNITS,
    ) / WR_STAGE2_PROGRESS_CLAMP_UNITS
    return WR_STAGE2_BONUS_COEFFICIENT * new_progress_fraction


def localized_drift_assistance_reward(transition: RewardTransition) -> float:
    """WR-chase Stage 2: V4 plus a bounded two-zone drift-attempt bonus."""
    return signed_progress_efficiency_reward(transition) + localized_drift_bonus(
        transition
    )
