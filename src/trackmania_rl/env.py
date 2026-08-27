"""Gymnasium environment for the Phase 1 TrackMania integration smoke test."""

from __future__ import annotations

import json
import math
import threading
import time
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import gymnasium as gym
import numpy as np

from trackmania_rl.observations import (
    OBSERVATION_SIZE,
    ObservationDiagnostics,
    ReferencePath,
    build_observation,
)
from trackmania_rl.game_launch import confirm_a01_solo
from trackmania_rl.rewards import (
    RewardFunction,
    RewardTransition,
    V6_REVERSAL_WINDOW_SECONDS,
    V6_STEERING_DELTA_DEADBAND,
    phase1_smoke_reward,
)
from trackmania_rl.tmi_bridge import MessageType, ProtocolError, TmiBridgeClient

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE_PATH = WORKSPACE_ROOT / "data" / "tracks" / "a01_reference_path.csv"


@dataclass(frozen=True)
class EnvironmentConfig:
    port: int = 8478
    simulation_speed: float = 6.0
    step_period_ms: int = 100
    bridge_response_timeout_ms: int = 30_000
    max_episode_ms: int = 45_000
    max_lateral_offset: float = 50.0
    max_vertical_drop: float = 10.0
    max_start_progress: float = 25.0
    max_start_lateral_offset: float = 10.0
    max_start_vertical_offset: float = 5.0
    max_start_speed: int = 5
    auto_respawn_on_connect: bool = True
    max_initial_respawn_steps: int = 100
    wait_for_race_start_on_connect: bool = False
    # Live entry points may load A01 after starting ModLoader so the owner never
    # has to navigate the game menus manually.
    map_to_load: str | None = None
    # Disabled for V0-V2. V3 enables this with its pre-registered 2,000 ms window.
    stuck_window_ms: int | None = None
    stuck_progress_gain_units: float = 1.0
    stuck_world_distance_units: float = 2.0
    # Compatibility only for models trained before the signed Gas direction was
    # verified. New training must use the corrected default (False).
    legacy_reversed_pedal_mapping: bool = False


@dataclass(frozen=True)
class SessionStep:
    state: object
    race_time_ms: int
    race_finished: bool
    applied_steer: int
    applied_gas: int


@dataclass(frozen=True)
class PlaybackStep:
    state: object
    race_time_ms: int
    race_finished: bool


class EpisodeSession(Protocol):
    def prepare(self) -> object: ...

    def reset(self, diagnostics: ObservationDiagnostics | None = None) -> object: ...

    def advance(self, action: np.ndarray) -> SessionStep: ...

    def close(self) -> None: ...


class LiveTmiSession:
    """Own one synchronous bridge connection and its pending run-step callback."""

    def __init__(self, config: EnvironmentConfig) -> None:
        self.config = config
        self.client = TmiBridgeClient(port=config.port, timeout_seconds=30.0)
        self._connected = False
        self._pending_step = False
        self._current_state: object | None = None
        self._current_race_time: int | None = None
        self._initial_snapshot: bytes | None = None
        self._map_confirmation_stop = threading.Event()
        self._map_confirmation_thread: threading.Thread | None = None

    def _start_map_confirmation(self) -> None:
        """Confirm TMNF's post-map loading screen until callbacks resume."""
        self._map_confirmation_stop.clear()

        def confirm() -> None:
            deadline = time.monotonic() + 30.0
            while not self._map_confirmation_stop.wait(0.75):
                if time.monotonic() >= deadline:
                    return
                try:
                    confirm_a01_solo()
                except (OSError, RuntimeError):
                    continue

        self._map_confirmation_thread = threading.Thread(
            target=confirm,
            name="trackmania-map-confirmation",
            daemon=True,
        )
        self._map_confirmation_thread.start()

    def _connect(self) -> None:
        if self._connected:
            return
        self.client.connect()
        self._connected = True
        self._wait_for_run_step()

    def _handle_non_step(self, message_type: MessageType) -> None:
        if message_type is MessageType.SC_ON_CONNECT_SYNC:
            self.client.set_response_timeout(
                self.config.bridge_response_timeout_ms
            )
            self.client.execute_command("set unfocused_fps_limit false")
            self.client.execute_command("set disable_forced_camera true")
            self.client.set_speed(self.config.simulation_speed)
            self.client.set_on_step_period(self.config.step_period_ms)
            if self.config.map_to_load is not None:
                self.client.execute_command(f"map {self.config.map_to_load}")
                self._start_map_confirmation()
        elif message_type is MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC:
            self.client.read_int32()
            self.client.read_int32()
        elif message_type is MessageType.SC_LAP_COUNT_CHANGED_SYNC:
            self.client.read_int32()
            self.client.read_int32()
        self.client.respond(message_type)

    def _wait_for_run_step(self) -> None:
        if self._pending_step:
            raise RuntimeError("cannot wait while a run-step callback is pending")
        while True:
            message_type = self.client.read_message_type()
            if message_type is MessageType.SC_RUN_STEP_SYNC:
                self._current_race_time = self.client.read_int32()
                self._current_state = self.client.get_simulation_state()
                self._pending_step = True
                self._map_confirmation_stop.set()
                return
            self._handle_non_step(message_type)

    def prepare(self) -> object:
        """Connect, optionally respawn, and expose a snapshot candidate."""
        self._connect()
        if not self._pending_step or self._current_state is None:
            raise RuntimeError("prepare requires a pending simulation step")
        if self.config.wait_for_race_start_on_connect:
            for _ in range(self.config.max_initial_respawn_steps):
                if self._current_race_time is not None and self._current_race_time >= 0:
                    break
                self.client.set_continuous_input(
                    steer=0.0,
                    throttle=0.0,
                    brake=0.0,
                )
                self.client.respond(MessageType.SC_RUN_STEP_SYNC)
                self._pending_step = False
                self._wait_for_run_step()
            else:
                raise RuntimeError(
                    "fresh map did not reach nonnegative race time within "
                    f"{self.config.max_initial_respawn_steps} steps"
                )
        if self.config.auto_respawn_on_connect:
            self.client.give_up()
            if self.config.max_initial_respawn_steps <= 0:
                raise ValueError("max_initial_respawn_steps must be positive")
            countdown_observed = False
            for _ in range(self.config.max_initial_respawn_steps):
                self.client.set_continuous_input(
                    steer=0.0,
                    throttle=0.0,
                    brake=0.0,
                )
                self.client.respond(MessageType.SC_RUN_STEP_SYNC)
                self._pending_step = False
                self._wait_for_run_step()
                if self._current_race_time is None:
                    raise RuntimeError("respawn callback did not include race time")
                if self._current_race_time < 0:
                    countdown_observed = True
                elif countdown_observed:
                    break
            else:
                raise RuntimeError(
                    "initial respawn did not complete a negative-to-nonnegative "
                    "countdown transition within "
                    f"{self.config.max_initial_respawn_steps} steps"
                )
        return self._current_state

    def reset(self, diagnostics: ObservationDiagnostics | None = None) -> object:
        self._connect()
        if not self._pending_step or self._current_state is None:
            raise RuntimeError("reset requires a pending simulation step")

        if self._initial_snapshot is None:
            if diagnostics is None:
                raise RuntimeError("first reset requires start-position diagnostics")
            if diagnostics.progress > self.config.max_start_progress:
                raise RuntimeError(
                    f"car is {diagnostics.progress:.3f} units along the path; "
                    "restart A01 before capturing the episode snapshot"
                )
            if (
                abs(diagnostics.lateral_offset)
                > self.config.max_start_lateral_offset
            ):
                raise RuntimeError(
                    f"car is {diagnostics.lateral_offset:.3f} units from the "
                    "reference path; restart A01 before capturing the episode snapshot"
                )
            if int(self._current_state.display_speed) > self.config.max_start_speed:
                raise RuntimeError(
                    f"car speed is {self._current_state.display_speed}; "
                    "wait at the A01 start before capturing the episode snapshot"
                )
            if (
                abs(diagnostics.vertical_offset)
                > self.config.max_start_vertical_offset
            ):
                raise RuntimeError(
                    f"car is {diagnostics.vertical_offset:.3f} vertical units from "
                    "the reference path; wait for a clean A01 respawn"
                )
            self._initial_snapshot = bytes(self._current_state.data)

        self.client.rewind_to_state(self._initial_snapshot)
        self.client.set_continuous_input(steer=0.0, throttle=0.0, brake=0.0)
        self.client.respond(MessageType.SC_RUN_STEP_SYNC)
        self._pending_step = False
        self._wait_for_run_step()
        return self._current_state

    def advance(self, action: np.ndarray) -> SessionStep:
        if (
            not self._pending_step
            or self._current_state is None
            or self._current_race_time is None
        ):
            raise RuntimeError("advance requires a pending simulation step")
        throttle = float(action[1])
        brake = float(action[2])
        if self.config.legacy_reversed_pedal_mapping:
            throttle, brake = brake, throttle
        applied_steer, applied_gas = self.client.set_continuous_input(
            steer=float(action[0]),
            throttle=throttle,
            brake=brake,
        )
        self.client.respond(MessageType.SC_RUN_STEP_SYNC)
        self._pending_step = False
        self._wait_for_run_step()
        race_finished = self.client.race_finished()
        return SessionStep(
            state=self._current_state,
            race_time_ms=int(self._current_race_time),
            race_finished=race_finished,
            applied_steer=applied_steer,
            applied_gas=applied_gas,
        )

    def advance_playback(self) -> PlaybackStep:
        """Advance a loaded TMInterface input file without overriding its inputs."""
        if (
            not self._pending_step
            or self._current_state is None
            or self._current_race_time is None
        ):
            raise RuntimeError("playback advance requires a pending simulation step")
        self.client.respond(MessageType.SC_RUN_STEP_SYNC)
        self._pending_step = False
        self._wait_for_run_step()
        return PlaybackStep(
            state=self._current_state,
            race_time_ms=int(self._current_race_time),
            race_finished=self.client.race_finished(),
        )

    def recover_inputs(self, filename: str) -> None:
        """Persist the current episode's TMInterface input replay."""
        if not self._connected or not self._pending_step:
            raise RuntimeError("input recovery requires a pending live race step")
        self.client.recover_inputs(filename)

    def close(self) -> None:
        self._map_confirmation_stop.set()
        if not self._connected:
            return
        try:
            self.client.set_continuous_input(steer=0.0, throttle=0.0, brake=0.0)
            self.client.set_speed(1.0)
            if self._pending_step:
                self.client.respond(MessageType.SC_RUN_STEP_SYNC)
                self._pending_step = False
        except OSError:
            pass
        finally:
            self.client.close()
            self._connected = False


class JsonlActionLogger:
    """Flush one raw-action audit record per environment step."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8", newline="\n")

    @staticmethod
    def _json_value(value: float) -> float | str:
        if np.isfinite(value):
            return float(value)
        return str(value)

    def log_invalid(
        self, *, episode: int, step: int, action: np.ndarray, reason: str
    ) -> None:
        self._write(
            {
                "episode": episode,
                "step": step,
                "raw_action": [self._json_value(value) for value in action.ravel()],
                "finite": bool(np.isfinite(action).all()),
                "within_range": False,
                "valid": False,
                "error": reason,
            }
        )

    def log_step(self, record: dict[str, Any]) -> None:
        self._write(record)

    def _write(self, record: dict[str, Any]) -> None:
        self._file.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


class TrackmaniaEnv(gym.Env[np.ndarray, np.ndarray]):
    """Synchronous Phase 1 environment with owner-approved temporary reward."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        config: EnvironmentConfig | None = None,
        reference_path: ReferencePath | None = None,
        session: EpisodeSession | None = None,
        action_log_path: Path | None = None,
        reward_function: RewardFunction = phase1_smoke_reward,
    ) -> None:
        super().__init__()
        self.config = config or EnvironmentConfig()
        if self.config.stuck_window_ms is not None:
            if self.config.stuck_window_ms <= 0:
                raise ValueError("stuck_window_ms must be positive when enabled")
            if (
                not math.isfinite(self.config.stuck_progress_gain_units)
                or self.config.stuck_progress_gain_units <= 0.0
            ):
                raise ValueError("stuck_progress_gain_units must be finite and positive")
            if (
                not math.isfinite(self.config.stuck_world_distance_units)
                or self.config.stuck_world_distance_units <= 0.0
            ):
                raise ValueError(
                    "stuck_world_distance_units must be finite and positive"
                )
        self.reference_path = reference_path or ReferencePath.from_csv(
            DEFAULT_REFERENCE_PATH
        )
        self.session = session or LiveTmiSession(self.config)
        self.reward_function = reward_function
        self.reward_name = getattr(
            reward_function,
            "__name__",
            type(reward_function).__name__,
        )
        self.action_space = gym.spaces.Box(
            low=np.asarray([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self._logger = JsonlActionLogger(action_log_path) if action_log_path else None
        self._episode = -1
        self._step = 0
        self._episode_start_race_time = 0
        self._last_step_race_time_ms: int | None = None
        self._last_diagnostics: ObservationDiagnostics | None = None
        self._maximum_progress: float | None = None
        self._last_position: np.ndarray | None = None
        self._last_steer_action: float | None = None
        self._last_steering_delta_direction = 0
        self._steering_reversal_steps: list[int] = []
        self._stuck_history: list[tuple[int, float, float]] = []

    def _observation(
        self, state: object
    ) -> tuple[np.ndarray, ObservationDiagnostics]:
        observation, diagnostics = build_observation(state, self.reference_path)
        if not self.observation_space.contains(observation):
            raise ProtocolError("observation is outside the declared finite shape/dtype")
        return observation, diagnostics

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options

        if self._episode < 0:
            # Inspect the current state before allowing the session to capture it.
            state = self.session.prepare()
            observation, diagnostics = self._observation(state)
            state = self.session.reset(diagnostics)
            observation, diagnostics = self._observation(state)
        else:
            state = self.session.reset(self._last_diagnostics)
            observation, diagnostics = self._observation(state)

        self._episode += 1
        self._step = 0
        self._episode_start_race_time = int(state.race_time)
        self._last_step_race_time_ms = int(state.race_time)
        self._last_diagnostics = diagnostics
        self._maximum_progress = diagnostics.progress
        self._last_position = np.asarray(state.position, dtype=np.float64).copy()
        self._last_steer_action = None
        self._last_steering_delta_direction = 0
        self._steering_reversal_steps = []
        self._stuck_history = [(0, diagnostics.progress, 0.0)]
        return observation, self._info(state, diagnostics)

    def _steering_reversal_status(
        self,
        current_steer: float,
    ) -> tuple[int, bool, int]:
        previous_steer = self._last_steer_action
        steering_delta = (
            0.0 if previous_steer is None else current_steer - previous_steer
        )
        delta_direction = (
            1
            if steering_delta >= V6_STEERING_DELTA_DEADBAND
            else -1
            if steering_delta <= -V6_STEERING_DELTA_DEADBAND
            else 0
        )
        reversal = bool(
            delta_direction
            and self._last_steering_delta_direction
            and delta_direction != self._last_steering_delta_direction
        )
        if delta_direction:
            self._last_steering_delta_direction = delta_direction
        if reversal:
            self._steering_reversal_steps.append(self._step)

        window_steps = round(
            V6_REVERSAL_WINDOW_SECONDS * 1000.0 / self.config.step_period_ms
        )
        first_in_window = self._step - window_steps
        self._steering_reversal_steps = [
            step
            for step in self._steering_reversal_steps
            if step >= first_in_window
        ]
        return delta_direction, reversal, len(self._steering_reversal_steps)

    def _stuck_status(
        self,
        *,
        elapsed_ms: int,
        state: object,
        diagnostics: ObservationDiagnostics,
    ) -> tuple[bool, float | None, float | None]:
        """Evaluate V3's frozen rolling-window truncation without shaping reward."""
        position = np.asarray(state.position, dtype=np.float64)
        if self._last_position is None or not self._stuck_history:
            raise RuntimeError("stuck detection requires reset position history")
        cumulative_distance = self._stuck_history[-1][2] + float(
            np.linalg.norm(position - self._last_position)
        )
        self._last_position = position.copy()
        self._stuck_history.append(
            (elapsed_ms, diagnostics.progress, cumulative_distance)
        )

        window_ms = self.config.stuck_window_ms
        if window_ms is None:
            return False, None, None
        target_ms = elapsed_ms - window_ms
        history_times = [sample[0] for sample in self._stuck_history]
        start = bisect_left(history_times, target_ms)
        start_time, start_progress, start_distance = self._stuck_history[start]
        if elapsed_ms - start_time < window_ms:
            return False, None, None
        progress_gain = diagnostics.progress - start_progress
        world_distance = cumulative_distance - start_distance
        stuck = bool(
            progress_gain < self.config.stuck_progress_gain_units
            and world_distance < self.config.stuck_world_distance_units
        )
        return stuck, progress_gain, world_distance

    def _validated_action(self, action: np.ndarray) -> np.ndarray:
        raw = np.asarray(action)
        converted = np.asarray(action, dtype=np.float32)
        reason: str | None = None
        if converted.shape != (3,):
            reason = f"action must have shape (3,), got {converted.shape}"
        elif not np.isfinite(converted).all():
            reason = "action contains NaN or infinite values"
        elif not self.action_space.contains(converted):
            reason = f"action is outside {self.action_space}"
        if reason is not None:
            if self._logger is not None:
                self._logger.log_invalid(
                    episode=self._episode, step=self._step, action=raw, reason=reason
                )
            raise ValueError(reason)
        return converted

    @staticmethod
    def _full_simstate_log(state: object) -> dict[str, Any]:
        """Extract live dynamics for reward calculation and audit, not observation."""
        try:
            velocity = np.asarray(state.velocity, dtype=np.float64)
            rotation = np.asarray(state.rotation_matrix, dtype=np.float64)
            angular_velocity = np.asarray(
                state.dyna.current_state.angular_speed,
                dtype=np.float64,
            )
            wheel_contacts = [
                bool(wheel.real_time_state.has_ground_contact)
                for wheel in state.simulation_wheels
            ]
            wheel_sliding = [
                bool(wheel.real_time_state.is_sliding)
                for wheel in state.simulation_wheels
            ]
        except AttributeError:
            return {"full_simstate_available": False}
        if len(wheel_contacts) != 4 or len(wheel_sliding) != 4:
            raise ProtocolError("live SimState must expose exactly four wheels")
        values = np.concatenate((velocity, rotation.ravel(), angular_velocity))
        if not np.isfinite(values).all():
            raise ProtocolError("full SimState audit telemetry is nonfinite")
        local_velocity = rotation.T @ velocity
        forward_speed = float(local_velocity[2])
        right_speed = float(local_velocity[0])
        return {
            "full_simstate_available": True,
            "velocity": velocity.tolist(),
            "rotation_matrix": rotation.tolist(),
            "angular_velocity": angular_velocity.tolist(),
            "local_forward_velocity": forward_speed,
            "local_right_velocity": right_speed,
            "local_up_velocity": float(local_velocity[1]),
            "slip_angle_degrees": math.degrees(
                math.atan2(right_speed, max(abs(forward_speed), 1e-9))
            ),
            "body_up_yaw_rate": float(np.dot(angular_velocity, rotation[:, 1])),
            "wheel_ground_contacts": wheel_contacts,
            "wheel_sliding": wheel_sliding,
            "ground_contact_count": sum(wheel_contacts),
            "sliding_wheel_count": sum(wheel_sliding),
        }

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        validated = self._validated_action(action)
        previous_steer = self._last_steer_action
        steering_rate_change = (
            0.0
            if previous_steer is None
            else abs(float(validated[0]) - previous_steer)
        )
        (
            steering_delta_direction,
            steering_slope_reversal,
            steering_reversals_in_window,
        ) = self._steering_reversal_status(float(validated[0]))
        result = self.session.advance(validated)
        race_clock_boundary = bool(
            self._last_step_race_time_ms is not None
            and (
                result.race_time_ms < self._last_step_race_time_ms
                or (
                    self._last_step_race_time_ms < 0
                    and result.race_time_ms >= 0
                )
            )
        )
        if race_clock_boundary:
            previous_steer = None
            steering_rate_change = 0.0
            steering_delta_direction = 0
            steering_slope_reversal = False
            steering_reversals_in_window = 0
            self._last_steering_delta_direction = 0
            self._steering_reversal_steps = []
        observation, diagnostics = self._observation(result.state)
        full_simstate = self._full_simstate_log(result.state)
        elapsed_ms = max(0, result.race_time_ms - self._episode_start_race_time)
        timed_out = elapsed_ms >= self.config.max_episode_ms
        off_track = abs(diagnostics.lateral_offset) > self.config.max_lateral_offset
        below_reference = (
            diagnostics.vertical_offset < -self.config.max_vertical_drop
        )
        terminated = bool(result.race_finished)
        stuck_candidate, stuck_progress_gain, stuck_world_distance = (
            self._stuck_status(
                elapsed_ms=elapsed_ms,
                state=result.state,
                diagnostics=diagnostics,
            )
        )
        # A jump arc below one recorded human trajectory is not itself a fall.
        # Confirm the car has also stopped progressing and moving before using
        # the vertical deviation as a terminal failure signal.
        fallen = bool(not terminated and below_reference and stuck_candidate)
        stuck = bool(not terminated and stuck_candidate and not fallen)
        truncated = bool(
            not terminated and (timed_out or off_track or fallen or stuck)
        )
        if self._last_diagnostics is None:
            raise RuntimeError("step requires reset diagnostics")
        if self._maximum_progress is None:
            raise RuntimeError("step requires reset progress high-water mark")
        maximum_progress_before_step = self._maximum_progress
        new_high_water_progress_delta = max(
            0.0,
            diagnostics.progress - maximum_progress_before_step,
        )
        transition = RewardTransition(
            previous_diagnostics=self._last_diagnostics,
            diagnostics=diagnostics,
            display_speed=int(result.state.display_speed),
            elapsed_ms=elapsed_ms,
            terminated=terminated,
            truncated=truncated,
            timed_out=timed_out,
            off_track=off_track,
            fallen=fallen,
            stuck=stuck,
            steering_rate_change=steering_rate_change,
            steering_slope_reversal=steering_slope_reversal,
            steering_reversals_in_window=steering_reversals_in_window,
            new_high_water_progress_delta=new_high_water_progress_delta,
            full_simstate_available=bool(
                full_simstate["full_simstate_available"]
            ),
            ground_contact_count=int(full_simstate.get("ground_contact_count", 0)),
            sliding_wheel_count=int(full_simstate.get("sliding_wheel_count", 0)),
            slip_angle_degrees=float(full_simstate.get("slip_angle_degrees", 0.0)),
            body_up_yaw_rate=float(full_simstate.get("body_up_yaw_rate", 0.0)),
        )
        reward = float(self.reward_function(transition))
        if not math.isfinite(reward):
            raise ValueError(f"reward function {self.reward_name} returned {reward}")

        info = self._info(result.state, diagnostics)
        info.update(
            {
                "elapsed_ms": elapsed_ms,
                "applied_steer": result.applied_steer,
                "applied_gas": result.applied_gas,
                "timeout": timed_out,
                "off_track": off_track,
                "below_reference": below_reference,
                "fallen": fallen,
                "stuck": stuck,
                "stuck_window_progress_gain": stuck_progress_gain,
                "stuck_window_world_distance": stuck_world_distance,
                "race_finished": terminated,
                "reward_function": self.reward_name,
                "steering_rate_change": steering_rate_change,
                "steering_delta_direction": steering_delta_direction,
                "steering_slope_reversal": steering_slope_reversal,
                "steering_reversals_in_window": steering_reversals_in_window,
                "race_clock_boundary": race_clock_boundary,
            }
        )
        if self._logger is not None:
            self._logger.log_step(
                {
                    "episode": self._episode,
                    "step": self._step,
                    "race_time_ms": result.race_time_ms,
                    "raw_action": validated.tolist(),
                    "input_steer": float(validated[0]),
                    "input_throttle": float(validated[1]),
                    "input_brake": float(validated[2]),
                    "previous_steer": previous_steer,
                    "steering_rate_change": steering_rate_change,
                    "steering_delta_direction": steering_delta_direction,
                    "steering_slope_reversal": steering_slope_reversal,
                    "steering_reversals_in_window": steering_reversals_in_window,
                    "race_clock_boundary": race_clock_boundary,
                    "finite": True,
                    "within_range": True,
                    "valid": True,
                    "applied_steer": result.applied_steer,
                    "applied_gas": result.applied_gas,
                    "display_speed": int(result.state.display_speed),
                    "position": np.asarray(
                        result.state.position,
                        dtype=np.float64,
                    ).tolist(),
                    "upright_cosine": float(
                        np.asarray(
                            result.state.rotation_matrix,
                            dtype=np.float64,
                        )[1, 1]
                    ),
                    "reward": reward,
                    "reward_function": self.reward_name,
                    "terminated": terminated,
                    "truncated": truncated,
                    "timeout": timed_out,
                    "off_track": off_track,
                    "below_reference": below_reference,
                    "fallen": fallen,
                    "stuck": stuck,
                    "stuck_window_progress_gain": stuck_progress_gain,
                    "stuck_window_world_distance": stuck_world_distance,
                    "race_finished": terminated,
                    "previous_progress": self._last_diagnostics.progress,
                    "progress": diagnostics.progress,
                    "new_high_water_progress_delta": (
                        new_high_water_progress_delta
                    ),
                    "lateral_offset": diagnostics.lateral_offset,
                    "vertical_offset": diagnostics.vertical_offset,
                    "heading_error": diagnostics.heading_error,
                    **full_simstate,
                }
            )

        self._last_diagnostics = diagnostics
        self._maximum_progress = max(
            maximum_progress_before_step,
            diagnostics.progress,
        )
        self._last_steer_action = float(validated[0])
        self._last_step_race_time_ms = int(result.race_time_ms)
        self._step += 1
        return observation, reward, terminated, truncated, info

    def _info(
        self, state: object, diagnostics: ObservationDiagnostics
    ) -> dict[str, Any]:
        return {
            "race_time_ms": int(state.race_time),
            "display_speed": int(state.display_speed),
            "progress": diagnostics.progress,
            "lateral_offset": diagnostics.lateral_offset,
            "vertical_offset": diagnostics.vertical_offset,
            "heading_error": diagnostics.heading_error,
            "segment_index": diagnostics.segment_index,
        }

    def close(self) -> None:
        try:
            self.session.close()
        finally:
            if self._logger is not None:
                self._logger.close()
