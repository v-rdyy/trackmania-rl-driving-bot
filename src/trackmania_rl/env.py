"""Gymnasium environment for the Phase 1 TrackMania integration smoke test."""

from __future__ import annotations

import json
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
from trackmania_rl.tmi_bridge import MessageType, ProtocolError, TmiBridgeClient

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE_PATH = WORKSPACE_ROOT / "data" / "tracks" / "a01_reference_path.csv"


@dataclass(frozen=True)
class EnvironmentConfig:
    port: int = 8478
    simulation_speed: float = 6.0
    step_period_ms: int = 100
    max_episode_ms: int = 45_000
    max_lateral_offset: float = 50.0
    max_start_progress: float = 25.0
    max_start_lateral_offset: float = 10.0
    max_start_speed: int = 5


@dataclass(frozen=True)
class SessionStep:
    state: object
    race_time_ms: int
    race_finished: bool
    applied_steer: int
    applied_gas: int


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

    def _connect(self) -> None:
        if self._connected:
            return
        self.client.connect()
        self._connected = True
        self._wait_for_run_step()

    def _handle_non_step(self, message_type: MessageType) -> None:
        if message_type is MessageType.SC_ON_CONNECT_SYNC:
            self.client.execute_command("set unfocused_fps_limit false")
            self.client.execute_command("set disable_forced_camera true")
            self.client.set_speed(self.config.simulation_speed)
            self.client.set_on_step_period(self.config.step_period_ms)
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
                return
            self._handle_non_step(message_type)

    def prepare(self) -> object:
        """Connect and expose the current state without capturing a snapshot."""
        self._connect()
        if not self._pending_step or self._current_state is None:
            raise RuntimeError("prepare requires a pending simulation step")
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
        applied_steer, applied_gas = self.client.set_continuous_input(
            steer=float(action[0]),
            throttle=float(action[1]),
            brake=float(action[2]),
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

    def close(self) -> None:
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
    ) -> None:
        super().__init__()
        self.config = config or EnvironmentConfig()
        self.reference_path = reference_path or ReferencePath.from_csv(
            DEFAULT_REFERENCE_PATH
        )
        self.session = session or LiveTmiSession(self.config)
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
        self._last_diagnostics: ObservationDiagnostics | None = None

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
        self._last_diagnostics = diagnostics
        return observation, self._info(state, diagnostics)

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

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        validated = self._validated_action(action)
        result = self.session.advance(validated)
        observation, diagnostics = self._observation(result.state)
        elapsed_ms = max(0, result.race_time_ms - self._episode_start_race_time)
        timed_out = elapsed_ms >= self.config.max_episode_ms
        off_track = abs(diagnostics.lateral_offset) > self.config.max_lateral_offset
        terminated = bool(result.race_finished)
        truncated = bool(not terminated and (timed_out or off_track))
        reward = float(result.state.display_speed) / 1000.0
        if truncated:
            reward -= 1.0

        info = self._info(result.state, diagnostics)
        info.update(
            {
                "elapsed_ms": elapsed_ms,
                "applied_steer": result.applied_steer,
                "applied_gas": result.applied_gas,
                "timeout": timed_out,
                "off_track": off_track,
            }
        )
        if self._logger is not None:
            self._logger.log_step(
                {
                    "episode": self._episode,
                    "step": self._step,
                    "race_time_ms": result.race_time_ms,
                    "raw_action": validated.tolist(),
                    "finite": True,
                    "within_range": True,
                    "valid": True,
                    "applied_steer": result.applied_steer,
                    "applied_gas": result.applied_gas,
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "progress": diagnostics.progress,
                    "lateral_offset": diagnostics.lateral_offset,
                    "heading_error": diagnostics.heading_error,
                }
            )

        self._last_diagnostics = diagnostics
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
            "heading_error": diagnostics.heading_error,
            "segment_index": diagnostics.segment_index,
        }

    def close(self) -> None:
        try:
            self.session.close()
        finally:
            if self._logger is not None:
                self._logger.close()
