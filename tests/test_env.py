from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from trackmania_rl.env import (
    EnvironmentConfig,
    LiveTmiSession,
    SessionStep,
    TrackmaniaEnv,
)
from trackmania_rl.observations import ReferencePath
from trackmania_rl.rewards import sparse_finish_reward
from trackmania_rl.tmi_bridge import MessageType


def state(
    *,
    x: float,
    z: float,
    speed: int,
    race_time: int,
    y: float = 0.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        position=np.asarray([x, y, z]),
        velocity=np.asarray([float(speed) / 3.6, 0.0, 0.0]),
        rotation_matrix=np.asarray(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        ),
        display_speed=speed,
        race_time=race_time,
        data=bytearray(b"state"),
    )


class FakeSession:
    def __init__(
        self,
        states: list[SimpleNamespace],
        *,
        race_finished: bool = False,
    ) -> None:
        self.states = states
        self.race_finished = race_finished
        self.index = 0
        self.actions: list[np.ndarray] = []
        self.closed = False

    def prepare(self) -> SimpleNamespace:
        return self.states[0]

    def reset(self, diagnostics=None) -> SimpleNamespace:
        del diagnostics
        self.index = 0
        return self.states[0]

    def advance(self, action: np.ndarray) -> SessionStep:
        self.actions.append(action.copy())
        self.index = min(self.index + 1, len(self.states) - 1)
        return SessionStep(
            state=self.states[self.index],
            race_time_ms=self.states[self.index].race_time,
            race_finished=self.race_finished,
            applied_steer=round(float(action[0]) * 65536),
            applied_gas=round(float(action[2] - action[1]) * 65536),
        )

    def close(self) -> None:
        self.closed = True


class RecordingBridgeClient:
    def __init__(self) -> None:
        self.calls: list[object] = []
        self.race_is_finished = False

    def give_up(self) -> None:
        self.calls.append("give_up")

    def set_response_timeout(self, timeout_ms) -> None:
        self.calls.append(("response_timeout", timeout_ms))

    def execute_command(self, command) -> None:
        self.calls.append(("command", command))

    def set_speed(self, speed) -> None:
        self.calls.append(("speed", speed))

    def set_on_step_period(self, period_ms) -> None:
        self.calls.append(("step_period", period_ms))

    def race_finished(self) -> bool:
        self.calls.append("race_finished")
        return self.race_is_finished

    def set_continuous_input(self, *, steer, throttle, brake):
        self.calls.append(("input", steer, throttle, brake))
        return round(steer * 65536), round((brake - throttle) * 65536)

    def respond(self, message_type) -> None:
        self.calls.append(("respond", message_type))


class TrackmaniaEnvTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = ReferencePath(
            distances=np.asarray([0.0, 100.0]),
            points=np.asarray([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]),
        )

    def test_step_applies_continuous_action_reward_and_jsonl_audit(self) -> None:
        session = FakeSession(
            [
                state(x=0, z=0, speed=0, race_time=0),
                state(x=1, z=0, speed=100, race_time=100),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "actions.jsonl"
            env = TrackmaniaEnv(
                reference_path=self.path,
                session=session,
                action_log_path=log_path,
            )
            observation, _ = env.reset()
            next_observation, reward, terminated, truncated, info = env.step(
                np.asarray([0.5, 0.75, 0.25], dtype=np.float32)
            )
            env.close()

            self.assertEqual(observation.shape, (26,))
            self.assertEqual(next_observation.shape, (26,))
            self.assertAlmostEqual(reward, 0.1)
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["applied_steer"], 32768)
            self.assertEqual(info["applied_gas"], -32768)
            record = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertEqual(record["raw_action"], [0.5, 0.75, 0.25])
            self.assertTrue(record["finite"])
            self.assertTrue(record["within_range"])
            self.assertFalse(record["timeout"])
            self.assertFalse(record["off_track"])
            self.assertFalse(record["fallen"])
            self.assertFalse(record["race_finished"])

    def test_invalid_action_is_logged_and_rejected_before_session(self) -> None:
        session = FakeSession([state(x=0, z=0, speed=0, race_time=0)])
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "actions.jsonl"
            env = TrackmaniaEnv(
                reference_path=self.path,
                session=session,
                action_log_path=log_path,
            )
            env.reset()

            with self.assertRaisesRegex(ValueError, "NaN"):
                env.step(np.asarray([np.nan, 0.5, 0.0], dtype=np.float32))
            env.close()

            self.assertEqual(session.actions, [])
            record = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertFalse(record["finite"])
            self.assertFalse(record["valid"])

    def test_off_track_step_is_truncated_with_owner_approved_penalty(self) -> None:
        session = FakeSession(
            [
                state(x=0, z=0, speed=0, race_time=0),
                state(x=1, z=-60, speed=100, race_time=100),
            ]
        )
        env = TrackmaniaEnv(
            config=EnvironmentConfig(max_lateral_offset=50.0),
            reference_path=self.path,
            session=session,
        )
        env.reset()

        _, reward, terminated, truncated, info = env.step(
            np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        )
        env.close()

        self.assertAlmostEqual(reward, -0.9)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertTrue(info["off_track"])

    def test_timeout_step_is_truncated_with_owner_approved_penalty(self) -> None:
        session = FakeSession(
            [
                state(x=0, z=0, speed=0, race_time=0),
                state(x=1, z=0, speed=100, race_time=500),
            ]
        )
        env = TrackmaniaEnv(
            config=EnvironmentConfig(max_episode_ms=500),
            reference_path=self.path,
            session=session,
        )
        env.reset()

        _, reward, terminated, truncated, info = env.step(
            np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        )
        env.close()

        self.assertAlmostEqual(reward, -0.9)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertTrue(info["timeout"])

    def test_vertical_fall_is_truncated_before_timeout(self) -> None:
        session = FakeSession(
            [
                state(x=0, z=0, speed=0, race_time=0),
                state(x=1, y=-11, z=0, speed=100, race_time=100),
            ]
        )
        env = TrackmaniaEnv(
            config=EnvironmentConfig(max_vertical_drop=10.0),
            reference_path=self.path,
            session=session,
        )
        env.reset()

        _, reward, terminated, truncated, info = env.step(
            np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        )
        env.close()

        self.assertAlmostEqual(reward, -0.9)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertTrue(info["fallen"])
        self.assertFalse(info["timeout"])

    def test_reward_function_is_injected_and_named_in_info(self) -> None:
        session = FakeSession(
            [
                state(x=0, z=0, speed=0, race_time=0),
                state(x=1, z=0, speed=100, race_time=100),
            ],
            race_finished=True,
        )
        env = TrackmaniaEnv(
            reference_path=self.path,
            session=session,
            reward_function=sparse_finish_reward,
        )
        env.reset()

        _, reward, terminated, truncated, info = env.step(
            np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        )
        env.close()

        self.assertEqual(reward, 1.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["reward_function"], "sparse_finish_reward")

    def test_nonfinite_injected_reward_fails_fast(self) -> None:
        session = FakeSession(
            [
                state(x=0, z=0, speed=0, race_time=0),
                state(x=1, z=0, speed=100, race_time=100),
            ]
        )

        def invalid_reward(_transition):
            return np.nan

        env = TrackmaniaEnv(
            reference_path=self.path,
            session=session,
            reward_function=invalid_reward,
        )
        env.reset()

        with self.assertRaisesRegex(ValueError, "invalid_reward returned nan"):
            env.step(np.asarray([0.0, 0.0, 0.0], dtype=np.float32))
        env.close()


class LiveTmiSessionTests(unittest.TestCase):
    def _ready_session(
        self,
        *,
        legacy: bool = False,
    ) -> tuple[LiveTmiSession, RecordingBridgeClient]:
        session = LiveTmiSession(
            EnvironmentConfig(legacy_reversed_pedal_mapping=legacy)
        )
        client = RecordingBridgeClient()
        current_state = state(x=0, z=0, speed=0, race_time=0)
        session.client = client
        session._connected = True
        session._pending_step = True
        session._current_state = current_state
        session._current_race_time = 0

        def complete_step() -> None:
            session._pending_step = True

        session._wait_for_run_step = complete_step
        return session, client

    def test_advance_uses_corrected_throttle_and_brake_semantics(self) -> None:
        session, client = self._ready_session()

        result = session.advance(
            np.asarray([0.25, 0.75, 0.125], dtype=np.float32)
        )

        self.assertEqual(client.calls[0], ("input", 0.25, 0.75, 0.125))
        self.assertEqual(result.applied_gas, -40960)

    def test_advance_can_reproduce_legacy_reversed_pedal_mapping(self) -> None:
        session, client = self._ready_session(legacy=True)

        result = session.advance(
            np.asarray([0.25, 0.75, 0.125], dtype=np.float32)
        )

        self.assertEqual(client.calls[0], ("input", 0.25, 0.125, 0.75))
        self.assertEqual(result.applied_gas, 40960)

    def test_connect_callback_extends_timeout_before_training_work(self) -> None:
        config = EnvironmentConfig(
            simulation_speed=100.0,
            step_period_ms=100,
            bridge_response_timeout_ms=30_000,
        )
        session = LiveTmiSession(config)
        client = RecordingBridgeClient()
        session.client = client

        session._handle_non_step(MessageType.SC_ON_CONNECT_SYNC)

        self.assertEqual(
            client.calls,
            [
                ("response_timeout", 30_000),
                ("command", "set unfocused_fps_limit false"),
                ("command", "set disable_forced_camera true"),
                ("speed", 100.0),
                ("step_period", 100),
                ("respond", MessageType.SC_ON_CONNECT_SYNC),
            ],
        )

    def test_prepare_respawns_and_waits_neutral_through_countdown(self) -> None:
        session = LiveTmiSession(EnvironmentConfig(auto_respawn_on_connect=True))
        client = RecordingBridgeClient()
        initial_state = state(x=20, z=0, speed=100, race_time=1000)
        countdown_states = iter(
            [
                state(x=0, z=0, speed=0, race_time=1000),
                state(x=0, z=0, speed=0, race_time=-200),
                state(x=0, z=0, speed=0, race_time=-100),
                state(x=0, z=0, speed=0, race_time=0),
            ]
        )
        session.client = client
        session._connected = True
        session._pending_step = True
        session._current_state = initial_state

        def complete_respawn_step() -> None:
            respawned_state = next(countdown_states)
            session._current_state = respawned_state
            session._current_race_time = respawned_state.race_time
            session._pending_step = True

        session._wait_for_run_step = complete_respawn_step

        prepared = session.prepare()

        self.assertEqual(prepared.race_time, 0)
        self.assertEqual(
            client.calls,
            [
                "give_up",
                ("input", 0.0, 0.0, 0.0),
                ("respond", MessageType.SC_RUN_STEP_SYNC),
                ("input", 0.0, 0.0, 0.0),
                ("respond", MessageType.SC_RUN_STEP_SYNC),
                ("input", 0.0, 0.0, 0.0),
                ("respond", MessageType.SC_RUN_STEP_SYNC),
                ("input", 0.0, 0.0, 0.0),
                ("respond", MessageType.SC_RUN_STEP_SYNC),
            ],
        )

    def test_prepare_fails_if_respawn_countdown_never_finishes(self) -> None:
        session = LiveTmiSession(
            EnvironmentConfig(
                auto_respawn_on_connect=True,
                max_initial_respawn_steps=2,
            )
        )
        client = RecordingBridgeClient()
        countdown_state = state(x=0, z=0, speed=0, race_time=-1000)
        session.client = client
        session._connected = True
        session._pending_step = True
        session._current_state = countdown_state

        def stalled_countdown_step() -> None:
            session._current_state = countdown_state
            session._current_race_time = countdown_state.race_time
            session._pending_step = True

        session._wait_for_run_step = stalled_countdown_step

        with self.assertRaisesRegex(RuntimeError, "countdown transition"):
            session.prepare()

    def test_prepare_can_preserve_current_state_when_auto_respawn_disabled(self) -> None:
        session = LiveTmiSession(EnvironmentConfig(auto_respawn_on_connect=False))
        client = RecordingBridgeClient()
        current_state = state(x=20, z=0, speed=100, race_time=1000)
        session.client = client
        session._connected = True
        session._pending_step = True
        session._current_state = current_state

        prepared = session.prepare()

        self.assertIs(prepared, current_state)
        self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main()
