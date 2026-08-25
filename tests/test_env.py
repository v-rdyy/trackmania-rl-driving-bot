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
from trackmania_rl.rewards import (
    clamped_forward_progress_reward,
    clustered_reversal_frequency_reward,
    sparse_finish_reward,
    steering_rate_smoothness_reward,
)
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
            self.assertEqual(record["upright_cosine"], 1.0)
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

    def test_steering_rate_change_is_zero_on_reset_then_tracks_frame_delta(self) -> None:
        session = FakeSession(
            [
                state(x=0, z=0, speed=0, race_time=0),
                state(x=1, z=0, speed=100, race_time=100),
                state(x=2, z=0, speed=100, race_time=200),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "actions.jsonl"
            env = TrackmaniaEnv(
                reference_path=self.path,
                session=session,
                action_log_path=log_path,
                reward_function=steering_rate_smoothness_reward,
            )
            env.reset()
            _, first_reward, _, _, first_info = env.step(
                np.asarray([0.75, 1.0, 0.0], dtype=np.float32)
            )
            _, second_reward, _, _, second_info = env.step(
                np.asarray([-0.25, 1.0, 0.0], dtype=np.float32)
            )
            env.reset()
            _, reset_reward, _, _, reset_info = env.step(
                np.asarray([-0.25, 1.0, 0.0], dtype=np.float32)
            )
            env.close()

            records = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(first_info["steering_rate_change"], 0.0)
        self.assertEqual(second_info["steering_rate_change"], 1.0)
        self.assertEqual(reset_info["steering_rate_change"], 0.0)
        self.assertAlmostEqual(first_reward, 0.0)
        self.assertAlmostEqual(second_reward, -0.05)
        self.assertAlmostEqual(reset_reward, 0.0)
        self.assertIsNone(records[0]["previous_steer"])
        self.assertEqual(records[1]["previous_steer"], 0.75)
        self.assertIsNone(records[2]["previous_steer"])

    def test_clustered_reversal_history_is_event_triggered_and_clears_on_reset(self) -> None:
        session = FakeSession(
            [state(x=index, z=0, speed=100, race_time=index * 100) for index in range(8)]
        )
        actions = [0.0, 0.10, 0.0, 0.10, 0.0, 0.10]
        env = TrackmaniaEnv(
            reference_path=self.path,
            session=session,
            reward_function=clustered_reversal_frequency_reward,
        )
        env.reset()
        infos = []
        rewards = []
        for steer in actions:
            _, reward, _, _, info = env.step(
                np.asarray([steer, 1.0, 0.0], dtype=np.float32)
            )
            infos.append(info)
            rewards.append(reward)
        env.reset()
        _, _, _, _, reset_info = env.step(
            np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        )
        env.close()

        self.assertEqual(
            [info["steering_slope_reversal"] for info in infos],
            [False, False, True, True, True, True],
        )
        self.assertEqual(
            [info["steering_reversals_in_window"] for info in infos],
            [0, 0, 1, 2, 3, 4],
        )
        self.assertAlmostEqual(rewards[-1], rewards[-2] - 0.05)
        self.assertFalse(reset_info["steering_slope_reversal"])
        self.assertEqual(reset_info["steering_reversals_in_window"], 0)

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

    def test_vertical_drop_requires_stalled_confirmation(self) -> None:
        session = FakeSession(
            [
                state(x=0, z=0, speed=0, race_time=0),
                state(x=1, y=-11, z=0, speed=100, race_time=100),
                state(x=1, y=-11, z=0, speed=0, race_time=2100),
            ]
        )
        env = TrackmaniaEnv(
            config=EnvironmentConfig(
                max_vertical_drop=10.0,
                stuck_window_ms=2_000,
            ),
            reference_path=self.path,
            session=session,
        )
        env.reset()

        _, first_reward, terminated, truncated, first_info = env.step(
            np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        )
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(first_info["below_reference"])
        self.assertFalse(first_info["fallen"])

        _, reward, terminated, truncated, info = env.step(
            np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        )
        env.close()

        self.assertAlmostEqual(first_reward, 0.1)
        self.assertAlmostEqual(reward, -1.0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertTrue(info["below_reference"])
        self.assertTrue(info["fallen"])
        self.assertFalse(info["stuck"])
        self.assertFalse(info["timeout"])

    def test_low_jump_arc_is_not_a_fall_while_progress_continues(self) -> None:
        session = FakeSession(
            [
                state(x=0, z=0, speed=0, race_time=0),
                state(x=1, y=-11, z=0, speed=100, race_time=100),
                state(x=15, y=-11, z=0, speed=100, race_time=2100),
            ]
        )
        env = TrackmaniaEnv(
            config=EnvironmentConfig(
                max_vertical_drop=10.0,
                stuck_window_ms=2_000,
            ),
            reference_path=self.path,
            session=session,
        )
        env.reset()
        env.step(np.asarray([0.0, 0.0, 0.0], dtype=np.float32))

        _, _, terminated, truncated, info = env.step(
            np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        )
        env.close()

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["below_reference"])
        self.assertFalse(info["fallen"])
        self.assertFalse(info["stuck"])

    def test_live_config_can_request_a_map_without_changing_offline_sessions(self) -> None:
        config = EnvironmentConfig(map_to_load="A01-Race.Challenge.Gbx")

        self.assertEqual(config.map_to_load, "A01-Race.Challenge.Gbx")

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

    def test_v3_stuck_window_truncates_stationary_episode_at_two_seconds(self) -> None:
        session = FakeSession(
            [
                state(x=0, z=0, speed=0, race_time=time_ms)
                for time_ms in range(0, 2100, 100)
            ]
        )
        env = TrackmaniaEnv(
            config=EnvironmentConfig(stuck_window_ms=2_000),
            reference_path=self.path,
            session=session,
            reward_function=clamped_forward_progress_reward,
        )
        env.reset()

        final_result = None
        for _ in range(20):
            final_result = env.step(
                np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
            )
        env.close()

        assert final_result is not None
        _, reward, terminated, truncated, info = final_result
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertTrue(info["stuck"])
        self.assertEqual(info["elapsed_ms"], 2_000)
        self.assertEqual(info["stuck_window_progress_gain"], 0.0)
        self.assertEqual(info["stuck_window_world_distance"], 0.0)

    def test_v3_world_motion_prevents_false_stuck_truncation(self) -> None:
        session = FakeSession(
            [
                state(x=0, z=index * 0.2, speed=0, race_time=index * 100)
                for index in range(21)
            ]
        )
        env = TrackmaniaEnv(
            config=EnvironmentConfig(stuck_window_ms=2_000),
            reference_path=self.path,
            session=session,
            reward_function=clamped_forward_progress_reward,
        )
        env.reset()

        final_result = None
        for _ in range(20):
            final_result = env.step(
                np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
            )
        env.close()

        assert final_result is not None
        _, _, terminated, truncated, info = final_result
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["stuck"])
        self.assertAlmostEqual(info["stuck_window_progress_gain"], 0.0)
        self.assertAlmostEqual(info["stuck_window_world_distance"], 4.0)

    def test_v3_progress_prevents_false_stuck_truncation(self) -> None:
        session = FakeSession(
            [
                state(x=index * 0.075, z=0, speed=0, race_time=index * 100)
                for index in range(21)
            ]
        )
        env = TrackmaniaEnv(
            config=EnvironmentConfig(stuck_window_ms=2_000),
            reference_path=self.path,
            session=session,
            reward_function=clamped_forward_progress_reward,
        )
        env.reset()

        final_result = None
        for _ in range(20):
            final_result = env.step(
                np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
            )
        env.close()

        assert final_result is not None
        _, _, terminated, truncated, info = final_result
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["stuck"])
        self.assertAlmostEqual(info["stuck_window_progress_gain"], 1.5)
        self.assertAlmostEqual(info["stuck_window_world_distance"], 1.5)

    def test_v3_finish_takes_precedence_over_stuck_candidate(self) -> None:
        session = FakeSession(
            [
                state(x=0, z=0, speed=0, race_time=0),
                state(x=0, z=0, speed=0, race_time=2_000),
            ],
            race_finished=True,
        )
        env = TrackmaniaEnv(
            config=EnvironmentConfig(stuck_window_ms=2_000),
            reference_path=self.path,
            session=session,
            reward_function=clamped_forward_progress_reward,
        )
        env.reset()

        _, _, terminated, truncated, info = env.step(
            np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        )
        env.close()

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["stuck"])

    def test_stuck_thresholds_are_validated_when_enabled(self) -> None:
        session = FakeSession([state(x=0, z=0, speed=0, race_time=0)])
        with self.assertRaisesRegex(ValueError, "stuck_window_ms"):
            TrackmaniaEnv(
                config=EnvironmentConfig(stuck_window_ms=0),
                reference_path=self.path,
                session=session,
            )
        with self.assertRaisesRegex(ValueError, "stuck_progress_gain_units"):
            TrackmaniaEnv(
                config=EnvironmentConfig(
                    stuck_window_ms=2_000,
                    stuck_progress_gain_units=float("nan"),
                ),
                reference_path=self.path,
                session=session,
            )


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

    def test_playback_advance_does_not_override_loaded_inputs(self) -> None:
        session, client = self._ready_session()

        result = session.advance_playback()

        self.assertEqual(
            client.calls,
            [
                ("respond", MessageType.SC_RUN_STEP_SYNC),
                "race_finished",
            ],
        )
        self.assertEqual(result.race_time_ms, 0)

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

    def test_connect_callback_can_load_and_confirm_a_map(self) -> None:
        session = LiveTmiSession(
            EnvironmentConfig(map_to_load="A01-Race.Challenge.Gbx")
        )
        client = RecordingBridgeClient()
        session.client = client
        confirmations: list[bool] = []
        session._start_map_confirmation = lambda: confirmations.append(True)

        session._handle_non_step(MessageType.SC_ON_CONNECT_SYNC)

        self.assertIn(
            ("command", "map A01-Race.Challenge.Gbx"),
            client.calls,
        )
        self.assertEqual(confirmations, [True])

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

    def test_prepare_can_wait_for_a_fresh_map_countdown_without_respawning(self) -> None:
        session = LiveTmiSession(
            EnvironmentConfig(
                auto_respawn_on_connect=False,
                wait_for_race_start_on_connect=True,
            )
        )
        client = RecordingBridgeClient()
        countdown_states = iter(
            [
                state(x=0, z=0, speed=0, race_time=-100),
                state(x=0, z=0, speed=0, race_time=0),
            ]
        )
        session.client = client
        session._connected = True
        session._pending_step = True
        session._current_state = state(x=0, z=0, speed=0, race_time=-200)
        session._current_race_time = -200

        def complete_countdown_step() -> None:
            current = next(countdown_states)
            session._current_state = current
            session._current_race_time = current.race_time
            session._pending_step = True

        session._wait_for_run_step = complete_countdown_step

        prepared = session.prepare()

        self.assertEqual(prepared.race_time, 0)
        self.assertNotIn("give_up", client.calls)

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
