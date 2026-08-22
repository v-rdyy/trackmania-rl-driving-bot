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

from trackmania_rl.env import EnvironmentConfig, SessionStep, TrackmaniaEnv
from trackmania_rl.observations import ReferencePath


def state(*, x: float, z: float, speed: int, race_time: int) -> SimpleNamespace:
    return SimpleNamespace(
        position=np.asarray([x, 0.0, z]),
        velocity=np.asarray([float(speed) / 3.6, 0.0, 0.0]),
        rotation_matrix=np.asarray(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        ),
        display_speed=speed,
        race_time=race_time,
        data=bytearray(b"state"),
    )


class FakeSession:
    def __init__(self, states: list[SimpleNamespace]) -> None:
        self.states = states
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
            race_finished=False,
            applied_steer=round(float(action[0]) * 65536),
            applied_gas=round(float(action[1] - action[2]) * 65536),
        )

    def close(self) -> None:
        self.closed = True


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
            self.assertEqual(info["applied_gas"], 32768)
            record = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertEqual(record["raw_action"], [0.5, 0.75, 0.25])
            self.assertTrue(record["finite"])
            self.assertTrue(record["within_range"])

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


if __name__ == "__main__":
    unittest.main()
