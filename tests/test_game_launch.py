from __future__ import annotations

import unittest
import sys
from collections import namedtuple
from pathlib import Path
from unittest.mock import Mock, patch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from trackmania_rl.game_launch import (
    _bridge_is_listening,
    click_trackmania_client,
    ensure_trackmania_running,
    tmloader_command,
)
from trackmania_rl.video_capture import VideoCaptureError, WindowTarget


class GameLaunchTests(unittest.TestCase):
    @patch("trackmania_rl.game_launch.psutil.net_connections")
    def test_bridge_listener_check_does_not_open_a_client_connection(
        self,
        net_connections: Mock,
    ) -> None:
        Address = namedtuple("Address", "ip port")
        Connection = namedtuple("Connection", "laddr status")
        net_connections.return_value = [
            Connection(Address("127.0.0.1", 8478), "LISTEN")
        ]

        self.assertTrue(_bridge_is_listening(8478))
        net_connections.assert_called_once_with(kind="tcp")

    def test_client_click_rejects_coordinates_outside_the_window(self) -> None:
        with self.assertRaisesRegex(ValueError, "fractions"):
            click_trackmania_client(1.1, 0.5)

    def test_profile_command_uses_modloader_run_mode(self) -> None:
        self.assertEqual(
            tmloader_command(
                executable=Path("TMLoader.exe"),
                game="TmForever",
                profile="default",
            ),
            ["TMLoader.exe", "run", "TmForever", "default"],
        )

    @patch("trackmania_rl.game_launch.confirm_a01_solo")
    @patch("trackmania_rl.game_launch._bridge_is_listening", return_value=True)
    @patch("trackmania_rl.game_launch.subprocess.Popen")
    @patch("trackmania_rl.game_launch.find_trackmania_window")
    def test_launches_missing_game_and_clears_startup_screen(
        self,
        find_window: Mock,
        popen: Mock,
        _bridge: Mock,
        confirm_solo: Mock,
    ) -> None:
        target = WindowTarget(1, "TrackMania", 0, 0, 640, 480)
        find_window.side_effect = [VideoCaptureError("missing"), target]
        executable = Path("TMLoader.exe")
        with patch.object(Path, "is_file", return_value=True):
            observed, launched = ensure_trackmania_running(
                executable=executable,
                timeout_seconds=1.0,
            )

        self.assertIs(observed, target)
        self.assertTrue(launched)
        popen.assert_called_once_with(
            ["TMLoader.exe", "run", "TmForever", "default"]
        )
        confirm_solo.assert_called_once_with(target)

    @patch("trackmania_rl.game_launch.find_trackmania_window")
    def test_existing_game_is_not_touched(self, find_window: Mock) -> None:
        target = WindowTarget(1, "TrackMania", 0, 0, 640, 480)
        find_window.return_value = target

        observed, launched = ensure_trackmania_running()

        self.assertIs(observed, target)
        self.assertFalse(launched)

    @patch("trackmania_rl.game_launch.time.sleep")
    @patch("trackmania_rl.game_launch.confirm_a01_solo")
    @patch("trackmania_rl.game_launch.find_trackmania_window")
    def test_existing_interrupted_load_can_be_confirmed(
        self,
        find_window: Mock,
        confirm_solo: Mock,
        _sleep: Mock,
    ) -> None:
        target = WindowTarget(1, "TrackMania", 0, 0, 640, 480)
        find_window.return_value = target

        observed, launched = ensure_trackmania_running(confirm_existing=True)

        self.assertIs(observed, target)
        self.assertFalse(launched)
        confirm_solo.assert_called_once_with(target)
