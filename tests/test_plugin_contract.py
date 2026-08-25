from __future__ import annotations

import re
import unittest
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_PATH = WORKSPACE_ROOT / "vendor" / "tminterface" / "python_link.as"


class PythonLinkPluginContractTests(unittest.TestCase):
    def test_queued_startup_handshake_recovers_after_missed_menu_event(self) -> None:
        source = PLUGIN_PATH.read_text(encoding="utf-8")
        render_body = re.search(
            r"void Render\(\)\{(?P<body>.*?)\n\}",
            source,
            re.DOTALL,
        ).group("body")

        self.assertIn("on_connect_queued", render_body)
        self.assertIn("GetCurrentGameState() != TM::GameState::StartUp", render_body)
        self.assertIn("OnConnect()", render_body)

    def test_snapshot_count_callbacks_are_not_nested_socket_exchanges(self) -> None:
        source = PLUGIN_PATH.read_text(encoding="utf-8")

        for callback in ("OnCheckpointCountChanged", "OnLapCountChanged"):
            match = re.search(
                rf"void {callback}\([^)]*\)\{{(?P<body>.*?)\n\}}",
                source,
                re.DOTALL,
            )
            self.assertIsNotNone(match, f"missing {callback}")
            body = match.group("body")
            self.assertNotIn("clientSock.Write", body)
            self.assertNotIn("WaitForResponse", body)

        checkpoint_body = re.search(
            r"void OnCheckpointCountChanged\([^)]*\)\{(?P<body>.*?)\n\}",
            source,
            re.DOTALL,
        ).group("body")
        self.assertIn("simManager.PreventSimulationFinish()", checkpoint_body)
        self.assertIn("race_finished_pending = true", checkpoint_body)


if __name__ == "__main__":
    unittest.main()
