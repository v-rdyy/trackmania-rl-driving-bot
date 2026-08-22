from __future__ import annotations

import socket
import struct
import sys
import threading
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from trackmania_rl.tmi_bridge import MessageType, ProtocolError, TmiBridgeClient


class TmiBridgeClientTests(unittest.TestCase):
    def test_read_message_type_accepts_known_message(self) -> None:
        server, client_socket = socket.socketpair()
        self.addCleanup(server.close)
        client = TmiBridgeClient()
        client._socket = client_socket
        self.addCleanup(client_socket.close)

        server.sendall(struct.pack("<i", MessageType.SC_ON_CONNECT_SYNC))

        self.assertIs(client.read_message_type(), MessageType.SC_ON_CONNECT_SYNC)

    def test_recv_exact_reassembles_fragmented_payload(self) -> None:
        server, client_socket = socket.socketpair()
        self.addCleanup(server.close)
        client = TmiBridgeClient()
        client._socket = client_socket
        self.addCleanup(client_socket.close)

        def send_fragments() -> None:
            server.sendall(b"ab")
            server.sendall(b"cdef")

        sender = threading.Thread(target=send_fragments)
        sender.start()
        self.assertEqual(client._recv_exact(6), b"abcdef")
        sender.join()

    def test_read_message_type_rejects_unknown_message(self) -> None:
        server, client_socket = socket.socketpair()
        self.addCleanup(server.close)
        client = TmiBridgeClient()
        client._socket = client_socket
        self.addCleanup(client_socket.close)

        server.sendall(struct.pack("<i", 999))

        with self.assertRaisesRegex(ProtocolError, "unknown bridge message"):
            client.read_message_type()

    def test_race_finished_uses_protocol_request_and_decodes_response(self) -> None:
        server, client_socket = socket.socketpair()
        self.addCleanup(server.close)
        client = TmiBridgeClient()
        client._socket = client_socket
        self.addCleanup(client_socket.close)

        requests: list[int] = []

        def answer_request() -> None:
            requests.append(struct.unpack("<i", server.recv(4))[0])
            server.sendall(struct.pack("<i", 1))

        responder = threading.Thread(target=answer_request)
        responder.start()
        self.assertTrue(client.race_finished())
        responder.join()
        self.assertEqual(requests, [MessageType.C_RACE_FINISHED])

    def test_set_input_state_encodes_four_digital_inputs(self) -> None:
        server, client_socket = socket.socketpair()
        self.addCleanup(server.close)
        client = TmiBridgeClient()
        client._socket = client_socket
        self.addCleanup(client_socket.close)

        client.set_input_state(left=True, accelerate=True)

        self.assertEqual(
            server.recv(8),
            struct.pack("<iBBBB", MessageType.C_SET_INPUT_STATE, 1, 0, 1, 0),
        )

    def test_set_speed_encodes_float_multiplier(self) -> None:
        server, client_socket = socket.socketpair()
        self.addCleanup(server.close)
        client = TmiBridgeClient()
        client._socket = client_socket
        self.addCleanup(client_socket.close)

        client.set_speed(4.0)

        self.assertEqual(
            server.recv(8), struct.pack("<if", MessageType.C_SET_SPEED, 4.0)
        )

    def test_set_speed_rejects_unsafe_multiplier(self) -> None:
        client = TmiBridgeClient()

        with self.assertRaisesRegex(ValueError, "speed must be"):
            client.set_speed(0.0)

    def test_set_on_step_period_encodes_valid_tick_multiple(self) -> None:
        server, client_socket = socket.socketpair()
        self.addCleanup(server.close)
        client = TmiBridgeClient()
        client._socket = client_socket
        self.addCleanup(client_socket.close)

        client.set_on_step_period(100)

        self.assertEqual(
            server.recv(8),
            struct.pack("<ii", MessageType.C_SET_ON_STEP_PERIOD, 100),
        )

    def test_set_on_step_period_rejects_non_tick_value(self) -> None:
        client = TmiBridgeClient()

        with self.assertRaisesRegex(ValueError, "multiple of 10"):
            client.set_on_step_period(25)

    def test_set_response_timeout_encodes_unsigned_milliseconds(self) -> None:
        server, client_socket = socket.socketpair()
        self.addCleanup(server.close)
        client = TmiBridgeClient()
        client._socket = client_socket
        self.addCleanup(client_socket.close)

        client.set_response_timeout(30_000)

        self.assertEqual(
            server.recv(8),
            struct.pack("<iI", MessageType.C_SET_TIMEOUT, 30_000),
        )

    def test_set_response_timeout_rejects_invalid_uint32(self) -> None:
        client = TmiBridgeClient()

        with self.assertRaisesRegex(ValueError, "positive uint32"):
            client.set_response_timeout(0)
        with self.assertRaisesRegex(ValueError, "positive uint32"):
            client.set_response_timeout(0x1_0000_0000)

    def test_recover_inputs_sends_safe_tm_interface_command(self) -> None:
        server, client_socket = socket.socketpair()
        self.addCleanup(server.close)
        client = TmiBridgeClient()
        client._socket = client_socket
        self.addCleanup(client_socket.close)

        client.recover_inputs("v2_speed_final_take_001_ep_01.txt")

        command = b"recover_inputs v2_speed_final_take_001_ep_01.txt"
        self.assertEqual(
            server.recv(8 + len(command)),
            struct.pack("<ii", MessageType.C_EXECUTE_COMMAND, len(command)) + command,
        )

    def test_recover_inputs_rejects_paths_and_non_txt_names(self) -> None:
        client = TmiBridgeClient()

        for filename in ("../replay.txt", "folder/replay.txt", "replay.gbx", ""):
            with self.subTest(filename=filename):
                with self.assertRaisesRegex(ValueError, "simple alphanumeric"):
                    client.recover_inputs(filename)

    def test_rewind_to_state_encodes_snapshot_length_and_bytes(self) -> None:
        server, client_socket = socket.socketpair()
        self.addCleanup(server.close)
        client = TmiBridgeClient()
        client._socket = client_socket
        self.addCleanup(client_socket.close)
        snapshot = b"snapshot"

        client.rewind_to_state(snapshot)

        self.assertEqual(
            server.recv(8 + len(snapshot)),
            struct.pack("<ii", MessageType.C_REWIND_TO_STATE, len(snapshot)) + snapshot,
        )

    def test_continuous_input_encodes_analog_steer_and_gas(self) -> None:
        server, client_socket = socket.socketpair()
        self.addCleanup(server.close)
        client = TmiBridgeClient()
        client._socket = client_socket
        self.addCleanup(client_socket.close)

        applied = client.set_continuous_input(
            steer=-0.5, throttle=0.75, brake=0.25
        )

        self.assertEqual(applied, (-32768, 32768))
        self.assertEqual(
            server.recv(12),
            struct.pack(
                "<iii", MessageType.C_SET_ANALOG_INPUT_STATE, -32768, 32768
            ),
        )

    def test_continuous_input_rejects_nonfinite_and_out_of_range_values(self) -> None:
        client = TmiBridgeClient()

        with self.assertRaisesRegex(ValueError, "steer must be finite"):
            client.set_continuous_input(steer=float("nan"), throttle=0.0, brake=0.0)
        with self.assertRaisesRegex(ValueError, "throttle must be in"):
            client.set_continuous_input(steer=0.0, throttle=1.1, brake=0.0)
        with self.assertRaisesRegex(ValueError, "brake must be in"):
            client.set_continuous_input(steer=0.0, throttle=0.0, brake=-0.1)

    def test_give_up_sends_protocol_request(self) -> None:
        server, client_socket = socket.socketpair()
        self.addCleanup(server.close)
        client = TmiBridgeClient()
        client._socket = client_socket
        self.addCleanup(client_socket.close)

        client.give_up()

        self.assertEqual(
            server.recv(4), struct.pack("<i", MessageType.C_GIVE_UP)
        )


if __name__ == "__main__":
    unittest.main()
