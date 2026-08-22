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
