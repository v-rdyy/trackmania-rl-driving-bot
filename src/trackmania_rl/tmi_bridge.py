"""Small TCP client for the vendored TMInterface 2.x AngelScript bridge.

The wire protocol follows the maintained ``dersiwi/trackmania-gym`` bridge,
which in turn preserves the simulation-state structures from the legacy
``tminterface`` Python package. This module intentionally exposes only the
Phase 0 telemetry surface; input control is added and verified separately.
"""

from __future__ import annotations

import socket
import struct
from enum import IntEnum, auto

from tminterface.structs import CheckpointData, SimStateData


class MessageType(IntEnum):
    SC_RUN_STEP_SYNC = auto()
    SC_CHECKPOINT_COUNT_CHANGED_SYNC = auto()
    SC_LAP_COUNT_CHANGED_SYNC = auto()
    SC_REQUESTED_FRAME_SYNC = auto()
    SC_ON_CONNECT_SYNC = auto()
    C_SET_SPEED = auto()
    C_REWIND_TO_STATE = auto()
    C_REWIND_TO_CURRENT_STATE = auto()
    C_GET_SIMULATION_STATE = auto()
    C_SET_INPUT_STATE = auto()
    C_GIVE_UP = auto()
    C_PREVENT_SIMULATION_FINISH = auto()
    C_SHUTDOWN = auto()
    C_EXECUTE_COMMAND = auto()
    C_SET_TIMEOUT = auto()
    C_RACE_FINISHED = auto()
    C_REQUEST_FRAME = auto()
    C_RESET_CAMERA = auto()
    C_SET_ON_STEP_PERIOD = auto()
    C_UNREQUEST_FRAME = auto()
    C_TOGGLE_INTERFACE = auto()
    C_IS_IN_MENUS = auto()
    C_GET_INPUTS = auto()


class ProtocolError(RuntimeError):
    """Raised when the bridge sends an incomplete or unexpected message."""


class TmiBridgeClient:
    """Synchronous loopback client for one TMInterface ``python_link.as``."""

    def __init__(self, port: int = 8478, timeout_seconds: float = 15.0) -> None:
        if not 1 <= port <= 65535:
            raise ValueError(f"port must be between 1 and 65535, got {port}")
        self.port = port
        self.timeout_seconds = timeout_seconds
        self._socket: socket.socket | None = None

    def connect(self) -> None:
        if self._socket is not None:
            raise RuntimeError("client is already connected")
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(self.timeout_seconds)
        client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        client.connect(("127.0.0.1", self.port))
        self._socket = client

    def close(self) -> None:
        if self._socket is None:
            return
        try:
            self._socket.sendall(struct.pack("<i", MessageType.C_SHUTDOWN))
        except OSError:
            pass
        finally:
            self._socket.close()
            self._socket = None

    def __enter__(self) -> TmiBridgeClient:
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def read_message_type(self) -> MessageType:
        raw_type = self.read_int32()
        try:
            return MessageType(raw_type)
        except ValueError as error:
            raise ProtocolError(f"unknown bridge message type: {raw_type}") from error

    def read_int32(self) -> int:
        return struct.unpack("<i", self._recv_exact(4))[0]

    def respond(self, message_type: MessageType) -> None:
        self._send_int32(message_type)

    def execute_command(self, command: str) -> None:
        payload = command.encode("utf-8")
        self._send(struct.pack("<ii", MessageType.C_EXECUTE_COMMAND, len(payload)))
        self._send(payload)

    def set_input_state(
        self,
        *,
        left: bool = False,
        right: bool = False,
        accelerate: bool = False,
        brake: bool = False,
    ) -> None:
        """Set the four digital driving inputs consumed by the bridge."""

        self._send(
            struct.pack(
                "<iBBBB",
                MessageType.C_SET_INPUT_STATE,
                left,
                right,
                accelerate,
                brake,
            )
        )

    def give_up(self) -> None:
        """Reset the current local run through TMInterface."""

        self._send_int32(MessageType.C_GIVE_UP)

    def get_simulation_state(self) -> SimStateData:
        self._send_int32(MessageType.C_GET_SIMULATION_STATE)
        state_length = self.read_int32()
        if state_length <= 0:
            raise ProtocolError(f"invalid simulation-state length: {state_length}")
        state = SimStateData(self._recv_exact(state_length))
        state.cp_data.resize(CheckpointData.cp_states_field, state.cp_data.cp_states_length)
        state.cp_data.resize(CheckpointData.cp_times_field, state.cp_data.cp_times_length)
        return state

    def race_finished(self) -> bool:
        """Return whether TMInterface considers the current race complete."""

        self._send_int32(MessageType.C_RACE_FINISHED)
        return self.read_int32() != 0

    def _send_int32(self, value: int | MessageType) -> None:
        self._send(struct.pack("<i", int(value)))

    def _send(self, payload: bytes) -> None:
        if self._socket is None:
            raise RuntimeError("client is not connected")
        self._socket.sendall(payload)

    def _recv_exact(self, byte_count: int) -> bytes:
        if self._socket is None:
            raise RuntimeError("client is not connected")
        chunks = bytearray()
        while len(chunks) < byte_count:
            chunk = self._socket.recv(byte_count - len(chunks))
            if not chunk:
                raise ProtocolError(
                    f"bridge disconnected after {len(chunks)} of {byte_count} bytes"
                )
            chunks.extend(chunk)
        return bytes(chunks)
