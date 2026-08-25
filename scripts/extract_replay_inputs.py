"""Extract a checksum-pinned TMNF replay into TMInterface input commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from trackmania_rl.game_launch import close_trackmania, ensure_trackmania_running
from trackmania_rl.tmi_bridge import (
    MessageType,
    ProtocolError,
    TmiBridgeClient,
)

A02_REPLAY = Path(
    r"C:\Program Files (x86)\Steam\steamapps\common\TrackMania Nations Forever"
    r"\GameData\Tracks\Campaigns\Nations\White\A02-Race.Replay.gbx"
)
A02_REPLAY_SHA256 = (
    "7546E19CE9CA0D36E074406256A547A256F20B23602985085E98082EE7D178E3"
)
DEFAULT_TMI_SCRIPTS = Path.home() / "Documents" / "TMInterface" / "Scripts"
DEFAULT_OUTPUT = (
    WORKSPACE_ROOT / "artifacts" / "replays" / "a02_reference" / "nadeo_author.txt"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-replay", type=Path, default=A02_REPLAY)
    parser.add_argument(
        "--expected-source-sha256",
        default=A02_REPLAY_SHA256,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tmi-scripts-dir", type=Path, default=DEFAULT_TMI_SCRIPTS)
    parser.add_argument("--port", type=int, default=8478)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--reuse-game",
        action="store_true",
        help="reuse a verified stable game/menu instead of launching a fresh process",
    )
    return parser.parse_args()


def wait_for_file(path: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file() and path.stat().st_size > 0:
            return
        time.sleep(0.05)
    raise ProtocolError(f"TMInterface did not create extracted inputs: {path}")


def main() -> int:
    args = parse_args()
    args.source_replay = args.source_replay.resolve()
    args.output = args.output.resolve()
    args.tmi_scripts_dir = args.tmi_scripts_dir.resolve()
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    if not args.source_replay.is_file():
        raise SystemExit(f"source replay does not exist: {args.source_replay}")
    source_hash = sha256(args.source_replay)
    if source_hash != args.expected_source_sha256.upper():
        raise SystemExit(
            f"source replay hash mismatch: expected "
            f"{args.expected_source_sha256.upper()}, got {source_hash}"
        )
    if not args.tmi_scripts_dir.is_dir():
        raise SystemExit(f"TMInterface Scripts directory missing: {args.tmi_scripts_dir}")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite extracted inputs: {args.output}")

    replay_name = "a02_nadeo_author.Replay.Gbx"
    input_name = "a02_nadeo_author.txt"
    external_replay = args.tmi_scripts_dir / replay_name
    external_inputs = args.tmi_scripts_dir / input_name
    if external_replay.exists() and sha256(external_replay) != source_hash:
        raise SystemExit(f"existing TMInterface replay has wrong hash: {external_replay}")
    if external_inputs.exists():
        raise SystemExit(f"refusing to overwrite TMInterface inputs: {external_inputs}")
    if not external_replay.exists():
        shutil.copy2(args.source_replay, external_replay)

    if not args.reuse_game:
        close_trackmania()
    _, launched = ensure_trackmania_running(
        port=args.port,
        confirm_existing=not args.reuse_game,
    )
    print(f"TrackMania ready (launched={launched})", flush=True)
    with TmiBridgeClient(port=args.port, timeout_seconds=args.timeout) as client:
        message_type = client.read_message_type()
        if message_type is not MessageType.SC_ON_CONNECT_SYNC:
            raise ProtocolError(
                f"expected menu connect callback, received {message_type.name}"
            )
        extraction_command = (
            f'dump_inputs "{args.source_replay}" {external_inputs.name}'
        )
        client.execute_command(extraction_command)
        client.respond(message_type)
        wait_for_file(external_inputs, args.timeout)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(external_inputs, args.output)
    manifest = {
        "track": "A02-Race",
        "source": str(args.source_replay),
        "source_sha256": source_hash,
        "source_bytes": args.source_replay.stat().st_size,
        "extraction_command": extraction_command,
        "tminterface_minimum_version": "2.2.0",
        "output": str(args.output.relative_to(WORKSPACE_ROOT)),
        "output_sha256": sha256(args.output),
        "output_bytes": args.output.stat().st_size,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"extracted {manifest['output_bytes']} bytes of A02 author inputs to "
        f"{args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
