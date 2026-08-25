"""Extract a checksum-pinned TMNF replay into TMInterface input commands.

TMInterface 2.2.1's direct-file ``dump_inputs`` path is not used here.  The
installed client rejects even a replay it saved itself when that replay is
passed back by filename, while its Replay UI extracts the same bytes.  This
offline path uses the TMInterface author's ``pygbx`` parser and the conversion
semantics from his public ``gbxtools/generate_input_file.py`` utility.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pygbx import Gbx, GbxType

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

A02_REPLAY = Path(
    r"C:\Program Files (x86)\Steam\steamapps\common\TrackMania Nations Forever"
    r"\GameData\Tracks\Campaigns\Nations\White\A02-Race.Replay.gbx"
)
A02_REPLAY_SHA256 = (
    "7546E19CE9CA0D36E074406256A547A256F20B23602985085E98082EE7D178E3"
)
DEFAULT_OUTPUT = (
    WORKSPACE_ROOT / "artifacts" / "replays" / "a02_reference" / "nadeo_author.txt"
)


@dataclass(frozen=True)
class ReplayInputEvent:
    time: int
    event_name: str
    enabled: int
    flags: int


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
    return parser.parse_args()


def _event_time(event: ReplayInputEvent) -> int:
    """Match TMInterface's 10 ms replay-input boundary convention."""
    event_time = (int(event.time) // 10) * 10
    if event.event_name == "Respawn":
        return event_time - 10 if int(event.time) % 10 == 0 else event_time
    return event_time - 10


def _next_matching_event(
    events: list[ReplayInputEvent],
    start_index: int,
) -> ReplayInputEvent | None:
    target_name = events[start_index].event_name
    for candidate in events[start_index + 1 :]:
        if candidate.event_name == target_name:
            return candidate
    return None


def _analog_value(event: ReplayInputEvent) -> int:
    packed = np.int32((int(event.flags) << 16) | int(event.enabled))
    return int(-((packed << np.int32(8)) >> np.int32(8)))


def replay_events_to_script(
    events: list[ReplayInputEvent],
    race_time_ms: int,
) -> str:
    """Convert one finishing ghost's controls to a TMInterface input script."""
    lines: list[str] = []
    invert_axis = any(event.event_name == "_FakeDontInverseAxis" for event in events)
    interface_offset = (
        0xFFFF
        if any(
            event.event_name == "_FakeIsRaceRunning" and event.time % 10 == 5
            for event in events
        )
        else 0
    )
    normalized = [
        ReplayInputEvent(
            time=int(event.time) - interface_offset,
            event_name=event.event_name,
            enabled=int(event.enabled),
            flags=int(event.flags),
        )
        for event in events
    ]

    for index, event in enumerate(normalized):
        if event.event_name in {"AccelerateReal", "BrakeReal"}:
            if event.flags != 1:
                continue
        elif event.event_name.startswith("_Fake") or event.enabled == 0:
            continue

        next_event = _next_matching_event(normalized, index)
        end_ms = _event_time(next_event) if next_event is not None else race_time_ms
        start_ms = _event_time(event)
        if start_ms < 0:
            if end_ms < 0 and next_event is not None:
                continue
            start_ms = 0
        start_ms = (start_ms // 10) * 10
        end_ms = (end_ms // 10) * 10

        if event.event_name in {"Steer", "Gas"}:
            value = _analog_value(event)
            if invert_axis:
                value = -value
            lines.append(f"{start_ms} {event.event_name.lower()} {value}")
            continue

        key_by_event = {
            "Accelerate": "up",
            "AccelerateReal": "up",
            "SteerLeft": "left",
            "SteerRight": "right",
            "Brake": "down",
            "BrakeReal": "down",
            "Respawn": "enter",
        }
        key = key_by_event.get(event.event_name)
        if key is None or event.event_name == "Horn":
            continue
        if next_event is None and end_ms == -1:
            lines.append(f"{start_ms} press {key}")
        else:
            lines.append(f"{start_ms}-{end_ms} press {key}")

    if not lines:
        raise ValueError("selected replay ghost produced no usable input commands")
    return "\n".join(lines) + "\n"


def extract_fastest_ghost(source_replay: Path) -> tuple[str, dict[str, object]]:
    replay = Gbx(str(source_replay))
    ghosts = replay.get_classes_by_ids(
        [GbxType.CTN_GHOST, GbxType.CTN_GHOST_OLD]
    )
    finishing_ghosts = [
        ghost
        for ghost in ghosts
        if int(getattr(ghost, "race_time", 0)) > 0
        and getattr(ghost, "control_entries", None)
    ]
    if not finishing_ghosts:
        raise ValueError("source replay contains no finishing ghost with inputs")
    ghost = min(finishing_ghosts, key=lambda item: int(item.race_time))
    events = [
        ReplayInputEvent(
            time=int(event.time),
            event_name=str(event.event_name),
            enabled=int(event.enabled),
            flags=int(event.flags),
        )
        for event in ghost.control_entries
    ]
    script = replay_events_to_script(events, int(ghost.race_time))
    metadata: dict[str, object] = {
        "available_ghost_count": len(ghosts),
        "finishing_ghost_count": len(finishing_ghosts),
        "selected_ghost_index": ghosts.index(ghost),
        "selected_ghost_login": str(getattr(ghost, "login", "")),
        "selected_ghost_race_time_ms": int(ghost.race_time),
        "selected_ghost_control_entries": len(events),
    }
    return script, metadata


def main() -> int:
    args = parse_args()
    args.source_replay = args.source_replay.resolve()
    args.output = args.output.resolve()
    if not args.source_replay.is_file():
        raise SystemExit(f"source replay does not exist: {args.source_replay}")
    source_hash = sha256(args.source_replay)
    if source_hash != args.expected_source_sha256.upper():
        raise SystemExit(
            f"source replay hash mismatch: expected "
            f"{args.expected_source_sha256.upper()}, got {source_hash}"
        )
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite extracted inputs: {args.output}")
    script, ghost_metadata = extract_fastest_ghost(args.source_replay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(script, encoding="utf-8", newline="\n")
    manifest = {
        "track": "A02-Race",
        "source": str(args.source_replay),
        "source_sha256": source_hash,
        "source_bytes": args.source_replay.stat().st_size,
        "extraction_method": "pygbx-fastest-finishing-ghost",
        "extraction_provenance": (
            "TMInterface author donadigo's public gbxtools/generate_input_file.py"
        ),
        "pygbx_version": importlib.metadata.version("pygbx"),
        "python_lzo_version": importlib.metadata.version("python-lzo"),
        "output": str(args.output.relative_to(WORKSPACE_ROOT)),
        "output_sha256": sha256(args.output),
        "output_bytes": args.output.stat().st_size,
        **ghost_metadata,
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
