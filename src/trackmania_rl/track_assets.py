"""Install checksum-pinned built-in tracks where TMInterface can load them."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def ensure_pinned_track_copy(
    source: Path,
    destination: Path,
    expected_sha256: str,
) -> bool:
    """Copy a built-in challenge into the user's indexed track directory.

    Returns ``True`` only when a copy was created. Existing destinations are
    accepted only when they match the pinned source bytes.
    """
    source = source.resolve()
    destination = destination.resolve()
    expected = expected_sha256.upper()
    if not source.is_file():
        raise FileNotFoundError(f"source challenge is missing: {source}")
    source_hash = sha256(source)
    if source_hash != expected:
        raise ValueError(
            f"source challenge hash mismatch: expected {expected}, got {source_hash}"
        )
    if destination.exists():
        destination_hash = sha256(destination)
        if destination_hash != expected:
            raise ValueError(
                "refusing to overwrite a different user challenge: "
                f"{destination} has {destination_hash}"
            )
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256(destination) != expected:
        raise OSError(f"copied challenge failed checksum verification: {destination}")
    return True
