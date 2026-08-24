"""Launch the configured Windows TrackMania/TMInterface profile on demand."""

from __future__ import annotations

import ctypes
import socket
import subprocess
import time
from pathlib import Path

from trackmania_rl.video_capture import (
    VideoCaptureError,
    WindowTarget,
    find_trackmania_window,
    focus_window,
    wait_for_foreground,
)


DEFAULT_TMLOADER = Path.home() / "AppData" / "Local" / "TMLoader" / "TMLoader.exe"
DEFAULT_GAME = "TmForever"
DEFAULT_PROFILE = "default"
VK_RETURN = 0x0D


def tmloader_command(
    *,
    executable: Path = DEFAULT_TMLOADER,
    game: str = DEFAULT_GAME,
    profile: str = DEFAULT_PROFILE,
) -> list[str]:
    """Return the profile command that actually launches the instrumented game."""
    return [str(executable), "run", game, profile]


def _bridge_is_listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.25)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def _press_startup_enter(target: WindowTarget) -> None:
    focus_window(target)
    wait_for_foreground(target)
    user32 = ctypes.windll.user32
    scan_code = user32.MapVirtualKeyW(VK_RETURN, 0)
    user32.keybd_event(VK_RETURN, scan_code, 0, 0)
    time.sleep(0.05)
    user32.keybd_event(VK_RETURN, scan_code, 0x0002, 0)


def ensure_trackmania_running(
    *,
    port: int = 8478,
    executable: Path = DEFAULT_TMLOADER,
    game: str = DEFAULT_GAME,
    profile: str = DEFAULT_PROFILE,
    timeout_seconds: float = 60.0,
) -> tuple[WindowTarget, bool]:
    """Start ModLoader's profile when needed and clear its startup confirmation.

    Returns the visible game window and whether this call launched it. An existing
    game is left untouched so Enter is never injected into an active race.
    """
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    try:
        return find_trackmania_window(), False
    except VideoCaptureError:
        pass

    if not executable.is_file():
        raise FileNotFoundError(f"TrackMania ModLoader is missing: {executable}")
    subprocess.Popen(tmloader_command(executable=executable, game=game, profile=profile))

    deadline = time.monotonic() + timeout_seconds
    target: WindowTarget | None = None
    while time.monotonic() < deadline:
        try:
            target = find_trackmania_window()
        except VideoCaptureError:
            time.sleep(0.25)
            continue
        if _bridge_is_listening(port):
            break
        time.sleep(0.25)
    else:
        raise TimeoutError(
            f"TrackMania profile {profile!r} did not expose its window and port {port} "
            f"within {timeout_seconds:g} seconds"
        )

    if target is None:
        raise RuntimeError("TrackMania launch completed without a visible window")
    time.sleep(0.5)
    _press_startup_enter(target)
    time.sleep(0.75)
    return target, True
