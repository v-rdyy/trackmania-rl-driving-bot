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
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004


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


def press_trackmania_key(
    virtual_key: int,
    target: WindowTarget | None = None,
) -> WindowTarget:
    """Focus TrackMania and synthesize one keyboard press."""
    if target is None:
        target = find_trackmania_window()
    focus_window(target)
    wait_for_foreground(target)
    user32 = ctypes.windll.user32
    scan_code = user32.MapVirtualKeyW(virtual_key, 0)
    user32.keybd_event(virtual_key, scan_code, 0, 0)
    time.sleep(0.05)
    user32.keybd_event(virtual_key, scan_code, 0x0002, 0)
    return target


def click_trackmania_client(
    x_fraction: float,
    y_fraction: float,
    target: WindowTarget | None = None,
) -> WindowTarget:
    """Click a normalized point inside TrackMania's client area."""
    if not 0.0 <= x_fraction <= 1.0 or not 0.0 <= y_fraction <= 1.0:
        raise ValueError("client click fractions must lie in [0, 1]")
    if target is None:
        target = find_trackmania_window()
    focus_window(target)
    wait_for_foreground(target)
    x = round(target.left + target.width * x_fraction)
    y = round(target.top + target.height * y_fraction)
    user32 = ctypes.windll.user32
    user32.SetCursorPos(x, y)
    user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.05)
    user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    return target


def confirm_a01_solo(target: WindowTarget | None = None) -> WindowTarget:
    """Clear A01's loading screen and explicitly choose the `None` opponent."""
    if target is None:
        target = find_trackmania_window()
    click_trackmania_client(0.50, 0.62, target)
    press_trackmania_key(VK_RETURN, target)
    time.sleep(0.75)
    click_trackmania_client(0.50, 0.62, target)
    press_trackmania_key(VK_RETURN, target)
    return target


def ensure_trackmania_running(
    *,
    port: int = 8478,
    executable: Path = DEFAULT_TMLOADER,
    game: str = DEFAULT_GAME,
    profile: str = DEFAULT_PROFILE,
    timeout_seconds: float = 60.0,
    confirm_existing: bool = False,
) -> tuple[WindowTarget, bool]:
    """Start ModLoader's profile when needed and clear its startup confirmation.

    Returns the visible game window and whether this call launched it. An existing
    game is left untouched so Enter is never injected into an active race.
    """
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    try:
        target = find_trackmania_window()
        if confirm_existing:
            confirm_a01_solo(target)
            time.sleep(0.75)
        return target, False
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
    confirm_a01_solo(target)
    time.sleep(0.75)
    return target, True
