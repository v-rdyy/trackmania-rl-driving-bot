"""Windows TrackMania window capture for reproducible progress videos."""

from __future__ import annotations

import ctypes
import hashlib
import multiprocessing
import queue
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageGrab


class VideoCaptureError(RuntimeError):
    """Raised when a progress video cannot be captured or verified."""


@dataclass(frozen=True)
class WindowTarget:
    handle: int
    title: str
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def scaled_even_size(width: int, height: int, max_width: int) -> tuple[int, int]:
    if width <= 0 or height <= 0 or max_width <= 0:
        raise ValueError("capture dimensions must be positive")
    scale = min(1.0, max_width / width)
    output_width = max(2, int(round(width * scale)))
    output_width -= output_width % 2
    output_height = max(2, int(round(height * output_width / width)))
    output_height -= output_height % 2
    return output_width, output_height


def find_trackmania_window() -> WindowTarget:
    """Return the visible TMNF client area on the interactive Windows desktop."""
    if not hasattr(ctypes, "windll"):
        raise VideoCaptureError("TrackMania window capture requires Windows")

    from ctypes import wintypes

    user32 = ctypes.windll.user32
    candidates: list[tuple[int, int, str]] = []
    callback_type = ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HWND,
        wintypes.LPARAM,
    )

    def visit(handle: int, _: int) -> bool:
        if not user32.IsWindowVisible(handle):
            return True
        class_name = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(handle, class_name, len(class_name))
        if class_name.value != "TmForever":
            return True
        title_length = user32.GetWindowTextLengthW(handle)
        title = ctypes.create_unicode_buffer(title_length + 1)
        user32.GetWindowTextW(handle, title, title_length + 1)
        client = wintypes.RECT()
        if not user32.GetClientRect(handle, ctypes.byref(client)):
            return True
        area = max(0, client.right - client.left) * max(
            0,
            client.bottom - client.top,
        )
        if area:
            candidates.append((area, int(handle), title.value))
        return True

    callback = callback_type(visit)
    if not user32.EnumWindows(callback, 0):
        raise VideoCaptureError("could not enumerate desktop windows")
    if not candidates:
        raise VideoCaptureError(
            "no visible TrackMania window found; keep TMNF running before capture"
        )

    _, handle, title = max(candidates)
    client = wintypes.RECT()
    origin = wintypes.POINT(0, 0)
    if not user32.GetClientRect(handle, ctypes.byref(client)):
        raise VideoCaptureError("could not read the TrackMania client rectangle")
    if not user32.ClientToScreen(handle, ctypes.byref(origin)):
        raise VideoCaptureError("could not map the TrackMania window to the screen")
    target = WindowTarget(
        handle=handle,
        title=title,
        left=int(origin.x),
        top=int(origin.y),
        right=int(origin.x + client.right),
        bottom=int(origin.y + client.bottom),
    )
    if target.width <= 0 or target.height <= 0:
        raise VideoCaptureError(f"invalid TrackMania client area: {target.bbox}")
    return target


def focus_window(target: WindowTarget) -> None:
    user32 = ctypes.windll.user32
    user32.ShowWindow(target.handle, 9)  # SW_RESTORE
    user32.SetForegroundWindow(target.handle)


def restart_trackmania_race() -> WindowTarget:
    """Send TrackMania's Delete restart key before a checkpoint stage."""
    target = find_trackmania_window()
    focus_window(target)
    user32 = ctypes.windll.user32
    if not user32.PostMessageW(target.handle, 0x0100, 0x2E, 0):  # WM_KEYDOWN
        raise VideoCaptureError("could not send the TrackMania restart key down")
    if not user32.PostMessageW(target.handle, 0x0101, 0x2E, 0):  # WM_KEYUP
        raise VideoCaptureError("could not send the TrackMania restart key up")
    time.sleep(0.75)
    return target


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    filename = "segoeuib.ttf" if bold else "segoeui.ttf"
    path = Path("C:/Windows/Fonts") / filename
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def _draw_overlay(
    frame: Image.Image,
    *,
    label: str,
    status: dict[str, Any],
    title_font: ImageFont.ImageFont,
    detail_font: ImageFont.ImageFont,
) -> None:
    state = str(status.get("state", "running"))
    elapsed_ms = int(status.get("elapsed_ms", 0))
    speed = int(status.get("display_speed", 0))
    progress = float(status.get("progress", 0.0))
    detail = (
        f"{state}  |  t={elapsed_ms / 1000:.1f}s  |  "
        f"speed={speed}  |  progress={progress:.0f}"
    )
    draw = ImageDraw.Draw(frame)
    draw.rounded_rectangle((18, 18, 985, 102), radius=12, fill=(0, 0, 0))
    draw.text((36, 28), label, fill=(255, 255, 255), font=title_font)
    draw.text((36, 66), detail, fill=(220, 230, 240), font=detail_font)


def _capture_worker(
    output_path: str,
    label: str,
    fps: int,
    output_size: tuple[int, int],
    target: WindowTarget,
    status_queue: Any,
    stop_event: Any,
    ready_queue: Any,
    result_queue: Any,
) -> None:
    """Capture in a separate process so ImageGrab cannot starve the bridge."""
    writer = imageio_ffmpeg.write_frames(
        output_path,
        output_size,
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        fps=fps,
        quality=7,
        codec="libx264",
        macro_block_size=2,
        ffmpeg_log_level="warning",
        output_params=["-preset", "veryfast", "-movflags", "+faststart"],
    )
    frame_count = 0
    maximum_frame_standard_deviation = 0.0
    ready_sent = False
    try:
        writer.send(None)
        ready_queue.put({"ok": True})
        ready_sent = True
        interval = 1.0 / fps
        next_frame = time.perf_counter()
        title_font = _font(28, bold=True)
        detail_font = _font(21)
        status: dict[str, Any] = {"state": "preparing A01"}
        while not stop_event.is_set():
            while True:
                try:
                    status = status_queue.get_nowait()
                except queue.Empty:
                    break
            frame = ImageGrab.grab(
                bbox=target.bbox,
                include_layered_windows=True,
                all_screens=True,
            ).convert("RGB")
            if frame.size != output_size:
                frame = frame.resize(output_size, Image.Resampling.BILINEAR)
            array = np.asarray(frame, dtype=np.uint8)
            if frame_count < fps:
                maximum_frame_standard_deviation = max(
                    maximum_frame_standard_deviation,
                    float(array.std()),
                )
            _draw_overlay(
                frame,
                label=label,
                status=status,
                title_font=title_font,
                detail_font=detail_font,
            )
            writer.send(np.asarray(frame, dtype=np.uint8))
            frame_count += 1
            next_frame += interval
            delay = next_frame - time.perf_counter()
            if delay > 0:
                stop_event.wait(delay)
            else:
                next_frame = time.perf_counter()
    except BaseException:
        error = traceback.format_exc()
        if not ready_sent:
            ready_queue.put({"ok": False, "error": error})
            ready_sent = True
        result_queue.put({"ok": False, "error": error})
    else:
        result_queue.put(
            {
                "ok": True,
                "frame_count": frame_count,
                "maximum_frame_standard_deviation": (
                    maximum_frame_standard_deviation
                ),
            }
        )
    finally:
        try:
            writer.close()
        except BaseException:
            pass


def probe_video(path: Path) -> dict[str, Any]:
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    try:
        metadata = next(reader)
        frame = next(reader)
    except (StopIteration, OSError) as error:
        raise VideoCaptureError(f"encoded video has no readable frame: {path}") from error
    finally:
        reader.close()
    expected_bytes = int(metadata["size"][0]) * int(metadata["size"][1]) * 3
    if len(frame) != expected_bytes:
        raise VideoCaptureError(
            f"decoded frame has {len(frame)} bytes, expected {expected_bytes}"
        )
    return {
        "codec": metadata.get("codec"),
        "fps": metadata.get("fps"),
        "duration_seconds": metadata.get("duration"),
        "size": list(metadata["size"]),
    }


class ProgressVideoRecorder:
    """Capture one annotated evaluation episode to an H.264 MP4."""

    def __init__(
        self,
        output_path: Path,
        *,
        label: str,
        fps: int = 20,
        max_width: int = 1280,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.output_path = output_path
        self.label = label
        self.fps = fps
        self.max_width = max_width
        self.target: WindowTarget | None = None
        self.output_size: tuple[int, int] | None = None
        self.frame_count = 0
        self.maximum_frame_standard_deviation = 0.0
        self._status: dict[str, Any] = {"state": "preparing A01"}
        self._context = multiprocessing.get_context("spawn")
        self._status_queue: Any | None = None
        self._stop_event: Any | None = None
        self._ready_queue: Any | None = None
        self._result_queue: Any | None = None
        self._process: Any | None = None
        self._started_at = 0.0

    def update(self, **status: Any) -> None:
        self._status.update(status)
        if self._status_queue is None:
            return
        try:
            self._status_queue.put_nowait(dict(self._status))
        except queue.Full:
            try:
                self._status_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._status_queue.put_nowait(dict(self._status))
            except queue.Full:
                pass

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("recorder has already been started")
        self.target = find_trackmania_window()
        focus_window(self.target)
        time.sleep(0.4)
        self.output_size = scaled_even_size(
            self.target.width,
            self.target.height,
            self.max_width,
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._started_at = time.perf_counter()
        self._status_queue = self._context.Queue(maxsize=4)
        self._stop_event = self._context.Event()
        self._ready_queue = self._context.Queue(maxsize=1)
        self._result_queue = self._context.Queue(maxsize=1)
        self.update()
        self._process = self._context.Process(
            target=_capture_worker,
            args=(
                str(self.output_path),
                self.label,
                self.fps,
                self.output_size,
                self.target,
                self._status_queue,
                self._stop_event,
                self._ready_queue,
                self._result_queue,
            ),
            name="trackmania-video-capture",
            daemon=True,
        )
        self._process.start()
        try:
            ready = self._ready_queue.get(timeout=10.0)
        except queue.Empty as error:
            self._process.terminate()
            self._process.join(timeout=5.0)
            raise VideoCaptureError(
                "video encoder did not start within 10 seconds"
            ) from error
        if not ready.get("ok"):
            self._process.join(timeout=5.0)
            raise VideoCaptureError(
                f"video encoder failed during startup: {ready.get('error')}"
            )

    def stop(self) -> dict[str, Any]:
        if self._process is None or self._stop_event is None:
            raise RuntimeError("recorder has not been started")
        assert self._result_queue is not None
        self._stop_event.set()
        self._process.join(timeout=15.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5.0)
            raise VideoCaptureError("video encoder did not stop within 15 seconds")
        try:
            result = self._result_queue.get(timeout=2.0)
        except queue.Empty as error:
            raise VideoCaptureError("video process returned no capture result") from error
        if not result.get("ok"):
            raise VideoCaptureError(f"video capture failed: {result.get('error')}")
        self.frame_count = int(result["frame_count"])
        self.maximum_frame_standard_deviation = float(
            result["maximum_frame_standard_deviation"]
        )
        if self.frame_count < self.fps:
            raise VideoCaptureError(
                f"captured only {self.frame_count} frames; expected at least one second"
            )
        if self.maximum_frame_standard_deviation < 2.0:
            raise VideoCaptureError("captured frames appear blank or nearly uniform")
        encoded = probe_video(self.output_path)
        encoded.update(
            {
                "path": str(self.output_path),
                "sha256": sha256(self.output_path),
                "frames_written": self.frame_count,
                "capture_wall_seconds": time.perf_counter() - self._started_at,
                "maximum_frame_standard_deviation": (
                    self.maximum_frame_standard_deviation
                ),
                "window_title": self.target.title if self.target else None,
                "window_bbox": list(self.target.bbox) if self.target else None,
            }
        )
        return encoded
