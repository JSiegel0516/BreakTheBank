"""Get frames from a camera, capture card, or video file.

Live devices buffer frames. If detection runs slower than the camera, a plain
read() loop falls further and further behind. So for live sources a background
thread keeps draining the device and we only ever hand back the newest frame.
Video files are read one frame at a time so nothing is skipped.

rotate() turns a frame by quarter turns, for a camera mounted at an angle
(CAMERA_ROTATION in settings.py, keys < and > in read.py).
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np

# clockwise quarter turns -> cv2.rotate code
ROTATE_CODES = {1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_COUNTERCLOCKWISE}

class Camera:
    """One video source: a device index, a video file path, or a stream URL.

    Raises RuntimeError if it cannot be opened. width and height are requested
    from a live device, which may ignore them; `size` is what it actually gives.
    A live source is `stale` once no fresh frame has come for `stale_after`
    seconds (a cable pulled, a phone locked); reopen() tries it again.
    Call release() when done, or use it as a context manager."""

    stale_after = 2.0

    def __init__(self, source: int | str = 0, width: int | None = None, height: int | None = None):
        self.source, self.width, self.height = source, width, height
        self.is_file = isinstance(source, str) and Path(source).exists()
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source {source!r}")
        self._request_size()

        self._frame = None
        self._ok = False
        self._fresh_at = time.monotonic()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        if not self.is_file:
            self._start_pump()

    def _request_size(self):
        if self.width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def _start_pump(self):
        # _fresh_at is left alone: after a reopen the source stays stale until the new capture
        # really delivers a frame. Each pump gets its own capture and stop flag, so one left over
        # from before a reopen cannot read the new capture.
        started = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._pump, args=(self.cap, self._stop), daemon=True)
        self._thread.start()
        # Give the device a moment to deliver a first frame.
        for _ in range(50):
            if self._fresh_at > started:
                break
            time.sleep(0.02)

    def _stop_pump(self):
        # Stopping under the lock means a pump still blocked in read() cannot publish once this returns.
        with self._lock:
            self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
            self._thread = None

    def _pump(self, cap, stop: threading.Event):
        while not stop.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                if stop.is_set():
                    return
                self._frame, self._ok, self._fresh_at = frame, True, time.monotonic()

    def read(self):
        """Return (ok, frame). For live sources this is the newest available frame,
        which stays the last one received while the source is stale."""
        if self.is_file:
            return self.cap.read()
        with self._lock:
            if not self._ok:
                return False, None
            return True, self._frame.copy()

    @property
    def age(self) -> float:
        """Seconds since the last fresh frame (since opening, if none came yet). Always 0 for a video file."""
        if self.is_file:
            return 0.0
        return time.monotonic() - self._fresh_at

    @property
    def stale(self) -> bool:
        """True when a live source has sent no fresh frame for `stale_after` seconds."""
        return self.age > self.stale_after

    def reopen(self) -> bool:
        """Release the source and open it again with the same requested width and height.

        Returns False if it cannot be opened. True means it opened, not that it
        sends frames: the source stays `stale` until the new capture delivers a
        fresh one. Either way the object stays usable: read() keeps returning the
        last frame until a fresh one arrives, and reopen() can be called again."""
        self._stop_pump()
        self.cap.release()
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            return False
        self._request_size()
        if not self.is_file:
            self._start_pump()
        return True

    @property
    def size(self) -> tuple[int, int]:
        return (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def release(self):
        self._stop_pump()
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


def rotate(frame: np.ndarray, quarter_turns: int) -> np.ndarray:
    """The frame turned clockwise by quarter_turns (negative: anticlockwise; 4 is a full turn).

    A clockwise quarter turn of a W x H frame moves pixel (x, y) to (H - 1 - y, x).
    No turn returns the frame itself; any other is a new array."""
    turns = quarter_turns % 4
    return frame if turns == 0 else cv2.rotate(frame, ROTATE_CODES[turns])


def parse_source(text: str) -> int | str:
    """'0' -> 0 (device index); anything else is a path or URL."""
    return int(text) if text.isdigit() else text


def snapshot_cameras(max_index: int = 5) -> list[tuple[int, object]]:
    """Open cameras 0, 1, 2 ... in turn and grab one frame from each.

    Returns [(index, frame or None)] for every index that opened. Camera
    numbers depend on what is connected (an iPhone joining or leaving can swap
    them), so this is the quickest way to see which number is which."""
    found = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue
        frame = None
        for _ in range(30):  # cameras need a moment to adjust exposure after opening
            ok, grabbed = cap.read()
            if ok:
                frame = grabbed
            time.sleep(0.03)
        cap.release()
        found.append((i, frame))
    return found
