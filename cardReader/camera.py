"""Get frames from a camera, capture card, or video file.

Live devices buffer frames. If detection runs slower than the camera, a plain
read() loop falls further and further behind. So for live sources a background
thread keeps draining the device and we only ever hand back the newest frame.
Video files are read one frame at a time so nothing is skipped.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2


class Camera:
    def __init__(self, source: int | str = 0, width: int | None = None, height: int | None = None):
        self.source = source
        self.is_file = isinstance(source, str) and Path(source).exists()
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source {source!r}")
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self._frame = None
        self._ok = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        if not self.is_file:
            self._thread = threading.Thread(target=self._pump, daemon=True)
            self._thread.start()
            # Give the device a moment to deliver a first frame.
            for _ in range(50):
                if self._ok:
                    break
                time.sleep(0.02)

    def _pump(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame, self._ok = frame, True

    def read(self):
        """Return (ok, frame). For live sources this is the newest available frame."""
        if self.is_file:
            return self.cap.read()
        with self._lock:
            if not self._ok:
                return False, None
            return True, self._frame.copy()

    @property
    def size(self) -> tuple[int, int]:
        return (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def release(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


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
            ok, f = cap.read()
            if ok:
                frame = f
            time.sleep(0.03)
        cap.release()
        found.append((i, frame))
    return found
