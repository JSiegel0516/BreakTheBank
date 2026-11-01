import threading
import time

import cv2
import numpy as np
import pytest

from cardReader import camera
from cardReader.camera import Camera


class FakeCapture:
    """Stands in for cv2.VideoCapture: sends frames while `live`, nothing once unplugged."""

    def __init__(self, opens=True, live=True):
        self.opens, self.live, self.sizes = opens, live, {}

    def isOpened(self):  # noqa: N802 (OpenCV's name)
        return self.opens

    def set(self, prop, value):
        self.sizes[prop] = value

    def get(self, prop):
        return self.sizes.get(prop, 0)

    def read(self):
        time.sleep(0.002)
        if not self.live:
            return False, None
        return True, np.zeros((4, 6, 3), np.uint8)

    def release(self):
        self.live = False


class StuckCapture(FakeCapture):
    """Sends one frame, then blocks in read() until `unblock` is set and returns a white frame."""

    def __init__(self):
        super().__init__()
        self.reads, self.unblock = 0, threading.Event()

    def read(self):
        self.reads += 1
        if self.reads > 1:
            self.unblock.wait()
            return True, np.full((4, 6, 3), 255, np.uint8)
        return super().read()


@pytest.fixture
def captures(monkeypatch):
    """Every capture opened, in order. Set `next_opens` to make the next open fail, `next_live`
    to make it open but send nothing, or `next_capture` to hand out that capture once."""
    opened = []

    def open_capture(source):
        capture = open_capture.next_capture or FakeCapture(open_capture.next_opens, open_capture.next_live)
        open_capture.next_capture = None
        opened.append(capture)
        return capture

    open_capture.next_opens, open_capture.next_live, open_capture.next_capture = True, True, None
    monkeypatch.setattr(camera.cv2, "VideoCapture", open_capture)
    return opened, open_capture


def wait_until(condition, seconds=1.0):
    end = time.monotonic() + seconds
    while not condition() and time.monotonic() < end:
        time.sleep(0.01)
    return condition()


def test_live_camera_goes_stale_when_frames_stop_and_reopen_brings_them_back(captures):
    opened, _ = captures
    with Camera(0, 1920, 1080) as cam:
        cam.stale_after = 0.1
        assert cam.read()[0] and not cam.stale
        opened[0].live = False                                    # cable pulled
        assert wait_until(lambda: cam.stale)
        assert cam.read()[0]                                      # the last frame is still there to show
        assert cam.reopen() is True
        assert len(opened) == 2 and not cam.stale
        assert opened[1].sizes == {cv2.CAP_PROP_FRAME_WIDTH: 1920, cv2.CAP_PROP_FRAME_HEIGHT: 1080}
        time.sleep(0.2)
        assert not cam.stale                                      # fresh frames keep coming


def test_failed_reopen_returns_false_and_can_be_tried_again(captures):
    opened, open_capture = captures
    with Camera(0) as cam:
        cam.stale_after = 0.1
        opened[0].live = False
        open_capture.next_opens = False
        assert cam.reopen() is False
        assert wait_until(lambda: cam.stale)
        assert cam.read()[0]                                      # still usable: the last frame
        open_capture.next_opens = True
        assert cam.reopen() is True
        assert wait_until(lambda: not cam.stale)


def test_reopen_that_opens_but_sends_nothing_stays_stale(captures):
    opened, open_capture = captures
    with Camera(0) as cam:
        cam.stale_after = 0.1
        opened[0].live = False
        assert wait_until(lambda: cam.stale)
        open_capture.next_live = False                            # opens, but no frame ever comes
        reopened_at = time.monotonic()
        assert cam.reopen() is True
        assert cam.age > time.monotonic() - reopened_at           # the last fresh frame is still the old one
        assert cam.stale


def test_a_stopped_pump_cannot_publish_a_frame_that_arrives_late(captures):
    opened, open_capture = captures
    open_capture.next_capture = StuckCapture()
    with Camera(0) as cam:
        cam.stale_after = 0.1
        stuck_pump = cam._thread
        assert wait_until(lambda: opened[0].reads > 1)            # the pump is now blocked in read()
        open_capture.next_live = False
        assert cam.reopen() is True                               # gives up waiting for the stuck pump
        assert stuck_pump.is_alive()
        fresh_before = cam.age
        opened[0].unblock.set()                                   # the old read finally returns a white frame
        stuck_pump.join(timeout=1)
        assert not stuck_pump.is_alive()
        assert cam.read()[1].max() == 0                           # the old pump did not overwrite the frame ...
        assert cam.age >= fresh_before and cam.stale              # ... nor make it look fresh


def test_video_file_is_never_stale(captures, tmp_path):
    video = tmp_path / "round1.mp4"
    video.touch()
    with Camera(str(video)) as cam:
        cam.stale_after = 0.0
        time.sleep(0.05)
        assert cam.is_file and cam.age == 0.0 and not cam.stale


@pytest.mark.parametrize("quarter_turns", [0, 1, 2, 3, -1, 4])
def test_rotate_turns_the_frame_clockwise_in_quarter_turns(quarter_turns):
    frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    expected = np.rot90(frame, -quarter_turns)               # np.rot90 turns anticlockwise for positive k
    assert (camera.rotate(frame, quarter_turns) == expected).all()
    assert camera.rotate(frame, 0) is frame
