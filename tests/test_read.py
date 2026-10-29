import sys
import time
from types import ModuleType, SimpleNamespace

import cv2
import numpy as np
import pytest

import read
from cardReader.cards import Detection
from cardReader.hud import Hud
from read import STAGES, StageTimer, reconnect


class FakeCamera:
    """A live camera sending flat frames of grey `level`. From read `stale_from` on it is stale until reopened."""

    is_file, size = False, (640, 360)

    def __init__(self, level=0, stale_from=None):
        self.level, self.stale_from = level, stale_from
        self.reads, self.stale, self.reopened = 0, False, 0

    def read(self):
        self.reads += 1
        self.stale = self.stale or self.reads == self.stale_from
        return True, np.full((360, 640, 3), self.level, np.uint8)

    def reopen(self):
        self.reopened += 1
        self.stale = False
        return True

    def release(self):
        pass


def run_reader(monkeypatch, tmp_path, cam, keys, detect, *args):
    """Run read.main() on a fake camera and detector with no real window, pressing one of `keys` per frame.
    args: extra command-line arguments. Saved frames go to tmp_path / "frames"."""
    monkeypatch.setattr(read, "Camera", lambda *a: cam)
    detect.device_used = "mps"
    detector = ModuleType("cardReader.detector")
    detector.Detector = lambda *a: detect
    monkeypatch.setitem(sys.modules, "cardReader.detector", detector)
    monkeypatch.setattr(read, "live_model", lambda path: None)
    monkeypatch.setattr(read.SeatLayout, "load", staticmethod(lambda *a: None))
    monkeypatch.setattr(read.settings, "FRAMES_DIR", str(tmp_path / "frames"))
    pressed = iter(keys)
    monkeypatch.setattr(read.cv2, "waitKey", lambda ms: next(pressed))
    for name in ("imshow", "namedWindow", "setMouseCallback", "destroyAllWindows"):
        monkeypatch.setattr(read.cv2, name, lambda *a: None)
    (tmp_path / "x.pt").touch()
    monkeypatch.setattr(sys, "argv", ["read.py", "--model", str(tmp_path / "x.pt"), "--players", "3", *args])
    read.main()


def test_stale_camera_shows_no_signal_feeds_nothing_to_the_tracker_and_c_reconnects(monkeypatch, capsys, tmp_path):
    cam = FakeCamera(stale_from=5)
    seen = []                                               # (camera stale, "fed" to the tracker or no_signal drawn)
    process, draw = read.Reader.process, read.Hud.draw

    def watched_process(self, *args):
        seen.append((cam.stale, "fed"))
        return process(self, *args)

    def watched_draw(self, *args, **kw):
        seen.append((cam.stale, kw["no_signal"]))
        return draw(self, *args, **kw)

    monkeypatch.setattr(read.Reader, "process", watched_process)
    monkeypatch.setattr(read.Hud, "draw", watched_draw)
    keys = [255] * 6 + [ord("c")] + [255] * 3 + [ord("q")]
    run_reader(monkeypatch, tmp_path, cam, keys, lambda frame: [Detection("5H", 0.9, 100, 100)])

    assert cam.reopened == 1
    assert (True, "fed") not in seen                         # nothing from a stale frame reached the tracker
    assert [no_signal for stale, no_signal in seen if stale] == [True, True, True]   # reads 5 to 7, until C
    assert (False, "fed") in seen and (False, False) in seen
    out = capsys.readouterr().out
    assert f"model {tmp_path / 'x.pt'} on mps" in out
    assert "no signal from source" in out and "reconnected to source" in out
    assert "average time per frame: read" in out and "detect" in out


def test_detector_gets_the_preprocessed_frame_shift_v_shows_the_raw_one_and_s_saves_raw(monkeypatch, tmp_path):
    raw, brightened = 50, round(255 * (50 / 255) ** read.settings.PREPROCESS_BRIGHTEN_GAMMA)
    detected, drawn = [], []
    draw = read.Hud.draw

    def detect(frame):
        detected.append(int(frame[0, 0, 0]))
        return []

    def watched_draw(self, frame, *args, **kw):
        drawn.append((int(frame[0, 0, 0]), kw["preprocess"], kw["showing_raw"]))
        return draw(self, frame, *args, **kw)

    monkeypatch.setattr(read.Hud, "draw", watched_draw)
    keys = [255, ord("V"), 255, ord("s"), ord("q")]
    run_reader(monkeypatch, tmp_path, FakeCamera(level=raw), keys, detect, "--preprocess", "gamma")

    assert detected == [brightened] * 5
    assert drawn == [(brightened, "gamma", False)] * 2 + [(raw, "gamma", True)] * 3
    saved = list((tmp_path / "frames").glob("frame_*.jpg"))
    assert len(saved) == 1
    assert abs(int(cv2.imread(str(saved[0]))[0, 0, 0]) - raw) <= 2


def test_editing_detects_every_third_frame_reuses_the_cards_and_drains_the_mouse(monkeypatch, tmp_path):
    cam = FakeCamera()
    detected, fed, drained = [], [], []
    process, drain = read.Reader.process, read.wait_key_draining_mouse

    def detect(frame):
        detected.append(cam.reads)
        return [Detection("5H", 0.9, 100, 100)]

    def watched_process(self, frame, dets):
        fed.append(len(dets))
        return process(self, frame, dets)

    def watched_drain(budget_ms):
        drained.append(cam.reads)
        return drain(budget_ms)

    monkeypatch.setattr(read.Reader, "process", watched_process)
    monkeypatch.setattr(read, "wait_key_draining_mouse", watched_drain)
    monkeypatch.setattr(read.settings, "SEATS_FILE", str(tmp_path / "seats.json"))
    keys = [255, ord("e")] + [255] * 5 + [ord("e"), 255, ord("q")]  # E after read 2 starts editing, after 8 ends it
    run_reader(monkeypatch, tmp_path, cam, keys, detect)

    assert detected == [1, 2, 4, 7, 9, 10]   # every frame, then one in three while editing, then every frame again
    assert fed == [1] * 10                  # frames without a detection pass reuse the last cards
    assert drained == [3, 4, 5, 6, 7, 8]
    assert (tmp_path / "seats.json").exists()


def test_draining_the_mouse_stops_at_a_key_or_when_the_time_is_up(monkeypatch):
    calls = []

    def wait_key(ms):
        calls.append(ms)
        time.sleep(0.002)
        return ord("v") if len(calls) == 3 else -1

    monkeypatch.setattr(read.cv2, "waitKey", wait_key)
    assert read.wait_key_draining_mouse(1000) == ord("v") and calls == [1, 1, 1]
    calls.clear()
    monkeypatch.setattr(read.cv2, "waitKey", lambda ms: calls.append(ms) or -1)
    start = time.perf_counter()
    assert read.wait_key_draining_mouse(20) == -1
    assert time.perf_counter() - start >= 0.02 and len(calls) > 1


def test_preprocess_flag_only_takes_a_known_mode(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["read.py", "--preprocess", "sharpen"])
    with pytest.raises(SystemExit):
        read.main()
    assert "invalid choice: 'sharpen'" in capsys.readouterr().err


@pytest.mark.parametrize(("opens", "said", "flashed"), [(True, "reconnected to source 1 at 1920x1080",
                                                          "camera reconnected"),
                                                         (False, "could not reopen source 1",
                                                          "reconnect failed: press C to try again")])
def test_reconnect_prints_and_flashes_how_it_went(capsys, opens, said, flashed):
    cam = SimpleNamespace(reopen=lambda: opens, size=(1920, 1080))
    hud = Hud()
    reconnect(cam, hud, 1)
    assert said in capsys.readouterr().out
    assert hud._message == flashed


def test_stage_timer_smooths_for_the_bar_and_averages_for_the_summary():
    timer = StageTimer()
    timer.add("detect", 20.0)                     # the first sample is taken as it is
    assert timer.smoothed["detect"] == 20.0
    timer.add("detect", 30.0)
    assert timer.smoothed["detect"] == pytest.approx(21.0)
    assert timer.averages()["detect"] == pytest.approx(25.0)
    assert timer.averages()["preprocess"] == 0.0  # a stage never timed averages 0
    assert timer.summary() == "read 0.0 ms, preprocess 0.0 ms, detect 25.0 ms, draw 0.0 ms"
    assert list(timer.averages()) == list(STAGES)


def test_lap_records_the_time_since_the_last_lap_and_returns_now():
    timer = StageTimer()
    start = time.perf_counter()
    time.sleep(0.01)
    end = timer.lap("read", start)
    assert end > start
    assert timer.smoothed["read"] == pytest.approx(1000 * (end - start))
    assert timer.smoothed["read"] >= 10.0
