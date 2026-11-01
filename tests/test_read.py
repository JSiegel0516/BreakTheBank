import gc
import json
import sys
import time
import urllib.request
import weakref
from types import ModuleType, SimpleNamespace

import cv2
import numpy as np
import pytest

import read
from cardReader.cards import Detection
from cardReader.hud import SIDEBAR, Hud
from cardReader.sidebar.server import SidebarServer
from read import STAGES, StageTimer, fps_note, reconnect


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


def run_reader(monkeypatch, tmp_path, cam, keys, detect, *args, make_detector=None):
    """Run read.main() on a fake camera and detector with no real window, pressing one of `keys` per frame.
    args: extra command-line arguments. Saved frames go to tmp_path / "frames". Every model load
    returns detect, unless make_detector is given: then each load calls make_detector(weights)."""
    monkeypatch.setattr(read, "Camera", lambda *a: cam)
    detect.device_used = "mps"
    detector = ModuleType("cardReader.detector")
    detector.Detector = (lambda weights, *a: make_detector(weights)) if make_detector else (lambda *a: detect)
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


def test_quit_summary_prints_the_measured_fps_when_far_from_the_setting(monkeypatch, capsys, tmp_path):
    run_reader(monkeypatch, tmp_path, FakeCamera(), [255, 255, ord("q")], lambda frame: [])
    out = capsys.readouterr().out                           # a fake camera runs far faster than 12.4 fps
    assert "fps, but FPS in settings.py is 12.4" in out


def test_fps_note_is_empty_within_a_fifth_of_the_setting():
    assert fps_note(12.4, 12.4) == ""
    assert fps_note(14.8, 12.4) == "" and fps_note(10.0, 12.4) == ""
    assert fps_note(30.0, 12.4) == ("measured 30.0 fps, but FPS in settings.py is 12.4: "
                                    "set FPS to about 30.0 so the *_SECONDS settings take the time they say")
    assert fps_note(9.0, 12.4).startswith("measured 9.0 fps")


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


def test_frames_are_turned_by_the_setting_and_the_keys_before_detection_and_s_saves_them_turned(monkeypatch,
                                                                                                  tmp_path):
    monkeypatch.setattr(read.settings, "CAMERA_ROTATION", 90)
    detected = []

    def detect(frame):
        detected.append(frame.shape[:2])
        return []

    keys = [ord("s"), ord(">"), 255, ord("q")]
    run_reader(monkeypatch, tmp_path, FakeCamera(), keys, detect)

    assert detected == [(640, 360)] * 2 + [(360, 640)] * 2  # 90 degrees from the setting, then 180 after >
    saved = list((tmp_path / "frames").glob("frame_*.jpg"))
    assert len(saved) == 1 and cv2.imread(str(saved[0])).shape[:2] == (640, 360)


def test_a_turn_while_editing_detects_on_the_next_frame_so_cards_keep_their_numbers(monkeypatch, tmp_path):
    cam = FakeCamera()
    detected, fed, events = [], [], []
    process = read.Reader.process

    def detect(frame):
        detected.append(cam.reads)
        upright = frame.shape[:2] == (360, 640)
        x, y = (100, 50) if upright else (359 - 50, 100)  # the same card, turned with the frame
        return [Detection("KH", 0.9, x, y)]

    def watched_process(self, frame, dets):
        fed.append((frame.shape[:2], [(det.cx, det.cy) for det in dets]))
        found = process(self, frame, dets)
        events.extend((cam.reads, event.kind, event.card.order) for event in found if event.card)
        return found

    monkeypatch.setattr(read.Reader, "process", watched_process)
    monkeypatch.setattr(read.settings, "SEATS_FILE", str(tmp_path / "seats.json"))
    # 7 frames to place the card, E after read 8 starts editing, > after read 10 (a detect frame) turns
    keys = [255] * 7 + [ord("e"), 255, ord(">")] + [255] * 6 + [ord("q")]
    run_reader(monkeypatch, tmp_path, cam, keys, detect)

    assert 11 in detected                                # the frame after the turn is detected, not skipped
    turned = [dets for shape, dets in fed[10:]]
    assert all(shape == (640, 360) for shape, _ in fed[10:]) and turned == [[(309, 100)]] * len(turned)
    assert [kind for _, kind, _ in events] == ["placed"]  # no hidden, returned, or second placed event
    assert events[0][2] == 1


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


class FakeServer:
    """Stands in for SidebarServer: records what read.py publishes."""

    def __init__(self, host, port):
        self.host, self.port, self.published, self.closed = host, port, [], False
        self.url = f"http://{host}:{port}/"

    def publish(self, snapshot):
        self.published.append(snapshot)

    def drain_actions(self):
        return []

    def close(self):
        self.closed = True


@pytest.mark.parametrize("web", [False, True])
def test_web_moves_the_sidebar_to_the_page_and_publishes_every_frame(monkeypatch, capsys, tmp_path, web):
    servers, widths = [], []
    draw = read.Hud.draw

    def watched_draw(self, *args, **kw):
        canvas = draw(self, *args, **kw)
        widths.append(canvas.shape[1])
        return canvas

    monkeypatch.setattr(read.Hud, "draw", watched_draw)
    monkeypatch.setattr(read, "SidebarServer", lambda *a: servers.append(FakeServer(*a)) or servers[-1])
    keys = [255, ord("r"), 255, ord("q")]
    run_reader(monkeypatch, tmp_path, FakeCamera(), keys, lambda frame: [], *(["--web"] if web else []))

    if not web:
        assert servers == [] and widths == [640 + round(SIDEBAR * 640)] * 4
        return
    server = servers[0]
    assert (server.host, server.port, server.closed) == ("127.0.0.1", read.settings.WEB_PORT, True)
    assert f"dashboard at http://127.0.0.1:{read.settings.WEB_PORT}/" in capsys.readouterr().out
    assert widths == [640] * 4
    assert len(server.published) == 4
    flashes = [snapshot["flash"] for snapshot in server.published]
    assert flashes == [None, None, {"text": "new round: numbering restarts at 1", "seconds": 1.5}, None]


class RecordingServer(SidebarServer):
    """A real dashboard server that also keeps every snapshot read.py publishes."""

    def __init__(self, host, port):
        super().__init__(host, port)
        self.published = []

    def publish(self, snapshot):
        self.published.append(snapshot)
        super().publish(snapshot)


class PostingCamera(FakeCamera):
    """A live camera that presses page buttons: each action in posts[n] is POSTed to the dashboard
    just before read n, so several can wait for the reader at once. An action is a bare name or
    the whole body as a dict."""

    def __init__(self, servers, posts):
        super().__init__()
        self.servers, self.posts = servers, posts

    def read(self):
        for action in self.posts.get(self.reads + 1, []):
            body = json.dumps(action if isinstance(action, dict) else {"action": action}).encode()
            request = urllib.request.Request(self.servers[0].url + "action", data=body,
                                             headers={"Content-Type": "application/json"})
            assert urllib.request.urlopen(request, timeout=5).status == 204
        return super().read()


def run_with_page(monkeypatch, tmp_path, posts, keys, *args, detect=lambda frame: [], make_detector=None):
    """Run read.main() with a real dashboard on a free port, pressing page buttons per `posts`.
    make_detector: as for run_reader.
    Returns (the server, the key codes Reader.handle_key received other than 255, the camera)."""
    servers, handled = [], []
    handle_key = read.Reader.handle_key

    def watched_handle_key(self, key, frame):
        if key != 255:
            handled.append(key)
        return handle_key(self, key, frame)

    monkeypatch.setattr(read.Reader, "handle_key", watched_handle_key)
    def open_server(host, port):  # a free port in place of WEB_PORT, so the test never clashes with a running reader
        servers.append(RecordingServer(host, 0))
        return servers[-1]

    monkeypatch.setattr(read, "SidebarServer", open_server)
    cam = PostingCamera(servers, posts)
    run_reader(monkeypatch, tmp_path, cam, keys, detect, "--web", *args, make_detector=make_detector)
    return servers[0], handled, cam


def test_page_buttons_reach_the_key_handler_and_flash_like_their_keys(monkeypatch, capsys, tmp_path):
    posts = {2: ["new_round"], 3: ["players_5"], 4: ["reconnect"], 5: ["dance"], 6: ["quit"]}
    server, handled, cam = run_with_page(monkeypatch, tmp_path, posts, [255] * 10)

    assert handled == [ord("r"), ord("5"), ord("q")]       # reconnect is read.py's, like the C key
    assert cam.reopened == 1
    flashes = [snapshot["flash"]["text"] for snapshot in server.published if snapshot["flash"]]
    assert flashes == ["new round: numbering restarts at 1", "5 players: template layout", "camera reconnected"]
    assert server.published[-1]["players"] == 5
    assert "ignored unknown page action 'dance'" in capsys.readouterr().out


def test_a_page_rename_reaches_the_reader_and_the_next_snapshot(monkeypatch, tmp_path):
    seats_file = tmp_path / "seats.json"
    monkeypatch.setattr(read.settings, "SEATS_FILE", str(seats_file))
    posts = {2: [{"action": "rename_seat", "seat": "P1", "name": "Ann"}],
             3: [{"action": "rename_seat", "seat": "P9", "name": "Bo"}], 4: ["quit"]}
    server, handled, _ = run_with_page(monkeypatch, tmp_path, posts, [255] * 10)

    assert handled == [ord("q")]                                     # a rename presses no key
    names = [{hand["seat"]: hand["name"] for hand in snapshot["hands"]} for snapshot in server.published]
    assert names[1]["P1"] == "P1" and names[2]["P1"] == "Ann"       # shown on the frame after the post
    flashes = [snapshot["flash"]["text"] for snapshot in server.published if snapshot["flash"]]
    assert flashes == ["P1 is now Ann", "no seat P9 at a 3-player table"]
    assert json.loads(seats_file.read_text())["names"] == {"P1": "Ann"}


def test_no_display_with_web_runs_from_the_page_alone_and_quits_from_it(monkeypatch, tmp_path):
    drawn = []
    monkeypatch.setattr(read.Hud, "draw", lambda *a, **kw: drawn.append(1))
    server, handled, cam = run_with_page(monkeypatch, tmp_path, {3: ["new_shoe"], 4: ["quit"]}, [], "--no-display")

    assert handled == [ord("R"), ord("q")]                  # no window: cv2.waitKey is never called
    assert drawn == [] and cam.reads == 4
    assert [snapshot["flash"] for snapshot in server.published][-1] == {"text": "new shoe: count reset to 0",
                                                                         "seconds": 1.5}


def test_buttons_pressed_together_each_flash_on_the_page(monkeypatch, tmp_path):
    posts = {2: ["new_round", "preprocess", "grey_boxes"], 6: ["quit"]}
    server, handled, _ = run_with_page(monkeypatch, tmp_path, posts, [255] * 10, "--preprocess", "gamma")

    assert handled == [ord("r"), ord("p"), ord("g"), ord("q")]
    flashes = [snapshot["flash"]["text"] for snapshot in server.published if snapshot["flash"]]
    assert flashes == ["new round: numbering restarts at 1", "preprocess: clahe", "unmatched boxes hidden"]


def test_queued_turns_and_a_turn_then_players_each_get_the_frame_the_turn_left(monkeypatch, tmp_path):
    readers, turns, player_changes = [], [], []
    turn, set_players = read.Reader.turn, read.Reader.set_players

    def watched_turn(self, quarter_turns, width, height):
        readers.append(self)
        turns.append((width, height))
        return turn(self, quarter_turns, width, height)

    def watched_set_players(self, n, width, height):
        player_changes.append((n, width, height))
        return set_players(self, n, width, height)

    monkeypatch.setattr(read.Reader, "turn", watched_turn)
    monkeypatch.setattr(read.Reader, "set_players", watched_set_players)
    cam_reads = []

    def detect(frame):  # a KH at (100, 50) on the upright 640x360 frame for 8 frames, then nothing
        cam_reads.append(1)
        return [Detection("KH", 0.9, 100, 50)] if len(cam_reads) <= 8 else []

    posts = {9: ["turn_right", "turn_right"], 12: ["turn_right", "players_4"], 14: ["quit"]}
    run_with_page(monkeypatch, tmp_path, posts, [255] * 20, detect=detect)

    assert turns == [(640, 360), (360, 640), (640, 360)]  # each turn sees the frame the one before left
    assert player_changes == [(4, 360, 640)]                # three quarter turns leave the frame 360 wide
    reader = readers[0]
    assert (reader.layout.width, reader.layout.height, reader.layout.players) == (360, 640, 4)
    (card,) = reader.tracker.round
    assert (card.label, card.x, card.y) == ("KH", 50, 539)  # (100, 50) turned 180 is (539, 309), then 90 more


def test_a_taken_port_stops_the_reader_with_a_message():
    first = SidebarServer("127.0.0.1", 0)
    try:
        port = int(first.url.rsplit(":", 1)[1].strip("/"))
        with pytest.raises(SystemExit, match="set WEB_PORT in settings.py to a free port"):
            read.start_web("127.0.0.1", port)
    finally:
        first.close()


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


def make_models(root, *runs):
    """A models folder under root with weights for each run; "cards" is promoted from model_1."""
    for run in runs:
        (root / "models" / run / "weights").mkdir(parents=True)
        (root / "models" / run / "weights" / "best.pt").write_bytes(b"weights")
    if "cards" in runs:
        (root / "models" / "cards" / "LIVE.txt").write_text("model_1\n")
    return root / "models"


def test_switch_model_loads_a_run_and_refuses_paths_outside_models_or_without_weights(monkeypatch, capsys,
                                                                                       tmp_path):
    monkeypatch.setattr(read.settings, "PROJECT_ROOT", tmp_path)
    models_dir = make_models(tmp_path, "cards", "model_1")
    (models_dir / "model_2").mkdir()                                  # a run that never finished
    (tmp_path / "elsewhere.pt").write_bytes(b"weights")
    loaded = []

    def load(path):
        loaded.append(path)
        if "broken" in path:
            raise RuntimeError("not a YOLO file")
        return SimpleNamespace(device_used="cpu")

    models = read.LiveDetector(load, models_dir)
    hud = Hud()
    assert models.detector is None
    assert models.snapshot() == {"current": None, "choices": [
        {"path": "models/cards/weights/best.pt", "label": "cards (live: model_1)"},
        {"path": "models/model_1/weights/best.pt", "label": "model_1"}]}

    assert models.switch("models/model_1/weights/best.pt", hud) is True
    first = models.detector
    assert loaded == [str(models_dir / "model_1" / "weights" / "best.pt")]
    assert hud._message == "model: model_1" and "model model_1 on cpu" in capsys.readouterr().out
    assert models.snapshot()["current"] == "models/model_1/weights/best.pt"

    refusals = {"elsewhere.pt": "not a model in models/: elsewhere.pt",
                "models/../elsewhere.pt": "not a model in models/: models/../elsewhere.pt",
                "models/model_2/weights/best.pt": "no weights at models/model_2/weights/best.pt"}
    for path, message in refusals.items():
        assert models.switch(path, hud) is False
        assert hud._message == message
    (models_dir / "broken" / "weights").mkdir(parents=True)
    (models_dir / "broken" / "weights" / "best.pt").write_bytes(b"junk")
    assert models.switch("models/broken/weights/best.pt", hud) is False
    assert hud._message == "could not load broken: see the terminal"
    assert models.detector is first and len(loaded) == 2
    assert models.snapshot()["current"] == "models/model_1/weights/best.pt"
    assert [choice["label"] for choice in models.snapshot()["choices"]] == [
        "cards (live: model_1)", "broken", "model_1"]                 # the list is read again on each switch


def test_choosing_a_model_on_the_page_starts_detection_and_keeps_the_cards(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(read.settings, "PROJECT_ROOT", tmp_path)
    make_models(tmp_path, "model_1")
    rounds = []
    record = read.Reader.process

    def watched_process(self, frame, dets):
        events = record(self, frame, dets)
        rounds.append([(card.order, card.label) for card in self.tracker.round])
        return events

    monkeypatch.setattr(read.Reader, "process", watched_process)
    switch = {"action": "switch_model", "path": "models/model_1/weights/best.pt"}
    posts = {4: [switch], 20: [switch], 24: ["quit"]}
    server, _, _ = run_with_page(monkeypatch, tmp_path, posts, [255] * 40, "--model", str(tmp_path / "none.pt"),
                                 detect=lambda frame: [Detection("KH", 0.9, 100, 50)])

    capture = [snapshot["capture_mode"] for snapshot in server.published]
    assert capture[:4] == [True] * 4 and not any(capture[4:])     # detection starts on the frame after the post
    assert rounds[18] == [(1, "KH")]                                # confirmed before the second switch
    assert server.published[0]["model"]["current"] is None
    assert server.published[-1]["model"] == {"current": "models/model_1/weights/best.pt", "choices": [
        {"path": "models/model_1/weights/best.pt", "label": "model_1"}]}
    flashes = [snapshot["flash"]["text"] for snapshot in server.published if snapshot["flash"]]
    assert flashes == ["model: model_1", "model: model_1"]
    assert rounds[-1] == [(1, "KH")]                                # the second switch kept the card and its number
    assert "model model_1 on mps" in capsys.readouterr().out


class FakeDetector:
    """A distinct detector per load, so a test can tell the start-up model from a switched-to one."""

    device_used = "cpu"

    def __init__(self, weights):
        self.weights = weights

    def __call__(self, frame):
        return []


def test_the_start_up_model_is_freed_after_switching_away(monkeypatch, tmp_path):
    monkeypatch.setattr(read.settings, "PROJECT_ROOT", tmp_path)
    make_models(tmp_path, "model_1")
    refs, alive = [], []

    def make_detector(weights):  # keeps only a weak reference, so the reader holds the sole strong one
        detector = FakeDetector(weights)
        refs.append(weakref.ref(detector))
        return detector

    process = read.Reader.process

    def watched_process(self, frame, dets):
        gc.collect()
        alive.append([ref() is not None for ref in refs])
        return process(self, frame, dets)

    monkeypatch.setattr(read.Reader, "process", watched_process)
    posts = {3: [{"action": "switch_model", "path": "models/model_1/weights/best.pt"}], 6: ["quit"]}
    run_with_page(monkeypatch, tmp_path, posts, [255] * 10, make_detector=make_detector)

    assert len(refs) == 2                                           # the start-up model, then model_1
    assert alive[1] == [True]                                       # before the switch: the start-up model runs
    assert alive[-1] == [False, True]                               # after it: only model_1 is still held
