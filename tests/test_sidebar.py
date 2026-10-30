import http.client
import json
import queue
import urllib.error
import urllib.request

import numpy as np
import pytest

from cardReader.cards import Detection
from cardReader.hud import EDIT_HELP, KEY_LIST, Hud
from cardReader.reader import Reader
from cardReader.seats import SeatLayout
from cardReader.sidebar import server as server_module
from cardReader.sidebar.server import SidebarServer
from cardReader.sidebar.snapshot import ACTION_KEYS, BUTTONS, build_snapshot
from cardReader.tracker import CardTracker

W, H = 1280, 720


def place(reader, *cards):
    """Show the (label, seat index, x offset) cards for two frames, enough to confirm them."""
    dets = []
    for label, seat, dx in cards:
        cx, cy = reader.layout.seats[seat].polygon.mean(axis=0)
        dets.append(Detection(label, 0.9, float(cx) + dx, float(cy)))
    frame = np.zeros((H, W, 3), np.uint8)
    reader.process(frame, dets)
    reader.process(frame, dets)


@pytest.fixture
def reader(tmp_path):
    """Round 1 dealt a 2S, which is locked. Round 2: dealer 6C, P1 KS 6H, and a 2S again in P2 (a duplicate)."""
    tracker = CardTracker(confirm_frames=2, drop_frames=5, match_dist=20)
    reader = Reader(tracker, SeatLayout(3, W, H), Hud(), tmp_path / "seats.json", tmp_path / "frames")
    empty = np.zeros((H, W, 3), np.uint8)
    place(reader, ("2S", 1, 0))
    for _ in range(5):
        reader.process(empty, [])                  # the 2S is swept away
    reader.handle_key(ord("r"), empty)
    place(reader, ("6C", 0, 0), ("KS", 1, -30), ("6H", 1, 30), ("2S", 2, 0))
    return reader


def snapshot_of(reader, flash=None):
    return build_snapshot(reader, fps=12.345, stage_ms={"detect": 31.26, "preprocess": 0.44}, no_signal=False,
                          capture_mode=False, showing_raw=True, flash=flash)


def test_snapshot_holds_the_count_grid_hands_and_status(reader):
    snapshot = snapshot_of(reader, flash=("new round: numbering restarts at 1", 1.5))
    assert snapshot["count"] == {"headline": "+2.2", "cue": "favours player", "running": "+2", "cards_left": 48,
                                 "deck_size": 52}
    rows = {row["rank"]: row for row in snapshot["grid"]}
    assert [row["rank"] for row in snapshot["grid"]] == ["2", "3", "4", "5", "6", "7", "8", "9", "10",
                                                         "J", "Q", "K", "A"]
    assert rows["2"]["cards"][0] == {"name": "2S", "rank": "2", "suit": "S", "state": "locked", "duplicate": True}
    assert [card["state"] for card in rows["6"]["cards"]] == ["unseen", "pending", "unseen", "pending"]
    assert (rows["2"]["left"], rows["6"]["left"], rows["K"]["left"], rows["A"]["left"]) == (3, 2, 3, 4)
    assert (rows["2"]["hi_lo"], rows["8"]["hi_lo"], rows["A"]["hi_lo"]) == (1, 0, -1)
    assert snapshot["unrevealed"] == 0

    hands = {hand["seat"]: hand for hand in snapshot["hands"]}
    assert list(hands) == ["DEALER", "P1", "P2", "P3"]
    assert hands["DEALER"]["kind"] == "dealer" and hands["DEALER"]["hint"] is None
    assert [card["label"] for card in hands["P1"]["cards"]] == ["KS", "6H"]
    assert hands["P1"]["total"] == "16" and hands["P1"]["hint"] == "stand"   # hard 16 against a 6
    assert hands["P2"]["hint"] is None                                       # one card: no hint
    assert hands["P3"] == {"seat": "P3", "kind": "player", "cards": [], "total": "-", "hint": None}

    assert snapshot["players"] == 3 and snapshot["edit_help"] == []
    keys = {entry["key"]: entry for entry in snapshot["keys"]}
    assert [(entry["key"], entry["label"]) for entry in snapshot["keys"]] == KEY_LIST
    assert keys["R"]["buttons"] == [{"action": "new_round", "text": "r"}]
    assert keys["X"]["buttons"] == [{"action": "rescan", "text": "x"}]
    assert [button["text"] for button in keys["1-7"]["buttons"]] == ["1", "2", "3", "4", "5", "6", "7"]
    assert keys["E"]["buttons"] == keys["S"]["buttons"] == keys["V"]["buttons"] == keys["L"]["buttons"] == []
    assert (snapshot["preprocess"], snapshot["showing_raw"], snapshot["no_signal"]) == ("off", True, False)
    assert snapshot["flash"] == {"text": "new round: numbering restarts at 1", "seconds": 1.5}
    assert snapshot["timing"] == {"fps": 12.3, "detect_ms": 31.3, "preprocess_ms": 0.4}
    assert json.loads(json.dumps(snapshot)) == snapshot


def test_snapshot_shows_the_edit_help_while_editing(reader):
    reader.editor.active = True
    assert snapshot_of(reader)["edit_help"] == EDIT_HELP
    assert snapshot_of(reader)["flash"] is None


@pytest.fixture
def server():
    server = SidebarServer("127.0.0.1", 0)
    yield server
    server.close()


def test_server_serves_the_page_and_streams_the_last_snapshot(server):
    page = urllib.request.urlopen(server.url, timeout=5).read().decode()
    assert "new EventSource(\"/events\")" in page
    with pytest.raises(urllib.error.HTTPError, match="404"):
        urllib.request.urlopen(server.url + "missing", timeout=5)

    snapshot = {"count": {"headline": "+1.0"}, "timing": {"fps": 12.0}}
    server.publish(snapshot)
    port = int(server.url.rsplit(":", 1)[1].strip("/"))
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    connection.request("GET", "/events")
    response = connection.getresponse()
    assert response.getheader("Content-Type") == "text/event-stream"
    line = response.readline().decode()
    assert line.startswith("data: ") and json.loads(line[len("data: "):]) == snapshot
    connection.close()


def post(server, body: bytes, content_type="application/json") -> int:
    """POST body to /action; returns the status code."""
    request = urllib.request.Request(server.url + "action", data=body, headers={"Content-Type": content_type})
    try:
        return urllib.request.urlopen(request, timeout=5).status
    except urllib.error.HTTPError as error:
        return error.code


def test_posted_actions_are_drained_in_order_and_bad_posts_are_refused(server):
    assert server.drain_actions() == []
    assert post(server, b'{"action": "new_round"}') == 204
    assert post(server, b'{"action": "quit"}') == 204
    assert server.drain_actions() == ["new_round", "quit"]
    assert server.drain_actions() == []

    assert post(server, b'{"action": "quit"}', content_type="text/plain") == 415   # what another site could send
    assert post(server, b"not json") == 400
    assert post(server, b'{"action": 7}') == 400
    assert post(server, b'{"button": "quit"}') == 400
    assert post(server, b'{"action": "' + b"x" * 2000 + b'"}') == 400
    assert server.drain_actions() == []


def test_every_button_presses_a_key_the_reader_handles():
    assert ACTION_KEYS["players_3"] == ord("3") and ACTION_KEYS["new_shoe"] == ord("R")
    assert set(ACTION_KEYS) == {f"players_{n}" for n in range(1, 8)} | {
        "grey_boxes", "new_round", "rescan", "new_shoe", "reconnect", "preprocess", "raw_or_enhanced",
        "turn_left", "turn_right", "quit"}
    assert set(BUTTONS) <= {key for key, _ in KEY_LIST}


def received(client: queue.Queue) -> list[dict]:
    messages = []
    while not client.empty():
        messages.append(json.loads(client.get_nowait()))
    return messages


def test_publish_sends_changes_at_once_and_unchanged_state_only_on_the_heartbeat(server, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(server_module.time, "monotonic", lambda: clock[0])
    client = server._subscribe()
    for seconds, count, fps in [(0.0, 1, 12.0), (0.1, 1, 12.5), (0.1, 2, 12.5), (0.1, 2, 12.6), (0.5, 2, 13.0)]:
        clock[0] += seconds
        server.publish({"count": count, "timing": {"fps": fps}})
    assert received(client) == [{"count": 1, "timing": {"fps": 12.0}},   # first
                                {"count": 2, "timing": {"fps": 12.5}},   # changed
                                {"count": 2, "timing": {"fps": 13.0}}]   # heartbeat; fps alone waits for it


def test_a_flash_is_sent_once(server):
    client = server._subscribe()
    flash = {"text": "new round", "seconds": 1.5}
    for snapshot in [{"flash": None}, {"flash": flash}, {"flash": None}, {"flash": None}]:
        server.publish(snapshot)
    assert [message["flash"] for message in received(client)] == [None, flash, None]


def test_a_page_that_falls_behind_skips_events_instead_of_growing_its_queue(server, monkeypatch):
    monkeypatch.setattr(server_module, "QUEUE_SIZE", 2)
    client = server._subscribe()
    for count in range(5):
        server.publish({"count": count})
    assert received(client) == [{"count": 0}, {"count": 1}]
