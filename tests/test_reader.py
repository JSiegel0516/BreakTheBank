import csv
import io

import numpy as np

from cardReader.cards import Detection
from cardReader.hud import Hud
from cardReader.reader import LOG_HEADER, Reader
from cardReader.seats import SeatLayout
from cardReader.tracker import CardTracker

W, H = 640, 360


def make_reader(tmp_path, players=3, log=None):
    tracker = CardTracker(confirm_frames=2, drop_frames=5, match_dist=20)
    layout = SeatLayout(players, W, H)
    return Reader(tracker, layout, Hud(), tmp_path / "seats.json", tmp_path / "frames", log=log)


def frame():
    return np.zeros((H, W, 3), np.uint8)


def test_process_places_cards_logs_them_with_their_seat_and_counts(tmp_path):
    buf = io.StringIO()
    r = make_reader(tmp_path, log=csv.writer(buf))
    seat = r.layout.seats[1].polygon.mean(axis=0)  # inside P1
    dets = [Detection("5H", 0.9, float(seat[0]), float(seat[1]))]
    assert r.process(frame(), dets) == []
    events = r.process(frame(), dets)
    assert [(e.kind, e.card.label) for e in events] == [("placed", "5H")]
    assert r.count.running == 1
    rows = list(csv.reader(io.StringIO(buf.getvalue())))
    assert len(rows) == 1 and len(rows[0]) == len(LOG_HEADER)
    assert rows[0][2:7] == ["placed", "1", "5H", "P1", "1"]


def test_duplicate_is_logged_with_its_seat_and_leaves_the_count(tmp_path):
    buf = io.StringIO()
    r = make_reader(tmp_path, log=csv.writer(buf))
    seat = r.layout.seats[2].polygon.mean(axis=0)  # inside P2
    dets = [Detection("KD", 0.9, float(seat[0]), float(seat[1]))]
    r.process(frame(), dets)
    r.process(frame(), dets)
    for _ in range(5):
        r.process(frame(), [])                     # the KD is swept away
    r.handle_key(ord("r"), frame())
    r.process(frame(), dets)
    events = r.process(frame(), dets)              # a KD again in the next round
    assert [e.kind for e in events] == ["placed", "duplicate"]
    rows = list(csv.reader(io.StringIO(buf.getvalue())))
    assert rows[-1][2:7] == ["duplicate", "1", "KD", "P2", "-1"]
    assert r.count.running == -1


def test_removed_card_is_printed_and_logged_and_leaves_the_count(tmp_path, capsys):
    buf = io.StringIO()
    tracker = CardTracker(confirm_frames=2, drop_frames=2, match_dist=20, hidden_frames=3)
    r = Reader(tracker, SeatLayout(3, W, H), Hud(), tmp_path / "seats.json", tmp_path / "frames",
               log=csv.writer(buf))
    seat = r.layout.seats[1].polygon.mean(axis=0)  # inside P1
    dets = [Detection("KD", 0.9, float(seat[0]), float(seat[1]))]
    r.process(frame(), dets)
    r.process(frame(), dets)
    for _ in range(2 + 3):
        r.process(frame(), [])                     # hidden, then removed
    assert r.tracker.round == [] and r.count.running == -1
    assert "removed  #1   KD (P1)" in capsys.readouterr().out
    rows = list(csv.reader(io.StringIO(buf.getvalue())))
    assert rows[-1][2:7] == ["removed", "1", "KD", "P1", "-1"]


def test_face_down_card_takes_one_card_off_cards_left(tmp_path):
    r = make_reader(tmp_path)
    dets = [Detection("BACK", 0.9, 100, 100), Detection("5H", 0.9, 300, 100)]
    r.process(frame(), dets)
    r.process(frame(), dets)
    assert (r.count.running, r.count.cards_left) == (1, 50)


def test_frame_of_another_size_rescales_the_layout(tmp_path):
    r = make_reader(tmp_path)
    r.process(np.zeros((H * 2, W * 2, 3), np.uint8), [])
    assert (r.layout.width, r.layout.height) == (W * 2, H * 2)
    assert r.editor.layout is r.layout and r.table.layout is r.layout


def test_number_keys_switch_players_and_prefer_a_saved_layout(tmp_path):
    r = make_reader(tmp_path)
    assert r.handle_key(ord("5"), frame()) is True
    assert r.layout.players == 5 and r.editor.layout.players == 5
    r.layout.seats[1].polygon += (40, 0)                 # edit ...
    r.handle_key(ord("e"), frame())                      # enter edit mode
    assert r.editor.active
    r.handle_key(ord("e"), frame())                      # leave: saves
    assert not r.editor.active and (tmp_path / "seats.json").exists()
    moved = r.layout.seats[1].polygon.copy()
    r.handle_key(ord("3"), frame())
    r.handle_key(ord("5"), frame())                      # back to 5: the saved zones come back
    assert np.abs(r.layout.seats[1].polygon - moved).max() <= 1


def test_round_shoe_legend_view_save_and_quit_keys(tmp_path):
    r = make_reader(tmp_path)
    seat = r.layout.seats[1].polygon.mean(axis=0)
    dets = [Detection("KD", 0.9, float(seat[0]), float(seat[1]))]
    r.process(frame(), dets)
    r.process(frame(), dets)
    assert r.tracker.round and r.count.running == -1
    r.handle_key(ord("r"), frame())
    assert r.tracker.round == [] and r.tracker.shoe_seen == ["KD"]
    r.handle_key(ord("R"), frame())
    assert r.tracker.shoe_seen == [] and r.count.running == 0
    legend = r.hud.show_legend
    r.handle_key(ord("l"), frame())
    assert r.hud.show_legend is not legend
    assert r.hud.view == "cards_left"
    r.handle_key(ord("v"), frame())
    assert r.hud.view == "game" and r.hud._message == "Game view"
    r.handle_key(ord("v"), frame())
    assert r.hud.view == "cards_left" and r.hud._message == "Cards Left view"
    r.handle_key(ord("s"), frame())
    assert len(list((tmp_path / "frames").glob("frame_*.jpg"))) == 1
    assert r.handle_key(ord("x"), frame()) is True       # unknown keys are ignored
    assert r.handle_key(ord("q"), frame()) is False


def test_p_cycles_the_preprocess_modes_and_shift_v_toggles_the_displayed_frame(tmp_path):
    r = make_reader(tmp_path)
    assert (r.preprocess, r.show_enhanced) == ("off", True)
    modes = []
    for _ in range(4):
        r.handle_key(ord("p"), frame())
        modes.append(r.preprocess)
        assert r.hud._message == f"preprocess: {r.preprocess}"
    assert modes == ["gamma", "clahe", "auto", "off"]
    r.handle_key(ord("V"), frame())
    assert r.show_enhanced is False and r.hud._message == "showing the raw camera frame"
    r.handle_key(ord("V"), frame())
    assert r.show_enhanced is True and r.hud._message == "showing the frame the detector sees"
    assert r.hud.view == "cards_left"                    # Shift+V is not V


def test_g_hides_and_shows_the_unmatched_boxes(tmp_path):
    r = make_reader(tmp_path)
    assert r.hud.show_unmatched is True
    r.handle_key(ord("g"), frame())
    assert r.hud.show_unmatched is False and r.hud._message == "unmatched boxes hidden"
    r.handle_key(ord("g"), frame())
    assert r.hud.show_unmatched is True and r.hud._message == "unmatched boxes shown"


def test_reader_starts_in_the_mode_it_is_given(tmp_path):
    r = Reader(CardTracker(), SeatLayout(3, W, H), Hud(), tmp_path / "seats.json", tmp_path / "frames",
               preprocess="clahe")
    r.handle_key(ord("p"), frame())
    assert r.preprocess == "auto"
