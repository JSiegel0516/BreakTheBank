import csv
import io

import numpy as np
import pytest

from cardReader.cards import Detection
from cardReader.hud import GRID_RANKS, DealtCards, Hud
from cardReader.reader import LOG_HEADER, Reader
from cardReader.seats import SeatLayout
from cardReader.tracker import CardTracker, TrackerConfig

FRAME = 1 / TrackerConfig.nominal_fps  # one update() given no time

W, H = 640, 360


def make_reader(tmp_path, players=3, log=None):
    tracker = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME, match_dist=20)
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


@pytest.mark.parametrize("decks, cards_left", [(1, 51), (2, 103), (3, 155)])
def test_reader_counts_cards_left_from_the_trackers_decks(tmp_path, decks, cards_left):
    tracker = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME, match_dist=20, decks=decks)
    r = Reader(tracker, SeatLayout(3, W, H), Hud(), tmp_path / "seats.json", tmp_path / "frames")
    assert r.decks == decks and r.count.cards_left == 52 * decks
    seat = r.layout.seats[1].polygon.mean(axis=0)
    dets = [Detection("5H", 0.9, float(seat[0]), float(seat[1]))]
    r.process(frame(), dets)
    r.process(frame(), dets)
    assert r.count.cards_left == cards_left


def test_rename_seat_trims_saves_and_an_empty_name_restores_the_default(tmp_path):
    r = make_reader(tmp_path)
    assert r.seat_names == {}
    assert r.rename_seat("P1", "  Ann  ") is True
    assert r.seat_names == {"P1": "Ann"} and r.hud._message == "P1 is now Ann"
    assert r.rename_seat("dealer", "x" * 16) is True
    assert make_reader(tmp_path).seat_names == {"P1": "Ann", "dealer": "x" * 16}   # survives a restart
    assert r.rename_seat("P1", "   ") is True
    assert r.seat_names == {"dealer": "x" * 16} and r.hud._message == "P1 is P1 again"
    assert make_reader(tmp_path).seat_names == {"dealer": "x" * 16}


def test_rename_seat_refuses_an_unknown_seat_or_a_long_name(tmp_path):
    r = make_reader(tmp_path, players=3)
    assert r.rename_seat("P4", "Ann") is False
    assert r.hud._message == "no seat P4 at a 3-player table"
    assert r.rename_seat("OTHER", "Ann") is False
    assert r.rename_seat("P1", "x" * 17) is False
    assert r.hud._message == "names can be at most 16 characters"
    assert r.seat_names == {} and not (tmp_path / "seats.json").exists()


def test_names_are_shared_by_every_player_count_and_kept_when_zones_are_saved(tmp_path):
    r = make_reader(tmp_path, players=3)
    r.rename_seat("P2", "Bo")
    r.handle_key(ord("5"), frame())
    r.handle_key(ord("e"), frame())
    r.handle_key(ord("e"), frame())                      # saves the 5-player zones
    assert r.seat_names == {"P2": "Bo"}
    assert make_reader(tmp_path, players=5).seat_names == {"P2": "Bo"}
    assert SeatLayout.load(tmp_path / "seats.json", W, H).players == 5


def test_a_renamed_seat_is_still_logged_and_printed_by_its_id(tmp_path, capsys):
    buf = io.StringIO()
    r = make_reader(tmp_path, log=csv.writer(buf))
    r.rename_seat("P1", "Ann")
    seat = r.layout.seats[1].polygon.mean(axis=0)
    dets = [Detection("5H", 0.9, float(seat[0]), float(seat[1]))]
    r.process(frame(), dets)
    r.process(frame(), dets)
    rows = list(csv.reader(io.StringIO(buf.getvalue())))
    assert rows[0][2:6] == ["placed", "1", "5H", "P1"]
    out = capsys.readouterr().out
    assert "(P1)" in out and "Ann" not in out


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
    tracker = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=2 * FRAME, match_dist=20, hidden_seconds=3 * FRAME)
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


def test_a_held_copy_is_printed_and_logged_and_its_count_follows_when_counted(tmp_path, capsys):
    buf = io.StringIO()
    tracker = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME, match_dist=20, decks=2,
                          copy_hold_seconds=2 * FRAME)
    r = Reader(tracker, SeatLayout(3, W, H), Hud(), tmp_path / "seats.json", tmp_path / "frames",
               log=csv.writer(buf))
    x, y = map(float, r.layout.seats[1].polygon.mean(axis=0))  # inside P1
    first = [Detection("KH", 0.9, x, y), Detection("KH", 0.9, x + 30, y + 40)]  # both corners of a KH
    copy = Detection("KH", 0.9, x + 40, y)                     # a second KH fanned over it
    for _ in range(2):
        r.process(frame(), first)
    events = [e.kind for _ in range(3) for e in r.process(frame(), [*first, copy])]
    assert events == ["placed", "held", "counted"]
    out = capsys.readouterr().out
    assert "held     #2   KH (P1)" in out and "counted  #2   KH (P1) count -2" in out
    rows = list(csv.reader(io.StringIO(buf.getvalue())))
    assert [row[2:7] for row in rows[-3:]] == [["placed", "2", "KH", "P1", "-1"], ["held", "2", "KH", "P1", "-1"],
                                               ["counted", "2", "KH", "P1", "-2"]]


def test_switching_the_counting_system_recounts_the_shoe_and_the_log_follows(tmp_path, capsys):
    buf = io.StringIO()
    r = make_reader(tmp_path, log=csv.writer(buf))
    seat = r.layout.seats[1].polygon.mean(axis=0)  # inside P1
    five = [Detection("5H", 0.9, float(seat[0]), float(seat[1]))]
    r.process(frame(), five)
    r.process(frame(), five)
    assert (r.count_system, r.count.running) == ("hi_lo", 1)

    assert r.set_count_system("uston_apc")
    assert (r.count_system, r.count.system, r.count.running) == ("uston_apc", "uston_apc", 3)
    assert "--- counting with Uston APC: running count +3 ---" in capsys.readouterr().out
    assert r.hud.pop_flash() == ("counting with Uston APC", 1.5)

    ten = [Detection("5H", 0.9, float(seat[0]), float(seat[1])), Detection("10S", 0.9, float(seat[0]) + 60, 100)]
    r.process(frame(), ten)
    r.process(frame(), ten)
    rows = list(csv.reader(io.StringIO(buf.getvalue())))
    assert rows[-1][2:7] == ["placed", "2", "10S", "", "0"]       # +3 for the 5, -3 for the 10
    assert "count +0" in capsys.readouterr().out


def test_cycling_the_decks_recounts_the_cards_seen_and_keeps_the_shoe(tmp_path, capsys):
    r = make_reader(tmp_path)
    seat = r.layout.seats[1].polygon.mean(axis=0)
    five = [Detection("5H", 0.9, float(seat[0]), float(seat[1]))]
    r.process(frame(), five)
    r.process(frame(), five)
    shown = []
    for _ in range(4):
        decks = r.cycle_decks()
        shown.append((decks, r.decks, r.tracker.cfg.decks, r.count.cards_left, r.count.decks, r.hud.pop_flash()))
    assert shown == [(2, 2, 2, 103, 2, ("2 decks", 1.5)), (3, 3, 3, 155, 3, ("3 decks", 1.5)),
                     (1, 1, 1, 51, 1, ("1 deck", 1.5)), (2, 2, 2, 103, 2, ("2 decks", 1.5))]
    assert r.tracker.shoe_seen == ["5H"] and r.count.running == 1   # recounted, not cleared
    assert "--- 2 decks: running count +1, 103 cards left ---" in capsys.readouterr().out


def test_fewer_decks_mid_shoe_count_extra_copies_once_and_more_decks_bring_them_back(tmp_path):
    tracker = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME, match_dist=20, decks=3)
    r = Reader(tracker, SeatLayout(3, W, H), Hud(), tmp_path / "seats.json", tmp_path / "frames")
    p1, p2 = r.layout.seats[1].polygon.mean(axis=0), r.layout.seats[2].polygon.mean(axis=0)
    two_kings = [Detection("KH", 0.9, float(p1[0]), float(p1[1])), Detection("KH", 0.9, float(p2[0]), float(p2[1]))]
    r.process(frame(), two_kings)
    r.process(frame(), two_kings)
    assert r.tracker.shoe_seen == ["KH", "KH"] and (r.count.cards_left, r.count.running) == (154, -2)

    def agree():
        """The count, the grid's copies left, and the odds' unseen cards tell the same story."""
        dealt = DealtCards.from_tracker(r.tracker)
        return r.count.cards_left == len(dealt.unseen()) == dealt.left(GRID_RANKS)

    assert r.cycle_decks() == 1
    assert (r.count.cards_left, r.count.running) == (51, -1) and agree()
    r.process(frame(), two_kings)                                   # a later frame refreshes the count
    assert (r.count.cards_left, r.count.running) == (51, -1) and agree()
    assert r.tracker.shoe_seen == ["KH", "KH"]                      # the shoe's history is kept
    assert r.cycle_decks() == 2
    assert (r.count.cards_left, r.count.running) == (102, -2) and agree()
    assert r.cycle_decks() == 3
    assert (r.count.cards_left, r.count.running) == (154, -2) and agree()


def test_an_unknown_counting_system_changes_nothing(tmp_path):
    r = make_reader(tmp_path)
    assert not r.set_count_system("ko")
    assert (r.count_system, r.count.system) == ("hi_lo", "hi_lo")
    assert r.hud.pop_flash() == ("no counting system 'ko'", 2.5)


def test_the_reader_starts_in_the_counting_system_it_is_given(tmp_path):
    tracker = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME, match_dist=20)
    r = Reader(tracker, SeatLayout(3, W, H), Hud(), tmp_path / "seats.json", tmp_path / "frames",
               count_system="hi_opt_ii")
    assert (r.count_system, r.count.system) == ("hi_opt_ii", "hi_opt_ii")


def test_face_down_card_takes_one_card_off_cards_left(tmp_path):
    r = make_reader(tmp_path)
    dets = [Detection("BACK", 0.9, 100, 100), Detection("5H", 0.9, 300, 100)]
    r.process(frame(), dets)
    r.process(frame(), dets)
    assert (r.count.running, r.count.cards_left) == (1, 50)   # the true count's divisor: cards still in the shoe
    assert r.cards_left_text() == "51 cards left, 1 face down"  # unseen copies, as the grid and the page show


def test_the_terminal_gives_cards_left_as_the_grid_counts_them(tmp_path, capsys):
    r = make_reader(tmp_path)
    assert r.cards_left_text() == "52 cards left"
    dets = [Detection("BACK", 0.9, 100, 100), Detection("5H", 0.9, 300, 100)]
    r.process(frame(), dets)
    r.process(frame(), dets)
    r.cycle_decks()
    assert "--- 2 decks: running count +1, 103 cards left, 1 face down ---" in capsys.readouterr().out


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
    assert r.handle_key(ord("z"), frame()) is True       # unknown keys are ignored
    assert r.handle_key(ord("q"), frame()) is False


def test_x_rescans_the_round_recounts_and_logs_a_rescan_row(tmp_path, capsys):
    buf = io.StringIO()
    r = make_reader(tmp_path, log=csv.writer(buf))
    seat = r.layout.seats[1].polygon.mean(axis=0)
    dets = [Detection("5H", 0.9, float(seat[0]), float(seat[1]))]
    r.process(frame(), dets)
    r.process(frame(), dets)
    assert r.count.running == 1
    capsys.readouterr()
    assert r.handle_key(ord("x"), frame()) is True
    assert r.tracker.round == [] and r.tracker.shoe_seen == [] and r.count.running == 0
    assert "--- rescan" in capsys.readouterr().out
    assert r.hud._message.startswith("rescan")
    rows = list(csv.reader(io.StringIO(buf.getvalue())))
    assert len(rows[-1]) == len(LOG_HEADER)
    assert rows[-1][1:] == [str(r.tracker.frame), "rescan", "", "", "", "0", "", ""]


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


def test_n_toggles_the_placement_badges_with_a_message(tmp_path):
    r = make_reader(tmp_path)
    assert r.hud.show_order is False
    r.handle_key(ord("n"), frame())
    assert r.hud.show_order is True and r.hud._message == "placement numbers shown"
    r.handle_key(ord("n"), frame())
    assert r.hud.show_order is False and r.hud._message == "placement numbers hidden"


def test_keys_6_and_7_do_nothing(tmp_path):
    r = make_reader(tmp_path)
    for key in "67":
        r.handle_key(ord(key), frame())
        assert r.layout.players == 3


def test_reader_starts_in_the_mode_it_is_given(tmp_path):
    r = Reader(CardTracker(), SeatLayout(3, W, H), Hud(), tmp_path / "seats.json", tmp_path / "frames",
               preprocess="clahe")
    r.handle_key(ord("p"), frame())
    assert r.preprocess == "auto"


def test_angle_keys_turn_the_view_a_quarter_turn_and_wrap_around(tmp_path):
    r = make_reader(tmp_path)
    assert r.rotation == 0
    r.handle_key(ord(">"), frame())
    assert r.rotation == 1
    assert r.hud._message == "camera turned to 90 degrees: CAMERA_ROTATION = 90 keeps it"
    turned = np.zeros((W, H, 3), np.uint8)               # the next frame arrives turned
    for _ in range(3):
        r.handle_key(ord(">"), turned)
        turned = np.zeros(turned.shape[1::-1] + (3,), np.uint8)
    assert r.rotation == 0 and r.hud._message.startswith("camera turned to 0 degrees")
    r.handle_key(ord("<"), frame())
    assert r.rotation == 3 and r.hud._message == "camera turned to 270 degrees: CAMERA_ROTATION = 270 keeps it"


def test_a_quarter_turn_reloads_the_layout_for_the_turned_size(tmp_path):
    r = make_reader(tmp_path)
    r.handle_key(ord("4"), frame())
    r.handle_key(ord(">"), frame())
    assert (r.layout.width, r.layout.height, r.layout.players) == (H, W, 4)
    assert r.editor.layout is r.layout and r.table.layout is r.layout
    assert (r.layout.seats[0].polygon == SeatLayout(4, H, W).seats[0].polygon).all()  # the template, not stretched
    r.layout.seats[1].polygon += (5, 0)
    r.layout.save(r.seats_file)
    saved = r.layout.seats[1].polygon.copy()
    r.handle_key(ord("<"), np.zeros((W, H, 3), np.uint8))
    r.handle_key(ord(">"), frame())
    assert np.abs(r.layout.seats[1].polygon - saved).max() <= 1  # the saved layout for 4 players


def test_a_card_on_the_table_keeps_its_number_through_a_turn(tmp_path, capsys):
    r = make_reader(tmp_path)
    dets = [Detection("KH", 0.9, 100, 50)]
    r.process(frame(), dets)
    r.process(frame(), dets)
    r.handle_key(ord(">"), frame())
    turned = np.zeros((W, H, 3), np.uint8)
    capsys.readouterr()
    for _ in range(3):
        assert r.process(turned, [Detection("KH", 0.9, H - 1 - 50, 100)]) == []
    assert [(c.order, c.label, c.x, c.y) for c in r.tracker.round] == [(1, "KH", H - 1 - 50, 100)]
    assert r.tracker.cfg.match_dist == 20 * H / W


def test_reader_starts_at_the_rotation_it_is_given(tmp_path):
    r = Reader(CardTracker(), SeatLayout(3, H, W), Hud(), tmp_path / "seats.json", tmp_path / "frames",
               rotation=1)
    r.handle_key(ord(">"), np.zeros((W, H, 3), np.uint8))
    assert r.rotation == 2 and "180 degrees" in r.hud._message


def deal(r, *cards, seat=1):
    """Show cards (label, dx, dy), placed around the centre of seat i, until they are confirmed.
    Every card still on the table must be in cards, or it goes hidden."""
    cx, cy = r.layout.seats[seat].polygon.mean(axis=0)
    dets = [Detection(label, 0.9, float(cx + dx), float(cy + dy)) for label, dx, dy in cards]
    r.process(frame(), dets)
    return r.process(frame(), dets)


def p1_hands(r):
    return [(h.part, h.labels) for h in r.table.hands(r.tracker.round) if h.seat == "P1"]


def test_split_seat_makes_two_hands_and_later_cards_go_to_the_nearest_seed(tmp_path, capsys):
    buf = io.StringIO()
    r = make_reader(tmp_path, log=csv.writer(buf))
    pair = [("8H", 25, 0), ("8S", -25, 0)]
    deal(r, *pair)
    assert r.split_seat("P1") is True
    assert r.hud._message == "P1 split: 8S | 8H"
    assert "--- P1 split: 8S | 8H ---" in capsys.readouterr().out
    assert p1_hands(r) == [(1, ["8S"]), (2, ["8H"])]
    deal(r, *pair, ("3D", -25, 30), ("KC", 25, -30))
    assert p1_hands(r) == [(1, ["8S", "3D"]), (2, ["8H", "KC"])]
    rows = list(csv.reader(io.StringIO(buf.getvalue())))
    assert [row[4:6] for row in rows] == [["8H", "P1"], ["8S", "P1"], ["3D", "P1"], ["KC", "P1"]]


def test_split_seat_allows_a_jack_and_a_queen(tmp_path):
    r = make_reader(tmp_path)
    deal(r, ("JS", -25, 0), ("QH", 25, 0))
    assert r.split_seat("P1") is True
    assert p1_hands(r) == [(1, ["JS"]), (2, ["QH"])]


@pytest.mark.parametrize(("seat", "cards", "message"), [
    ("P1", [("8S", 0, 0)], "P1 can split two cards, not 1"),
    ("P1", [("8S", -25, 0), ("8H", 25, 0), ("8D", 0, 30)], "P1 can split two cards, not 3"),
    ("P1", [("8S", -25, 0), ("BACK", 25, 0)], "P1 cannot split a face-down card"),
    ("P1", [("8S", -25, 0), ("9H", 25, 0)], "P1 cannot split 8S and 9H: not a pair"),
    ("P4", [("8S", -25, 0), ("8H", 25, 0)], "no seat P4 at a 3-player table"),
    ("dealer", [("8S", -25, 0), ("8H", 25, 0)], "the dealer's hand cannot be split"),
])
def test_split_seat_refuses_anything_but_a_face_up_pair(tmp_path, seat, cards, message):
    r = make_reader(tmp_path)
    deal(r, *cards, seat=0 if seat == "dealer" else 1)
    assert r.split_seat(seat) is False
    assert r.hud._message == message and r.table.splits == {}


def test_split_seat_refuses_a_seat_already_split(tmp_path):
    r = make_reader(tmp_path)
    deal(r, ("8S", -25, 0), ("8H", 25, 0))
    r.split_seat("P1")
    splits = dict(r.table.splits)
    assert r.split_seat("P1") is False
    assert r.hud._message == "P1 is already split" and r.table.splits == splits


def test_join_seat_and_toggle_split_undo_a_split(tmp_path, capsys):
    r = make_reader(tmp_path)
    deal(r, ("8S", -25, 0), ("8H", 25, 0))
    assert r.join_seat("P1") is False and r.hud._message == "P1 is not split"
    r.split_seat("P1")
    assert r.join_seat("P1") is True
    assert r.hud._message == "P1 joined" and "--- P1 joined: 8S | 8H ---" in capsys.readouterr().out
    assert p1_hands(r) == [(None, ["8S", "8H"])]
    assert r.toggle_split("P1") is True and r.table.splits
    assert r.toggle_split("P1") is True and r.table.splits == {}


def test_shift_digit_keys_toggle_the_split_of_p1_to_p5(tmp_path):
    r = make_reader(tmp_path)
    deal(r, ("8S", -25, 0), ("8H", 25, 0), seat=2)
    assert r.handle_key(ord("@"), frame()) is True
    assert r.hud._message == "P2 split: 8S | 8H" and set(r.table.splits) == {"P2"}
    r.handle_key(ord("@"), frame())
    assert r.hud._message == "P2 joined" and r.table.splits == {}
    r.handle_key(ord("!"), frame())
    assert r.hud._message == "P1 can split two cards, not 0"
    r.handle_key(ord("%"), frame())
    assert r.hud._message == "no seat P5 at a 3-player table"


@pytest.mark.parametrize("key", ["r", "x", "R", "5"])
def test_round_rescan_shoe_and_player_count_keys_clear_every_split(tmp_path, key):
    r = make_reader(tmp_path)
    dets = []
    for seat, (left, right) in [(1, ("8S", "8H")), (2, ("5C", "5D"))]:
        cx, cy = r.layout.seats[seat].polygon.mean(axis=0)
        dets += [Detection(left, 0.9, float(cx - 25), float(cy)), Detection(right, 0.9, float(cx + 25), float(cy))]
    r.process(frame(), dets)
    r.process(frame(), dets)
    r.split_seat("P1")
    r.split_seat("P2")
    assert set(r.table.splits) == {"P1", "P2"}
    r.handle_key(ord(key), frame())
    assert r.table.splits == {}


def test_the_end_of_a_round_clears_every_split(tmp_path):
    r = make_reader(tmp_path)
    deal(r, ("8S", -25, 0), ("8H", 25, 0))
    r.split_seat("P1")
    kinds = []
    while "round_over" not in kinds and r.tracker.frame < 200:
        kinds = [e.kind for e in r.process(frame(), [])]
    assert "round_over" in kinds and r.table.splits == {}
