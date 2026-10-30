import re

import cv2
import numpy as np
import pytest

from cardReader.blackjack import shoe_count
from cardReader.cards import Card, Detection
from cardReader.hud import GREY, KEY_LIST, KEYS, RED, SIDEBAR, WHITE, DealtCards, Hud, count_cue, status_text, view_tabs
from cardReader.seats import SeatLayout, ZoneEditor


def test_draw_returns_video_plus_sidebar_with_hands_tags_and_edit_handles():
    W, H = 1280, 720
    layout = SeatLayout(4, W, H)
    p1, dealer = layout.seats[1].polygon.mean(axis=0), layout.seats[0].polygon.mean(axis=0)
    cards = [Card(1, "KH", float(p1[0]), float(p1[1]), 1), Card(2, "BACK", float(dealer[0]), float(dealer[1]), 2),
             Card(3, "AS", float(p1[0]) + 30, float(p1[1]), 3, hidden=True), Card(4, "2C", 3, 3, 4)]  # 2C loose
    dets = [Detection("KH", 0.9, float(p1[0]), float(p1[1]), (p1[0] - 10, p1[1] - 15, p1[0] + 10, p1[1] + 15)),
            Detection("9D", 0.5, 400, 400, (390, 385, 410, 415))]  # not confirmed: thin grey box
    editor = ZoneEditor(layout)
    editor.active, editor.selected = True, 1
    hud = Hud()
    hud.flash("hello")
    frame = np.zeros((H, W, 3), np.uint8)
    canvas = hud.draw(frame, dets, cards, 29.7, layout, shoe_count(["KH"]), capture_mode=False, editor=editor)
    assert canvas.shape == (H, W + round(SIDEBAR * W), 3)
    assert canvas[:, W:].any()                      # the sidebar has content
    assert frame.sum() == 0                          # the input frame is not drawn on


def test_boxes_take_the_colour_of_their_bound_card_and_g_hides_the_unmatched():
    W, H = 1280, 720
    layout = SeatLayout(3, W, H)
    frame = np.zeros((H, W, 3), np.uint8)
    six = Card(1, "6H", 120, 130, 1)
    misread = [Detection("9H", 0.9, 120, 130, (100, 100, 140, 160))]  # the 6H's corner, read as 9H this frame
    edge = (130, 100)                                                   # a pixel on the box's left edge
    hud = Hud()
    empty = hud.draw(frame, [], [six], 30.0, layout, shoe_count(["6H"]))
    assert tuple(hud.draw(frame, misread, [six], 30.0, layout, shoe_count(["6H"]), bound=[six])[edge]) == RED
    assert tuple(hud.draw(frame, misread, [six], 30.0, layout, shoe_count(["6H"]), bound=[None])[edge]) == GREY
    assert tuple(hud.draw(frame, misread, [six], 30.0, layout, shoe_count(["6H"]))[edge]) == GREY
    hud.show_unmatched = False
    hidden = hud.draw(frame, misread, [six], 30.0, layout, shoe_count(["6H"]), bound=[None])
    assert (hidden[edge] == empty[edge]).all()
    assert tuple(hud.draw(frame, misread, [six], 30.0, layout, shoe_count(["6H"]), bound=[six])[edge]) == RED
    assert ("G", "grey boxes") in KEY_LIST


@pytest.mark.parametrize("legend", [True, False])
@pytest.mark.parametrize("editing", [False, True])
@pytest.mark.parametrize("view", ["cards_left", "game"])
@pytest.mark.parametrize(("W", "H"), [(640, 360), (1280, 720), (1920, 1080), (3840, 2160)])
def test_both_views_draw_pending_locked_and_duplicate_cards_at_every_size(W, H, view, editing, legend):
    layout = SeatLayout(3, W, H)
    p1 = layout.seats[1].polygon.mean(axis=0)
    cards = [Card(1, "QS", float(p1[0]), float(p1[1]), 1), Card(2, "KH", float(p1[0]) + 40, float(p1[1]), 2)]
    dealt = DealtCards(locked={"KH", "2S", "AC"}, pending={"QS"}, duplicates={"KH"}, unrevealed=1)
    editor = ZoneEditor(layout)
    editor.active = editing
    hud = Hud()
    hud.view, hud.show_legend = view, legend
    frame = np.zeros((H, W, 3), np.uint8)
    canvas = hud.draw(frame, [], cards, 30.0, layout, shoe_count(["KH", "2S", "AC", "QS"], unrevealed=1),
                      editor=editor, dealt=dealt)
    assert canvas.shape == (H, W + round(SIDEBAR * W), 3)
    assert frame.sum() == 0


def test_cards_left_grid_shows_the_dealt_cards():
    W, H = 1280, 720
    layout = SeatLayout(3, W, H)
    frame = np.zeros((H, W, 3), np.uint8)
    hud = Hud()
    assert hud.view == "cards_left"
    empty = hud.draw(frame, [], [], 30.0, layout, shoe_count([]))
    dealt = hud.draw(frame, [], [], 30.0, layout, shoe_count([]), dealt=DealtCards(locked={"5D"}))
    assert (empty[:, W:] != dealt[:, W:]).any()
    hud.view = "game"
    game = hud.draw(frame, [], [], 30.0, layout, shoe_count([]), dealt=DealtCards(locked={"5D"}))
    assert (game[:, W:] != dealt[:, W:]).any()
    assert (game[:, :W] == dealt[:, :W]).all()         # the video side is the same in both views


def test_view_tabs_name_both_views_and_mark_the_current_one():
    assert view_tabs("cards_left") == [("CARDS LEFT", True), ("GAME", False)]
    assert view_tabs("game") == [("CARDS LEFT", False), ("GAME", True)]


@pytest.mark.parametrize(("W", "H"), [(1280, 720), (3840, 2160)])
def test_the_header_shows_which_view_is_on(W, H):
    layout = SeatLayout(3, W, H)
    frame = np.zeros((H, W, 3), np.uint8)
    hud = Hud()
    cards_left = hud.draw(frame, [], [], 30.0, layout, shoe_count([]))
    hud.view = "game"
    game = hud.draw(frame, [], [], 30.0, layout, shoe_count([]))
    tab_rows = slice(0, round(40 * hud.s))             # the tab strip, above the count block
    assert (cards_left[tab_rows, W:] != game[tab_rows, W:]).any()
    count_rows = slice(round(50 * hud.s), round(120 * hud.s))
    assert (cards_left[count_rows, W:] == game[count_rows, W:]).all()  # the count below is the same


@pytest.mark.parametrize("view", ["cards_left", "game"])
def test_each_view_has_its_own_legend_and_l_hides_it(view):
    W, H = 1280, 720
    layout = SeatLayout(3, W, H)
    frame = np.zeros((H, W, 3), np.uint8)
    hud = Hud()
    hud.view = view
    shown = hud.draw(frame, [], [], 30.0, layout, shoe_count([]))
    hud.show_legend = False
    hidden = hud.draw(frame, [], [], 30.0, layout, shoe_count([]))
    assert (shown[:, W:] != hidden[:, W:]).any()
    hud.show_legend = True
    hud.view = "game" if view == "cards_left" else "cards_left"
    other = hud.draw(frame, [], [], 30.0, layout, shoe_count([]))
    legend_rows = slice(H // 2, H)                     # the legend sits at the bottom of the sidebar
    assert (shown[legend_rows, W:] != other[legend_rows, W:]).any()


def test_bottom_bar_fits_at_720p_in_capture_mode():
    W, H = 1280, 720
    hud = Hud()
    stage_ms = {"detect": 123.4, "preprocess": 12.3}
    canvas = hud.draw(np.zeros((H, W, 3), np.uint8), [], [], 30.0, SeatLayout(3, W, H), shoe_count([]),
                      capture_mode=True, stage_ms=stage_ms, preprocess="clahe", showing_raw=True)
    keys_w = hud._size(KEYS, 0.5)[0]
    status_w = hud._size(status_text(30.0, stage_ms, True, "clahe", True), 0.5)[0]
    assert keys_w + status_w + 3 * hud.pad <= canvas.shape[1]


def test_status_shows_fps_timings_the_preprocess_mode_and_a_raw_display():
    assert status_text(29.7, {"detect": 23.41, "preprocess": 0.0}, False) == \
        "29.7 fps  detect 23.4 ms  prep off 0.0 ms"
    assert status_text(29.7, None, False) == "29.7 fps  detect 0.0 ms  prep off 0.0 ms"
    assert status_text(29.7, {"preprocess": 3.96}, False, "clahe") == "29.7 fps  detect 0.0 ms  prep clahe 4.0 ms"
    assert status_text(29.7, None, False, "gamma", showing_raw=True).endswith("prep gamma 0.0 ms  showing raw")
    assert status_text(0.0, None, True).startswith("CAPTURE MODE (no model)")


def test_the_mode_and_raw_display_change_the_bottom_bar():
    W, H = 1280, 720
    layout = SeatLayout(3, W, H)
    frame = np.zeros((H, W, 3), np.uint8)
    plain = Hud().draw(frame, [], [], 30.0, layout, shoe_count([]))
    raw = Hud().draw(frame, [], [], 30.0, layout, shoe_count([]), preprocess="gamma", showing_raw=True)
    assert (plain[-40:] != raw[-40:]).any()
    assert ("P", "preprocess mode") in KEY_LIST and ("Shift+V", "raw or enhanced") in KEY_LIST


@pytest.mark.parametrize(("W", "H"), [(640, 360), (1280, 720), (1920, 1080)])
def test_no_signal_message_is_drawn_over_the_video_and_c_is_listed(W, H):
    layout = SeatLayout(3, W, H)
    frame = np.zeros((H, W, 3), np.uint8)
    live = Hud().draw(frame, [], [], 30.0, layout, shoe_count([]))
    lost = Hud().draw(frame, [], [], 30.0, layout, shoe_count([]), no_signal=True)
    centre = (slice(H // 3, 2 * H // 3), slice(W // 4, 3 * W // 4))
    assert (live[centre] != lost[centre]).any()
    assert frame.sum() == 0
    assert ("C", "reconnect camera") in KEY_LIST


def test_count_cue_follows_the_sign_of_the_true_count():
    assert count_cue(1.5)[0] == "favours player"
    assert count_cue(-0.2)[0] == "favours dealer"
    assert count_cue(0.0)[0] == "neutral"


def test_draw_in_capture_mode_with_nothing_on_the_table():
    W, H = 640, 360
    canvas = Hud().draw(np.zeros((H, W, 3), np.uint8), [], [], 0.0, SeatLayout(1, W, H), shoe_count([]),
                        capture_mode=True)
    assert canvas.shape[1] == W + round(SIDEBAR * W)


NAMES = ["AS", "10H", "QD", "7C", "BACK"]


@pytest.mark.parametrize("view", ["cards_left", "game"])
@pytest.mark.parametrize(("W", "H"), [(640, 360), (1280, 720), (1920, 1080), (3840, 2160)])
def test_card_names_draw_the_rank_and_suit_shape_never_a_suit_letter(monkeypatch, W, H, view):
    layout = SeatLayout(3, W, H)
    p1 = layout.seats[1].polygon.mean(axis=0)
    cards = [Card(i + 1, name, float(p1[0]) + 60 * i, float(p1[1]), i + 1, hidden=name == "7C")
             for i, name in enumerate(NAMES)]
    texts = []
    put_text = cv2.putText
    monkeypatch.setattr(cv2, "putText", lambda img, text, *a: texts.append(text) or put_text(img, text, *a))
    hud = Hud()
    hud.view = view
    frame = np.zeros((H, W, 3), np.uint8)
    bare = hud.draw(frame, [], [], 30.0, layout, shoe_count([]))
    texts.clear()
    tagged = hud.draw(frame, [], cards, 30.0, layout, shoe_count(["AS", "10H", "QD", "7C"], unrevealed=1),
                      dealt=DealtCards(pending={"AS", "10H", "QD", "7C"}))
    assert (bare[:, :W] != tagged[:, :W]).any()
    assert not [text for text in texts if re.fullmatch(r"(10|[2-9AJQK])[SHDC]", text)]
    assert {"A", "10", "Q", "7", "BACK"} <= set(texts)


def test_a_name_is_as_wide_as_its_rank_and_suit_shape():
    hud = Hud()
    hud.draw(np.zeros((720, 1280, 3), np.uint8), [], [], 30.0, SeatLayout(3, 1280, 720), shoe_count([]))
    assert hud._name_size("10H", 0.8)[0] > hud._size("10", 0.8)[0]
    assert hud._name_size("10H", 0.8)[0] == hud._name_size("10S", 0.8)[0]
    assert hud._name_size("BACK", 0.8) == hud._size("BACK", 0.8)
    assert hud._name(np.zeros((50, 200, 3), np.uint8), "10H", 5, 25, 0.8, RED) == hud._name_size("10H", 0.8)[0]


@pytest.mark.parametrize(("W", "H"), [(640, 360), (1280, 720), (3840, 2160)])
def test_each_grid_cell_is_a_mini_card_that_shows_its_state(W, H):
    hud = Hud()
    hud.draw(np.zeros((H, W, 3), np.uint8), [], [], 30.0, SeatLayout(3, W, H), shoe_count([]))
    h = round(26 * hud.s)
    w = round(h * 2.5 / 3.5)
    drawn = {}
    for state, duplicate in [("unseen", False), ("pending", False), ("locked", False), ("unseen", True)]:
        cell = np.full((h, w, 3), 25, np.uint8)
        hud._cell_mark(cell, "KH", state, duplicate, 0, 0, w, h)
        drawn[state, duplicate] = cell
    assert (drawn["unseen", False] == WHITE).all(axis=2).any()           # a white card face
    assert (drawn["unseen", False] == RED).all(axis=2).any()             # a heart in red
    spade = np.full((h, w, 3), 25, np.uint8)
    hud._cell_mark(spade, "KS", "unseen", False, 0, 0, w, h)
    assert not (spade == RED).all(axis=2).any() and (spade == 0).all(axis=2).any()  # a spade in black
    cells = list(drawn.values())
    for i, a in enumerate(cells):
        for b in cells[i + 1:]:
            assert (a != b).any()


def test_rotation_keys_are_listed():
    assert ("<", "turn left") in KEY_LIST and (">", "turn right") in KEY_LIST
