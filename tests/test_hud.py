import re
from collections import Counter

import cv2
import numpy as np
import pytest

from cardReader.blackjack import COUNT_SYSTEMS, shoe_count
from cardReader.cards import Card, Detection
from cardReader.hud import (
    BADGE,
    DIM,
    DUPLICATE,
    GREEN,
    GREY,
    HELD_MARK,
    KEY_LIST,
    KEYS,
    PENDING_FILL,
    RED,
    SHADE_ALPHA,
    SIDEBAR,
    WHITE,
    DealtCards,
    Hud,
    count_cue,
    seat_title,
    status_text,
    value_legend,
    value_shade,
    view_tabs,
)
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


@pytest.mark.parametrize(("W", "H"), [(640, 360), (1280, 720), (3840, 2160)])
def test_draw_without_the_sidebar_is_the_video_width_with_the_same_overlay(W, H):
    layout = SeatLayout(3, W, H)
    p1 = layout.seats[1].polygon.mean(axis=0)
    cards = [Card(1, "KH", float(p1[0]), float(p1[1]), 1)]
    frame = np.zeros((H, W, 3), np.uint8)
    with_sidebar = Hud().draw(frame, [], cards, 30.0, layout, shoe_count(["KH"]))
    video_only = Hud().draw(frame, [], cards, 30.0, layout, shoe_count(["KH"]), sidebar=False)
    assert video_only.shape == (H, W, 3)
    assert (video_only[:H // 2] == with_sidebar[:H // 2, :W]).all()   # tags and zones as before
    assert (video_only[-10:] != frame[-10:]).any()                    # the bottom bar is drawn
    assert frame.sum() == 0


def test_pop_flash_returns_the_latest_message_once():
    hud = Hud()
    assert hud.pop_flash() is None
    hud.flash("new round")
    hud.flash("camera reconnected", 2.5)
    assert hud.pop_flash() == ("camera reconnected", 2.5)
    assert hud.pop_flash() is None
    assert hud._message == "camera reconnected"              # still shown in the window until it times out


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


def test_count_cue_says_which_cards_are_left_and_whom_that_favours():
    assert count_cue(1.5) == ("high cards left \u00b7 favours player", GREEN)
    assert count_cue(-0.2) == ("low cards left \u00b7 favours dealer", RED)
    assert count_cue(0.0) == ("even deck", WHITE)


def test_row_shading_darkens_with_the_size_of_the_value():
    assert value_shade(0) == (GREY, SHADE_ALPHA)
    assert [value_shade(value)[0] for value in (1, 3, -1, -3)] == [GREEN, GREEN, RED, RED]
    alphas = [value_shade(value)[1] for value in (1, 2, 3)]
    assert alphas[0] == SHADE_ALPHA and alphas[0] < alphas[1] < alphas[2]
    assert value_shade(-3)[1] > value_shade(-1)[1]


def test_the_legend_names_each_value_and_its_ranks():
    assert [text for _, _, text in value_legend(COUNT_SYSTEMS["hi_lo"])] == ["2 to 6: +1", "7 to 9: 0", "10 to A: -1"]
    assert [text for _, _, text in value_legend(COUNT_SYSTEMS["hi_opt_ii"])] == \
        ["4, 5: +2", "2, 3, 6, 7: +1", "8, 9, A: 0", "10 to K: -2"]
    assert [(value, text) for _, value, text in value_legend(COUNT_SYSTEMS["uston_apc"])] == \
        [(3, "5: +3"), (2, "3, 4, 6, 7: +2"), (1, "2, 8: +1"), (0, "A: 0"), (-1, "9: -1"), (-3, "10 to K: -3")]


def test_the_window_names_the_counting_system_and_shows_its_values(monkeypatch):
    W, H = 1280, 720
    texts = []
    put_text = cv2.putText
    monkeypatch.setattr(cv2, "putText", lambda img, text, *a: texts.append(text) or put_text(img, text, *a))
    layout = SeatLayout(3, W, H)
    Hud().draw(np.zeros((H, W, 3), np.uint8), [], [], 30.0, layout, shoe_count([]))
    assert "Hi-Lo" in texts and "+3" not in texts
    texts.clear()
    Hud().draw(np.zeros((H, W, 3), np.uint8), [], [], 30.0, layout, shoe_count([], system="uston_apc"))
    assert "Uston APC" in texts and {"+3", "-3", "-1"} <= set(texts)


def test_dealt_cards_count_the_unseen_cards_of_given_ranks():
    dealt = DealtCards(locked={"AS", "KH"}, pending={"10D"})
    assert dealt.left(["A"]) == 3
    assert dealt.left(["10", "J", "Q", "K", "A"]) == 17
    assert DealtCards().left(["A"]) == 4


def test_dealt_cards_count_copies_with_more_decks():
    dealt = DealtCards(locked=Counter({"AS": 2, "KH": 1}), pending=Counter({"AH": 1}), decks=3)
    assert (dealt.seen("AS"), dealt.copies_left("AS"), dealt.state("AS")) == (2, 1, "locked")
    assert (dealt.seen("AH"), dealt.state("AH"), dealt.seen("AD")) == (1, "pending", 0)
    assert dealt.left(["A"]) == 12 - 3 and dealt.left(["K"]) == 11
    unseen = dealt.unseen()
    assert len(unseen) == 156 - 4 and unseen.count("AS") == 1 and unseen.count("2S") == 3
    assert DealtCards(locked=Counter({"AS": 2}), decks=1).copies_left("AS") == 0  # fewer decks chosen mid-shoe


def test_dealt_cards_show_a_pending_copy_over_a_locked_one():
    assert DealtCards(locked={"KH"}, pending={"KH"}, decks=2).state("KH") == "pending"


def test_the_window_counts_cards_left_out_of_the_whole_shoe(monkeypatch):
    texts = []
    put_text = cv2.putText
    monkeypatch.setattr(cv2, "putText", lambda img, text, *a: texts.append(text) or put_text(img, text, *a))
    Hud().draw(np.zeros((720, 1280, 3), np.uint8), [], [], 30.0, SeatLayout(3, 1280, 720),
               shoe_count(["KH"], decks=2), dealt=DealtCards(pending={"KH"}, decks=2))
    assert "103/104" in texts
    assert "7" in texts and "8" in texts                    # the K row's copies left, and every other row's
    assert not [text for text in texts if text.endswith(" face down")]   # no suffix with none face down


@pytest.mark.parametrize("view", ["cards_left", "game"])
@pytest.mark.parametrize(("W", "H"), [(640, 360), (1280, 720), (3840, 2160)])
def test_the_window_counts_face_down_cards_in_cards_left_and_names_them_beside_it(monkeypatch, view, W, H):
    drawn = []
    put_text = cv2.putText
    monkeypatch.setattr(cv2, "putText",
                        lambda img, text, org, *a: drawn.append((text, org)) or put_text(img, text, org, *a))
    hud = Hud()
    hud.view = view
    canvas = hud.draw(np.zeros((H, W, 3), np.uint8), [], [], 30.0, SeatLayout(3, W, H),
                      shoe_count(["KH"], decks=2, unrevealed=2),
                      dealt=DealtCards(pending={"KH"}, unrevealed=2, decks=2))
    where = dict(drawn)
    assert "103/104" in where and "2 face down" in where   # the grid's unseen copies, whole at every size
    suffix_x, suffix_y = where["2 face down"]
    assert suffix_x + hud._size("2 face down", 0.55)[0] <= canvas.shape[1] - hud.pad   # inside the sidebar
    beside = suffix_y == where["103/104"][1]
    assert beside == (W >= 1280)                            # a narrow sidebar puts it on a line of its own


def test_without_dealt_cards_the_window_counts_a_whole_shoe(monkeypatch):
    texts = []
    put_text = cv2.putText
    monkeypatch.setattr(cv2, "putText", lambda img, text, *a: texts.append(text) or put_text(img, text, *a))
    Hud().draw(np.zeros((720, 1280, 3), np.uint8), [], [], 30.0, SeatLayout(3, 1280, 720), shoe_count([], decks=3))
    assert "156/156" in texts


def test_a_mini_card_with_copies_left_is_filled_by_the_share_seen():
    hud = Hud()
    hud.draw(np.zeros((2160, 3840, 3), np.uint8), [], [], 30.0, SeatLayout(3, 3840, 2160), shoe_count([]))
    h = round(26 * hud.s)        # 4K: a cell big enough that the fill shows around the markings
    w = round(h * 2.5 / 3.5)

    def cell(state, seen):
        img = np.full((h, w, 3), 25, np.uint8)
        hud._cell_mark(img, "KS", state, False, 0, 0, w, h, seen=seen)
        return img

    half = cell("locked", 0.5)
    mid = h // 2
    filled = (half[mid] == DIM).all(axis=1).nonzero()[0]
    white = (half[mid] == WHITE).all(axis=1).nonzero()[0]
    assert filled.size and white.size and filled.max() < white.min()  # dim on the left, white on the right
    assert (cell("locked", 1 / 3) != half).any() and (half != cell("locked", 1.0)).any()
    assert (cell("pending", 0.5) == PENDING_FILL).all(axis=2).any()
    assert (cell("unseen", 0.0) == cell("unseen", 1.0)).all()


@pytest.mark.parametrize(("seen", "lines"), [(["2S"], {"high cards left", "favours player"}),
                                              (["KS"], {"low cards left", "favours dealer"}),
                                              ([], {"even deck"})])
def test_the_window_draws_the_cue_one_half_per_line_in_plain_ascii(monkeypatch, seen, lines):
    W, H = 1280, 720
    texts = []
    put_text = cv2.putText
    monkeypatch.setattr(cv2, "putText", lambda img, text, *a: texts.append(text) or put_text(img, text, *a))
    Hud().draw(np.zeros((H, W, 3), np.uint8), [], [], 30.0, SeatLayout(3, W, H), shoe_count(seen))
    assert lines <= set(texts)
    assert not [text for text in texts if not text.isascii()]   # OpenCV cannot draw the middle dot


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
    assert ("< >", "turn view") in KEY_LIST


def test_rescan_key_is_listed_and_on_the_bottom_bar():
    assert ("X", "rescan round") in KEY_LIST and "[X] rescan" in KEYS


def test_video_tags_show_the_placement_badge_only_when_show_order_is_on():
    W, H = 1280, 720
    layout = SeatLayout(3, W, H)
    frame = np.zeros((H, W, 3), np.uint8)
    six = Card(1, "6H", 300, 300, 1)
    hud = Hud()
    assert hud.show_order is False
    plain = hud.draw(frame, [], [six], 30.0, layout, shoe_count(["6H"]), sidebar=False)
    hud.show_order = True
    badged = hud.draw(frame, [], [six], 30.0, layout, shoe_count(["6H"]), sidebar=False)
    video = slice(0, H // 2)                                     # above the bottom bar
    assert not (plain[video] == BADGE).all(axis=2).any()
    assert (badged[video] == BADGE).all(axis=2).any()
    assert ("N", "placement numbers") in KEY_LIST


@pytest.mark.parametrize("show_order", [False, True])
def test_a_held_copy_has_a_magenta_question_mark_after_its_name_on_the_video(show_order):
    W, H = 1280, 720
    layout = SeatLayout(3, W, H)
    frame = np.zeros((H, W, 3), np.uint8)
    hud = Hud()
    hud.show_order = show_order
    counted = Card(2, "KH", 300, 300, 1)
    held = Card(2, "KH", 300, 300, 1, held=True)
    video = slice(0, H // 2)                                     # above the bottom bar
    for card, marked in [(counted, False), (held, True)]:
        img = hud.draw(frame, [], [card], 30.0, layout, shoe_count(["KH"], decks=2), sidebar=False)
        assert (img[video] == DUPLICATE).all(axis=2).any() is np.bool_(marked)


def test_game_legend_lists_a_held_copy(monkeypatch):
    W, H = 1280, 720
    texts = []
    put_text = cv2.putText
    monkeypatch.setattr(cv2, "putText", lambda img, text, *a: texts.append(text) or put_text(img, text, *a))
    hud = Hud()
    hud.view = "game"
    hud.draw(np.zeros((H, W, 3), np.uint8), [], [], 30.0, SeatLayout(3, W, H), shoe_count([]))
    assert "copy held, not counted" in texts and HELD_MARK in texts


@pytest.mark.parametrize("show_order", [False, True])
def test_game_legend_lists_the_placement_number_only_when_show_order_is_on(monkeypatch, show_order):
    W, H = 1280, 720
    texts = []
    put_text = cv2.putText
    monkeypatch.setattr(cv2, "putText", lambda img, text, *a: texts.append(text) or put_text(img, text, *a))
    hud = Hud()
    hud.view, hud.show_order = "game", show_order
    hud.draw(np.zeros((H, W, 3), np.uint8), [], [], 30.0, SeatLayout(3, W, H), shoe_count([]))
    assert ("placement number" in texts) is show_order
    assert "covered, still counted" in texts


def test_the_game_view_and_the_video_zone_labels_show_display_names(monkeypatch):
    W, H = 1280, 720
    texts = []
    put_text = cv2.putText
    monkeypatch.setattr(cv2, "putText", lambda img, text, *a: texts.append(text) or put_text(img, text, *a))
    hud = Hud()
    hud.view = "game"
    hud.draw(np.zeros((H, W, 3), np.uint8), [], [], 30.0, SeatLayout(3, W, H), shoe_count([]),
             seat_names={"P1": "Ann", "dealer": "House"})
    for title in ("Ann", "House", "P2", "P3"):
        assert texts.count(title) == 2                             # the Game view and the zone label
    assert "P1" not in texts and "DEALER" not in texts
    for W, H in ((1280, 720), (640, 360)):
        texts.clear()
        hud.draw(np.zeros((H, W, 3), np.uint8), [], [], 30.0, SeatLayout(5, W, H), shoe_count([]), sidebar=False)
        assert texts[:6] == ["DEALER", "P1", "P2", "P3", "P4", "P5"]  # no names: the whole seat ids


def _top_edge_midpoint(polygon: np.ndarray) -> tuple[float, float]:
    ends = np.roll(polygon, -1, axis=0)
    midpoints = (polygon + ends) / 2
    return tuple(midpoints[np.argmin(midpoints[:, 1])])


def _tilted_rectangle(cx, cy, half_w, half_h, degrees) -> np.ndarray:
    angle = np.radians(degrees)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    corners = np.array([(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)])
    return (corners @ rotation.T + (cx, cy)).astype(np.int32)


@pytest.mark.parametrize(("W", "H", "layout_of", "names"), [
    (1280, 720, lambda W, H: SeatLayout(3, W, H), {"P1": "Ann", "dealer": "House"}),
    (640, 360, lambda W, H: SeatLayout(5, W, H), {f"P{i}": "W" * 16 for i in range(1, 6)}),
    (1280, 720, lambda W, H: SeatLayout(2, W, H, [SeatLayout(2, W, H).seats[0].polygon,
                                                  _tilted_rectangle(300, 500, 120, 90, 25),
                                                  _tilted_rectangle(900, 500, 120, 90, -30)]), {}),
])
def test_each_zone_label_is_centred_on_its_top_most_edge_inside_the_polygon(monkeypatch, W, H, layout_of, names):
    layout = layout_of(W, H)
    drawn = []
    put_text = cv2.putText
    monkeypatch.setattr(cv2, "putText",
                        lambda img, text, org, *a: drawn.append((text, org)) or put_text(img, text, org, *a))
    hud = Hud()
    hud.draw(np.zeros((H, W, 3), np.uint8), [], [], 30.0, layout, shoe_count([]), sidebar=False, seat_names=names)
    labels = drawn[:len(layout.seats)]                              # the zone labels come first
    for seat, (text, (x, y)) in zip(layout.seats, labels, strict=True):
        title = seat_title(seat.name, names)
        assert text == title or (text.endswith("..") and title.startswith(text[:-2]))
        w, h, _ = hud._size(text, 0.6)
        mid_x, _ = _top_edge_midpoint(seat.polygon.astype(float))
        assert abs(x + w / 2 - mid_x) <= 1
        half_pad = hud.pad // 2
        panel = [(x - half_pad, y - h - half_pad), (x + w + half_pad, y - h - half_pad),
                 (x + w + half_pad, y + half_pad), (x - half_pad, y + half_pad)]
        contour = seat.polygon.reshape(-1, 1, 2)
        assert all(cv2.pointPolygonTest(contour, (float(px), float(py)), False) >= 0 for px, py in panel)


def test_a_zone_label_moves_below_a_notch_cut_into_a_concave_zone(monkeypatch):
    W, H = 1600, 900                                                   # text scale 1, padding 10
    dealer = SeatLayout(1, W, H).seats[0].polygon
    notched = np.array([(100, 100), (300, 100), (300, 110), (190, 110), (190, 120), (300, 120),
                        (300, 300), (100, 300)], np.int32)
    drawn = []
    put_text = cv2.putText
    monkeypatch.setattr(cv2, "putText",
                        lambda img, text, org, *a: drawn.append((text, org)) or put_text(img, text, org, *a))
    hud = Hud()
    hud.draw(np.zeros((H, W, 3), np.uint8), [], [], 30.0, SeatLayout(1, W, H, [dealer, notched]), shoe_count([]),
             sidebar=False, seat_names={"P1": "Ann"})
    text, (x, y) = drawn[1]
    assert text == "Ann"
    w, h, _ = hud._size(text, 0.6)
    half_pad = hud.pad // 2
    contour = notched.reshape(-1, 1, 2)
    panel_points = [(float(px), float(py)) for px in range(x - half_pad, x + w + half_pad + 1)
                    for py in range(y - h - half_pad, y + half_pad + 1)]
    assert all(cv2.pointPolygonTest(contour, point, False) >= 0 for point in panel_points)


def test_a_zone_label_moves_down_where_the_zone_widens_and_is_left_out_where_nothing_fits(monkeypatch):
    W, H = 1280, 720
    dealer = SeatLayout(2, W, H).seats[0].polygon
    narrow_top = np.array([(300, 400), (310, 400), (500, 600), (100, 600)], np.int32)
    tiny = np.array([(900, 400), (904, 400), (902, 404)], np.int32)
    layout = SeatLayout(2, W, H, [dealer, narrow_top, tiny])
    drawn = []
    put_text = cv2.putText
    monkeypatch.setattr(cv2, "putText",
                        lambda img, text, org, *a: drawn.append((text, org)) or put_text(img, text, org, *a))
    hud = Hud()
    hud.draw(np.zeros((H, W, 3), np.uint8), [], [], 30.0, layout, shoe_count([]), sidebar=False,
             seat_names={"P1": "Ann", "P2": "Bo"})
    texts = [text for text, _ in drawn]
    assert texts[:2] == ["DEALER", "Ann"]
    assert "Bo" not in texts and "B.." not in texts                     # the tiny zone has no label
    _, (x, y) = drawn[1]
    w, h, _ = hud._size("Ann", 0.6)
    half_pad = hud.pad // 2
    assert abs(x + w / 2 - 305) <= 1 and y - h - half_pad > 400 + hud.pad   # moved below the narrow edge
    panel = [(x - half_pad, y - h - half_pad), (x + w + half_pad, y - h - half_pad),
             (x + w + half_pad, y + half_pad), (x - half_pad, y + half_pad)]
    contour = narrow_top.reshape(-1, 1, 2)
    assert all(cv2.pointPolygonTest(contour, (float(px), float(py)), False) >= 0 for px, py in panel)


@pytest.mark.parametrize(("W", "H", "name"), [(1280, 720, "Ann"), (640, 360, "W" * 16)])
def test_game_view_shows_a_split_seat_as_two_numbered_lines(monkeypatch, W, H, name):
    layout = SeatLayout(3, W, H)
    cx, cy = map(float, layout.seats[1].polygon.mean(axis=0))
    cards = [Card(1, "8S", cx - 20, cy, 1), Card(2, "8H", cx + 20, cy, 2)]
    texts = []
    put_text = cv2.putText
    monkeypatch.setattr(cv2, "putText", lambda img, text, *a: texts.append(text) or put_text(img, text, *a))
    hud = Hud()
    hud.view, hud.show_legend = "game", False
    splits = {"P1": (1, 2)}
    hud.draw(np.zeros((H, W, 3), np.uint8), [], cards, 30.0, layout, shoe_count(["8S", "8H"]),
             seat_names={"P1": name}, splits=splits)
    sidebar_texts = texts[len(layout.seats):]                           # after the zone labels on the video
    titles = [text for text in sidebar_texts if text.startswith(name[:2])]
    assert len(titles) == 2 and titles[0].endswith(" 1") and titles[1].endswith(" 2")   # a long name keeps its part
    assert splits == {"P1": (1, 2)}
    hud.draw(np.zeros((H, W, 3), np.uint8), [], cards[:1], 30.0, layout, shoe_count(["8S"]), splits=splits)
    assert splits == {}                                                 # a seed left: the split is dropped


def test_split_keys_are_listed():
    assert ("Shift+1-5", "split seat") in KEY_LIST


def test_a_wide_display_name_is_shortened_so_every_hand_keeps_room_for_its_cards(monkeypatch):
    W, H = 640, 360                                                    # the narrowest sidebar: 192 pixels
    layout = SeatLayout(3, W, H)
    cards = [Card(i + 1, label, *map(float, layout.seats[seat].polygon.mean(axis=0)), i + 1)
             for i, (label, seat) in enumerate([("KS", 1), ("QH", 2)])]
    drawn = []
    put_text = cv2.putText
    monkeypatch.setattr(cv2, "putText", lambda img, text, org, *a: drawn.append((img.shape[1], text, org[0]))
                        or put_text(img, text, org, *a))
    hud = Hud()
    hud.view, hud.show_legend = "game", False                          # at 360 high the legend leaves no room
    hud.draw(np.zeros((H, W, 3), np.uint8), [], cards, 30.0, layout, shoe_count(["KS", "QH"]),
             seat_names={"P1": "W" * 16})
    side_w = round(SIDEBAR * W)
    sidebar = [(text, x) for width, text, x in drawn if width == side_w]
    titles = [text for text, _ in sidebar if text.startswith("W")]
    assert titles and titles[0].endswith("..") and len(titles[0]) < 18   # shortened, not the full name
    assert {"DEALER", "P2", "P3"} <= {text for text, _ in sidebar}      # the default names fit whole
    for rank in ("K", "Q"):                                             # each hand's card starts well inside
        x = next(x for text, x in sidebar if text == rank)
        assert x + hud._size(rank, 0.8)[0] < side_w - hud.pad
