import numpy as np

from card_reader.blackjack import shoe_count
from card_reader.cards import Card, Detection
from card_reader.hud import SIDEBAR, Hud
from card_reader.seats import SeatLayout, ZoneEditor


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


def test_draw_in_capture_mode_with_nothing_on_the_table():
    W, H = 640, 360
    canvas = Hud().draw(np.zeros((H, W, 3), np.uint8), [], [], 0.0, SeatLayout(1, W, H), shoe_count([]),
                        capture_mode=True)
    assert canvas.shape[1] == W + round(SIDEBAR * W)
