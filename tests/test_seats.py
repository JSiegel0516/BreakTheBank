import cv2
import numpy as np

from cardReader.seats import SeatLayout, ZoneEditor

W, H = 1920, 1080
DOWN, MOVE, UP, RIGHT = cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONDOWN


def editor(players=3):
    e = ZoneEditor(SeatLayout(players, W, H))
    e.active = True
    return e


def drag(e, start, end):
    e.on_mouse(DOWN, *start)
    e.on_mouse(MOVE, (start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
    e.on_mouse(MOVE, *end)
    e.on_mouse(UP, *end)


def test_drag_one_corner_moves_only_that_corner():
    e = editor()
    p1 = e.layout.seats[1]
    before = p1.polygon.copy()
    corner = tuple(int(v) for v in before[2])
    drag(e, corner, (corner[0] + 120, corner[1] + 40))
    assert tuple(p1.polygon[2]) == (corner[0] + 120, corner[1] + 40)
    assert np.array_equal(np.delete(p1.polygon, 2, axis=0), np.delete(before, 2, axis=0))


def test_drag_inside_moves_the_whole_zone():
    e = editor()
    dealer = e.layout.seats[0]
    before = dealer.polygon.copy()
    cx, cy = (int(v) for v in before.mean(axis=0))
    drag(e, (cx, cy), (cx + 100, cy + 50))
    assert np.array_equal(dealer.polygon, before + (100, 50))


def test_whole_zone_stops_at_the_frame_edge():
    e = editor()
    dealer = e.layout.seats[0]
    cx, cy = (int(v) for v in dealer.polygon.mean(axis=0))
    drag(e, (cx, cy), (cx, cy - 5000))
    assert dealer.polygon[:, 1].min() == 0  # pushed up to the top edge, shape kept
    assert dealer.polygon[:, 1].max() - dealer.polygon[:, 1].min() == \
        np.ptp(SeatLayout(3, W, H).seats[0].polygon[:, 1])


def test_right_click_resets_a_zone():
    e = editor()
    p2 = e.layout.seats[2]
    template = p2.polygon.copy()
    cx, cy = (int(v) for v in template.mean(axis=0))
    drag(e, (cx, cy), (cx + 200, cy - 100))
    assert not np.array_equal(p2.polygon, template)
    e.on_mouse(RIGHT, cx + 200, cy - 100)
    assert np.array_equal(p2.polygon, template)


def test_nothing_happens_outside_edit_mode():
    e = editor()
    e.active = False
    before = [s.polygon.copy() for s in e.layout.seats]
    cx, cy = (int(v) for v in before[0].mean(axis=0))
    drag(e, (cx, cy), (cx + 300, cy + 300))
    assert all(np.array_equal(s.polygon, b) for s, b in zip(e.layout.seats, before))


def test_clicks_on_the_sidebar_are_ignored():
    e = editor()
    before = e.layout.seats[0].polygon.copy()
    e.on_mouse(DOWN, W + 50, 100)  # the sidebar is to the right of the video
    e.on_mouse(MOVE, W + 80, 200)
    e.on_mouse(UP, W + 80, 200)
    assert np.array_equal(e.layout.seats[0].polygon, before)


def test_edited_corner_changes_which_seat_a_card_is_in():
    e = editor()
    p1 = e.layout.seats[1]
    far = (int(p1.polygon[:, 0].max()) + 150, int(p1.polygon[:, 1].mean()))
    assert e.layout.assign(*far) != "P1"
    corner = tuple(int(v) for v in p1.polygon[int(p1.polygon[:, 0].argmax())])
    drag(e, corner, (far[0] + 50, far[1]))
    assert e.layout.assign(*far) == "P1"


def test_save_and_load_keep_edits_at_any_resolution(tmp_path):
    e = editor(players=5)
    corner = tuple(int(v) for v in e.layout.seats[3].polygon[0])
    drag(e, corner, (corner[0] - 60, corner[1] + 30))
    path = tmp_path / "seats.json"
    e.layout.save(path)
    same = SeatLayout.load(path, W, H)
    assert same.players == 5 and same.names() == e.layout.names()
    assert np.abs(same.seats[3].polygon - e.layout.seats[3].polygon).max() <= 1
    half = SeatLayout.load(path, W // 2, H // 2)
    assert np.abs(half.seats[3].polygon * 2 - e.layout.seats[3].polygon).max() <= 2
    assert SeatLayout.load(tmp_path / "missing.json", W, H) is None


def test_resized_scales_every_zone():
    layout = SeatLayout(4, W, H)
    small = layout.resized(W // 2, H // 2)
    for a, b in zip(layout.seats, small.seats):
        assert np.abs(b.polygon * 2 - a.polygon).max() <= 2
