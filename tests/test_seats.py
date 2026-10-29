import itertools

import cv2
import numpy as np
import pytest

from cardReader.seats import MAX_PLAYERS, SeatLayout, ZoneEditor

W, H = 1920, 1080
DOWN, MOVE, UP, RIGHT = cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONDOWN
COUNTS = range(1, MAX_PLAYERS + 1)
EVERY_SEAT = [(players, i) for players in COUNTS for i in range(players + 1)]  # i = 0 is the dealer


def editor(players=3):
    e = ZoneEditor(SeatLayout(players, W, H))
    e.active = True
    return e


def drag(e, start, end):
    e.on_mouse(DOWN, *start)
    e.on_mouse(MOVE, (start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
    e.on_mouse(MOVE, *end)
    e.on_mouse(UP, *end)


def inward(point, step=30):
    """point moved step pixels towards the middle of the frame on both axes."""
    x, y = (int(v) for v in point)
    return x + (step if x < W / 2 else -step), y + (step if y < H / 2 else -step)


@pytest.mark.parametrize(("width", "height"), [(1920, 1080), (3840, 2160)])
@pytest.mark.parametrize("players", COUNTS)
def test_template_seats_lie_inside_the_frame_and_apart(players, width, height):
    layout = SeatLayout(players, width, height)
    for seat in layout.seats:
        xs, ys = seat.polygon[:, 0], seat.polygon[:, 1]
        assert (xs > 0).all() and (xs < width - 1).all() and (ys > 0).all() and (ys < height - 1).all(), seat.name
    for a, b in itertools.combinations(layout.seats, 2):
        overlap, _ = cv2.intersectConvexConvex(a.polygon.astype(np.float32), b.polygon.astype(np.float32))
        assert overlap == 0, (a.name, b.name)


@pytest.mark.parametrize(("players", "i"), EVERY_SEAT)
def test_every_corner_of_every_seat_can_be_dragged(players, i):
    for corner_i in range(4):
        e = editor(players)
        seat = e.layout.seats[i]
        before = seat.polygon.copy()
        target = inward(before[corner_i])
        drag(e, tuple(int(v) for v in before[corner_i]), target)
        assert tuple(seat.polygon[corner_i]) == target, (seat.name, corner_i)
        assert np.array_equal(np.delete(seat.polygon, corner_i, axis=0), np.delete(before, corner_i, axis=0))


@pytest.mark.parametrize(("players", "i"), EVERY_SEAT)
def test_every_plus_of_every_seat_adds_a_corner(players, i):
    for edge in range(4):
        e = editor(players)
        seat = e.layout.seats[i]
        before = seat.polygon.copy()
        mid = ZoneEditor.midpoints(before)[edge]
        target = inward(mid)
        drag(e, tuple(int(v) for v in mid), target)
        assert len(seat.polygon) == 5 and tuple(seat.polygon[edge + 1]) == target, (seat.name, edge)


@pytest.mark.parametrize(("players", "i"), EVERY_SEAT)
def test_every_seat_can_be_moved_by_its_body(players, i):
    e = editor(players)
    seat = e.layout.seats[i]
    before = seat.polygon.copy()
    centre = tuple(int(v) for v in before.mean(axis=0))
    target = inward(centre, step=20)
    drag(e, centre, target)
    assert e.selected == i
    assert np.array_equal(seat.polygon, before + np.subtract(target, centre))


def test_a_corner_dragged_past_the_edge_stops_there_and_is_grabbed_from_just_beyond():
    e = editor()
    last = e.layout.seats[-1]
    right = int(last.polygon[:, 0].argmax())
    y = int(last.polygon[right, 1])
    drag(e, tuple(int(v) for v in last.polygon[right]), (W + 200, y))
    assert tuple(last.polygon[right]) == (W - 1, y)
    drag(e, (W + 10, y), (W - 100, y))  # pressed on the sidebar, next to the handle
    assert tuple(last.polygon[right]) == (W - 100, y)

    low = int(last.polygon[:, 1].argmax())
    x = int(last.polygon[low, 0])
    drag(e, (x, int(last.polygon[low, 1])), (x, H + 200))
    assert tuple(last.polygon[low]) == (x, H - 1)
    drag(e, (x, H + 10), (x, H - 80))  # pressed on the bottom bar, next to the handle
    assert tuple(last.polygon[low]) == (x, H - 80)


def test_the_nearer_of_two_close_corners_is_grabbed():
    e = editor()
    p1, p2 = e.layout.seats[1], e.layout.seats[2]
    p2_corner = tuple(int(v) for v in p2.polygon[0])
    near = (p2_corner[0] - 12, p2_corner[1])
    drag(e, tuple(int(v) for v in p1.polygon[1]), near)
    assert e._hit(*near) == (1, 1)
    assert e._hit(*p2_corner) == (2, 0)


@pytest.mark.parametrize("players", COUNTS)
def test_edited_layouts_save_and_reload_at_the_same_shape(players, tmp_path):
    e = editor(players)
    for seat in e.layout.seats:
        drag(e, tuple(int(v) for v in seat.polygon[0]), inward(seat.polygon[0], step=25))
    path = tmp_path / "seats.json"
    e.layout.save(path)
    for width, height in ((W, H), (2 * W, 2 * H)):
        loaded = SeatLayout.load(path, width, height, players)
        scale = width / W
        for a, b in zip(e.layout.seats, loaded.seats, strict=True):
            assert np.abs(b.polygon - a.polygon * scale).max() <= scale, a.name


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
    assert all(np.array_equal(s.polygon, b) for s, b in zip(e.layout.seats, before, strict=True))


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
    for a, b in zip(layout.seats, small.seats, strict=True):
        assert np.abs(b.polygon * 2 - a.polygon).max() <= 2


def test_drag_a_midpoint_adds_a_corner_there():
    e = editor()
    p1 = e.layout.seats[1]
    before = p1.polygon.copy()
    mid = tuple(int(v) for v in ZoneEditor.midpoints(before)[1])  # middle of the edge from corner 1 to 2
    drag(e, mid, (mid[0] + 80, mid[1] + 20))
    assert len(p1.polygon) == 5
    assert tuple(p1.polygon[2]) == (mid[0] + 80, mid[1] + 20)  # the new corner sits between old corners 1 and 2
    assert np.array_equal(np.delete(p1.polygon, 2, axis=0), before)


def test_clicking_a_midpoint_without_dragging_adds_nothing():
    e = editor()
    p1 = e.layout.seats[1]
    mid = tuple(int(v) for v in ZoneEditor.midpoints(p1.polygon)[0])
    e.on_mouse(DOWN, *mid)
    e.on_mouse(UP, *mid)
    assert len(p1.polygon) == 4


def test_midpoint_of_the_last_edge_inserts_before_the_first_corner():
    e = editor()
    dealer = e.layout.seats[0]
    mid = tuple(int(v) for v in ZoneEditor.midpoints(dealer.polygon)[3])  # edge from corner 3 back to corner 0
    drag(e, mid, (mid[0] - 30, mid[1]))
    assert len(dealer.polygon) == 5 and tuple(dealer.polygon[4]) == (mid[0] - 30, mid[1])


def test_right_click_a_corner_removes_it_but_keeps_three():
    e = editor()
    p2 = e.layout.seats[2]
    for expected in (3, 3):  # 4 -> 3, then a 3-corner zone keeps its corners
        corner = tuple(int(v) for v in p2.polygon[0])
        e.on_mouse(RIGHT, *corner)
        assert len(p2.polygon) == expected


def test_right_click_inside_resets_added_corners_too():
    e = editor()
    p2 = e.layout.seats[2]
    template = p2.polygon.copy()
    mid = tuple(int(v) for v in ZoneEditor.midpoints(template)[0])
    drag(e, mid, (mid[0], mid[1] - 60))
    assert len(p2.polygon) == 5
    cx, cy = (int(v) for v in template.mean(axis=0))
    e.on_mouse(RIGHT, cx, cy)
    assert np.array_equal(p2.polygon, template)


def test_corner_wins_over_a_nearby_midpoint():
    e = editor()
    p1 = e.layout.seats[1]
    corner = tuple(int(v) for v in p1.polygon[0])
    assert e._hit(*corner) == (1, 0)


def test_each_player_count_keeps_its_own_layout(tmp_path):
    path = tmp_path / "seats.json"
    five = editor(players=5)
    c = tuple(int(v) for v in five.layout.seats[1].polygon[0])
    drag(five, c, (c[0] + 40, c[1]))
    five.layout.save(path)
    three = editor(players=3)
    c = tuple(int(v) for v in three.layout.seats[2].polygon[0])
    drag(three, c, (c[0], c[1] + 40))
    three.layout.save(path)

    again5 = SeatLayout.load(path, W, H, players=5)
    assert np.abs(again5.seats[1].polygon - five.layout.seats[1].polygon).max() <= 1
    again3 = SeatLayout.load(path, W, H, players=3)
    assert np.abs(again3.seats[2].polygon - three.layout.seats[2].polygon).max() <= 1
    assert SeatLayout.load(path, W, H, players=4) is None   # never saved: caller uses the template
    assert SeatLayout.load(path, W, H).players == 3         # no count given: the one saved last


def test_added_corners_survive_save_and_load(tmp_path):
    e = editor(players=2)
    mid = tuple(int(v) for v in ZoneEditor.midpoints(e.layout.seats[1].polygon)[2])
    drag(e, mid, (mid[0] + 25, mid[1] + 25))
    e.layout.save(tmp_path / "seats.json")
    loaded = SeatLayout.load(tmp_path / "seats.json", W, H)
    assert len(loaded.seats[1].polygon) == 5


def test_old_single_layout_file_still_loads(tmp_path):
    path = tmp_path / "seats.json"
    old = {"players": 4, "seats": [{"name": s.name, "polygon": (s.polygon / [W, H]).tolist()}
                                   for s in SeatLayout(4, W, H).seats]}
    path.write_text(__import__("json").dumps(old))
    loaded = SeatLayout.load(path, W, H)
    assert loaded.players == 4 and len(loaded.seats) == 5
