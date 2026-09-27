from cardReader.tracker import CardTracker, Detection


def det(label, x, y, conf=0.9):
    return Detection(label, conf, x, y)


def labels(tr):
    return [c.label for c in tr.table()]


def test_card_needs_confirm_frames():
    tr = CardTracker(confirm_frames=3, drop_frames=5)
    assert tr.update([det("AS", 50, 200)]) == []
    assert tr.update([det("AS", 51, 201)]) == []
    ev = tr.update([det("AS", 50, 200)])
    assert [(e.kind, e.card.label, e.card.order) for e in ev] == [("placed", "AS", 1)]
    assert labels(tr) == ["AS"]


def test_flicker_does_not_confirm():
    tr = CardTracker(confirm_frames=3, drop_frames=2)
    tr.update([det("AS", 50, 200)])
    tr.update([])
    tr.update([])  # dropped
    tr.update([det("AS", 50, 200)])
    tr.update([det("AS", 50, 200)])
    assert tr.table() == []


def test_placement_order_is_kept():
    tr = CardTracker(confirm_frames=2, drop_frames=5, match_dist=20)
    on_table = []
    for label, x in [("KH", 300), ("2C", 100), ("9D", 500), ("AS", 50)]:
        on_table.append(det(label, x, 200))
        for _ in range(3):
            tr.update(on_table)
    assert labels(tr) == ["KH", "2C", "9D", "AS"]
    assert [c.order for c in tr.table()] == [1, 2, 3, 4]


def test_hand_passing_over_does_not_remove():
    tr = CardTracker(confirm_frames=2, drop_frames=10)
    for _ in range(3):
        tr.update([det("KH", 50, 200)])
    for _ in range(5):
        tr.update([])  # occluded
    assert labels(tr) == ["KH"]
    removed = []
    for _ in range(10):
        removed += tr.update([])
    assert tr.table() == []
    assert [(e.kind, e.card.label) for e in removed] == [("removed", "KH")]
    assert tr.history[0].removed_frame is not None


def test_two_corners_collapse_to_one_card():
    tr = CardTracker(confirm_frames=2, drop_frames=5, match_dist=20)
    for _ in range(3):
        tr.update([det("7D", 50, 200), det("7D", 150, 300)])
    assert labels(tr) == ["7D"]


def test_second_corner_appearing_later_is_not_a_new_card():
    tr = CardTracker(confirm_frames=2, drop_frames=5, match_dist=20)
    for _ in range(3):
        tr.update([det("7D", 50, 200)])
    for _ in range(3):
        tr.update([det("7D", 50, 200), det("7D", 150, 300)])
    assert labels(tr) == ["7D"]
    assert len(tr.history) == 1


def test_label_vote_survives_one_misread():
    tr = CardTracker(confirm_frames=2, drop_frames=5)
    tr.update([det("6C", 50, 200)])
    tr.update([det("6C", 50, 200)])
    tr.update([det("8C", 50, 200)])
    assert labels(tr) == ["6C"]


def test_each_back_is_its_own_card():
    tr = CardTracker(confirm_frames=2, drop_frames=5, match_dist=20)
    for _ in range(3):
        tr.update([det("BACK", 50, 50), det("BACK", 200, 50)])
    assert labels(tr) == ["BACK", "BACK"]


def test_reset_starts_a_new_round():
    tr = CardTracker(confirm_frames=2, drop_frames=5)
    for _ in range(3):
        tr.update([det("QS", 50, 50)])
    tr.reset()
    for _ in range(3):
        tr.update([det("3H", 50, 50)])
    assert [(c.order, c.label) for c in tr.table()] == [(1, "3H")]
