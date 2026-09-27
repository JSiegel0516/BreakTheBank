from cardVision.tracker import CardTracker, Detection


def zone_fn(x, y):
    return "dealer" if y < 100 else "player1"


def det(label, x, y, conf=0.9):
    return Detection(label, conf, x, y)


def test_card_needs_confirm_frames():
    tr = CardTracker(zone_fn, confirm_frames=3, drop_frames=5)
    assert tr.update([det("AS", 50, 200)]) == []
    assert tr.update([det("AS", 51, 201)]) == []
    ev = tr.update([det("AS", 50, 200)])
    assert [(e.kind, e.label, e.zone) for e in ev] == [("placed", "AS", "player1")]
    assert tr.state() == {"player1": ["AS"]}


def test_flicker_does_not_confirm():
    tr = CardTracker(zone_fn, confirm_frames=3, drop_frames=2)
    tr.update([det("AS", 50, 200)])
    tr.update([])
    tr.update([])  # dropped
    tr.update([det("AS", 50, 200)])
    tr.update([det("AS", 50, 200)])
    assert tr.state() == {}


def test_hand_passing_over_does_not_remove():
    tr = CardTracker(zone_fn, confirm_frames=2, drop_frames=10)
    for _ in range(3):
        tr.update([det("KH", 50, 200)])
    for _ in range(5):
        tr.update([])  # occluded
    assert tr.state() == {"player1": ["KH"]}
    for _ in range(10):
        tr.update([])
    assert tr.state() == {}
    assert tr.events[-1].kind == "removed"


def test_two_corners_collapse_to_one_card():
    tr = CardTracker(zone_fn, confirm_frames=2, drop_frames=5, match_dist=20)
    for _ in range(3):
        tr.update([det("7D", 50, 200), det("7D", 150, 300)])
    assert tr.state() == {"player1": ["7D"]}


def test_deck_uniqueness_keeps_higher_confidence():
    tr = CardTracker(zone_fn, confirm_frames=2, drop_frames=5)
    for _ in range(3):
        tr.update([det("QS", 50, 50, conf=0.95), det("QS", 50, 300, conf=0.4)])
    assert tr.state() == {"dealer": ["QS"]}


def test_label_vote_survives_one_misread():
    tr = CardTracker(zone_fn, confirm_frames=2, drop_frames=5)
    tr.update([det("6C", 50, 200)])
    tr.update([det("6C", 50, 200)])
    tr.update([det("8C", 50, 200)])
    assert tr.state() == {"player1": ["6C"]}


def test_backs_counted_separately():
    tr = CardTracker(zone_fn, confirm_frames=2, drop_frames=5, match_dist=20)
    for _ in range(3):
        tr.update([det("BACK", 50, 50), det("AS", 200, 50)])
    assert tr.state() == {"dealer": ["AS", "BACK"]}


def test_named_zone_beats_no_zone():
    zf = lambda x, y: "player3" if x < 100 else None
    tr = CardTracker(zf, confirm_frames=2, drop_frames=5, match_dist=20)
    for _ in range(3):  # corner inside the seat is lower confidence than the one outside
        tr.update([det("JH", 50, 50, conf=0.5), det("JH", 150, 150, conf=0.95)])
    assert tr.state() == {"player3": ["JH"]}
