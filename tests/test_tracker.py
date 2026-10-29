from cardReader.blackjack import shoe_count
from cardReader.tracker import CardTracker, Detection


def det(label, x, y, conf=0.9):
    return Detection(label, conf, x, y)


def labels(tr):
    return [c.label for c in tr.table()]


def run(tr, dets, frames):
    events = []
    for _ in range(frames):
        events += tr.update(dets)
    return events


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
        run(tr, on_table, 3)
    assert labels(tr) == ["KH", "2C", "9D", "AS"]
    assert [c.order for c in tr.table()] == [1, 2, 3, 4]


def test_two_corners_collapse_to_one_card():
    tr = CardTracker(confirm_frames=2, drop_frames=5, match_dist=20)
    run(tr, [det("7D", 50, 200), det("7D", 150, 300)], 3)
    assert labels(tr) == ["7D"]


def test_second_corner_appearing_later_is_not_a_new_card():
    tr = CardTracker(confirm_frames=2, drop_frames=5, match_dist=20)
    run(tr, [det("7D", 50, 200)], 3)
    run(tr, [det("7D", 50, 200), det("7D", 150, 300)], 3)
    assert labels(tr) == ["7D"]
    assert len(tr.round) == 1


def test_label_vote_survives_one_misread():
    tr = CardTracker(confirm_frames=2, drop_frames=5)
    tr.update([det("6C", 50, 200)])
    tr.update([det("6C", 50, 200)])
    tr.update([det("8C", 50, 200)])
    assert labels(tr) == ["6C"]


def test_each_back_is_its_own_card():
    tr = CardTracker(confirm_frames=2, drop_frames=5, match_dist=20)
    run(tr, [det("BACK", 50, 50), det("BACK", 200, 50)], 3)
    assert labels(tr) == ["BACK", "BACK"]


def test_covered_card_is_hidden_not_removed():
    tr = CardTracker(confirm_frames=2, drop_frames=3, round_end_frames=100)
    run(tr, [det("KH", 50, 200), det("5C", 300, 200)], 3)
    ev = run(tr, [det("5C", 300, 200)], 5)  # a hand covers the KH
    assert [(e.kind, e.card.label) for e in ev] == [("hidden", "KH")]
    assert labels(tr) == ["5C"]                               # not visible
    assert [(c.order, c.label, c.hidden) for c in tr.round] == [(1, "KH", True), (2, "5C", False)]


def test_card_returning_keeps_its_number_and_is_not_recounted():
    tr = CardTracker(confirm_frames=2, drop_frames=3, round_end_frames=100)
    run(tr, [det("KH", 50, 200), det("5C", 300, 200)], 3)
    run(tr, [det("5C", 300, 200)], 10)                        # KH hidden for a while
    run(tr, [det("5C", 300, 200), det("9D", 500, 200)], 3)    # a new card meanwhile
    ev = run(tr, [det("5C", 300, 200), det("9D", 500, 200), det("KH", 60, 210)], 3)
    assert [(e.kind, e.card.label, e.card.order) for e in ev] == [("returned", "KH", 1)]
    assert [(c.order, c.label) for c in tr.table()] == [(1, "KH"), (2, "5C"), (3, "9D")]
    assert tr.shoe_seen == ["KH", "5C", "9D"]                 # KH counted once


def test_face_card_returning_elsewhere_is_still_the_same_card():
    tr = CardTracker(confirm_frames=2, drop_frames=3, round_end_frames=100)
    run(tr, [det("QS", 50, 200), det("2D", 300, 200)], 3)
    run(tr, [det("2D", 300, 200)], 5)
    ev = run(tr, [det("2D", 300, 200), det("QS", 900, 600)], 3)  # picked up and put down far away
    assert [(e.kind, e.card.order) for e in ev] == [("returned", 1)]
    assert len(tr.round) == 2


def test_face_down_card_returns_near_where_it_was():
    tr = CardTracker(confirm_frames=2, drop_frames=3, match_dist=20, round_end_frames=100)
    run(tr, [det("BACK", 100, 100), det("7H", 400, 100)], 3)
    run(tr, [det("7H", 400, 100)], 5)
    ev = run(tr, [det("7H", 400, 100), det("BACK", 110, 105)], 3)
    assert [(e.kind, e.card.order) for e in ev] == [("returned", 1)]
    assert [c.label for c in tr.round] == ["BACK", "7H"]


def test_face_down_card_far_away_is_a_new_card():
    tr = CardTracker(confirm_frames=2, drop_frames=3, match_dist=20, round_end_frames=100)
    run(tr, [det("BACK", 100, 100), det("7H", 400, 100)], 3)
    run(tr, [det("7H", 400, 100)], 5)
    ev = run(tr, [det("7H", 400, 100), det("BACK", 800, 500)], 3)
    assert [(e.kind, e.card.order) for e in ev] == [("placed", 3)]


def test_round_ends_when_table_stays_empty():
    tr = CardTracker(confirm_frames=2, drop_frames=3, round_end_frames=10)
    run(tr, [det("KH", 50, 200), det("4S", 300, 200)], 3)
    ev = run(tr, [], 3 + 10)  # swept: hidden after 3 frames, round over 10 frames later
    assert [e.kind for e in ev][-1] == "round_over"
    assert tr.round == []
    run(tr, [det("8C", 50, 200)], 3)
    assert [(c.order, c.label) for c in tr.table()] == [(1, "8C")]  # numbering restarts


def test_short_cover_does_not_end_the_round():
    tr = CardTracker(confirm_frames=2, drop_frames=3, round_end_frames=10)
    run(tr, [det("KH", 50, 200)], 3)
    run(tr, [], 8)                      # whole table covered, but not long enough
    ev = run(tr, [det("KH", 50, 200)], 3)
    assert [e.kind for e in ev] == ["returned"]
    assert [(c.order, c.label) for c in tr.round] == [(1, "KH")]


def test_card_seen_in_a_later_round_is_not_counted_again():
    tr = CardTracker(confirm_frames=2, drop_frames=3, round_end_frames=10)
    run(tr, [det("KH", 50, 200)], 3)
    run(tr, [], 20)                     # round over
    run(tr, [det("KH", 50, 200), det("3D", 300, 200)], 3)
    assert [(c.order, c.label) for c in tr.table()] == [(1, "KH"), (2, "3D")]
    assert tr.shoe_seen == ["KH", "3D"]  # one deck: the KH is the same physical card


def test_card_seen_in_a_later_round_is_reported_as_a_duplicate():
    tr = CardTracker(confirm_frames=2, drop_frames=3, round_end_frames=10)
    run(tr, [det("KH", 50, 200), det("3D", 300, 200)], 3)
    run(tr, [], 20)                     # round over
    ev = run(tr, [det("KH", 400, 200)], 3)
    assert [(e.kind, e.card.order, e.card.label) for e in ev] == [("placed", 1, "KH"), ("duplicate", 1, "KH")]
    assert tr.shoe_seen == ["KH", "3D"]
    assert shoe_count(tr.shoe_seen).running == 0


def test_rename_to_a_name_counted_in_an_earlier_round_is_a_duplicate():
    tr = CardTracker(confirm_frames=2, drop_frames=3, round_end_frames=5, label_window=10)
    run(tr, [det("KH", 50, 200)], 3)
    run(tr, [], 10)                     # round over
    run(tr, [det("BACK", 50, 200)], 3)
    ev = run(tr, [det("KH", 50, 200)], 10)  # the face-down card is turned over: a KH again
    assert [(e.kind, e.card.order, e.card.label) for e in ev] == [("renamed", 1, "KH"), ("duplicate", 1, "KH")]
    assert tr.shoe_seen == ["KH"]


def test_two_corners_of_one_card_are_not_a_duplicate():
    tr = CardTracker(confirm_frames=2, drop_frames=5, match_dist=20)
    ev = run(tr, [det("7D", 50, 200)], 3)
    ev += run(tr, [det("7D", 50, 200), det("7D", 150, 300)], 3)
    assert [e.kind for e in ev] == ["placed"]


def test_card_still_down_through_a_new_round_is_not_a_duplicate():
    tr = CardTracker(confirm_frames=2, drop_frames=5, match_dist=20)
    run(tr, [det("QS", 50, 50)], 3)
    tr.new_round()
    ev = run(tr, [det("QS", 50, 50)], 1)
    assert [e.kind for e in ev] == ["placed"]


def test_new_shoe_forgets_earlier_names():
    tr = CardTracker(confirm_frames=2, drop_frames=5)
    run(tr, [det("QS", 50, 50)], 3)
    run(tr, [], 10)
    tr.new_shoe()
    ev = run(tr, [det("QS", 50, 50)], 3)
    assert [e.kind for e in ev] == ["placed"]


def test_new_round_restarts_numbering_but_keeps_the_count():
    tr = CardTracker(confirm_frames=2, drop_frames=5)
    run(tr, [det("QS", 50, 50)], 3)
    run(tr, [], 5)                      # swept away
    tr.new_round()
    run(tr, [det("3H", 300, 50)], 3)
    assert [(c.order, c.label) for c in tr.round] == [(1, "3H")]
    assert tr.shoe_seen == ["QS", "3H"]


def test_new_shoe_resets_the_count():
    tr = CardTracker(confirm_frames=2, drop_frames=5)
    run(tr, [det("QS", 50, 50)], 3)
    tr.new_shoe()
    assert tr.shoe_seen == [] and tr.round == []
    run(tr, [det("QS", 50, 50)], 3)
    assert tr.shoe_seen == ["QS"]


def test_new_round_with_cards_still_down_renumbers_them():
    tr = CardTracker(confirm_frames=2, drop_frames=5, match_dist=20)
    run(tr, [det("QS", 50, 50), det("4H", 300, 50)], 3)
    tr.new_round()
    run(tr, [det("QS", 50, 50), det("4H", 300, 50)], 1)
    assert [(c.order, c.label) for c in tr.table()] == [(1, "QS"), (2, "4H")]
    assert tr.shoe_seen == ["QS", "4H"]  # not counted again


def test_face_down_cards_are_not_counted():
    tr = CardTracker(confirm_frames=2, drop_frames=5, match_dist=20)
    run(tr, [det("BACK", 50, 50), det("9S", 300, 50)], 3)
    assert tr.shoe_seen == ["9S"]


def test_flickering_detection_is_not_confirmed_when_seen_steadily_is_required():
    tr = CardTracker(confirm_frames=6, drop_frames=30, confirm_window=8)
    for _ in range(12):                 # seen every other frame: 12 sightings over 24 frames
        tr.update([det("KS", 50, 200)])
        tr.update([])
    assert tr.round == []
    run(tr, [det("KS", 50, 200)], 6)    # now seen steadily
    assert labels(tr) == ["KS"]


def test_disputed_name_waits_for_agreement():
    tr = CardTracker(confirm_frames=4, drop_frames=30, confirm_agreement=0.8)
    for label in ["6H", "9H"] * 4:      # 50/50 split
        tr.update([det(label, 50, 200)])
    assert tr.round == []
    run(tr, [det("6H", 50, 200)], 12)   # 16 of 20 reads say 6H
    assert labels(tr) == ["6H"]


def test_reads_split_between_suits_of_one_rank_agree():
    tr = CardTracker(confirm_frames=4, drop_frames=30, confirm_agreement=0.8)
    for label in ["10H", "10D", "10H"] * 2:   # suits disagree, the rank does not
        tr.update([det(label, 50, 200)])
    assert labels(tr) == ["10H"]              # shown under its most common name


def test_reads_split_between_ranks_of_one_suit_wait():
    tr = CardTracker(confirm_frames=4, drop_frames=30, confirm_agreement=0.8)
    for label in ["6H", "9H", "6H"] * 2:
        tr.update([det(label, 50, 200)])
    assert tr.round == []


def test_low_average_confidence_is_not_confirmed():
    tr = CardTracker(confirm_frames=4, drop_frames=30, confirm_window=4, confirm_confidence=0.7)
    run(tr, [det("QD", 50, 200, conf=0.55)], 10)
    assert tr.round == []
    run(tr, [det("QD", 50, 200, conf=0.9)], 4)
    assert labels(tr) == ["QD"]


def test_moving_card_is_confirmed_only_once_it_stops():
    tr = CardTracker(confirm_frames=5, drop_frames=30, match_dist=40, confirm_window=5, still_distance=6)
    for i in range(12):                 # slid 15 px a frame across the table
        tr.update([det("8C", 50 + 15 * i, 200)])
    assert tr.round == []
    run(tr, [det("8C", 50 + 15 * 12, 200)], 5)
    assert labels(tr) == ["8C"]
    assert abs(tr.round[0].x - (50 + 15 * 12)) < 20  # counted where it landed


def test_small_jitter_still_counts_as_held_still():
    tr = CardTracker(confirm_frames=5, drop_frames=30, confirm_window=5, still_distance=6)
    for dx, dy in [(0, 0), (2, -1), (-2, 1), (1, 2), (-1, -2)]:
        tr.update([det("2H", 300 + dx, 200 + dy)])
    assert labels(tr) == ["2H"]


def test_hole_card_turned_face_up_is_renamed_not_added():
    tr = CardTracker(confirm_frames=2, drop_frames=5, match_dist=20, label_window=10)
    run(tr, [det("BACK", 100, 100), det("7H", 400, 100)], 6)   # dealer: hole card and up card
    assert [(c.order, c.label) for c in tr.round] == [(1, "BACK"), (2, "7H")]
    ev = run(tr, [det("KS", 100, 100), det("7H", 400, 100)], 6)  # the hole card is turned over
    assert [(e.kind, e.card.order, e.card.label, e.was) for e in ev] == [("renamed", 1, "KS", "BACK")]
    assert [(c.order, c.label) for c in tr.round] == [(1, "KS"), (2, "7H")]
    assert tr.shoe_seen == ["7H", "KS"]


def test_corrected_misread_fixes_the_count_and_keeps_the_number():
    tr = CardTracker(confirm_frames=2, drop_frames=5, match_dist=20, label_window=10)
    run(tr, [det("9H", 50, 200)], 3)                     # confirmed under the wrong name
    run(tr, [det("9H", 50, 200), det("2C", 300, 200)], 3)
    assert tr.shoe_seen == ["9H", "2C"]
    ev = run(tr, [det("6H", 50, 200), det("2C", 300, 200)], 6)
    assert [(e.kind, e.card.order, e.was) for e in ev] == [("renamed", 1, "9H")]
    assert [(c.order, c.label) for c in tr.round] == [(1, "6H"), (2, "2C")]
    assert tr.shoe_seen == ["6H", "2C"]                  # the 9H was never on the table; order kept
    assert len(tr.round) == 2                            # no phantom second card


def test_rename_does_not_uncount_a_card_seen_in_an_earlier_round():
    tr = CardTracker(confirm_frames=2, drop_frames=3, round_end_frames=5, label_window=10)
    run(tr, [det("KH", 50, 200)], 3)
    run(tr, [], 10)                                       # round over; KH stays counted
    run(tr, [det("KH", 50, 200)], 3)                     # next round: a card misread as the KH
    run(tr, [det("4D", 50, 200)], 6)                     # it was really a 4D
    assert [c.label for c in tr.round] == ["4D"]
    assert tr.shoe_seen == ["KH", "4D"]                  # the real KH from round 1 is still counted


def test_two_corners_disagreeing_do_not_rename_until_the_vote_flips():
    tr = CardTracker(confirm_frames=2, drop_frames=5, match_dist=20, label_window=10)
    run(tr, [det("9H", 50, 200), det("9H", 150, 300)], 3)   # both corners
    assert len(tr.round) == 1
    ev = run(tr, [det("6H", 50, 200), det("9H", 150, 300)], 4)  # one corner changes its mind
    assert ev == [] and tr.round[0].label == "9H"         # 4 of 10 reads: not yet
    ev = run(tr, [det("6H", 50, 200), det("6H", 150, 300)], 6)
    assert [e.kind for e in ev] == ["renamed"] and tr.round[0].label == "6H"
    assert len(tr.round) == 1


def test_without_a_label_window_the_vote_is_over_every_read():
    tr = CardTracker(confirm_frames=2, drop_frames=5)
    run(tr, [det("BACK", 100, 100)], 20)
    run(tr, [det("KS", 100, 100)], 15)                   # 15 of 35 reads: BACK still wins
    assert tr.round[0].label == "BACK"


def real_tracker():
    """The thresholds read.py uses, for a 1920-pixel-wide frame (match distance about 58 px)."""
    import settings
    from cardReader.tracker import TrackerConfig
    return CardTracker(TrackerConfig.from_settings(settings, 1920))


def corners(label, cx, cy):
    """The two corner indices of a face-up card centred on (cx, cy): about half the diagonal
    of a 110 x 160 px card away from its centre, where a BACK box is centred."""
    return [det(label, cx - 45, cy - 80), det(label, cx + 45, cy + 80)]


def test_hole_card_turned_face_up_with_real_geometry_keeps_its_number():
    tr = real_tracker()
    run(tr, [det("BACK", 900, 300), *corners("7H", 1200, 300)], 30)   # dealer: hole card and up card
    assert [(c.order, c.label) for c in tr.round] == [(1, "BACK"), (2, "7H")]
    ev = run(tr, [*corners("KS", 900, 300), *corners("7H", 1200, 300)], 120)  # hole card turned over
    assert [(e.kind, e.card.order, e.was) for e in ev] == [("renamed", 1, "BACK")]
    assert [(c.order, c.label, c.hidden) for c in tr.round] == [(1, "KS", False), (2, "7H", False)]
    assert tr.shoe_seen == ["7H", "KS"] and tr.shoe_unrevealed == 0


def test_neighbouring_face_down_cards_revealed_in_turn_keep_their_own_numbers():
    tr = real_tracker()
    run(tr, [det("BACK", 800, 300), det("BACK", 930, 300)], 30)        # two cards side by side
    run(tr, [*corners("QH", 800, 300), det("BACK", 930, 300)], 20)     # the left one is turned over
    run(tr, [*corners("QH", 800, 300), *corners("4C", 930, 300)], 60)  # then the right one
    assert [(c.order, c.label, c.hidden) for c in tr.round] == [(1, "QH", False), (2, "4C", False)]
    assert tr.shoe_seen == ["QH", "4C"]


def test_label_stays_counted_while_another_card_still_holds_it():
    tr = CardTracker(confirm_frames=2, drop_frames=5, match_dist=20, label_window=10)
    run(tr, [det("9H", 50, 200), det("6H", 300, 200)], 3)
    run(tr, [det("9H", 50, 200), det("9H", 300, 200)], 8)   # the second card is renamed to 9H too
    run(tr, [det("5C", 50, 200), det("9H", 300, 200)], 8)   # the first card was really a 5C
    assert [(c.order, c.label) for c in tr.round] == [(1, "5C"), (2, "9H")]
    assert tr.shoe_seen == ["9H", "5C"]                     # 9H is still on the table


def test_face_down_card_swept_without_a_reveal_is_remembered_until_the_next_shoe():
    tr = CardTracker(confirm_frames=2, drop_frames=3, round_end_frames=5)
    run(tr, [det("BACK", 100, 100), det("9S", 300, 100)], 3)
    assert tr.shoe_unrevealed == 1
    run(tr, [], 10)                                          # swept face down: round over
    assert tr.round == [] and tr.shoe_unrevealed == 1
    tr.new_shoe()
    assert tr.shoe_unrevealed == 0


def test_face_down_card_still_down_through_a_new_round_is_dealt_only_once():
    tr = CardTracker(confirm_frames=2, drop_frames=5, match_dist=20, label_window=10)
    run(tr, [det("BACK", 100, 100)], 3)
    assert tr.shoe_unrevealed == 1
    tr.new_round()                                       # key r while the card is still face down
    run(tr, [det("BACK", 100, 100)], 3)
    assert [(c.order, c.label) for c in tr.round] == [(1, "BACK")]
    assert tr.shoe_unrevealed == 1                       # the same card, not a second one dealt
    run(tr, [det("KS", 100, 100)], 10)                   # then it is turned over
    assert [(c.order, c.label) for c in tr.round] == [(1, "KS")]
    assert tr.shoe_unrevealed == 0 and tr.shoe_seen == ["KS"]


def test_config_object_and_keyword_overrides_agree():
    from cardReader.tracker import TrackerConfig
    a = CardTracker(TrackerConfig(confirm_frames=3, match_dist=10), drop_frames=7)
    assert (a.cfg.confirm_frames, a.cfg.match_dist, a.cfg.drop_frames) == (3, 10, 7)

    class S:  # the fields settings.py provides
        CONFIRM_FRAMES, CONFIRM_WINDOW, CONFIRM_AGREEMENT, CONFIRM_CONFIDENCE = 15, 20, 0.8, 0.7
        STILL_DISTANCE, DROP_FRAMES, MATCH_DISTANCE, ROUND_END_FRAMES, LABEL_WINDOW = 0.01, 30, 0.05, 90, 45
    cfg = TrackerConfig.from_settings(S, 2000)
    assert (cfg.still_distance, cfg.match_dist, cfg.label_window) == (20.0, 100.0, 45)
