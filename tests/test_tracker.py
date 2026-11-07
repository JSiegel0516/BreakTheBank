import numpy as np
import pytest

from cardReader import camera
from cardReader.blackjack import shoe_count
from cardReader.cards import Event
from cardReader.deck import BACK
from cardReader.tracker import CardTracker, Detection, TrackerConfig

FRAME = 1 / TrackerConfig.nominal_fps  # seconds the clock moves on an update() given no time: one nominal frame


def det(label, x, y, conf=0.9):
    return Detection(label, conf, x, y)


def labels(tr):
    return [c.label for c in tr.table()]


def run(tr, dets, frames):
    events = []
    for _ in range(frames):
        events += tr.update(dets)
    return events


def run_timed(tr, dets, frames, interval):
    """Like run, on a camera sending a frame every `interval` seconds."""
    events = []
    for _ in range(frames):
        events += tr.update(dets, now=tr.now + interval)
    return events


@pytest.mark.parametrize(("interval", "frames_to_confirm"), [(FRAME, 7), (2 * FRAME, 4), (FRAME / 2, 13)])
def test_a_card_is_confirmed_once_seen_for_the_set_seconds_at_any_frame_rate(interval, frames_to_confirm):
    tr = CardTracker(confirm_seconds=0.5, confirm_window_seconds=0.7)
    run_timed(tr, [], 2, interval)                       # the camera was already running
    assert run_timed(tr, [det("KH", 50, 200)], frames_to_confirm - 1, interval) == []
    ev = run_timed(tr, [det("KH", 50, 200)], 1, interval)
    assert [(e.kind, e.card.label) for e in ev] == [("placed", "KH")]
    assert (frames_to_confirm - 1) * interval < 0.5 <= frames_to_confirm * interval  # never early, a frame late at most


@pytest.mark.parametrize(("confirm_seconds", "interval", "updates_to_confirm"),
                         [(0.5, FRAME / 2, 14), (0.01, 0.001, 11)])
def test_with_times_given_a_card_seen_from_the_first_frame_is_not_confirmed_early(confirm_seconds, interval,
                                                                                   updates_to_confirm):
    tr = CardTracker(confirm_seconds=confirm_seconds, confirm_window_seconds=0.7)
    tr.update([det("KH", 50, 200)], now=100.0)           # the first frame has no frame before it: it stands for no time
    assert run_timed(tr, [det("KH", 50, 200)], updates_to_confirm - 2, interval) == []
    assert [e.kind for e in run_timed(tr, [det("KH", 50, 200)], 1, interval)] == ["placed"]
    assert (updates_to_confirm - 2) * interval < confirm_seconds <= (updates_to_confirm - 1) * interval


def test_with_times_given_a_short_time_is_not_rounded_away():
    tr = CardTracker(confirm_seconds=0.01, drop_seconds=1, round_end_seconds=10, hidden_seconds=0.01)
    run_timed(tr, [], 2, 0.001)
    assert run_timed(tr, [det("KH", 50, 200)], 9, 0.001) == []          # 9 ms seen of 10
    assert [e.kind for e in run_timed(tr, [det("KH", 50, 200)], 1, 0.001)] == ["placed"]
    assert [e.kind for e in run_timed(tr, [], 1, 1.0)] == ["hidden"]
    assert run_timed(tr, [], 9, 0.001) == []                             # hidden 9 ms of 10
    assert [e.kind for e in run_timed(tr, [], 1, 0.001)] == ["removed"]


def test_two_decks_with_times_given_a_short_copy_hold_still_waits():
    tr = CardTracker(confirm_seconds=0.001, drop_seconds=1, match_dist=20, decks=2, copy_hold_seconds=0.01)
    first = [det("KH", 50, 200), det("KH", 80, 240)]    # both corners of one card, diagonal as corners are
    hand = [*first, det("KH", 110, 200)]                 # and a second KH beside it
    run_timed(tr, [], 2, 0.001)
    assert [e.kind for e in run_timed(tr, first, 1, 0.001)] == ["placed"]
    assert [e.kind for e in run_timed(tr, hand, 1, 0.001)] == ["placed", "held"]  # seen together 1 ms of 10
    assert run_timed(tr, hand, 8, 0.001) == []
    ev = run_timed(tr, hand, 1, 0.001)
    assert [e.kind for e in ev] == ["counted"] and tr.shoe_seen == ["KH", "KH"]


def test_a_fast_camera_keeps_every_sighting_a_long_confirm_window_needs():
    tr = CardTracker(confirm_seconds=5, confirm_window_seconds=6)
    run_timed(tr, [], 2, 1 / 60)
    assert run_timed(tr, [det("KH", 50, 200)], 299, 1 / 60) == []      # far more than 200 sightings
    assert [e.kind for e in run_timed(tr, [det("KH", 50, 200)], 1, 1 / 60)] == ["placed"]  # 5 s at 60 fps


def test_a_fast_camera_votes_over_the_whole_label_window():
    tr = CardTracker(confirm_seconds=0.1, drop_seconds=1, label_window_seconds=5)
    run_timed(tr, [], 2, 1 / 60)
    run_timed(tr, [det("BACK", 100, 100)], 300, 1 / 60)
    run_timed(tr, [det("KS", 100, 100)], 140, 1 / 60)
    assert labels(tr) == ["BACK"]                         # the last 5 s: 160 BACK reads, 140 KS
    run_timed(tr, [det("KS", 100, 100)], 30, 1 / 60)
    assert labels(tr) == ["KS"]                           # 130 BACK, 170 KS


def test_without_a_confirm_window_every_sighting_is_kept_until_the_track_is_confirmed():
    tr = CardTracker(confirm_seconds=100 * FRAME, confirm_confidence=0.7, label_window_seconds=10 * FRAME)
    run(tr, [det("QD", 50, 200, conf=0.5)], 250)
    assert len(tr.tracks[1].hits) == 250                 # unconfirmed: the average needs every read
    run(tr, [det("QD", 50, 200, conf=0.99)], 300)
    assert labels(tr) == ["QD"]
    assert len(tr.tracks[1].hits) == 10                  # confirmed: only the label window is kept


def test_a_slow_camera_hides_a_card_and_ends_the_round_after_the_set_seconds():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=4 * FRAME, round_end_seconds=10 * FRAME)
    run_timed(tr, [], 2, 2 * FRAME)
    run_timed(tr, [det("KH", 50, 200)], 1, 2 * FRAME)
    assert labels(tr) == ["KH"]
    ev = run_timed(tr, [], 6, 2 * FRAME)                 # at half the nominal rate: half the frames
    # hidden 4 frames' time after its last sighting, the round over 10 frames' time after the card was last there
    assert [(e.kind, e.frame) for e in ev] == [("hidden", tr.frame - 4), ("round_over", tr.frame)]


def test_without_a_time_each_update_is_one_nominal_frame():
    tr = CardTracker()
    run(tr, [det("KH", 50, 200)], 3)
    assert tr.now == pytest.approx(3 * FRAME) and tr.frame == 3


def test_card_needs_confirm_seconds():
    tr = CardTracker(confirm_seconds=3 * FRAME, drop_seconds=5 * FRAME)
    assert tr.update([det("AS", 50, 200)]) == []
    assert tr.update([det("AS", 51, 201)]) == []
    ev = tr.update([det("AS", 50, 200)])
    assert [(e.kind, e.card.label, e.card.order) for e in ev] == [("placed", "AS", 1)]
    assert labels(tr) == ["AS"]


def test_flicker_does_not_confirm():
    tr = CardTracker(confirm_seconds=3 * FRAME, drop_seconds=2 * FRAME)
    tr.update([det("AS", 50, 200)])
    tr.update([])
    tr.update([])  # dropped
    tr.update([det("AS", 50, 200)])
    tr.update([det("AS", 50, 200)])
    assert tr.table() == []


def test_placement_order_is_kept():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME, match_dist=20)
    on_table = []
    for label, x in [("KH", 300), ("2C", 100), ("9D", 500), ("AS", 50)]:
        on_table.append(det(label, x, 200))
        run(tr, on_table, 3)
    assert labels(tr) == ["KH", "2C", "9D", "AS"]
    assert [c.order for c in tr.table()] == [1, 2, 3, 4]


def test_two_corners_collapse_to_one_card():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME, match_dist=20)
    run(tr, [det("7D", 50, 200), det("7D", 150, 300)], 3)
    assert labels(tr) == ["7D"]


def test_second_corner_appearing_later_is_not_a_new_card():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME, match_dist=20)
    run(tr, [det("7D", 50, 200)], 3)
    run(tr, [det("7D", 50, 200), det("7D", 150, 300)], 3)
    assert labels(tr) == ["7D"]
    assert len(tr.round) == 1


def test_label_vote_survives_one_misread():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME)
    tr.update([det("6C", 50, 200)])
    tr.update([det("6C", 50, 200)])
    tr.update([det("8C", 50, 200)])
    assert labels(tr) == ["6C"]


def test_each_back_is_its_own_card():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME, match_dist=20)
    run(tr, [det("BACK", 50, 50), det("BACK", 200, 50)], 3)
    assert labels(tr) == ["BACK", "BACK"]


def test_covered_card_is_hidden_not_removed():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=3 * FRAME, round_end_seconds=100 * FRAME)
    run(tr, [det("KH", 50, 200), det("5C", 300, 200)], 3)
    ev = run(tr, [det("5C", 300, 200)], 5)  # a hand covers the KH
    assert [(e.kind, e.card.label) for e in ev] == [("hidden", "KH")]
    assert labels(tr) == ["5C"]                               # not visible
    assert [(c.order, c.label, c.hidden) for c in tr.round] == [(1, "KH", True), (2, "5C", False)]


def weak_tracker(**overrides):
    """A tracker that starts tracks only from boxes scoring 0.6 or more."""
    return CardTracker(start_confidence=0.6, confirm_seconds=3 * FRAME, drop_seconds=3 * FRAME,
                       round_end_seconds=100 * FRAME, **overrides)


def test_weak_boxes_keep_a_track_started_strong_from_missing_or_being_dropped():
    tr = weak_tracker()
    run(tr, [det("KH", 50, 200, conf=0.7)], 3)
    card = tr.round[0]
    ev = run(tr, [det("KH", 56, 206, conf=0.5)], 20)        # far longer than drop_seconds, all weak
    assert ev == [] and not card.hidden
    assert [(t.missed, round(t.cx), round(t.cy)) for t in tr.tracks.values()] == [(0, 56, 206)]  # moved with them
    assert tr.bound == [card]
    assert [e.kind for e in run(tr, [], 3)] == ["hidden"]   # dropped drop_seconds after the last weak sighting


def test_a_weak_box_with_no_track_in_reach_starts_none():
    tr = weak_tracker(match_dist=20)
    run(tr, [det("KH", 50, 200, conf=0.7)], 3)
    ev = run(tr, [det("KH", 50, 200, conf=0.7), det("9D", 300, 200, conf=0.5)], 20)
    assert ev == [] and labels(tr) == ["KH"] and len(tr.tracks) == 1
    assert tr.bound == [tr.round[0], None]


def test_a_weak_box_beside_a_track_a_strong_box_hit_starts_none():
    tr = weak_tracker()
    run(tr, [det("KH", 50, 200, conf=0.7)], 3)
    run(tr, [det("KH", 50, 200, conf=0.7), det("KH", 55, 200, conf=0.5)], 20)
    assert len(tr.tracks) == 1 and tr.bound == [tr.round[0], None]


def test_weak_sightings_add_nothing_to_confirmation():
    tr = weak_tracker()
    tr.update([det("KH", 50, 200, conf=0.9)])
    assert run(tr, [det("KH", 50, 200, conf=0.5)], 20) == []   # kept, but never confirmed by weak boxes alone
    assert tr.update([det("KH", 50, 200, conf=0.9)]) == []
    ev = tr.update([det("KH", 50, 200, conf=0.9)])             # the third strong sighting, as with no weak boxes
    assert [(e.kind, e.card.label) for e in ev] == [("placed", "KH")]


def test_confirm_confidence_averages_only_the_strong_sightings():
    tr = weak_tracker(confirm_confidence=0.7)
    for _ in range(10):
        tr.update([det("KH", 50, 200, conf=0.65)])               # strong enough to track, too weak to confirm
        tr.update([det("KH", 50, 200, conf=0.4)])
    assert tr.round == []
    tr = weak_tracker(confirm_confidence=0.7)
    for _ in range(3):
        tr.update([det("KH", 50, 200, conf=0.75)])
        tr.update([det("KH", 50, 200, conf=0.4)])                # would pull the average below 0.7 if counted
    assert labels(tr) == ["KH"]


def test_weak_reads_do_not_sway_a_cards_name():
    tr = weak_tracker(label_window_seconds=5 * FRAME)
    run(tr, [det("6H", 50, 200, conf=0.9)], 3)
    ev = run(tr, [det("9H", 50, 200, conf=0.5)], 20)
    assert ev == [] and labels(tr) == ["6H"]


def test_card_returning_keeps_its_number_and_is_not_recounted():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=3 * FRAME, round_end_seconds=100 * FRAME)
    run(tr, [det("KH", 50, 200), det("5C", 300, 200)], 3)
    run(tr, [det("5C", 300, 200)], 10)                        # KH hidden for a while
    run(tr, [det("5C", 300, 200), det("9D", 500, 200)], 3)    # a new card meanwhile
    ev = run(tr, [det("5C", 300, 200), det("9D", 500, 200), det("KH", 60, 210)], 3)
    assert [(e.kind, e.card.label, e.card.order) for e in ev] == [("returned", "KH", 1)]
    assert [(c.order, c.label) for c in tr.table()] == [(1, "KH"), (2, "5C"), (3, "9D")]
    assert tr.shoe_seen == ["KH", "5C", "9D"]                 # KH counted once


def test_card_hidden_too_long_is_removed_but_stays_counted():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=3 * FRAME, round_end_seconds=100 * FRAME,
                     hidden_seconds=5 * FRAME)
    run(tr, [det("KH", 50, 200), det("5C", 300, 200)], 3)
    ev = run(tr, [det("5C", 300, 200)], 3 + 4)               # hidden after 3 misses, 4 frames short of expiry
    assert [e.kind for e in ev] == ["hidden"]
    ev = tr.update([det("5C", 300, 200)])
    assert [(e.kind, e.card.order, e.card.label) for e in ev] == [("removed", 1, "KH")]
    assert [(c.order, c.label) for c in tr.round] == [(2, "5C")]
    assert tr.shoe_seen == ["KH", "5C"] and shoe_count(tr.shoe_seen).running == 0


def test_card_returning_before_the_expiry_keeps_its_number():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=3 * FRAME, round_end_seconds=100 * FRAME,
                     hidden_seconds=8 * FRAME)
    run(tr, [det("KH", 50, 200), det("5C", 300, 200)], 3)
    run(tr, [det("5C", 300, 200)], 3 + 6)
    ev = run(tr, [det("5C", 300, 200), det("KH", 50, 200)], 3)  # back on the expiry frame
    assert [(e.kind, e.card.order) for e in ev] == [("returned", 1)]


def test_card_placed_after_a_removal_gets_a_fresh_number():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=3 * FRAME, round_end_seconds=100 * FRAME,
                     hidden_seconds=2 * FRAME)
    run(tr, [det("KH", 50, 200), det("5C", 300, 200)], 3)
    run(tr, [det("5C", 300, 200)], 10)                       # KH removed
    ev = run(tr, [det("5C", 300, 200), det("9D", 500, 200)], 3)
    assert [(e.kind, e.card.order) for e in ev] == [("placed", 3)]
    assert [c.order for c in tr.round] == [2, 3]


def test_card_that_comes_back_after_removal_is_placed_again_but_not_recounted():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=3 * FRAME, round_end_seconds=100 * FRAME,
                     hidden_seconds=2 * FRAME)
    run(tr, [det("KH", 50, 200), det("5C", 300, 200)], 3)
    run(tr, [det("5C", 300, 200)], 10)                       # KH removed
    ev = run(tr, [det("5C", 300, 200), det("KH", 50, 200)], 3)
    assert [(e.kind, e.card.order, e.card.label) for e in ev] == [("placed", 3, "KH")]
    assert tr.shoe_seen == ["KH", "5C"]


def test_hidden_seconds_zero_removes_a_card_as_soon_as_it_is_hidden():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=3 * FRAME, round_end_seconds=100 * FRAME, hidden_seconds=0)
    run(tr, [det("KH", 50, 200), det("5C", 300, 200)], 3)
    ev = run(tr, [det("5C", 300, 200)], 3)
    assert [(e.kind, e.frame) for e in ev] == [("hidden", tr.frame), ("removed", tr.frame)]
    assert [c.label for c in tr.round] == ["5C"]


def test_round_ends_after_every_card_was_removed():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=3 * FRAME, round_end_seconds=10 * FRAME,
                     hidden_seconds=2 * FRAME)
    run(tr, [det("KH", 50, 200), det("4S", 300, 200)], 3)
    ev = run(tr, [], 3 + 9)                 # hidden after 3 frames, removed 2 later, round over 9 after hidden
    assert [e.kind for e in ev] == ["hidden", "hidden", "removed", "removed", "round_over"]
    assert tr.round == []
    run(tr, [det("8C", 50, 200)], 3)
    assert [(c.order, c.label) for c in tr.table()] == [(1, "8C")]  # numbering restarts


def test_removed_face_down_card_stays_off_cards_left():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=3 * FRAME, round_end_seconds=100 * FRAME,
                     hidden_seconds=2 * FRAME)
    run(tr, [det("BACK", 100, 100), det("9S", 300, 100)], 3)
    run(tr, [det("9S", 300, 100)], 10)
    assert [c.label for c in tr.round] == ["9S"]
    assert tr.shoe_unrevealed == 1


def test_removed_card_hands_its_count_to_a_card_sharing_its_name():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=3 * FRAME, match_dist=20,
                     label_window_seconds=10 * FRAME, round_end_seconds=100 * FRAME,
                     hidden_seconds=2 * FRAME)
    run(tr, [det("9H", 50, 200), det("6H", 300, 200)], 3)
    run(tr, [det("9H", 50, 200), det("9H", 300, 200)], 8)   # the second card is renamed to 9H too
    run(tr, [det("9H", 300, 200)], 10)                      # the first is picked up
    assert [(c.order, c.label, c.counted) for c in tr.round] == [(2, "9H", True)]
    assert tr.shoe_seen == ["9H"]


def test_face_card_returning_elsewhere_is_still_the_same_card():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=3 * FRAME, round_end_seconds=100 * FRAME)
    run(tr, [det("QS", 50, 200), det("2D", 300, 200)], 3)
    run(tr, [det("2D", 300, 200)], 5)
    ev = run(tr, [det("2D", 300, 200), det("QS", 900, 600)], 3)  # picked up and put down far away
    assert [(e.kind, e.card.order) for e in ev] == [("returned", 1)]
    assert len(tr.round) == 2


def test_face_down_card_returns_near_where_it_was():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=3 * FRAME, match_dist=20, round_end_seconds=100 * FRAME)
    run(tr, [det("BACK", 100, 100), det("7H", 400, 100)], 3)
    run(tr, [det("7H", 400, 100)], 5)
    ev = run(tr, [det("7H", 400, 100), det("BACK", 110, 105)], 3)
    assert [(e.kind, e.card.order) for e in ev] == [("returned", 1)]
    assert [c.label for c in tr.round] == ["BACK", "7H"]


def test_face_down_card_far_away_is_a_new_card():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=3 * FRAME, match_dist=20, round_end_seconds=100 * FRAME)
    run(tr, [det("BACK", 100, 100), det("7H", 400, 100)], 3)
    run(tr, [det("7H", 400, 100)], 5)
    ev = run(tr, [det("7H", 400, 100), det("BACK", 800, 500)], 3)
    assert [(e.kind, e.card.order) for e in ev] == [("placed", 3)]


def test_removed_face_down_card_put_back_where_it_was_is_not_dealt_again():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=3 * FRAME, match_dist=20, round_end_seconds=100 * FRAME,
                     hidden_seconds=2 * FRAME)
    run(tr, [det("BACK", 100, 100), det("9S", 300, 100)], 3)
    run(tr, [det("9S", 300, 100)], 10)                       # picked up: removed
    assert [c.label for c in tr.round] == ["9S"] and tr.shoe_unrevealed == 1
    ev = run(tr, [det("9S", 300, 100), det("BACK", 140, 120)], 3)  # put back within 3 match distances
    assert [(e.kind, e.card.order, e.card.label) for e in ev] == [("placed", 3, "BACK")]
    assert tr.shoe_unrevealed == 1                           # the same card, not a second one dealt
    run(tr, [det("9S", 300, 100), det("BACK", 140, 120), det("BACK", 150, 200)], 3)
    assert tr.shoe_unrevealed == 2                           # another face-down card is dealt


def test_two_removed_face_down_cards_put_back_each_find_their_own_spot():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=3 * FRAME, match_dist=20, round_end_seconds=100 * FRAME,
                     hidden_seconds=2 * FRAME)
    run(tr, [det("BACK", 100, 100), det("BACK", 200, 100), det("9S", 300, 300)], 3)
    run(tr, [det("9S", 300, 300)], 10)                       # both picked up
    assert tr.shoe_unrevealed == 2
    # (155, 100) is in reach of both spots but nearer (200, 100); (70, 100) reaches only (100, 100)
    run(tr, [det("9S", 300, 300), det("BACK", 155, 100), det("BACK", 70, 100)], 3)
    assert [c.label for c in tr.table()] == ["9S", "BACK", "BACK"]
    assert tr.shoe_unrevealed == 2


def test_a_face_down_card_put_back_later_takes_over_a_spot_another_could_give_up():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=3 * FRAME, match_dist=20, round_end_seconds=100 * FRAME,
                     hidden_seconds=2 * FRAME)
    run(tr, [det("BACK", 100, 100), det("BACK", 160, 100), det("9S", 300, 300)], 3)
    run(tr, [det("9S", 300, 300)], 10)                       # both picked up
    run(tr, [det("9S", 300, 300), det("BACK", 128, 100)], 3)  # nearest (100, 100), in reach of (160, 100) too
    assert tr.shoe_unrevealed == 2
    run(tr, [det("9S", 300, 300), det("BACK", 128, 100), det("BACK", 50, 100)], 3)  # reaches (100, 100) only
    assert tr.shoe_unrevealed == 2                           # the first moves over to (160, 100)
    run(tr, [det("9S", 300, 300), det("BACK", 128, 100), det("BACK", 50, 100), det("BACK", 130, 130)], 3)
    assert tr.shoe_unrevealed == 3                           # in reach of both spots, both taken: a new card


def test_removed_face_down_card_far_away_or_in_a_later_round_is_a_new_card():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=3 * FRAME, match_dist=20, round_end_seconds=100 * FRAME,
                     hidden_seconds=2 * FRAME)
    run(tr, [det("BACK", 100, 100), det("9S", 300, 100)], 3)
    run(tr, [det("9S", 300, 100)], 10)
    run(tr, [det("9S", 300, 100), det("BACK", 100, 300)], 3)    # 200 pixels away
    assert tr.shoe_unrevealed == 2
    run(tr, [det("9S", 300, 100)], 10)
    tr.new_round()
    run(tr, [det("BACK", 100, 300)], 3)                          # where one was removed, but a new round
    assert tr.shoe_unrevealed == 3


def test_round_ends_when_table_stays_empty():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=3 * FRAME, round_end_seconds=10 * FRAME)
    run(tr, [det("KH", 50, 200), det("4S", 300, 200)], 3)
    ev = run(tr, [], 3 + 10)  # swept: hidden after 3 frames, round over 10 frames later
    assert [e.kind for e in ev][-1] == "round_over"
    assert tr.round == []
    run(tr, [det("8C", 50, 200)], 3)
    assert [(c.order, c.label) for c in tr.table()] == [(1, "8C")]  # numbering restarts


def test_short_cover_does_not_end_the_round():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=3 * FRAME, round_end_seconds=10 * FRAME)
    run(tr, [det("KH", 50, 200)], 3)
    run(tr, [], 8)                      # whole table covered, but not long enough
    ev = run(tr, [det("KH", 50, 200)], 3)
    assert [e.kind for e in ev] == ["returned"]
    assert [(c.order, c.label) for c in tr.round] == [(1, "KH")]


def test_card_seen_in_a_later_round_is_not_counted_again():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=3 * FRAME, round_end_seconds=10 * FRAME)
    run(tr, [det("KH", 50, 200)], 3)
    run(tr, [], 20)                     # round over
    run(tr, [det("KH", 50, 200), det("3D", 300, 200)], 3)
    assert [(c.order, c.label) for c in tr.table()] == [(1, "KH"), (2, "3D")]
    assert tr.shoe_seen == ["KH", "3D"]  # one deck: the KH is the same physical card


def test_card_seen_in_a_later_round_is_reported_as_a_duplicate():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=3 * FRAME, round_end_seconds=10 * FRAME)
    run(tr, [det("KH", 50, 200), det("3D", 300, 200)], 3)
    run(tr, [], 20)                     # round over
    ev = run(tr, [det("KH", 400, 200)], 3)
    assert [(e.kind, e.card.order, e.card.label) for e in ev] == [("placed", 1, "KH"), ("duplicate", 1, "KH")]
    assert tr.shoe_seen == ["KH", "3D"]
    assert shoe_count(tr.shoe_seen).running == 0


def test_rename_to_a_name_counted_in_an_earlier_round_is_a_duplicate():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=3 * FRAME, round_end_seconds=5 * FRAME,
                     label_window_seconds=10 * FRAME)
    run(tr, [det("KH", 50, 200)], 3)
    run(tr, [], 10)                     # round over
    run(tr, [det("BACK", 50, 200)], 3)
    ev = run(tr, [det("KH", 50, 200)], 10)  # the face-down card is turned over: a KH again
    assert [(e.kind, e.card.order, e.card.label) for e in ev] == [("renamed", 1, "KH"), ("duplicate", 1, "KH")]
    assert tr.shoe_seen == ["KH"] and tr.shoe_duplicates == {"KH"}


def test_two_corners_of_one_card_are_not_a_duplicate():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME, match_dist=20)
    ev = run(tr, [det("7D", 50, 200)], 3)
    ev += run(tr, [det("7D", 50, 200), det("7D", 150, 300)], 3)
    assert [e.kind for e in ev] == ["placed"]


def test_card_still_down_through_a_new_round_is_not_a_duplicate():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME, match_dist=20)
    run(tr, [det("QS", 50, 50)], 3)
    tr.new_round()
    ev = run(tr, [det("QS", 50, 50)], 1)
    assert [e.kind for e in ev] == ["placed"]


def test_name_counted_this_round_is_pending_until_new_round():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME)
    run(tr, [det("QS", 50, 50)], 3)
    assert (tr.shoe_pending, tr.shoe_locked) == ({"QS"}, set())
    tr.new_round()
    assert (tr.shoe_pending, tr.shoe_locked) == (set(), {"QS"})


def test_name_counted_this_round_is_locked_after_round_over():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=3 * FRAME, round_end_seconds=10 * FRAME)
    run(tr, [det("QS", 50, 50)], 3)
    ev = run(tr, [], 20)
    assert "round_over" in [e.kind for e in ev]
    assert (tr.shoe_pending, tr.shoe_locked) == (set(), {"QS"})


def test_duplicate_name_is_remembered_until_new_shoe():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=3 * FRAME, round_end_seconds=10 * FRAME)
    run(tr, [det("KH", 50, 200)], 3)
    assert tr.shoe_duplicates == set()
    run(tr, [], 20)                     # round over
    run(tr, [det("KH", 400, 200)], 3)
    assert tr.shoe_duplicates == {"KH"}
    tr.new_shoe()
    assert tr.shoe_duplicates == set() and tr.shoe_locked == set() and tr.shoe_pending == set()


def test_new_shoe_forgets_earlier_names():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME)
    run(tr, [det("QS", 50, 50)], 3)
    run(tr, [], 10)
    tr.new_shoe()
    ev = run(tr, [det("QS", 50, 50)], 3)
    assert [e.kind for e in ev] == ["placed"]


def test_new_round_restarts_numbering_but_keeps_the_count():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME)
    run(tr, [det("QS", 50, 50)], 3)
    run(tr, [], 5)                      # swept away
    tr.new_round()
    run(tr, [det("3H", 300, 50)], 3)
    assert [(c.order, c.label) for c in tr.round] == [(1, "3H")]
    assert tr.shoe_seen == ["QS", "3H"]


def test_new_shoe_resets_the_count():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME)
    run(tr, [det("QS", 50, 50)], 3)
    tr.new_shoe()
    assert tr.shoe_seen == [] and tr.round == []
    run(tr, [det("QS", 50, 50)], 3)
    assert tr.shoe_seen == ["QS"]


def test_new_round_with_cards_still_down_renumbers_them():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME, match_dist=20)
    run(tr, [det("QS", 50, 50), det("4H", 300, 50)], 3)
    tr.new_round()
    run(tr, [det("QS", 50, 50), det("4H", 300, 50)], 1)
    assert [(c.order, c.label) for c in tr.table()] == [(1, "QS"), (2, "4H")]
    assert tr.shoe_seen == ["QS", "4H"]  # not counted again


def test_rescan_uncounts_this_rounds_names_and_keeps_earlier_ones_in_order():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME, match_dist=20)
    run(tr, [det("QS", 50, 50), det("4H", 300, 50)], 3)
    run(tr, [], 5)                      # swept away
    tr.new_round()
    run(tr, [det("9C", 50, 50), det("2D", 300, 50)], 3)
    frame = tr.frame
    tr.rescan()
    assert tr.shoe_seen == ["QS", "4H"]
    assert (tr.shoe_pending, tr.shoe_locked) == (set(), {"QS", "4H"})
    assert tr.round == [] and tr.tracks == {} and tr.bound == [] and tr.frame == frame


def test_rescan_restores_the_unrevealed_and_duplicates_of_the_round_start():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=3 * FRAME, round_end_seconds=10 * FRAME, match_dist=20)
    run(tr, [det("BACK", 50, 50), det("KH", 300, 50)], 3)
    run(tr, [], 20)                     # round over: one face-down card dealt, KH locked
    run(tr, [det("BACK", 50, 200), det("KH", 300, 200)], 3)
    assert (tr.shoe_unrevealed, tr.shoe_duplicates) == (2, {"KH"})
    tr.rescan()
    assert (tr.shoe_unrevealed, tr.shoe_duplicates) == (1, set())


def test_cards_on_the_table_come_back_after_a_rescan_numbered_from_1_and_counted_once():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME, match_dist=20)
    run(tr, [det("9C", 50, 50)], 3)
    run(tr, [det("9C", 50, 50), det("6D", 300, 50)], 3)
    run(tr, [det("9C", 50, 50), det("BACK", 300, 50)], 1)
    tr.rescan()
    assert run(tr, [det("6D", 300, 50), det("9C", 50, 50)], 1) == []  # confirmed again from scratch
    ev = run(tr, [det("6D", 300, 50), det("9C", 50, 50)], 1)
    assert [(e.kind, e.card.order, e.card.label) for e in ev] == [("placed", 1, "6D"), ("placed", 2, "9C")]
    assert tr.shoe_seen == ["6D", "9C"] and tr.shoe_unrevealed == 0


def test_face_down_card_carried_into_the_round_is_not_dealt_again_after_a_rescan():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME, match_dist=20)
    run(tr, [det("BACK", 100, 100)], 3)
    tr.new_round()                                       # key r while the card is still face down
    run(tr, [det("BACK", 100, 100), det("5H", 300, 100), det("BACK", 300, 300)], 3)
    assert tr.shoe_unrevealed == 2
    tr.rescan()
    assert tr.shoe_unrevealed == 1                       # back to the round's start: the carried card
    run(tr, [det("BACK", 100, 100), det("5H", 300, 100), det("BACK", 300, 300)], 3)
    assert [(c.order, c.label) for c in tr.round] == [(1, "BACK"), (2, "5H"), (3, "BACK")]
    assert tr.shoe_unrevealed == 2                       # the carried card once, this round's once
    tr.rescan()
    run(tr, [det("BACK", 100, 100)], 3)
    assert tr.shoe_unrevealed == 1                       # still once after a second rescan


def test_two_decks_undo_takes_back_a_phantom_copy_and_keeps_every_other_card():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME, match_dist=20, decks=2)
    table = [det("KH", 100, 100), det("9C", 400, 100)]
    run(tr, table, 3)
    before = shoe_count(tr.shoe_seen, decks=2).running
    phantom = det("KH", 300, 300)                        # far from the real KH: a second copy
    ev = run(tr, table + [phantom], 3)
    assert [(e.kind, e.card.order) for e in ev] == [("placed", 3)] and tr.shoe_seen == ["KH", "9C", "KH"]
    undone = tr.undo_last()
    assert (undone.order, undone.label) == (3, "KH")
    assert [(c.order, c.label) for c in tr.round] == [(1, "KH"), (2, "9C")]
    assert tr.shoe_seen == ["KH", "9C"] and shoe_count(tr.shoe_seen, decks=2).running == before
    assert run(tr, table + [phantom], 1) == []           # its tracks are forgotten: confirmed again first
    ev = run(tr, table + [phantom], 1)
    assert [(e.kind, e.card.order) for e in ev] == [("placed", 4)]


def test_undo_on_a_round_with_no_cards_does_nothing():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME, match_dist=20)
    run(tr, [det("9C", 50, 50)], 1)                      # seen, not yet confirmed
    assert tr.undo_last() is None
    assert tr.round == [] and tr.shoe_seen == [] and len(tr.tracks) == 1


def test_undo_takes_a_face_down_card_off_the_unrevealed_and_a_duplicate_off_the_duplicates():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=3 * FRAME, round_end_seconds=10 * FRAME, match_dist=20)
    run(tr, [det("KH", 300, 50)], 3)
    run(tr, [], 20)                                      # round over: KH locked
    run(tr, [det("BACK", 50, 200)], 3)
    assert tr.shoe_unrevealed == 1
    assert tr.undo_last().label == BACK and tr.shoe_unrevealed == 0
    ev = run(tr, [det("KH", 300, 200)], 3)
    assert [e.kind for e in ev] == ["placed", "duplicate"] and tr.shoe_duplicates == {"KH"}
    tr.undo_last()
    assert tr.shoe_duplicates == set() and tr.shoe_seen == ["KH"]


def test_an_undone_face_down_card_dealt_before_is_not_added_again_and_a_new_one_is_added_once():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME, match_dist=20)
    run(tr, [det("BACK", 100, 100)], 3)
    tr.new_round()                                       # key r while the card is still face down
    run(tr, [det("BACK", 100, 100)], 3)
    assert [(c.order, c.label) for c in tr.round] == [(1, BACK)] and tr.shoe_unrevealed == 1
    tr.undo_last()
    assert tr.shoe_unrevealed == 1                       # dealt in the round before: it stays dealt
    run(tr, [det("BACK", 100, 100)], 3)
    assert [c.order for c in tr.round] == [2] and tr.shoe_unrevealed == 1   # the same card, not added again
    run(tr, [det("BACK", 100, 100), det("BACK", 300, 300)], 3)
    assert [c.order for c in tr.round] == [2, 3] and tr.shoe_unrevealed == 2
    tr.undo_last()
    assert tr.shoe_unrevealed == 1                       # dealt this round: it comes off
    run(tr, [det("BACK", 100, 100), det("BACK", 300, 300)], 3)
    assert [c.order for c in tr.round] == [2, 4] and tr.shoe_unrevealed == 2   # placed again, added once


def test_merge_folds_one_card_into_another_and_its_number_is_not_reused():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME, match_dist=20, decks=2)
    table = [det("9C", 50, 50), det("KH", 300, 50), det("5D", 50, 300), det("KH", 300, 300)]
    run(tr, table, 3)
    assert [(c.order, c.label) for c in tr.round] == [(1, "9C"), (2, "KH"), (3, "5D"), (4, "KH")]
    tracks_of_4 = [track_id for track_id, card in tr._card_of.items() if card.order == 4]
    assert tr.merge(4, 2).order == 4
    assert [c.order for c in tr.round] == [1, 2, 3]
    assert tr.shoe_seen == ["9C", "KH", "5D"]            # the merged card's copy left the count
    assert all(tr._card_of[track_id].order == 2 for track_id in tracks_of_4)
    assert run(tr, table, 2) == []                       # both tracks stay with card 2: nothing placed again
    ev = run(tr, table + [det("7S", 500, 300)], 3)
    assert [(e.kind, e.card.order) for e in ev] == [("placed", 5)]


@pytest.mark.parametrize(("order_from", "order_to"), [(3, 1), (1, 3), (1, 1)])
def test_merge_with_a_missing_number_or_one_card_does_nothing(order_from, order_to):
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME, match_dist=20)
    run(tr, [det("9C", 50, 50), det("KH", 300, 50)], 3)
    assert tr.merge(order_from, order_to) is None
    assert [c.order for c in tr.round] == [1, 2] and tr.shoe_seen == ["9C", "KH"]


def test_new_shoe_forgets_the_face_down_cards_carried_into_the_round():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME, match_dist=20)
    run(tr, [det("BACK", 100, 100)], 3)
    tr.new_shoe()
    tr.rescan()
    run(tr, [det("BACK", 100, 100)], 3)
    assert tr.shoe_unrevealed == 1                       # dealt from the new shoe


def test_name_read_again_after_a_rescan_is_not_a_duplicate():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME, match_dist=20)
    run(tr, [det("9C", 50, 50)], 3)
    tr.rescan()
    ev = run(tr, [det("9C", 50, 50)], 2)
    assert [e.kind for e in ev] == ["placed"] and tr.shoe_duplicates == set()


def test_new_round_after_a_rescan_locks_the_names_read_again():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME, match_dist=20)
    run(tr, [det("9C", 50, 50)], 3)
    tr.rescan()
    run(tr, [det("6C", 50, 50)], 2)
    tr.new_round()
    assert (tr.shoe_seen, tr.shoe_pending, tr.shoe_locked) == (["6C"], set(), {"6C"})


def test_rescan_tells_whether_this_round_placed_a_card():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME, match_dist=20)
    run(tr, [det("9C", 50, 50)], 1)                      # seen, not yet confirmed
    assert tr.rescan() is False
    run(tr, [det("9C", 50, 50)], 2)
    tr.new_round()
    assert tr.rescan() is False
    run(tr, [det("6C", 300, 50)], 2)
    assert tr.rescan() is True


def test_face_down_cards_are_not_counted():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME, match_dist=20)
    run(tr, [det("BACK", 50, 50), det("9S", 300, 50)], 3)
    assert tr.shoe_seen == ["9S"]


def test_flickering_detection_is_not_confirmed_when_seen_steadily_is_required():
    tr = CardTracker(confirm_seconds=6 * FRAME, drop_seconds=30 * FRAME, confirm_window_seconds=8 * FRAME)
    for _ in range(12):                 # seen every other frame: 12 sightings over 24 frames
        tr.update([det("KS", 50, 200)])
        tr.update([])
    assert tr.round == []
    run(tr, [det("KS", 50, 200)], 6)    # now seen steadily
    assert labels(tr) == ["KS"]


def test_disputed_name_waits_for_agreement():
    tr = CardTracker(confirm_seconds=4 * FRAME, drop_seconds=30 * FRAME, confirm_agreement=0.8)
    for label in ["6H", "9H"] * 4:      # 50/50 split
        tr.update([det(label, 50, 200)])
    assert tr.round == []
    run(tr, [det("6H", 50, 200)], 12)   # 16 of 20 reads say 6H
    assert labels(tr) == ["6H"]


def test_reads_split_between_suits_of_one_rank_agree():
    tr = CardTracker(confirm_seconds=4 * FRAME, drop_seconds=30 * FRAME, confirm_agreement=0.8)
    for label in ["10H", "10D", "10H"] * 2:   # suits disagree, the rank does not
        tr.update([det(label, 50, 200)])
    assert labels(tr) == ["10H"]              # shown under its most common name


def test_reads_split_between_ranks_of_one_suit_wait():
    tr = CardTracker(confirm_seconds=4 * FRAME, drop_seconds=30 * FRAME, confirm_agreement=0.8)
    for label in ["6H", "9H", "6H"] * 2:
        tr.update([det(label, 50, 200)])
    assert tr.round == []


def test_low_average_confidence_is_not_confirmed():
    tr = CardTracker(confirm_seconds=4 * FRAME, drop_seconds=30 * FRAME, confirm_window_seconds=4 * FRAME,
                     confirm_confidence=0.7)
    run(tr, [det("QD", 50, 200, conf=0.55)], 10)
    assert tr.round == []
    run(tr, [det("QD", 50, 200, conf=0.9)], 4)
    assert labels(tr) == ["QD"]


def test_moving_card_is_confirmed_only_once_it_stops():
    tr = CardTracker(confirm_seconds=5 * FRAME, drop_seconds=30 * FRAME, match_dist=40,
                     confirm_window_seconds=5 * FRAME, still_distance=6)
    for i in range(12):                 # slid 15 px a frame across the table
        tr.update([det("8C", 50 + 15 * i, 200)])
    assert tr.round == []
    run(tr, [det("8C", 50 + 15 * 12, 200)], 5)
    assert labels(tr) == ["8C"]
    assert abs(tr.round[0].x - (50 + 15 * 12)) < 20  # counted where it landed


def test_small_jitter_still_counts_as_held_still():
    tr = CardTracker(confirm_seconds=5 * FRAME, drop_seconds=30 * FRAME, confirm_window_seconds=5 * FRAME,
                     still_distance=6)
    for dx, dy in [(0, 0), (2, -1), (-2, 1), (1, 2), (-1, -2)]:
        tr.update([det("2H", 300 + dx, 200 + dy)])
    assert labels(tr) == ["2H"]


def test_hole_card_turned_face_up_is_renamed_not_added():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME, match_dist=20, label_window_seconds=10 * FRAME)
    run(tr, [det("BACK", 100, 100), det("7H", 400, 100)], 6)   # dealer: hole card and up card
    assert [(c.order, c.label) for c in tr.round] == [(1, "BACK"), (2, "7H")]
    ev = run(tr, [det("KS", 100, 100), det("7H", 400, 100)], 6)  # the hole card is turned over
    assert [(e.kind, e.card.order, e.card.label, e.was) for e in ev] == [("renamed", 1, "KS", "BACK")]
    assert [(c.order, c.label) for c in tr.round] == [(1, "KS"), (2, "7H")]
    assert tr.shoe_seen == ["7H", "KS"]


def test_corrected_misread_fixes_the_count_and_keeps_the_number():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME, match_dist=20, label_window_seconds=10 * FRAME)
    run(tr, [det("9H", 50, 200)], 3)                     # confirmed under the wrong name
    run(tr, [det("9H", 50, 200), det("2C", 300, 200)], 3)
    assert tr.shoe_seen == ["9H", "2C"]
    ev = run(tr, [det("6H", 50, 200), det("2C", 300, 200)], 6)
    assert [(e.kind, e.card.order, e.was) for e in ev] == [("renamed", 1, "9H")]
    assert [(c.order, c.label) for c in tr.round] == [(1, "6H"), (2, "2C")]
    assert tr.shoe_seen == ["6H", "2C"]                  # the 9H was never on the table; order kept
    assert len(tr.round) == 2                            # no phantom second card


def test_rename_does_not_uncount_a_card_seen_in_an_earlier_round():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=3 * FRAME, round_end_seconds=5 * FRAME,
                     label_window_seconds=10 * FRAME)
    run(tr, [det("KH", 50, 200)], 3)
    run(tr, [], 10)                                       # round over; KH stays counted
    run(tr, [det("KH", 50, 200)], 3)                     # next round: a card misread as the KH
    run(tr, [det("4D", 50, 200)], 6)                     # it was really a 4D
    assert [c.label for c in tr.round] == ["4D"]
    assert tr.shoe_seen == ["KH", "4D"]                  # the real KH from round 1 is still counted


def test_two_corners_disagreeing_do_not_rename_until_the_vote_flips():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME, match_dist=20, label_window_seconds=10 * FRAME)
    run(tr, [det("9H", 50, 200), det("9H", 150, 300)], 3)   # both corners
    assert len(tr.round) == 1
    ev = run(tr, [det("6H", 50, 200), det("9H", 150, 300)], 4)  # one corner changes its mind
    assert ev == [] and tr.round[0].label == "9H"         # 4 of 10 reads: not yet
    ev = run(tr, [det("6H", 50, 200), det("6H", 150, 300)], 6)
    assert [e.kind for e in ev] == ["renamed"] and tr.round[0].label == "6H"
    assert len(tr.round) == 1


def test_without_a_label_window_the_vote_is_over_every_read():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME)
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
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME, match_dist=20, label_window_seconds=10 * FRAME)
    run(tr, [det("9H", 50, 200), det("6H", 300, 200)], 3)
    run(tr, [det("9H", 50, 200), det("9H", 300, 200)], 8)   # the second card is renamed to 9H too
    run(tr, [det("5C", 50, 200), det("9H", 300, 200)], 8)   # the first card was really a 5C
    assert [(c.order, c.label) for c in tr.round] == [(1, "5C"), (2, "9H")]
    assert tr.shoe_seen == ["9H", "5C"]                     # 9H is still on the table


def test_face_down_card_swept_without_a_reveal_is_remembered_until_the_next_shoe():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=3 * FRAME, round_end_seconds=5 * FRAME)
    run(tr, [det("BACK", 100, 100), det("9S", 300, 100)], 3)
    assert tr.shoe_unrevealed == 1
    run(tr, [], 10)                                          # swept face down: round over
    assert tr.round == [] and tr.shoe_unrevealed == 1
    tr.new_shoe()
    assert tr.shoe_unrevealed == 0


def test_face_down_card_still_down_through_a_new_round_is_dealt_only_once():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME, match_dist=20, label_window_seconds=10 * FRAME)
    run(tr, [det("BACK", 100, 100)], 3)
    assert tr.shoe_unrevealed == 1
    tr.new_round()                                       # key r while the card is still face down
    run(tr, [det("BACK", 100, 100)], 3)
    assert [(c.order, c.label) for c in tr.round] == [(1, "BACK")]
    assert tr.shoe_unrevealed == 1                       # the same card, not a second one dealt
    run(tr, [det("KS", 100, 100)], 10)                   # then it is turned over
    assert [(c.order, c.label) for c in tr.round] == [(1, "KS")]
    assert tr.shoe_unrevealed == 0 and tr.shoe_seen == ["KS"]


def deal_rounds(tr, *rounds):
    """Deal each list of detections as one round: confirmed, then swept until round over."""
    events = []
    for dets in rounds:
        events += run(tr, dets, 3)
        events += run(tr, [], 20)
    return events


@pytest.mark.parametrize("decks", [1, 2, 3])
def test_a_card_is_a_duplicate_only_once_every_deck_has_dealt_it(decks):
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=3 * FRAME, round_end_seconds=10 * FRAME, match_dist=20,
                     decks=decks)
    ev = deal_rounds(tr, *[[det("KH", 50 + 100 * i, 200)] for i in range(decks)])
    assert "duplicate" not in [e.kind for e in ev]
    assert tr.shoe_seen == ["KH"] * decks
    ev = run(tr, [det("KH", 400, 200)], 3)
    assert [(e.kind, e.card.label) for e in ev] == [("placed", "KH"), ("duplicate", "KH")]
    assert tr.shoe_seen == ["KH"] * decks and tr.shoe_duplicates == {"KH"}


def test_two_decks_two_far_apart_cards_of_one_name_are_both_counted_and_a_third_is_a_duplicate():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME, match_dist=20, decks=2)
    ev = run(tr, [det("KH", 50, 200), det("KH", 400, 200)], 3)
    assert [(e.kind, e.card.order) for e in ev] == [("placed", 1), ("placed", 2)]
    assert tr.shoe_seen == ["KH", "KH"] and shoe_count(tr.shoe_seen, decks=2).running == -2
    ev = run(tr, [det("KH", 50, 200), det("KH", 400, 200), det("KH", 600, 50)], 3)
    assert [(e.kind, e.card.order) for e in ev] == [("placed", 3), ("duplicate", 3)]
    assert tr.shoe_seen == ["KH", "KH"]


def test_two_decks_two_corners_of_one_card_are_still_one_card():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME, match_dist=20, decks=2)
    ev = run(tr, [det("7D", 50, 200)], 3)
    ev += run(tr, [det("7D", 50, 200), det("7D", 90, 260)], 3)  # 72 px away: within 4 match distances
    assert [e.kind for e in ev] == ["placed"]
    assert labels(tr) == ["7D"] and tr.shoe_seen == ["7D"]


def test_two_decks_corners_with_real_geometry_are_one_card_and_two_seats_are_two():
    tr = real_tracker()
    tr.cfg.decks = 2
    run(tr, [*corners("KS", 600, 500), *corners("KS", 1300, 500)], 30)
    assert [(c.order, c.label) for c in tr.round] == [(1, "KS"), (2, "KS")]
    assert tr.shoe_seen == ["KS", "KS"]


def test_two_decks_a_hole_card_reveal_joins_its_own_card_not_a_far_card_of_that_name():
    tr = real_tracker()
    tr.cfg.decks = 2
    run(tr, [det("BACK", 900, 300), *corners("QH", 1500, 300)], 30)
    run(tr, [*corners("QH", 900, 300), *corners("QH", 1500, 300)], 120)  # the hole card is a QH too
    assert [(c.order, c.label) for c in tr.round] == [(1, "QH"), (2, "QH")]
    assert tr.shoe_seen == ["QH", "QH"] and tr.shoe_unrevealed == 0


@pytest.mark.parametrize("decks", [2, 3])
def test_more_decks_two_cards_of_one_name_fanned_in_one_hand_are_two_cards(decks):
    tr = real_tracker()
    tr.cfg.decks = decks
    run(tr, corners("KH", 900, 500), 30)
    run(tr, [*corners("KH", 900, 500), *corners("KH", 950, 500)], 30)  # the second KH fanned 50 px over
    assert [(c.order, c.label) for c in tr.round] == [(1, "KH"), (2, "KH")]
    assert tr.shoe_seen == ["KH", "KH"] and tr.shoe_duplicates == set()


def test_three_decks_a_fourth_copy_fanned_in_one_hand_is_a_duplicate():
    tr = real_tracker()
    tr.cfg.decks = 3
    hand = []
    for i in range(3):
        hand += corners("KH", 900 + 50 * i, 500)
        run(tr, hand, 30)
    assert [c.order for c in tr.round] == [1, 2, 3] and tr.shoe_seen == ["KH"] * 3
    ev = run(tr, [*hand, *corners("KH", 1050, 500)], 30)
    assert [(e.kind, e.card.order) for e in ev] == [("placed", 4), ("duplicate", 4)]
    assert tr.shoe_seen == ["KH"] * 3 and tr.shoe_duplicates == {"KH"}


@pytest.mark.parametrize("extra_label", ["KH", "KD"])
def test_two_decks_a_second_box_on_a_bound_corner_joins_its_card(extra_label):
    tr = real_tracker()
    tr.cfg.decks = 2
    run(tr, corners("KH", 900, 500), 30)
    extra = det(extra_label, 870, 420)                   # 15 px from the top corner: a third track, same or misread
    run(tr, [*corners("KH", 900, 500), extra], 30)
    assert [(c.order, c.label) for c in tr.round] == [(1, "KH")]
    assert tr.shoe_seen == ["KH"] and tr.bound[-1] is tr.round[0]


def test_two_decks_a_corner_track_that_went_stale_is_replaced_not_copied():
    tr = real_tracker()
    tr.cfg.decks = 2
    run(tr, corners("KH", 900, 500), 30)
    top, bottom = corners("KH", 900, 500)
    run(tr, [det("KH", top.cx + 65, top.cy), bottom], 10)  # card straightened: the top corner's track goes stale
    assert [(c.order, c.label) for c in tr.round] == [(1, "KH")]
    assert tr.shoe_seen == ["KH"]


def test_two_decks_a_card_slid_far_while_its_old_tracks_live_is_the_same_card():
    tr = real_tracker()
    tr.cfg.decks = 2
    run(tr, corners("KH", 900, 500), 30)
    ev = run(tr, corners("KH", 1400, 500), 10)           # 500 px away, before the old tracks drop
    assert "placed" not in [e.kind for e in ev]
    assert [(c.order, c.label) for c in tr.round] == [(1, "KH")] and tr.shoe_seen == ["KH"]


def test_two_decks_a_hidden_card_returning_far_away_is_the_same_card():
    tr = real_tracker()
    tr.cfg.decks = 2
    run(tr, corners("KH", 900, 500), 30)
    run(tr, [], 20)                                      # covered: its tracks drop and the card is hidden
    assert tr.round[0].hidden
    ev = run(tr, corners("KH", 1400, 500), 10)
    assert [(e.kind, e.card.order) for e in ev] == [("returned", 1)]
    assert tr.shoe_seen == ["KH"]


def held_copy_tracker(hold_frames):
    """Two decks, a KH placed at (900, 500), and a second KH fanned 50 px over held against it
    on the frame it is confirmed. Returns the tracker, the hand's detections, and that frame's events."""
    tr = real_tracker()
    tr.cfg.decks, tr.cfg.copy_hold_seconds = 2, hold_frames * FRAME
    run(tr, corners("KH", 900, 500), 30)
    hand = [*corners("KH", 900, 500), *corners("KH", 950, 500)]
    return tr, hand, run(tr, hand, 6)


def test_two_decks_a_copy_beside_a_card_of_its_name_is_held_until_both_are_seen_together():
    tr, hand, ev = held_copy_tracker(5)
    assert [(e.kind, e.card.order) for e in ev] == [("placed", 2), ("held", 2)]
    assert [c.held for c in tr.round] == [False, True] and tr.shoe_seen == ["KH"]
    assert run(tr, hand, 3) == []                        # 4 frames seen together, counting the first
    ev = run(tr, hand, 1)
    assert [(e.kind, e.card.order) for e in ev] == [("counted", 2)]
    assert [c.held for c in tr.round] == [False, False] and tr.shoe_seen == ["KH", "KH"]


def test_two_decks_merging_a_card_into_its_held_copy_hands_the_copy_over():
    tr, hand, _ = held_copy_tracker(50)
    assert [(c.order, c.counted, c.held) for c in tr.round] == [(1, True, False), (2, False, True)]
    tr.merge(1, 2)
    assert [(c.order, c.counted) for c in tr.round] == [(2, True)] and tr.shoe_seen == ["KH"]
    ev = run(tr, hand, 3)                                # its hold ends as when the first copy is removed
    assert [(e.kind, e.card.order) for e in ev] == [("counted", 2)]   # all four corners are card 2's
    assert [(c.order, c.counted, c.held) for c in tr.round] == [(2, True, False)] and tr.shoe_seen == ["KH"]


def test_two_decks_a_hold_of_zero_frames_counts_a_copy_at_once():
    tr, _, ev = held_copy_tracker(0)
    assert [(e.kind, e.card.order) for e in ev] == [("placed", 2)]
    assert tr.shoe_seen == ["KH", "KH"]


def test_two_decks_a_held_copy_seen_without_its_first_copy_stays_held():
    tr, hand, _ = held_copy_tracker(5)
    run(tr, corners("KH", 950, 500), 10)                 # the first KH is covered
    assert tr.round[1].held and tr.shoe_seen == ["KH"]
    ev = run(tr, hand, 4)
    assert [(e.kind, e.card.order) for e in ev] == [("counted", 2)]


def test_two_decks_a_held_copy_takes_the_count_of_the_first_copy_when_it_is_removed():
    tr, _, _ = held_copy_tracker(50)
    tr.cfg.hidden_seconds = 2 * FRAME
    ev = run(tr, corners("KH", 950, 500), 20)            # the first KH was this one, picked up and put down
    assert [(e.kind, e.card.order) for e in ev] == [("hidden", 1), ("removed", 1), ("counted", 2)]
    assert [(c.order, c.held, c.counted) for c in tr.round] == [(2, False, True)]
    assert tr.shoe_seen == ["KH"]


def test_two_decks_a_held_copy_takes_the_count_of_the_first_copy_when_it_is_renamed_away():
    tr, hand, _ = held_copy_tracker(50)
    hand[:2] = corners("KD", 900, 500)                   # the first KH is really a KD
    ev = run(tr, hand, 19)
    assert [(e.kind, e.card.order) for e in ev] == [("renamed", 1), ("counted", 2)]
    assert [(c.label, c.held, c.counted) for c in tr.round] == [("KD", False, True), ("KH", False, True)]
    assert tr.shoe_seen == ["KH", "KD"]


def test_two_decks_a_held_copy_does_not_take_the_count_of_another_card_renamed_away():
    tr, hand, _ = held_copy_tracker(100)
    far = corners("KH", 1500, 500)                       # a second KH far away: counted at once
    run(tr, [*hand, *far], 6)
    assert tr.shoe_seen == ["KH", "KH"]
    ev = run(tr, [*hand, *corners("KD", 1500, 500)], 19)  # it is really a KD
    assert [(e.kind, e.card.order) for e in ev] == [("renamed", 3)]
    assert [(c.held, c.counted) for c in tr.round] == [(False, True), (True, False), (False, True)]
    assert tr.shoe_seen == ["KH", "KD"]


def test_two_decks_a_held_copy_does_not_take_the_count_of_another_card_removed():
    tr, hand, _ = held_copy_tracker(100)
    run(tr, [*hand, *corners("KH", 1500, 500)], 6)       # a second KH far away: counted at once
    tr.cfg.hidden_seconds = 2 * FRAME
    ev = run(tr, hand, 20)                               # it is picked up
    assert [(e.kind, e.card.order) for e in ev] == [("hidden", 3), ("removed", 3)]
    assert [(c.order, c.held, c.counted) for c in tr.round] == [(1, False, True), (2, True, False)]
    assert tr.shoe_seen == ["KH", "KH"]                  # a removed card stays counted


def test_two_decks_a_held_copy_counted_after_the_last_copy_was_dealt_is_a_duplicate():
    tr, hand, _ = held_copy_tracker(20)
    hand += corners("KH", 1500, 500)                     # a second KH far away: counted at once
    ev = run(tr, hand, 6)
    assert [(e.kind, e.card.order) for e in ev] == [("placed", 3)] and tr.shoe_seen == ["KH", "KH"]
    ev = run(tr, hand, 20)
    assert [(e.kind, e.card.order) for e in ev] == [("counted", 2), ("duplicate", 2)]
    assert tr.shoe_seen == ["KH", "KH"] and tr.shoe_duplicates == {"KH"}


def test_two_decks_a_copy_is_not_held_once_every_copy_was_dealt():
    tr = real_tracker()
    tr.cfg.decks, tr.cfg.copy_hold_seconds = 2, 5 * FRAME
    run(tr, [*corners("KH", 900, 500), *corners("KH", 1500, 500)], 30)
    ev = run(tr, [*corners("KH", 900, 500), *corners("KH", 1500, 500), *corners("KH", 950, 500)], 6)
    assert [(e.kind, e.card.order) for e in ev] == [("placed", 3), ("duplicate", 3)]


def test_two_decks_a_held_copy_is_forgotten_by_a_rescan_and_by_the_round_end():
    tr, hand, _ = held_copy_tracker(50)
    tr.rescan()
    assert tr.round == [] and tr.shoe_seen == []
    ev = run(tr, hand, 6)
    assert [(e.kind, e.card.order) for e in ev] == [("placed", 1), ("placed", 2), ("held", 2)]
    ev = run(tr, [], 50)
    assert [e.kind for e in ev] == ["hidden", "hidden", "round_over"]
    assert tr.round == [] and tr.shoe_seen == ["KH"]


def test_two_decks_a_held_copy_renamed_away_is_counted_at_once():
    tr, hand, _ = held_copy_tracker(50)
    hand[2:] = corners("KD", 950, 500)                   # the second KH is really a KD
    ev = run(tr, hand, 19)
    assert [(e.kind, e.card.order, e.card.label) for e in ev] == [("renamed", 2, "KD")]
    assert [c.held for c in tr.round] == [False, False] and tr.shoe_seen == ["KH", "KD"]


def test_two_decks_a_held_copy_renamed_beside_another_card_of_its_new_name_is_held_again():
    tr = real_tracker()
    tr.cfg.decks, tr.cfg.copy_hold_seconds = 2, 50 * FRAME
    table = [*corners("KH", 900, 500), *corners("QH", 1000, 500)]
    run(tr, table, 30)
    run(tr, [*table, *corners("KH", 950, 500)], 6)       # held against the KH
    ev = run(tr, [*table, *corners("QH", 950, 500)], 19)  # it is really a second QH
    assert [(e.kind, e.card.order) for e in ev] == [("renamed", 3), ("held", 3)]
    assert tr.round[2].held and tr.shoe_seen == ["KH", "QH"]


def test_two_decks_a_held_copy_still_down_through_a_new_round_is_placed_afresh():
    tr, hand, _ = held_copy_tracker(50)
    tr.new_round()                                       # key r with both KHs still on the table
    ev = run(tr, hand, 1)
    assert [(e.kind, e.card.order) for e in ev] == [("placed", 1), ("placed", 2)]
    assert tr.shoe_seen == ["KH", "KH"]                  # the first is locked, the second counted now


def test_two_decks_a_hole_card_reveal_beside_a_card_of_that_name_joins_its_own_card():
    tr = real_tracker()
    tr.cfg.decks = 2
    run(tr, [det("BACK", 900, 300), *corners("QH", 1030, 300)], 30)
    run(tr, [*corners("QH", 900, 300), *corners("QH", 1030, 300)], 120)  # the hole card is a QH too
    assert [(c.order, c.label) for c in tr.round] == [(1, "QH"), (2, "QH")]
    assert tr.shoe_seen == ["QH", "QH"] and tr.shoe_unrevealed == 0


def test_two_decks_a_card_still_down_through_a_new_round_is_not_counted_again():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME, match_dist=20, decks=2)
    run(tr, [det("QS", 50, 50)], 3)
    tr.new_round()
    ev = run(tr, [det("QS", 50, 50)], 1)
    assert [e.kind for e in ev] == ["placed"]
    assert tr.shoe_seen == ["QS"] and (tr.shoe_pending, tr.shoe_locked) == (set(), {"QS"})


def test_two_decks_a_second_copy_beside_a_carried_one_is_counted_not_a_duplicate():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME, match_dist=20, decks=2)
    run(tr, [det("KH", 50, 50)], 3)
    tr.new_round()                                       # the KH stays on the table
    ev = run(tr, [det("KH", 50, 50), det("KH", 400, 200)], 3)
    assert [(e.kind, e.card.order) for e in ev] == [("placed", 1), ("placed", 2)]
    assert tr.shoe_seen == ["KH", "KH"] and tr.shoe_duplicates == set()


def test_two_decks_renaming_a_copy_away_beside_a_carried_one_gives_back_only_its_own_copy():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME, match_dist=20,
                     label_window_seconds=10 * FRAME, decks=2)
    run(tr, [det("KH", 50, 50)], 3)
    tr.new_round()                                       # the KH stays on the table
    run(tr, [det("KH", 50, 50), det("KH", 400, 200)], 3)  # a second KH, really a misread 4D
    run(tr, [det("KH", 50, 50), det("4D", 400, 200)], 8)
    assert [(c.order, c.label) for c in tr.round] == [(1, "KH"), (2, "4D")]
    assert tr.shoe_seen == ["KH", "4D"]
    assert shoe_count(tr.shoe_seen, decks=2).cards_left == 102


def test_two_decks_a_carried_card_renamed_away_and_back_keeps_only_its_locked_copy():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME, match_dist=20, label_window_seconds=4 * FRAME,
                     decks=2)
    run(tr, [det("KH", 50, 50)], 3)
    tr.new_round()                                       # the KH stays on the table
    run(tr, [det("KH", 50, 50)], 1)
    run(tr, [det("4D", 50, 50)], 8)                      # misread for a while
    assert tr.shoe_seen == ["KH", "4D"]
    run(tr, [det("KH", 50, 50)], 8)                      # then read right again
    assert [(c.order, c.label) for c in tr.round] == [(1, "KH")]
    assert tr.shoe_seen == ["KH"] and tr.shoe_pending == set()
    assert shoe_count(tr.shoe_seen, decks=2).cards_left == 103


def test_two_decks_removing_a_copy_does_not_hand_it_to_a_carried_card():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=3 * FRAME, match_dist=20,
                     label_window_seconds=10 * FRAME, round_end_seconds=100 * FRAME,
                     hidden_seconds=2 * FRAME, decks=2)
    run(tr, [det("KH", 50, 50)], 3)
    tr.new_round()                                       # the KH stays on the table
    run(tr, [det("KH", 50, 50), det("KH", 400, 200)], 3)
    run(tr, [det("KH", 50, 50)], 10)                     # the second KH is picked up
    assert [(c.order, c.counted) for c in tr.round] == [(1, False)]
    assert tr.shoe_seen == ["KH", "KH"]                  # a removed card stays counted


def test_two_decks_a_name_can_be_locked_and_pending_at_once():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=3 * FRAME, round_end_seconds=10 * FRAME, match_dist=20,
                     decks=2)
    deal_rounds(tr, [det("KH", 50, 200)])
    run(tr, [det("KH", 400, 200)], 3)
    assert (tr.shoe_pending, tr.shoe_locked) == ({"KH"}, {"KH"})


def test_two_decks_rename_to_a_name_with_a_copy_left_is_counted_not_a_duplicate():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=3 * FRAME, round_end_seconds=5 * FRAME,
                     label_window_seconds=10 * FRAME, decks=2)
    deal_rounds(tr, [det("KH", 50, 200)])
    run(tr, [det("BACK", 50, 200)], 3)
    ev = run(tr, [det("KH", 50, 200)], 10)
    assert [(e.kind, e.card.label) for e in ev] == [("renamed", "KH")]
    assert tr.shoe_seen == ["KH", "KH"] and tr.shoe_unrevealed == 0


def test_two_decks_renaming_a_second_copy_away_keeps_the_first_counted_in_order():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=3 * FRAME, round_end_seconds=5 * FRAME,
                     label_window_seconds=10 * FRAME, decks=2)
    deal_rounds(tr, [det("KH", 50, 200), det("5C", 300, 200)])
    run(tr, [det("KH", 50, 200)], 3)                     # a second KH, really a misread 4D
    assert tr.shoe_seen == ["KH", "5C", "KH"]
    run(tr, [det("4D", 50, 200)], 6)
    assert tr.shoe_seen == ["KH", "5C", "4D"]


def test_two_decks_rescan_takes_back_only_this_rounds_copies():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=3 * FRAME, round_end_seconds=10 * FRAME, match_dist=20,
                     decks=2)
    deal_rounds(tr, [det("KH", 50, 200), det("4C", 300, 200)])
    run(tr, [det("KH", 50, 200), det("KH", 400, 200)], 3)  # the second copy, then a third: a duplicate
    assert tr.shoe_seen == ["KH", "4C", "KH"] and tr.shoe_duplicates == {"KH"}
    tr.rescan()
    assert tr.shoe_seen == ["KH", "4C"] and tr.shoe_duplicates == set()
    assert (tr.shoe_pending, tr.shoe_locked) == (set(), {"KH", "4C"})


def test_two_decks_new_shoe_clears_every_copy():
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=3 * FRAME, round_end_seconds=10 * FRAME, match_dist=20,
                     decks=2)
    deal_rounds(tr, [det("KH", 50, 200)], [det("KH", 50, 200)])
    assert tr.shoe_seen == ["KH", "KH"]
    tr.new_shoe()
    assert tr.shoe_seen == [] and tr.shoe_locked == set() and tr.shoe_pending == set()
    ev = deal_rounds(tr, [det("KH", 50, 200)], [det("KH", 50, 200)])
    assert "duplicate" not in [e.kind for e in ev] and tr.shoe_seen == ["KH", "KH"]


def test_config_object_and_keyword_overrides_agree():
    a = CardTracker(TrackerConfig(confirm_seconds=0.3, match_dist=10), drop_seconds=0.7)
    assert (a.cfg.confirm_seconds, a.cfg.match_dist, a.cfg.drop_seconds) == (0.3, 10, 0.7)

    class S:  # the fields settings.py provides
        CONFIRM_SECONDS, CONFIRM_WINDOW_SECONDS, CONFIRM_AGREEMENT, CONFIRM_CONFIDENCE = 0.01, 1.0, 0.8, 0.7
        STILL_DISTANCE, DROP_SECONDS, MATCH_DISTANCE, ROUND_END_SECONDS = 0.01, 3.0, 0.05, 9.0
        LABEL_WINDOW_SECONDS, HIDDEN_SECONDS, COPY_HOLD_SECONDS, CONFIDENCE, SAME_CARD_DISTANCE = 4.5, 0, 0, 0.6, 0.1
    cfg = TrackerConfig.from_settings(S, 2000)
    assert cfg.start_confidence == 0.6
    assert (cfg.still_distance, cfg.match_dist, cfg.same_card_dist) == (20.0, 100.0, 200.0)  # fractions become pixels
    assert (cfg.confirm_seconds, cfg.confirm_window_seconds, cfg.drop_seconds) == (0.01, 1.0, 3.0)  # seconds kept
    assert (cfg.round_end_seconds, cfg.label_window_seconds) == (9.0, 4.5)
    assert (cfg.hidden_seconds, cfg.copy_hold_seconds) == (0, 0)


def test_a_time_shorter_than_a_frame_still_takes_a_frame():
    tr = CardTracker(confirm_seconds=0.01, confirm_window_seconds=0.01, drop_seconds=0.01,
                     round_end_seconds=0.01, label_window_seconds=0.01)
    ev = tr.update([det("KH", 50, 200)])
    assert [(e.kind, e.card.label) for e in ev] == [("placed", "KH")]
    assert [e.kind for e in tr.update([])] == ["hidden", "round_over"]


def test_shipped_settings_at_the_nominal_rate_take_the_frames_they_took_as_frame_counts():
    tr = real_tracker()                                  # updates without a time: 12.4 frames a second
    kh, fc = corners("KH", 900, 500), corners("5C", 1400, 500)
    assert run(tr, [*kh, *fc], 5) == []
    assert [e.kind for e in tr.update([*kh, *fc])] == ["placed", "placed"]  # CONFIRM_SECONDS 0.5: 6 frames
    assert run(tr, kh, 11) == []
    assert [e.kind for e in tr.update(kh)] == ["hidden"]                     # DROP_SECONDS 1: 12 frames
    assert run(tr, kh, 61) == []
    assert [e.kind for e in tr.update(kh)] == ["removed"]                    # HIDDEN_SECONDS 5: 62 frames
    assert run(tr, [], 12 + 35) == [Event("hidden", tr.round[0], tr.frame - 35)]
    assert [e.kind for e in tr.update([])] == ["round_over"]                 # ROUND_END_SECONDS 3: 37 frames


def test_bound_gives_each_detection_its_confirmed_card_even_when_misread():
    tr = CardTracker(confirm_seconds=3 * FRAME, drop_seconds=5 * FRAME, match_dist=20)
    tr.update([det("6H", 100, 100)])
    assert tr.bound == [None]                            # seen once: not confirmed yet
    run(tr, [det("6H", 100, 100)], 2)
    card = tr.round[0]
    assert tr.bound == [card]
    tr.update([det("9H", 101, 100), det("KS", 400, 300)])  # a one-frame misread, and a new corner elsewhere
    assert tr.bound[0] is card and card.label == "6H"
    assert tr.bound[1] is None
    tr.update([])
    assert tr.bound == []


@pytest.mark.parametrize("quarter_turns", [1, 2, 3, -1])
def test_rotate_turns_cards_and_tracks_with_the_frame(quarter_turns):
    W, H = 64, 36
    tr = CardTracker(confirm_seconds=2 * FRAME, drop_seconds=5 * FRAME, match_dist=6.4, still_distance=0.64)
    run(tr, [det("KH", 10, 5)], 3)
    tr.update([det("2S", 40, 30)])                       # an unconfirmed track
    tr.rotate(quarter_turns, W, H)

    def turned(x, y):
        """Where the pixel at (x, y) lands when the frame is turned like read.py turns it."""
        marker = np.zeros((H, W), np.uint8)
        marker[y, x] = 255
        rows, cols = np.nonzero(camera.rotate(marker, quarter_turns))
        return float(cols[0]), float(rows[0])

    assert (tr.round[0].x, tr.round[0].y) == turned(10, 5)
    assert sorted((t.cx, t.cy) for t in tr.tracks.values()) == sorted([turned(10, 5), turned(40, 30)])
    new_w = H if quarter_turns % 2 else W
    assert tr.cfg.match_dist == pytest.approx(0.1 * new_w) and tr.cfg.still_distance == pytest.approx(0.01 * new_w)


def test_cards_keep_their_numbers_across_a_quarter_turn():
    W, H = 640, 360
    tr = CardTracker(confirm_seconds=3 * FRAME, drop_seconds=5 * FRAME, match_dist=20, still_distance=4)
    run(tr, [det("KH", 100, 50), det("BACK", 500, 300)], 4)
    tr.rotate(1, W, H)
    events = run(tr, [det("KH", H - 1 - 50, 100), det("BACK", H - 1 - 300, 500)], 6)
    assert events == []
    assert [(c.order, c.label) for c in tr.round] == [(1, "KH"), (2, "BACK")]
    assert tr.shoe_seen == ["KH"] and tr.shoe_unrevealed == 1


# One card read twice: a corner misread for a while, then corrected, or read as another suit for good.

def misread_corner(label, wrong, cx=900, cy=500):
    """A card's two corner indices with the bottom one reading `wrong`."""
    top, bottom = corners(label, cx, cy)
    return [top, det(wrong, bottom.cx, bottom.cy)]


@pytest.mark.parametrize("decks", [1, 2])
def test_a_corner_misread_as_another_suit_then_corrected_is_merged_back_into_its_card(decks):
    tr = real_tracker()
    tr.cfg.decks = decks
    run(tr, misread_corner("2S", "2C"), 12)              # the bottom corner reads 2C for a second
    assert [(c.order, c.label) for c in tr.round] == [(1, "2S"), (2, "2C")]
    ev = run(tr, corners("2S", 900, 500), 36)
    assert [(e.kind, e.card.order, e.into.order) for e in ev] == [("merged", 2, 1)]
    assert [(c.order, c.label) for c in tr.round] == [(1, "2S")] and tr.shoe_seen == ["2S"]
    assert tr.bound == [tr.round[0], tr.round[0]]
    assert shoe_count(tr.shoe_seen, decks=decks).running == 1


@pytest.mark.parametrize("decks", [1, 2])
def test_a_corner_misread_as_another_rank_then_corrected_is_merged_back_into_its_card(decks):
    tr = real_tracker()
    tr.cfg.decks = decks
    run(tr, misread_corner("QH", "JH"), 12)
    assert [(c.order, c.label) for c in tr.round] == [(1, "QH"), (2, "JH")]
    ev = run(tr, corners("QH", 900, 500), 36)
    assert [(e.kind, e.card.order, e.into.order) for e in ev] == [("merged", 2, 1)]
    assert [(c.order, c.label) for c in tr.round] == [(1, "QH")] and tr.shoe_seen == ["QH"]


def test_the_card_placed_first_keeps_its_number_when_it_is_the_one_renamed():
    tr = real_tracker()
    tr.cfg.decks = 2
    top, bottom = corners("2S", 900, 500)
    run(tr, [det("2C", top.cx, top.cy), bottom], 12)     # the top corner, confirmed first, reads 2C
    assert [(c.order, c.label) for c in tr.round] == [(1, "2C"), (2, "2S")]
    ev = run(tr, corners("2S", 900, 500), 36)
    assert [(e.kind, e.card.order, e.was, e.into and e.into.order) for e in ev] == [
        ("renamed", 1, "2C", None), ("merged", 2, None, 1)]
    assert [(c.order, c.label, c.counted) for c in tr.round] == [(1, "2S", True)] and tr.shoe_seen == ["2S"]


def test_two_decks_a_real_second_copy_misread_then_corrected_stays_a_second_card():
    tr = real_tracker()
    tr.cfg.decks = 2
    run(tr, corners("2S", 900, 500), 30)                 # both corners of the first 2S in view
    run(tr, [*corners("2S", 900, 500), *corners("2C", 950, 500)], 12)
    ev = run(tr, [*corners("2S", 900, 500), *corners("2S", 950, 500)], 36)
    assert [(e.kind, e.card.order) for e in ev] == [("renamed", 2), ("held", 2), ("counted", 2)]  # as if placed
    assert [(c.order, c.label) for c in tr.round] == [(1, "2S"), (2, "2S")] and tr.shoe_seen == ["2S", "2S"]


def test_a_renamed_card_is_not_merged_into_a_far_card_of_its_new_name():
    tr = real_tracker()
    tr.cfg.decks = 2
    run(tr, [*corners("2S", 400, 300), *corners("2C", 1400, 300)], 12)
    ev = run(tr, [*corners("2S", 400, 300), *corners("2S", 1400, 300)], 36)   # 1000 px apart: two cards
    assert [(e.kind, e.card.order) for e in ev] == [("renamed", 2)]
    assert tr.shoe_seen == ["2S", "2S"] and len(tr.round) == 2


def test_two_decks_a_hidden_card_takes_a_renamed_card_at_any_distance():
    tr = real_tracker()
    tr.cfg.decks = 2
    run(tr, corners("KH", 900, 500), 30)
    run(tr, [], 20)                                      # picked up: hidden
    assert tr.round[0].hidden
    run(tr, corners("KD", 1400, 500), 12)                # put down far away and misread
    assert [(c.order, c.label) for c in tr.round] == [(1, "KH"), (2, "KD")]
    ev = run(tr, corners("KH", 1400, 500), 36)
    assert [(e.kind, e.card.order) for e in ev] == [("merged", 2), ("returned", 1)]
    assert [(c.order, c.label, c.hidden) for c in tr.round] == [(1, "KH", False)] and tr.shoe_seen == ["KH"]


def test_a_revealed_hole_card_is_never_folded_into_a_neighbour_of_its_name():
    tr = real_tracker()
    tr.cfg.decks = 2
    top, bottom = corners("QH", 1030, 300)
    run(tr, [det("BACK", 900, 300), top], 30)            # the up card shows one corner only
    run(tr, [*corners("QH", 900, 300), top], 120)        # the hole card is a QH too
    assert [(c.order, c.label) for c in tr.round] == [(1, "QH"), (2, "QH")]
    assert tr.shoe_seen == ["QH", "QH"]


def test_two_decks_a_card_renamed_beside_a_card_showing_both_corners_is_held_as_a_placed_one_would_be():
    tr = real_tracker()
    tr.cfg.decks, tr.cfg.copy_hold_seconds = 2, 50 * FRAME
    run(tr, corners("KH", 900, 500), 30)
    stray = det("KD", 1000, 420)                         # a third box 100 px off, misread
    run(tr, [*corners("KH", 900, 500), stray], 12)
    assert [(c.order, c.label) for c in tr.round] == [(1, "KH"), (2, "KD")]
    ev = run(tr, [*corners("KH", 900, 500), det("KH", 1000, 420)], 36)
    assert [(e.kind, e.card.order) for e in ev] == [("renamed", 2), ("held", 2)]
    assert tr.round[1].held and tr.shoe_seen == ["KH"]


# The cards' size, measured from the cards themselves.

def test_the_corner_span_is_measured_from_cards_showing_both_corners():
    tr = real_tracker()
    run(tr, corners("KS", 400, 300), 10)
    assert tr.corner_span is None                        # too few frames of a card with both corners
    run(tr, corners("KS", 400, 300), 20)
    assert tr.corner_span == pytest.approx((90 ** 2 + 160 ** 2) ** 0.5)
    tr.new_shoe()
    assert tr.corner_span is not None                    # the camera did not move


def test_two_boxes_on_one_corner_do_not_measure_a_span():
    tr = real_tracker()
    run(tr, [det("KS", 400, 300), det("KD", 415, 300)], 30)
    assert len(tr.round) == 1 and tr.corner_span is None


@pytest.mark.parametrize("decks", [1, 2])
def test_a_corner_read_as_the_wrong_suit_joins_its_card_once_the_span_is_known(decks):
    tr = real_tracker()
    tr.cfg.decks = decks
    table = corners("KS", 400, 300)
    run(tr, table, 24)                                   # teaches the span
    ev = run(tr, [*table, *misread_corner("2S", "2C")], 48)   # the bottom corner reads 2C for good
    assert [(e.kind, e.card.label) for e in ev] == [("placed", "2S")]
    assert [(c.order, c.label) for c in tr.round] == [(1, "KS"), (2, "2S")] and tr.shoe_seen == ["KS", "2S"]
    assert tr.bound[2:] == [tr.round[1], tr.round[1]]


def test_a_card_whose_corners_read_two_suits_keeps_its_name_when_a_corner_is_missed():
    tr = real_tracker()
    table = corners("KS", 400, 300)
    run(tr, table, 24)
    hand = misread_corner("2S", "2C")
    run(tr, [*table, *hand], 24)
    card = tr.round[1]
    names = set()
    for i in range(60):
        run(tr, [*table, *hand] if i % 3 else [*table, hand[1]], 1)  # the 2S corner missed every third frame
        names.add(card.label)
    assert names == {"2S"} and tr.shoe_seen == ["KS", "2S"]


def test_a_pair_fanned_in_one_hand_stays_two_cards_with_the_span_known():
    tr = real_tracker()
    tr.cfg.decks = 2
    table = corners("KS", 400, 300)
    run(tr, table, 24)
    eight_c = det("8C", 900 - 45 + 50, 500 - 80)          # fanned 50 px over the 8S, one corner showing
    run(tr, [*table, *corners("8S", 900, 500), eight_c], 36)
    assert [(c.order, c.label) for c in tr.round] == [(1, "KS"), (2, "8S"), (3, "8C")]
    assert tr.shoe_seen == ["KS", "8S", "8C"]


def test_two_decks_a_copy_much_nearer_a_lone_corner_than_a_span_is_a_second_copy_not_the_other_corner():
    tr = real_tracker()
    tr.cfg.decks = 2
    table = corners("KS", 400, 300)
    run(tr, table, 24)
    lone = det("2S", 855, 420)                           # a 2S whose other corner is covered
    run(tr, [*table, lone], 36)
    ev = run(tr, [*table, lone, det("2S", 905, 420)], 36)   # a second 2S fanned 50 px over it
    assert [(e.kind, e.card.order) for e in ev] == [("placed", 3), ("held", 3), ("counted", 3)]
    assert tr.shoe_seen == ["KS", "2S", "2S"]


def test_without_a_span_a_copy_near_a_lone_corner_still_joins_as_before():
    tr = real_tracker()
    tr.cfg.decks = 2
    lone = det("2S", 855, 420)
    run(tr, [lone], 36)
    ev = run(tr, [lone, det("2S", 905, 420)], 36)
    assert ev == [] and tr.shoe_seen == ["2S"]


def big_card(label, cx, cy):
    """The corners of a 180 x 260 px card: a camera close to the table. Its diagonal, 316 px, is beyond
    four match distances (230 px at 1920 wide)."""
    return [det(label, cx - 90, cy - 130), det(label, cx + 90, cy + 130)]


def back_box(cx, cy, w, h):
    return Detection(BACK, 0.9, cx, cy, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))


def test_two_decks_a_card_bigger_than_the_same_card_distance_is_two_copies_unless_the_setting_is_raised():
    tr = real_tracker()
    tr.cfg.decks = 2
    run(tr, big_card("2S", 900, 500), 36)
    assert tr.shoe_seen == ["2S", "2S"] and len(tr.round) == 2
    tr = real_tracker()
    tr.cfg.decks, tr.cfg.same_card_dist = 2, 0.18 * 1920
    run(tr, big_card("2S", 900, 500), 36)
    assert tr.shoe_seen == ["2S"] and len(tr.round) == 1


def test_two_decks_a_face_down_card_widens_the_same_card_distance_for_bigger_cards():
    tr = real_tracker()
    tr.cfg.decks = 2
    assert tr.same_card_dist() == pytest.approx(0.12 * 1920)
    run(tr, [back_box(400, 300, 180, 260)], 24)         # the hole card shows how big the cards are
    assert tr.card_diagonal == pytest.approx((180 ** 2 + 260 ** 2) ** 0.5)
    assert tr.same_card_dist() == pytest.approx(1.2 * tr.card_diagonal)
    run(tr, [back_box(400, 300, 180, 260), *big_card("2S", 900, 500)], 36)
    assert [(c.order, c.label) for c in tr.round] == [(1, "BACK"), (2, "2S")] and tr.shoe_seen == ["2S"]


def test_face_down_boxes_without_a_size_measure_nothing():
    tr = real_tracker()
    run(tr, [det("BACK", 400, 300)], 30)
    assert tr.card_diagonal is None and tr.same_card_dist() == pytest.approx(0.12 * 1920)


def test_the_same_card_distance_turns_with_the_frame():
    tr = CardTracker(match_dist=20, same_card_dist=100)
    tr.rotate(1, 1920, 1080)
    assert (tr.cfg.match_dist, tr.cfg.same_card_dist) == (pytest.approx(11.25), pytest.approx(56.25))
    tr = CardTracker(match_dist=20)
    tr.rotate(1, 1920, 1080)
    assert tr.cfg.same_card_dist is None and tr.same_card_dist() == pytest.approx(45)


# A box on a card's body: its pips read as an index. Two decks unless said otherwise.

def on_body(label, cx, cy, along=0.5, across=0.0):
    """A box on the body of a card centred on (cx, cy): `along` of the way from its top-left corner index
    to its bottom-right one, and `across` of that distance to the side of the line between them."""
    top, bottom = corners(label, cx, cy)
    dx, dy = bottom.cx - top.cx, bottom.cy - top.cy
    return det(label, top.cx + along * dx + across * dy, top.cy + along * dy - across * dx)


def pips(label, cx, cy):
    """Boxes on the two pips of a 2 centred on (cx, cy): mid-width, a quarter and three quarters down."""
    return [det(label, cx, cy - 40), det(label, cx, cy + 40)]


@pytest.mark.parametrize("read_as", ["2C", "2S"])
def test_two_decks_a_box_on_a_cards_body_reading_its_rank_joins_the_card(read_as):
    tr = real_tracker()
    tr.cfg.decks = 2
    ev = run(tr, [*corners("2C", 900, 500), on_body(read_as, 900, 500)], 36)
    assert [(e.kind, e.card.label) for e in ev] == [("placed", "2C")]
    assert [(c.order, c.label) for c in tr.round] == [(1, "2C")] and tr.shoe_seen == ["2C"]
    assert tr.bound == [tr.round[0]] * 3


def test_two_decks_boxes_on_a_twos_pips_join_it():
    tr = real_tracker()
    tr.cfg.decks = 2
    run(tr, [*corners("2C", 900, 500), *pips("2C", 900, 500)], 36)
    assert [(c.order, c.label) for c in tr.round] == [(1, "2C")] and tr.shoe_seen == ["2C"]
    assert tr.bound == [tr.round[0]] * 4


def test_two_decks_a_pip_box_confirmed_before_the_second_corner_joins_once_the_corner_shows():
    tr = real_tracker()
    tr.cfg.decks = 2
    top, bottom = corners("2C", 900, 500)
    pip = on_body("2C", 900, 500)
    run(tr, [top, pip], 12)                              # no span known: the pip is taken for the second corner
    assert [(c.order, c.label) for c in tr.round] == [(1, "2C")]
    ev = run(tr, [top, pip, bottom], 36)                 # the real corner still joins: the pip lies between
    assert ev == []
    assert [(c.order, c.label) for c in tr.round] == [(1, "2C")] and tr.shoe_seen == ["2C"]
    assert tr.bound == [tr.round[0]] * 3


def test_two_decks_a_pip_box_placed_as_a_copy_is_folded_into_its_card_once_both_corners_show():
    tr = real_tracker()
    tr.cfg.decks = 2
    table = corners("KS", 400, 300)
    run(tr, table, 24)                                   # the span is known
    top, bottom = corners("2C", 900, 500)
    pip = on_body("2C", 900, 500)
    ev = run(tr, [*table, top, pip], 12)                 # half a span from the corner: a copy, held
    assert [(e.kind, e.card.order) for e in ev] == [("placed", 2), ("placed", 3), ("held", 3)]
    ev = run(tr, [*table, top, pip, bottom], 36)
    assert [(e.kind, e.card.order, e.into.order) for e in ev] == [("merged", 3, 2)]
    assert [(c.order, c.label) for c in tr.round] == [(1, "KS"), (2, "2C")] and tr.shoe_seen == ["KS", "2C"]


def test_two_decks_a_counted_pip_card_is_folded_too():
    tr = real_tracker()
    tr.cfg.decks, tr.cfg.copy_hold_seconds = 2, 0       # no hold: the copy is counted at once
    table = corners("KS", 400, 300)
    run(tr, table, 24)
    top, bottom = corners("2C", 900, 500)
    pip = on_body("2C", 900, 500)
    run(tr, [*table, top, pip], 12)
    assert tr.shoe_seen == ["KS", "2C", "2C"]
    ev = run(tr, [*table, top, pip, bottom], 36)
    assert [(e.kind, e.card.order) for e in ev] == [("merged", 3)] and tr.shoe_seen == ["KS", "2C"]


def test_a_box_of_another_rank_on_a_cards_body_is_a_card_of_its_own():
    tr = real_tracker()
    tr.cfg.decks = 2
    run(tr, [*corners("2C", 900, 500), on_body("7D", 900, 500)], 36)   # a card fanned diagonally over it
    assert [(c.order, c.label) for c in tr.round] == [(1, "2C"), (2, "7D")] and tr.shoe_seen == ["2C", "7D"]


def test_two_decks_a_same_rank_box_beside_the_body_line_is_a_copy():
    tr = real_tracker()
    tr.cfg.decks = 2
    run(tr, corners("2C", 900, 500), 30)
    ev = run(tr, [*corners("2C", 900, 500), on_body("2C", 900, 500, along=0.5, across=0.3)], 36)
    assert [(e.kind, e.card.order) for e in ev] == [("placed", 2), ("held", 2), ("counted", 2)]
    assert tr.shoe_seen == ["2C", "2C"]


@pytest.mark.parametrize(("dx", "dy"), [(50, 0), (80, 0), (0, 80)])
def test_two_decks_a_copy_fanned_straight_over_a_card_stays_a_second_card(dx, dy):
    tr = real_tracker()
    tr.cfg.decks = 2
    table = corners("KS", 400, 300)
    run(tr, table, 24)                                   # the span is known: a copy fanned 80 px down puts the
    run(tr, [*table, *corners("2C", 900, 500)], 30)      # first card's far corner where a pip would be otherwise
    run(tr, [*table, *corners("2C", 900, 500), *corners("2C", 900 + dx, 500 + dy)], 36)
    assert [(c.order, c.label) for c in tr.round] == [(1, "KS"), (2, "2C"), (3, "2C")]
    assert tr.shoe_seen == ["KS", "2C", "2C"]


def test_two_decks_a_same_rank_copy_fanned_diagonally_by_half_a_card_is_placed_from_its_far_corner():
    tr = real_tracker()
    tr.cfg.decks = 2
    run(tr, corners("2C", 900, 500), 30)
    run(tr, [*corners("2C", 900, 500), *corners("2C", 945, 580)], 36)  # its near corner lands mid-card
    assert [(c.order, c.label) for c in tr.round] == [(1, "2C"), (2, "2C")] and tr.shoe_seen == ["2C", "2C"]
    assert tr.bound == [tr.round[0]] * 3 + [tr.round[1]]  # the near corner joined card 1; the far one is card 2


def test_one_deck_a_box_on_a_cards_body_reading_another_suit_joins_it():
    tr = real_tracker()
    run(tr, [*corners("2C", 900, 500), on_body("2S", 900, 500)], 36)
    assert [(c.order, c.label) for c in tr.round] == [(1, "2C")] and tr.shoe_seen == ["2C"]


def test_a_card_carrying_a_pip_box_does_not_measure_a_span():
    tr = real_tracker()
    run(tr, [*corners("2C", 900, 500), on_body("2C", 900, 500)], 40)
    assert tr.corner_span is None
