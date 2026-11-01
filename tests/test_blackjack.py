import pytest

from cardReader.blackjack import (
    dealer_finished,
    describe,
    hand_value,
    hi_lo,
    is_blackjack,
    is_pair,
    milestone,
    outcome,
    shoe_count,
)
from cardReader.seats import DEALER, MAX_PLAYERS, SeatLayout


def test_hand_values():
    assert hand_value(["KH", "7S"]) == (17, False)
    assert hand_value(["AH", "7S"]) == (18, True)
    assert hand_value(["AH", "AS", "9D"]) == (21, True)
    assert hand_value(["AH", "KS", "5C"]) == (16, False)
    assert hand_value(["10H", "9S", "5C"]) == (24, False)


def test_describe():
    assert is_blackjack(["AH", "KS"])
    assert not is_blackjack(["AH", "5S", "5D"])
    assert describe(["AH", "KS"]) == "BLACKJACK"
    assert describe(["10H", "9S", "5C"]) == "BUST (24)"
    assert describe(["AH", "6C"]) == "soft 17"
    assert describe(["KS", "BACK"]) == "10 +1 hidden"
    assert describe(["BACK"]) == "1 hidden"
    assert describe([]) == "-"


def test_is_pair_needs_two_face_up_cards_of_equal_value():
    assert is_pair(["8S", "8H"]) and is_pair(["JS", "QH"]) and is_pair(["AS", "AD"])
    assert not is_pair(["8S", "9H"])
    assert not is_pair(["8S"]) and not is_pair(["8S", "8H", "8D"]) and not is_pair([])
    assert not is_pair(["BACK", "BACK"]) and not is_pair(["8S", "BACK"])


def test_dealer_finishes_on_17_soft_17_included_or_a_bust_with_no_card_face_down():
    assert not dealer_finished(["KS", "BACK"]) and not dealer_finished(["KS", "7H", "BACK"])
    assert not dealer_finished(["KS", "6H"]) and not dealer_finished(["6C"]) and not dealer_finished([])
    assert dealer_finished(["KS", "7H"])          # hard 17
    assert dealer_finished(["AS", "6H"])          # soft 17: the dealer stands
    assert dealer_finished(["KS", "6H", "9D"])    # bust


def test_milestone_marks_a_complete_hands_blackjack_21_or_bust():
    assert milestone(["AS", "KH"]) == "blackjack"
    assert milestone(["7S", "7H", "7D"]) == "twenty-one"
    assert milestone(["KS", "6H", "9D"]) == "bust"
    assert milestone(["KS", "8H"]) is None
    assert milestone(["AS"]) is None and milestone(["AS", "BACK"]) is None and milestone([]) is None


def test_a_bust_blackjack_or_21_shows_before_the_dealer_finishes():
    unfinished = ["6C", "BACK"]
    assert outcome(["KS", "6H", "9D"], unfinished) == "bust"
    assert outcome(["AS", "KH"], unfinished) == "blackjack"
    assert outcome(["7S", "7H", "7D"], unfinished) == "twenty-one"
    assert outcome(["KS", "8H"], unfinished) is None
    assert outcome(["KS", "6H", "9D"], ["KC", "6D", "8S"]) == "bust"   # a player bust loses even to a dealer bust


def test_a_blackjack_pushes_only_against_a_dealer_blackjack():
    assert outcome(["AS", "KH"], ["KC", "7D"]) == "blackjack"
    assert outcome(["AS", "KH"], ["7C", "7D", "7S"]) == "blackjack"
    assert outcome(["AS", "KH"], ["KC", "6D", "8S"]) == "blackjack"
    assert outcome(["AS", "KH"], ["AD", "QC"]) == "push"


def test_a_21_wins_pushes_or_loses_against_a_finished_dealer():
    three_card_21 = ["7S", "7H", "7D"]
    assert outcome(three_card_21, ["KC", "QD"]) == "win"
    assert outcome(three_card_21, ["5C", "6D", "KS"]) == "push"
    assert outcome(three_card_21, ["AD", "QC"]) == "lose"


def test_other_hands_wait_for_the_dealer_then_compare_totals():
    assert outcome(["KS", "8H"], ["9C", "BACK"]) is None
    assert outcome(["KS", "8H"], ["KC", "6D"]) is None                 # dealer on 16 still draws
    assert outcome(["KS", "8H"], ["KC", "7D"]) == "win"
    assert outcome(["KS", "8H"], ["KC", "8D"]) == "push"
    assert outcome(["KS", "8H"], ["KC", "9D"]) == "lose"
    assert outcome(["KS", "8H"], ["AC", "6D"]) == "win"                # soft 17 stands


def test_a_dealer_bust_makes_every_standing_hand_win():
    dealer_bust = ["KC", "6D", "8S"]
    assert outcome(["KS", "2H"], dealer_bust) == "win"
    assert outcome(["7S", "7H", "7D"], dealer_bust) == "win"
    assert outcome(["AS", "KH"], dealer_bust) == "blackjack"


def test_a_one_card_or_face_down_hand_has_no_outcome():
    finished = ["KC", "7D"]
    assert outcome(["KS"], finished) is None
    assert outcome(["KS", "BACK"], finished) is None
    assert outcome(["KS", "6H", "BACK"], ["KC", "6D", "8S"]) is None


def test_hi_lo_values():
    assert [hi_lo(c) for c in ["2S", "6H", "7D", "9C", "10S", "JH", "QD", "KC", "AS", "BACK"]] == \
        [1, 1, 0, 0, -1, -1, -1, -1, -1, 0]


def test_full_deck_counts_to_zero():
    deck = [r + s for s in "SHDC" for r in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]]
    c = shoe_count(deck)
    assert (c.running, c.cards_left, c.true) == (0, 0, 0.0)


def test_true_count_divides_by_decks_left():
    c = shoe_count(["2S", "3H", "4D", "5C"] + ["7S"] * 22)  # 26 seen, running +4, half a deck left
    assert c.running == 4 and c.cards_left == 26
    assert c.true == pytest.approx(8.0)


def test_true_count_floors_decks_left_at_a_quarter_deck():
    deck = [r + s for s in "SHDC" for r in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]]
    c = shoe_count([card for card in deck if card != "2H"])  # one card left: running -1
    assert (c.running, c.cards_left) == (-1, 1)
    assert c.true == pytest.approx(-4.0)                       # -1 / 0.25 deck, not -1 / (1/52)


def test_face_down_cards_never_revealed_leave_the_deck_but_not_the_count():
    c = shoe_count(["KH", "5D"], unrevealed=2)
    assert (c.running, c.cards_left) == (0, 48)


@pytest.mark.parametrize("players", range(1, MAX_PLAYERS + 1))
def test_seat_layout(players):
    W, H = 1920, 1080
    layout = SeatLayout(players, W, H)
    assert layout.names() == [DEALER] + [f"P{i + 1}" for i in range(players)]
    for seat in layout.seats:
        assert seat.polygon.min() >= 0 and seat.polygon[:, 0].max() < W and seat.polygon[:, 1].max() < H
        cx, cy = seat.polygon.mean(axis=0)
        assert layout.assign(cx, cy) == seat.name  # seats do not overlap at their centres
    xs = [s.polygon[:, 0].mean() for s in layout.seats[1:]]
    assert xs == sorted(xs)  # P1 on the left, last player on the right
    assert layout.assign(5, H - 5) is None  # a corner of the frame is in no seat
