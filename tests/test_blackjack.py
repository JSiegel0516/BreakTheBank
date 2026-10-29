import pytest

from card_reader.blackjack import describe, hand_value, hi_lo, is_blackjack, shoe_count
from card_reader.seats import DEALER, SeatLayout


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


@pytest.mark.parametrize("players", range(1, 8))
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
