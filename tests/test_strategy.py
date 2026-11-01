import pytest

from cardReader.strategy import hint

DEALER_UP_CARDS = ["2C", "3C", "4C", "5C", "6C", "7C", "8C", "9C", "10C", "AC"]
PLAYS = {"H": "hit", "S": "stand", "D": "double", "P": "split"}

# Two-card hands and the expected play against the dealer's 2 to A.
HARD = [
    (["2S", "3H"], "H H H H H H H H H H"),   # 5
    (["2S", "4H"], "H H H H H H H H H H"),   # 6
    (["2S", "5H"], "H H H H H H H H H H"),   # 7
    (["2S", "6H"], "H H H H H H H H H H"),   # 8
    (["2S", "7H"], "H D D D D H H H H H"),   # 9
    (["2S", "8H"], "D D D D D D D D H H"),   # 10
    (["2S", "9H"], "D D D D D D D D D H"),   # 11
    (["KS", "2H"], "H H S S S H H H H H"),   # 12
    (["KS", "3H"], "S S S S S H H H H H"),   # 13
    (["KS", "4H"], "S S S S S H H H H H"),   # 14
    (["KS", "5H"], "S S S S S H H H H H"),   # 15
    (["KS", "6H"], "S S S S S H H H H H"),   # 16
    (["KS", "7H"], "S S S S S S S S S S"),   # 17
    (["KS", "8H"], "S S S S S S S S S S"),   # 18
    (["KS", "9H"], "S S S S S S S S S S"),   # 19
    (["KS", "4H", "6D"], "S S S S S S S S S S"),   # 20
    (["KS", "5H", "6D"], "S S S S S S S S S S"),   # 21
]

SOFT = [
    (["AS", "2H"], "H H H D D H H H H H"),   # 13
    (["AS", "3H"], "H H H D D H H H H H"),   # 14
    (["AS", "4H"], "H H D D D H H H H H"),   # 15
    (["AS", "5H"], "H H D D D H H H H H"),   # 16
    (["AS", "6H"], "H D D D D H H H H H"),   # 17
    (["AS", "7H"], "S D D D D S S H H H"),   # 18
    (["AS", "8H"], "S S S S S S S S S S"),   # 19
    (["AS", "9H"], "S S S S S S S S S S"),   # 20
    (["AS", "5H", "5D"], "S S S S S S S S S S"),   # 21
]

PAIRS = [
    (["2S", "2H"], "P P P P P P H H H H"),
    (["3S", "3H"], "P P P P P P H H H H"),
    (["4S", "4H"], "H H H P P H H H H H"),
    (["5S", "5H"], "D D D D D D D D H H"),
    (["6S", "6H"], "P P P P P H H H H H"),
    (["7S", "7H"], "P P P P P P H H H H"),
    (["8S", "8H"], "P P P P P P P P P P"),
    (["9S", "9H"], "P P P P P S P P S S"),
    (["10S", "10H"], "S S S S S S S S S S"),
    (["JS", "QH"], "S S S S S S S S S S"),
    (["AS", "AH"], "P P P P P P P P P P"),
]


@pytest.mark.parametrize(("player", "expected"), HARD + SOFT + PAIRS)
def test_every_table_row(player, expected):
    plays = [hint(player, [up_card]) for up_card in DEALER_UP_CARDS]
    assert plays == [PLAYS[code] for code in expected.split()]


def test_double_becomes_hit_or_stand_after_two_cards():
    assert hint(["2S", "4H", "5D"], ["6C"]) == "hit"      # hard 11
    assert hint(["AS", "3H", "4D"], ["4C"]) == "stand"    # soft 18 against 3 to 6
    assert hint(["AS", "3H", "4D"], ["9C"]) == "hit"      # soft 18 against 9 to A


def test_three_card_hand_with_matching_values_is_not_a_pair():
    assert hint(["2S", "2H", "2D"], ["6C"]) == "hit"


def test_face_cards_count_as_ten_for_the_dealer():
    assert hint(["KS", "6H"], ["QC"]) == hint(["KS", "6H"], ["10C"]) == "hit"


def test_dealer_up_card_is_the_first_face_up_card():
    assert hint(["KS", "6H"], ["BACK", "6C"]) == "stand"
    assert hint(["KS", "6H"], ["7C", "BACK"]) == "hit"


@pytest.mark.parametrize(("player", "dealer"), [
    (["KS", "BACK"], ["6C"]),          # hidden card
    (["KS"], ["6C"]),                  # fewer than two cards
    ([], ["6C"]),
    (["KS", "QH", "5D"], ["6C"]),      # bust
    (["AS", "KH"], ["6C"]),            # blackjack
    (["KS", "6H"], ["BACK"]),          # no face-up dealer card
    (["KS", "6H"], []),
])
def test_no_hint(player, dealer):
    assert hint(player, dealer) is None
