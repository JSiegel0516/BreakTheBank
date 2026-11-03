import pytest

from cardReader.blackjack import (
    COUNT_SYSTEMS,
    dealer_final_chance,
    dealer_final_odds,
    dealer_finished,
    dealer_odds,
    describe,
    hand_value,
    hi_lo,
    insurance,
    is_blackjack,
    is_pair,
    last_card_odds,
    milestone,
    outcome,
    player_odds,
    shoe_count,
    stand_odds,
    stood_odds,
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


GRID_ORDER = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
DECK = [rank + suit for suit in "SHDC" for rank in GRID_ORDER]


@pytest.mark.parametrize(("system", "name", "values"), [
    ("hi_lo", "Hi-Lo", [1, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, -1, -1]),
    ("hi_opt_ii", "Hi-Opt II", [1, 1, 2, 2, 1, 1, 0, 0, -2, -2, -2, -2, 0]),
    ("uston_apc", "Uston APC", [1, 2, 2, 3, 2, 2, 1, -1, -3, -3, -3, -3, 0]),
])
def test_each_counting_system_gives_every_rank_its_value(system, name, values):
    counting = COUNT_SYSTEMS[system]
    assert counting.name == name
    assert [counting.value(rank + "S") for rank in GRID_ORDER] == values
    assert counting.value("BACK") == 0


@pytest.mark.parametrize("system", COUNT_SYSTEMS)
def test_every_counting_system_sums_to_zero_over_a_deck(system):
    deck = [rank + suit for suit in "SHDC" for rank in GRID_ORDER]
    c = shoe_count(deck, system=system)
    assert (c.running, c.true, c.system) == (0, 0.0, system)


def test_hi_lo_is_the_default_system_and_hi_lo_matches_its_entry():
    assert shoe_count(["2S"]).system == "hi_lo"
    assert all(hi_lo(rank + "H") == COUNT_SYSTEMS["hi_lo"].values[rank] for rank in GRID_ORDER)


def test_uston_apc_divides_by_half_decks_left_and_hi_opt_ii_by_decks():
    seen = ["5S", "5H"]                                        # 50 cards left
    uston = shoe_count(seen, system="uston_apc")
    assert uston.running == 6 and uston.true == pytest.approx(6 / (50 / 26))
    hi_opt = shoe_count(seen, system="hi_opt_ii")
    assert hi_opt.running == 4 and hi_opt.true == pytest.approx(4 / (50 / 52))


def test_uston_apc_floors_at_a_quarter_deck_which_is_half_a_half_deck():
    deck = [rank + suit for suit in "SHDC" for rank in GRID_ORDER]
    c = shoe_count([card for card in deck if card != "5H"], system="uston_apc")  # one card left: running -3
    assert (c.running, c.cards_left) == (-3, 1)
    assert c.true == pytest.approx(-6.0)                       # -3 / 0.5 half-decks


def test_a_live_hand_gets_the_next_cards_totals_and_bust_from_the_unseen_cards():
    unseen = ["5S", "4H", "AD", "KC"]                          # 16 + each: 21, 20, 17, bust
    assert player_odds(["10S", "6H"], [], unseen) == {"21": 25, "20": 25, "19": 0, "18": 0, "bust": 25}


def test_a_soft_hand_counts_its_ace_as_one_when_eleven_would_bust():
    unseen = ["3S", "2H", "10D", "9C", "AS", "8H"]             # soft 18 + each: 21, 20, 18, 17, 19, 16
    assert player_odds(["AH", "7S"], [], unseen) == {"21": 17, "20": 17, "19": 17, "18": 17, "bust": 0}


def test_an_empty_seat_gets_its_two_card_blackjack_chance():
    unseen = ["AS", "KH", "QD", "5C"]                          # 2 * 1/4 * 2/3 = 1/3
    assert player_odds([], [], unseen) == {"blackjack": 33}
    assert player_odds([], [], ["AS"]) == {"blackjack": 0}     # one card left cannot make two


@pytest.mark.parametrize(("card", "chance"), [("AS", 50), ("JH", 25), ("9D", 0)])
def test_a_one_card_hand_gets_the_chance_its_second_card_makes_blackjack(card, chance):
    unseen = ["KC", "10D", "AH", "5S"]                         # two tens, one ace in four
    assert player_odds([card], [], unseen) == {"blackjack": chance}


@pytest.mark.parametrize(("labels", "dealer"), [
    (["10S", "AH"], []),                                       # blackjack
    (["10S", "5H", "6D"], []),                                 # 21
    (["10S", "5H", "9D"], []),                                 # bust
    (["10S", "8H"], ["10C", "9D"]),                            # lost to a finished dealer
    (["10S", "BACK"], []),                                     # a face-down card
])
def test_a_finished_or_unknown_player_hand_has_no_odds(labels, dealer):
    assert player_odds(labels, dealer, ["2S", "3S"]) is None


def test_no_unseen_cards_means_no_odds():
    assert player_odds(["10S", "6H"], [], []) is None
    assert dealer_odds(["AS", "BACK"], []) is None


@pytest.mark.parametrize(("up", "chance"), [("AS", 40), ("QH", 20), ("7D", 0)])
def test_the_dealer_hole_card_makes_blackjack_by_its_up_card(up, chance):
    unseen = ["KC", "JD", "AH", "5S", "2S"]                    # two tens, one ace in five
    assert dealer_odds([up, "BACK"], unseen) == {"blackjack": chance}
    assert dealer_odds(["BACK", up], unseen) == {"blackjack": chance}


@pytest.mark.parametrize("labels", [["AS"], ["AS", "KH"], ["BACK"], ["BACK", "BACK"], ["AS", "BACK", "5H"], []])
def test_a_dealer_without_one_up_card_and_one_hole_card_has_no_odds(labels):
    assert dealer_odds(labels, ["KC", "AH"]) is None


def test_the_dealer_with_its_hole_card_down_gets_blackjack_apart_from_its_other_21s():
    # hole K: blackjack. Hole 5: soft 16 draws; K then 5 makes hard 21, 5 makes soft 21
    assert dealer_final_odds(["AS", "BACK"], ["KC", "5D", "5H"]) == {
        "blackjack": 33, "21": 67, "20": 0, "19": 0, "18": 0, "17": 0, "bust": 0}
    # hole K: blackjack. Hole 6: soft 17 stands. Hole 5: soft 16 draws to a bust either way
    assert dealer_final_odds(["BACK", "AS"], ["KC", "6D", "5H"]) == {
        "blackjack": 33, "21": 0, "20": 0, "19": 0, "18": 0, "17": 33, "bust": 33}


def test_the_dealer_over_a_fresh_deck_shows_an_ace():
    unseen = [card for card in DECK if card != "AS"]
    assert dealer_final_odds(["AS", "BACK"], unseen) == {
        "blackjack": 31, "21": 5, "20": 13, "19": 13, "18": 13, "17": 13, "bust": 12}


def test_a_dealer_with_two_cards_up_under_17_keeps_drawing():
    # 15 + 2: 17. + 6: 21. + K: bust. + A: 16 draws again from 2, 6, K: 18 or bust
    odds = dealer_final_odds(["10S", "5H"], ["2C", "6D", "AH", "KD"])
    assert odds == {"21": 25, "20": 0, "19": 0, "18": 8, "17": 25, "bust": 42}
    assert 99 <= sum(odds.values()) <= 101


def test_the_dealer_stands_on_a_soft_17_it_draws_to():
    # A2 + 4: soft 17 stands. + 5: soft 18. + 9: hard 12 draws 4 then 5 (21) or 5 (17)
    assert dealer_final_odds(["AS", "2H"], ["4C", "9D", "5S"]) == {
        "21": 17, "20": 0, "19": 0, "18": 33, "17": 50, "bust": 0}


def test_a_dealer_with_one_card_up_and_no_hole_card_draws_its_second_card():
    # 6 + K: 16 draws A (17) or 5 (21). + A: soft 17. + 5: 11 draws K (21) or A (12, then K busts)
    assert dealer_final_odds(["6S"], ["KC", "AH", "5D"]) == {
        "21": 33, "20": 0, "19": 0, "18": 0, "17": 50, "bust": 17}


@pytest.mark.parametrize("labels", [["10S", "7H"], ["AS", "6H"], ["10S", "6H", "KD"], ["BACK"], ["BACK", "BACK"], []])
def test_a_finished_dealer_or_one_with_no_card_up_has_no_final_odds(labels):
    assert dealer_final_odds(labels, ["KC", "AH", "5D"]) is None


@pytest.mark.parametrize(("dealer", "unseen"), [
    (["2S", "3H"], ["2C"]),                                    # 7 draws the last card and still needs more
    (["AS", "2H"], ["4C", "9D"]),                              # a 4 stands on soft 17, but a 9 makes 12, then 16
])
def test_a_dealer_the_unseen_cards_can_run_out_on_has_no_final_odds_or_stand_odds(dealer, unseen):
    assert dealer_final_odds(dealer, unseen) is None
    assert stand_odds(["KS", "8H"], dealer, unseen) is None


def test_no_unseen_cards_means_no_final_odds_stand_odds_or_insurance():
    assert dealer_final_odds(["AS", "BACK"], []) is None
    assert stand_odds(["10S", "8H"], ["6C"], []) is None
    assert insurance(["AS", "BACK"], []) is None


@pytest.mark.parametrize(("player", "dealer", "unseen", "odds"), [
    (["10S", "8H"], ["AS", "BACK"], ["KC", "6D", "5H"], {"win": 67, "push": 0, "lose": 33}),     # blackjack loses
    (["10S", "8H"], ["10C", "5H"], ["2C", "6D", "AH", "KD"], {"win": 67, "push": 8, "lose": 25}),
    (["10S", "8H"], ["6S"], ["KC", "AH", "5D"], {"win": 67, "push": 0, "lose": 33}),
    (["10S", "7H"], ["6S"], ["KC", "AH", "5D"], {"win": 17, "push": 50, "lose": 33}),
    (["10S", "6H"], ["6S"], ["KC", "AH", "5D"], {"win": 17, "push": 0, "lose": 83}),             # only a bust wins
])
def test_a_live_hand_gets_its_chances_if_it_stands_now(player, dealer, unseen, odds):
    assert stand_odds(player, dealer, unseen) == odds


@pytest.mark.parametrize(("player", "dealer"), [
    (["10S", "8H"], ["10C", "9D"]),                            # the dealer is finished: the hand lost
    (["10S", "8H"], ["BACK"]),                                 # no dealer card up
    (["10S", "8H"], []),
    (["10S"], ["6C"]),                                         # one card
    ([], ["6C"]),
    (["10S", "BACK"], ["6C"]),                                 # a face-down card
    (["10S", "5H", "6D"], ["6C"]),                             # 21
    (["10S", "5H", "9D"], ["6C"]),                             # bust
])
def test_a_hand_that_cannot_stand_now_has_no_stand_odds(player, dealer):
    assert stand_odds(player, dealer, ["KC", "AH", "5D"]) is None


@pytest.mark.parametrize(("player", "dealer", "unseen", "was"), [
    # 15 before the 6D; pool KC 6C 2D 6D makes 25, 21, 17, 21
    (["10S", "5H", "6D"], ["9C"], ["KC", "6C", "2D"], {"result": "21", "chance": 50}),
    # 15 before the 9D; pool KC QC 2D 9D makes 25, 25, 17, 24
    (["10S", "5H", "9D"], ["9C"], ["KC", "QC", "2D"], {"result": "bust", "chance": 75}),
    # an empty seat over QC 5D 2H AS KH: 2 * 1/5 aces * 2/4 tens
    (["AS", "KH"], ["9C"], ["QC", "5D", "2H"], {"result": "blackjack", "chance": 20}),
    # a won 18: 10 before the 8H; pool 8C 5D KH 8H makes 18, 15, 20, 18
    (["10S", "8H"], ["9C", "8D"], ["8C", "5D", "KH"], {"result": "18", "chance": 50}),
    # a soft hand: soft 16 before the 5D; pool 5D 5C KH makes 21, 21, 16
    (["AS", "5H", "5D"], ["9C"], ["5C", "KH"], {"result": "21", "chance": 67}),
])
def test_a_finished_hand_gets_the_chance_its_last_card_made_its_result(player, dealer, unseen, was):
    assert last_card_odds(player, dealer, unseen) == was


@pytest.mark.parametrize(("player", "dealer", "unseen"), [
    (["10S", "8H"], ["9C"], ["KC"]),                           # still playing
    (["10S", "BACK"], ["9C", "8D"], ["KC"]),                   # a card face down
    (["10S"], ["9C", "8D"], ["KC"]),                           # one card
    (["10S", "5H", "6D"], ["9C"], []),                         # nothing unseen
])
def test_an_unfinished_hand_or_nothing_unseen_has_no_last_card_odds(player, dealer, unseen):
    assert last_card_odds(player, dealer, unseen) is None


# The dealer 9C 8D replayed from its 9C over 8C 5D KH and its 8D: 17 for an 8 (1/2), 19 for the KH (1/4),
# and a bust after the 5D (1/4), since any third card busts 14.
@pytest.mark.parametrize(("player", "stood"), [
    (["10S", "8H"], {"result": "win", "chance": 75}),          # a 17 or a bust
    (["10S", "7H"], {"result": "push", "chance": 50}),         # a 17
    (["10S", "6H"], {"result": "lose", "chance": 75}),         # a 17 or a 19
])
def test_a_settled_hand_gets_the_chance_of_its_result_from_the_dealers_up_card(player, stood):
    assert stood_odds(player, ["9C", "8D"], ["8C", "5D", "KH"]) == stood


def test_a_blackjack_pushed_by_the_dealers_blackjack_gets_the_chance_of_the_hole_card_completing_it():
    # the dealer AD QC replayed from its AD over 5D JC 2H QC: a ten-value hole card is 2 in 4
    assert stood_odds(["AS", "KH"], ["AD", "QC"], ["5D", "JC", "2H"]) == {"result": "push", "chance": 50}


@pytest.mark.parametrize(("player", "dealer", "unseen"), [
    (["10S", "8H"], ["9C"], ["8C", "5D", "KH"]),               # the dealer is still drawing
    (["10S", "8H"], ["9C", "BACK"], ["8C", "5D", "KH"]),
    (["10S", "5H", "9D"], ["9C", "8D"], ["8C", "5D", "KH"]),   # a bust is settled before the dealer plays
    (["AS", "KH"], ["9C", "8D"], ["8C", "5D", "KH"]),          # a blackjack that was paid
    (["10S", "8H"], ["9C", "8D"], []),                         # nothing unseen
])
def test_a_hand_not_settled_against_the_dealer_has_no_stood_odds(player, dealer, unseen):
    assert stood_odds(player, dealer, unseen) is None


@pytest.mark.parametrize(("dealer", "unseen", "final"), [
    (["9C", "8D"], ["8C", "5D", "KH"], {"result": "17", "chance": 50}),          # as above
    # replayed from the 9C over 8C 8D 5D KH: the 5D's 14 busts on any third card
    (["9C", "5D", "KH"], ["8C", "8D"], {"result": "bust", "chance": 25}),
    (["AS", "KH"], ["5D", "QC", "2H"], {"result": "blackjack", "chance": 50}),   # QC or KH under the ace
])
def test_a_finished_dealer_gets_the_chance_of_its_final_from_its_up_card(dealer, unseen, final):
    assert dealer_final_chance(dealer, unseen) == final


@pytest.mark.parametrize(("dealer", "unseen"), [
    (["9C"], ["8C", "5D", "KH"]),                               # still drawing
    (["9C", "BACK"], ["8C", "5D", "KH"]),
    ([], ["8C", "5D", "KH"]),
    (["9C", "8D"], []),                                         # nothing unseen
])
def test_an_unfinished_dealer_or_nothing_unseen_has_no_final_chance(dealer, unseen):
    assert dealer_final_chance(dealer, unseen) is None


@pytest.mark.parametrize(("unseen", "call"), [
    (["KC", "QD", "5H"], "take"),                              # two thirds ten-value
    (["KC", "5H", "6D"], "skip"),                              # exactly one third: break-even
    (["KC", "5H", "6D", "7S"], "skip"),
])
def test_insurance_pays_when_more_than_a_third_of_the_unseen_cards_are_ten_value(unseen, call):
    assert insurance(["AS", "BACK"], unseen) == call
    assert insurance(["BACK", "AH"], unseen) == call


def test_insurance_over_a_fresh_deck_is_skipped_until_low_cards_leave():
    unseen = [card for card in DECK if card != "AS"]           # 16 tens in 51
    assert insurance(["AS", "BACK"], unseen) == "skip"
    for low in ["2S", "3S", "4S", "5S"]:                       # 16 tens in 47
        unseen.remove(low)
    assert insurance(["AS", "BACK"], unseen) == "take"


@pytest.mark.parametrize("dealer", [["KS", "BACK"], ["AS"], ["AS", "KH"], ["AS", "BACK", "5H"], ["BACK", "BACK"], []])
def test_insurance_is_offered_only_for_an_ace_over_a_hole_card(dealer):
    assert insurance(dealer, ["KC", "QD", "5H"]) is None


def test_an_unknown_counting_system_is_refused():
    with pytest.raises(KeyError):
        shoe_count(["2S"], system="ko")


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
