import pytest

from cardReader.blackjack import COUNT_SYSTEMS
from cardReader.deck import RANKS, SUITS
from cardReader.tips import CLOSE_GAP, FIRM_GAP, bet_tip, play_values, tip


def shoe_without(*shown, decks=1):
    """Every copy of a shoe of `decks` decks, less one copy of each shown face card."""
    unseen = [rank + suit for rank in RANKS for suit in SUITS for _ in range(decks)]
    for label in shown:
        unseen.remove(label)
    return unseen


def test_values_by_hand_twenty_stands_against_a_dealer_who_ends_on_17_or_18():
    # the hole card is a 7 (two of three) or an 8: the dealer stands on 17 or 18
    values = play_values(["KS", "QH"], ["10D", "BACK"], ["7S", "7H", "8S"])
    assert values == pytest.approx({"stand": 1.0, "hit": -1.0, "double": -2.0})


def test_values_by_hand_sixteen_hits_on_one_card_and_stands_after():
    # dealer finals: a hole 4 then a 7 (21) or an 8 (bust), 1/6 each; a hole 7 (17) or 8 (18), 1/3 each
    values = play_values(["10S", "6H"], ["10D", "BACK"], ["4S", "7H", "8C"])
    stand = 1 / 6 - 5 / 6                       # 16 wins on a dealer bust only
    after_four = 5 / 6 - 1 / 6                  # 20 beats 17, 18, and bust, loses to 21
    hit = 1 / 3 * after_four + 2 / 3 * -1       # a 7 or an 8 busts; 20 stands rather than hit again
    assert values == pytest.approx({"stand": stand, "hit": hit, "double": 2 * hit})


def test_double_needs_exactly_two_cards():
    values = play_values(["4S", "3H", "4D"], ["6C", "BACK"], shoe_without("4S", "3H", "4D", "6C"))
    assert set(values) == {"stand", "hit"}


def test_a_shoe_that_cannot_play_the_dealer_out_has_no_values():
    assert play_values(["10S", "6H"], ["6C", "BACK"], ["5H"]) is None   # 6 + 5 must draw again: nothing left
    assert play_values(["10S", "6H"], [], shoe_without("10S", "6H")) is None


@pytest.mark.parametrize(("player", "dealer", "play", "tone"), [
    (["10S", "QH"], "6C", "stand", "firm"),      # hard 20 against a 6
    (["10S", "6H"], "10D", None, "close"),       # hard 16 against a 10: stand and hit nearly level
    (["5S", "6H"], "6C", "double", "firm"),      # hard 11 against a 6
])
def test_the_tip_and_its_tone_on_a_full_two_deck_shoe(player, dealer, play, tone):
    unseen = shoe_without(*player, dealer, decks=2)
    result = tip(player, [dealer, "BACK"], unseen)
    assert result.tone == tone
    if play:
        assert result.play == play
    values = play_values(player, [dealer, "BACK"], unseen)
    best, second = sorted(values.values(), reverse=True)[:2]
    assert result.gap == round(100 * (best - second))
    assert (best - second >= FIRM_GAP) == (tone == "firm") and (best - second < CLOSE_GAP) == (tone == "close")


def test_tip_text_and_hover_follow_the_tone():
    firm = tip(["10S", "QH"], ["6C", "BACK"], shoe_without("10S", "QH", "6C"))
    assert firm.text == "Stand." and firm.gap > 10
    assert firm.title.startswith(f"stand +{firm.gap / 100:.2f} over ")
    assert firm.title.endswith(" against 49 unseen cards")
    close = tip(["10S", "6H"], ["10D", "BACK"], shoe_without("10S", "6H", "10D", decks=2))
    assert close.text == f"Close: {close.play}" and 0 <= close.gap < 3
    assert close.title.endswith(" against 101 unseen cards")


def test_a_normal_tip_is_the_play_in_sentence_case():
    # soft 18 against a 9: hitting is the play, by less than a tenth of a bet
    result = tip(["AS", "7H"], ["9C", "BACK"], shoe_without("AS", "7H", "9C"))
    assert (result.play, result.tone, result.text) == ("hit", "normal", "Hit")
    assert 3 <= result.gap < 10


def test_split_and_the_fallback_come_from_basic_strategy_with_no_number():
    split = tip(["8S", "8H"], ["6C", "BACK"], shoe_without("8S", "8H", "6C"))
    assert (split.play, split.tone, split.text, split.gap) == ("split", "normal", "Split", None)
    unsplit = tip(["8S", "8H"], ["6C", "BACK"], shoe_without("8S", "8H", "6C"), can_split=False)
    assert unsplit.play != "split" and unsplit.gap is not None    # a split hand never says split
    fallback = tip(["10S", "6H"], ["6C", "BACK"], ["5H"])
    assert (fallback.play, fallback.tone, fallback.text, fallback.gap) == ("stand", "normal", "Stand", None)
    assert "basic strategy" in fallback.title


def test_no_tip_where_basic_strategy_has_none():
    assert tip(["10S"], ["6C", "BACK"], shoe_without("10S", "6C")) is None          # one card
    assert tip(["AS", "KH"], ["6C", "BACK"], shoe_without("AS", "KH", "6C")) is None  # blackjack
    assert tip(["10S", "6H"], ["BACK"], shoe_without("10S", "6H")) is None          # no dealer card up


@pytest.mark.parametrize(("true", "play", "tone", "text"), [
    (4.0, "bet-more", "firm", "Bet more."), (2.0, "bet-more", "normal", "Bet more"),
    (1.9, "bet-flat", "normal", "Bet flat"), (-0.9, "bet-flat", "normal", "Bet flat"),
    (-1.0, "bet-less", "normal", "Bet less"), (-2.0, "bet-less", "firm", "Bet less."),
])
def test_bet_tips_follow_the_hi_lo_thresholds(true, play, tone, text):
    result = bet_tip(true, "hi_lo")
    assert (result.play, result.tone, result.text, result.gap) == (play, tone, text, None)
    assert result.title.startswith(f"true count {true:+.1f}")


def test_every_counting_system_has_bet_thresholds_either_side_of_zero():
    for system_id, system in COUNT_SYSTEMS.items():
        assert system.bet_less_at < 0 < system.bet_more_at
        assert bet_tip(system.bet_more_at, system_id).play == "bet-more"
        assert bet_tip(system.bet_less_at, system_id).play == "bet-less"
