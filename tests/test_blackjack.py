from cardVision.blackjack import describe, hand_value, is_blackjack


def test_values():
    assert hand_value(["KH", "7S"]) == (17, False)
    assert hand_value(["AH", "7S"]) == (18, True)
    assert hand_value(["AH", "AS", "9D"]) == (21, True)
    assert hand_value(["AH", "KS", "5C"]) == (16, False)
    assert hand_value(["10H", "9S", "5C"]) == (24, False)


def test_blackjack_and_hidden():
    assert is_blackjack(["AH", "KS"])
    assert not is_blackjack(["AH", "5S", "5D"])
    assert describe(["AH", "KS"]) == "BLACKJACK"
    assert describe(["10H", "9S", "5C"]) == "BUST (24)"
    assert describe(["KS", "BACK"]) == "10 +1 hidden"
    assert describe([]) == "-"
