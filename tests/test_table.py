from cardReader.cards import Card
from cardReader.seats import DEALER, SeatLayout
from cardReader.table import OTHER, Table


def card(order, label, seat_polygon, hidden=False):
    cx, cy = seat_polygon.mean(axis=0)
    return Card(order, label, float(cx), float(cy), 1, hidden=hidden)


def test_hands_follow_layout_order_and_collect_loose_cards_last():
    layout = SeatLayout(3, 1920, 1080)
    table = Table(layout)
    cards = [card(1, "KH", layout.seats[1].polygon), card(2, "7D", layout.seats[0].polygon),
             card(3, "AS", layout.seats[1].polygon), Card(4, "2C", 5, 1075, 1)]  # 2C in the frame corner
    hands = table.hands(cards)
    assert [h.seat for h in hands] == [DEALER, "P1", "P2", "P3", OTHER]
    assert hands[1].labels == ["KH", "AS"] and hands[1].total == "BLACKJACK"
    assert hands[0].labels == ["7D"] and hands[0].total == "7"
    assert hands[2].cards == [] and hands[2].total == "-"
    assert hands[4].labels == ["2C"] and hands[4].total == ""
    assert table.seat_of(cards[3]) is None


def test_hidden_cards_stay_in_their_hand():
    layout = SeatLayout(1, 1280, 720)
    cards = [card(1, "KS", layout.seats[1].polygon), card(2, "BACK", layout.seats[1].polygon, hidden=True)]
    h = Table(layout).hands(cards)[1]
    assert h.labels == ["KS", "BACK"] and h.total == "10 +1 hidden"


def test_no_other_hand_when_every_card_is_seated():
    layout = SeatLayout(2, 1280, 720)
    hands = Table(layout).hands([card(1, "5C", layout.seats[2].polygon)])
    assert [h.seat for h in hands] == [DEALER, "P1", "P2"]
