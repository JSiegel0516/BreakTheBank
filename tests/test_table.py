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


def split_table():
    """A one-player table with 8S left of 8H in P1, split. Returns the table, the cards, and P1's centre."""
    layout = SeatLayout(1, 1280, 720)
    cx, cy = layout.seats[1].polygon.mean(axis=0)
    cards = [Card(1, "8H", cx + 40, cy, 1), Card(2, "8S", cx - 40, cy, 1)]
    table = Table(layout)
    table.split("P1", cards)
    return table, cards, (cx, cy)


def test_a_split_seat_gives_two_hands_with_the_left_card_as_part_1():
    table, cards, _ = split_table()
    assert table.splits == {"P1": (2, 1)}
    hands = table.hands(cards)
    assert [(h.seat, h.part, h.labels) for h in hands] == [
        (DEALER, None, []), ("P1", 1, ["8S"]), ("P1", 2, ["8H"])]
    assert hands[1].total == hands[2].total == "8"


def test_later_cards_join_the_hand_of_the_nearest_seed():
    table, cards, (cx, cy) = split_table()
    cards += [Card(3, "3D", cx - 30, cy + 20, 1), Card(4, "KC", cx + 25, cy - 20, 1)]
    assert all(table.seat_of(c) == "P1" for c in cards)
    first, second = table.hands(cards)[1:]
    assert first.labels == ["8S", "3D"] and first.total == "11"
    assert second.labels == ["8H", "KC"] and second.total == "18"


def test_a_split_drops_back_to_one_hand_when_a_seed_leaves_the_seat():
    table, cards, _ = split_table()
    hands = table.hands(cards[1:])                       # the 8H is taken away
    assert [(h.seat, h.part, h.labels) for h in hands] == [(DEALER, None, []), ("P1", None, ["8S"])]
    assert table.splits == {}
    assert len(table.hands(cards)) == 2                  # the 8H back does not split the seat again


def test_join_and_clear_splits_make_one_hand_again():
    table, cards, _ = split_table()
    table.join("P1")
    assert [h.part for h in table.hands(cards)] == [None, None]
    table.split("P1", cards)
    table.clear_splits()
    assert table.splits == {} and len(table.hands(cards)) == 2
