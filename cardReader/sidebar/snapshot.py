"""The reader's state for one frame, as the plain dict the browser dashboard renders,
and the buttons the page offers in place of keys.

build_snapshot depends only on its arguments, so it can be tested without a camera,
a window, or a server. Every value is a JSON type. "timing" changes on almost every
frame, so the server leaves it out when deciding whether anything changed.
Each hand carries the figures for its odds tiles (see _hand): the next card's odds and the
stand odds for a player hand, and the dealer's final totals and an insurance call for the dealer.
A finished hand carries instead how likely its result was ("was").

A page button posts its action name; read.py presses ACTION_KEYS[name] for it, so a
button does exactly what its key does. Renaming a seat posts "rename_seat" with the
seat id and the new name, which read.py passes to Reader.rename_seat. A hand's split
and join buttons post "split_seat" and "join_seat" with the seat id, for
Reader.split_seat and Reader.join_seat; they stand in for keys Shift+1-5, which
have no button in the keys list. Choosing a model posts "switch_model" with its
weights path, which read.LiveDetector loads. Choosing a counting system posts
"switch_count" with its COUNT_SYSTEMS id, which read.py passes to Reader.set_count_system.
The decks button posts "cycle_decks", for Reader.cycle_decks; there is no key for it.

Each entry of the keys list names its group (KEY_GROUPS): table, overlay, image,
camera, or window. The page shows each group's keys side by side on one row.
"""
from __future__ import annotations

from ..blackjack import (
    CARDS_PER_DECK,
    COUNT_SYSTEMS,
    CountSystem,
    dealer_final_chance,
    dealer_final_odds,
    insurance,
    is_pair,
    last_card_odds,
    milestone,
    outcome,
    player_odds,
    stand_odds,
    stood_odds,
)
from ..deck import SUITS
from ..hud import EDIT_HELP, GRID_RANKS, HIGH_RANKS, KEY_LIST, DealtCards, count_cue, seat_title
from ..reader import Reader
from ..seats import DEALER, MAX_PLAYERS
from ..table import OTHER, Hand
from ..tips import Tip, bet_tip, tip

# The page's buttons, by the KEY_LIST key they stand for: (action name, the key it presses).
# Keys left out (E, V, L, S) work in the window only; Shift+1-5 has its buttons on the hands instead.
BUTTONS = {
    f"1-{MAX_PLAYERS}": [(f"players_{n}", str(n)) for n in range(1, MAX_PLAYERS + 1)],
    "G": [("grey_boxes", "g")],
    "N": [("placement_numbers", "n")],
    "R": [("new_round", "r")],
    "X": [("rescan", "x")],
    "Shift+R": [("new_shoe", "R")],
    "C": [("reconnect", "c")],
    "P": [("preprocess", "p")],
    "Shift+V": [("raw_or_enhanced", "V")],
    "< >": [("turn_left", "<"), ("turn_right", ">")],
    "Q": [("quit", "q")],
}
ACTION_KEYS = {action: ord(key) for buttons in BUTTONS.values() for action, key in buttons}  # name -> key code
# The page's row for each KEY_LIST key. Rows appear in KEY_LIST order, which keeps each group's keys together.
KEY_GROUPS = {
    f"1-{MAX_PLAYERS}": "table", "R": "table", "X": "table", "Shift+R": "table", f"Shift+1-{MAX_PLAYERS}": "table",
    "G": "overlay", "N": "overlay",
    "P": "image", "Shift+V": "image", "< >": "image",
    "C": "camera", "Q": "camera",
    "E": "window", "V": "window", "L": "window", "S": "window",
}


def build_snapshot(reader: Reader, *, fps: float, stage_ms: dict[str, float], no_signal: bool,
                   capture_mode: bool, showing_raw: bool, flash: tuple[str, float] | None, model: dict) -> dict:
    """The dashboard's view of reader (its tracker, layout, count, and modes) this frame.

    count: the true count as "headline" with its "cue" (hud.count_cue), the running count,
    "cards_left" of "deck_size" (52 per deck), and the unseen tens and aces ("high_left" of
    "high_total") and aces ("aces_left" of "aces_total"), counted in copies, which match
    the grid's "left" column. cards_left is every unseen copy (DealtCards.unseen), so it is
    the sum of the grid's "left" column and the odds' card pool; it keeps the face-down cards
    never revealed ("unrevealed", shown beside it), unlike the true count's divisor
    (ShoeCount.cards_left). decks: the decks in the shoe, shown on the decks button.
    Each grid card gives "seen" of its "copies" (one per deck).
    count_system: the counting-system dropdown, {"current": the COUNT_SYSTEMS id reader counts
    in, "choices": [{"id", "name"}]}. Each grid row's "value" is its rank's value in that system.

    stage_ms: smoothed milliseconds per stage; "detect" and "preprocess" are shown.
    showing_raw: the window shows the raw camera frame rather than the preprocessed one.
    flash: Hud.pop_flash(), a message (text, seconds) flashed since the last snapshot, or None.
    model: the model dropdown, {"current": weights path or None, "choices": [{"path", "label"}]}
    (read.LiveDetector.snapshot), passed through as it is.
    """
    count = reader.count
    dealt = DealtCards.from_tracker(reader.tracker)
    hands = reader.table.hands(reader.tracker.round)
    dealer_labels = next((hand.labels for hand in hands if hand.seat == DEALER), [])
    unseen = dealt.unseen()
    bet = bet_tip(count.true, count.system)
    return {
        "count": {
            "headline": f"{count.true:+.1f}",
            "cue": count_cue(count.true)[0],
            "running": f"{count.running:+d}",
            "cards_left": len(unseen),
            "deck_size": count.decks * CARDS_PER_DECK,
            "high_left": dealt.left(HIGH_RANKS),
            "high_total": len(HIGH_RANKS) * len(SUITS) * dealt.decks,
            "aces_left": dealt.left(["A"]),
            "aces_total": len(SUITS) * dealt.decks,
        },
        "decks": reader.decks,
        "count_system": {
            "current": count.system,
            "choices": [{"id": system_id, "name": system.name} for system_id, system in COUNT_SYSTEMS.items()],
        },
        "grid": [_grid_row(rank, dealt, COUNT_SYSTEMS[count.system]) for rank in GRID_RANKS],
        "unrevealed": dealt.unrevealed,
        "hands": [_hand(hand, dealer_labels, reader.seat_names, unseen, bet) for hand in hands],
        "players": reader.layout.players,
        "edit_help": EDIT_HELP if reader.editor.active else [],
        "keys": [_key(key, label) for key, label in KEY_LIST],
        "preprocess": reader.preprocess,
        "showing_raw": showing_raw,
        "capture_mode": capture_mode,
        "model": model,
        "no_signal": no_signal,
        "flash": {"text": flash[0], "seconds": flash[1]} if flash else None,
        "timing": {
            "fps": round(fps, 1),
            "detect_ms": round(stage_ms.get("detect", 0.0), 1),
            "preprocess_ms": round(stage_ms.get("preprocess", 0.0), 1),
        },
    }


def _key(key: str, label: str) -> dict:
    """One entry of the keys list: its page buttons (none for a window-only key) and the row it sits on."""
    buttons = [{"action": action, "text": text} for action, text in BUTTONS.get(key, [])]
    return {"key": key, "label": label, "buttons": buttons, "group": KEY_GROUPS[key]}


def _grid_row(rank: str, dealt: DealtCards, counting: CountSystem) -> dict:
    """One rank's row of the Cards Left grid: its value in the counting system, each suit's state
    and copies seen, and how many copies are left."""
    cards = []
    for suit in SUITS:
        name = rank + suit
        cards.append({"name": name, "rank": rank, "suit": suit, "state": dealt.state(name),
                      "duplicate": name in dealt.duplicates, "seen": dealt.seen(name), "copies": dealt.decks})
    left = sum(dealt.copies_left(card["name"]) for card in cards)
    return {"rank": rank, "value": counting.values[rank], "cards": cards, "left": left}


def _hand(hand: Hand, dealer_labels: list[str], names: dict[str, str], unseen: list[str], bet: Tip) -> dict:
    """One hand's block: its seat id ("dealer", "P1", "OTHER"), the seat's display name, and its cards.

    part: 1 or 2 for a hand of a split seat, else None; the page shows it after the name.
    can_split: a player seat that is not split and holds a pair, so the page offers a split button.
    hint and tip: only player seats get them. A player hand gets tips.tip against the unseen
    cards, which never says split for a hand of a split seat; an empty player seat gets bet,
    the bet tip for the true count (tips.bet_tip). hint is the tip's play, which names the
    pill's colour ("stand", "bet-more", ...), and tip is {"text", "tone", "gap", "title"}
    (see tips.Tip); both None for no tip.
    outcome: blackjack.outcome against the dealer for a player hand; the dealer's own
    blackjack.milestone ('blackjack', 'twenty-one', 'bust', or None); None for OTHER.
    odds: the hand's chances from the unseen card labels: blackjack.player_odds for a player
    hand, such as {"blackjack": 7}, and blackjack.dealer_final_odds for the dealer, such as
    {"blackjack": 31, "21": 5, "20": 13, "19": 13, "18": 13, "17": 13, "bust": 12};
    None for OTHER and for a hand with no odds row.
    stand: blackjack.stand_odds for a player hand, {"win", "push", "lose"} if it stands now, else None.
    insurance: blackjack.insurance for the dealer, "take" or "skip" while an ace shows over a
    face-down card, else None. The dealer's finals are cached in blackjack, so the dealer's
    odds and every player's stand share one computation per frame.
    was: how likely a finished hand's result was, for the page's "Odds were" row. A player:
    blackjack.last_card_odds with a "stand" entry, blackjack.stood_odds or None, such as
    {"result": "21", "chance": 8, "stand": {"result": "win", "chance": 42}}. The dealer:
    blackjack.dealer_final_chance, such as {"result": "19", "chance": 13}. None for OTHER, for
    an unfinished hand, and when nothing is unseen."""
    kind = {DEALER: "dealer", OTHER: "other"}.get(hand.seat, "player")
    cards = [{"order": card.order, "label": card.label, "hidden": card.hidden} for card in hand.cards]
    unsplit = hand.part is None
    hand_tip = None
    if kind == "player":
        hand_tip = tip(hand.labels, dealer_labels, unseen, can_split=unsplit) if hand.cards else bet
    return {
        "seat": hand.seat,
        "name": seat_title(hand.seat, names),
        "part": hand.part,
        "kind": kind,
        "cards": cards,
        "total": hand.total,
        "hint": hand_tip.play if hand_tip else None,
        "tip": _tip_fields(hand_tip),
        "can_split": kind == "player" and unsplit and is_pair(hand.labels),
        "outcome": _outcome(kind, hand.labels, dealer_labels),
        "odds": _odds(kind, hand.labels, dealer_labels, unseen),
        "stand": stand_odds(hand.labels, dealer_labels, unseen) if kind == "player" else None,
        "insurance": insurance(hand.labels, unseen) if kind == "dealer" else None,
        "was": _was(kind, hand.labels, dealer_labels, unseen),
    }


def _tip_fields(hand_tip: Tip | None) -> dict | None:
    """What the page shows of a tip beside its colour: its text, tone, gap, and hover title."""
    if hand_tip is None:
        return None
    return {"text": hand_tip.text, "tone": hand_tip.tone, "gap": hand_tip.gap, "title": hand_tip.title}


def _outcome(kind: str, labels: list[str], dealer_labels: list[str]) -> str | None:
    if kind == "player":
        return outcome(labels, dealer_labels)
    if kind == "dealer":
        return milestone(labels)
    return None


def _odds(kind: str, labels: list[str], dealer_labels: list[str], unseen: list[str]) -> dict[str, int] | None:
    if kind == "player":
        return player_odds(labels, dealer_labels, unseen)
    if kind == "dealer":
        return dealer_final_odds(labels, unseen)
    return None


def _was(kind: str, labels: list[str], dealer_labels: list[str], unseen: list[str]) -> dict | None:
    if kind == "dealer":
        return dealer_final_chance(labels, unseen)
    if kind != "player":
        return None
    last_card = last_card_odds(labels, dealer_labels, unseen)
    if last_card is None:
        return None
    return last_card | {"stand": stood_odds(labels, dealer_labels, unseen)}
