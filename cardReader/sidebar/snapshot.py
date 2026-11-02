"""The reader's state for one frame, as the plain dict the browser dashboard renders,
and the buttons the page offers in place of keys.

build_snapshot depends only on its arguments, so it can be tested without a camera,
a window, or a server. Every value is a JSON type. "timing" changes on almost every
frame, so the server leaves it out when deciding whether anything changed.

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
    dealer_odds,
    is_pair,
    milestone,
    outcome,
    player_odds,
)
from ..deck import SUITS
from ..hud import EDIT_HELP, GRID_RANKS, HIGH_RANKS, KEY_LIST, DealtCards, count_cue, seat_title
from ..reader import Reader
from ..seats import DEALER, MAX_PLAYERS
from ..strategy import hint
from ..table import OTHER, Hand

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
    cards left of "deck_size" (52 per deck), and the unseen tens and aces ("high_left" of
    "high_total") and aces ("aces_left" of "aces_total"), counted in copies, which match
    the grid's "left" column. decks: the decks in the shoe, shown on the decks button.
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
    return {
        "count": {
            "headline": f"{count.true:+.1f}",
            "cue": count_cue(count.true)[0],
            "running": f"{count.running:+d}",
            "cards_left": count.cards_left,
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
        "hands": [_hand(hand, dealer_labels, reader.seat_names, unseen) for hand in hands],
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


def _hand(hand: Hand, dealer_labels: list[str], names: dict[str, str], unseen: list[str]) -> dict:
    """One hand's block: its seat id ("dealer", "P1", "OTHER"), the seat's display name, and its cards.

    part: 1 or 2 for a hand of a split seat, else None; the page shows it after the name.
    can_split: a player seat that is not split and holds a pair, so the page offers a split button.
    Only player seats get a strategy hint, which never says split for a hand of a split seat.
    outcome: blackjack.outcome against the dealer for a player hand; the dealer's own
    blackjack.milestone ('blackjack', 'twenty-one', 'bust', or None); None for OTHER.
    odds: the hand's chances from the unseen card labels, blackjack.player_odds for a player
    hand and blackjack.dealer_odds for the dealer, such as {"blackjack": 7}; None for OTHER
    and for a hand with no odds row."""
    kind = {DEALER: "dealer", OTHER: "other"}.get(hand.seat, "player")
    cards = [{"order": card.order, "label": card.label, "hidden": card.hidden} for card in hand.cards]
    unsplit = hand.part is None
    return {
        "seat": hand.seat,
        "name": seat_title(hand.seat, names),
        "part": hand.part,
        "kind": kind,
        "cards": cards,
        "total": hand.total,
        "hint": hint(hand.labels, dealer_labels, can_split=unsplit) if kind == "player" else None,
        "can_split": kind == "player" and unsplit and is_pair(hand.labels),
        "outcome": _outcome(kind, hand.labels, dealer_labels),
        "odds": _odds(kind, hand.labels, dealer_labels, unseen),
    }


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
        return dealer_odds(labels, unseen)
    return None
