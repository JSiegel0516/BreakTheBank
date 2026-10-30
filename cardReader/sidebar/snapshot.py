"""The reader's state for one frame, as the plain dict the browser dashboard renders,
and the buttons the page offers in place of keys.

build_snapshot depends only on its arguments, so it can be tested without a camera,
a window, or a server. Every value is a JSON type. "timing" changes on almost every
frame, so the server leaves it out when deciding whether anything changed.

A page button posts its action name; read.py presses ACTION_KEYS[name] for it, so a
button does exactly what its key does.
"""
from __future__ import annotations

from ..blackjack import CARDS_PER_DECK, hi_lo
from ..deck import SUITS
from ..hud import EDIT_HELP, GRID_RANKS, KEY_LIST, DealtCards, count_cue, seat_title
from ..reader import Reader
from ..seats import DEALER, MAX_PLAYERS
from ..strategy import hint
from ..table import OTHER, Hand

# The page's buttons, by the KEY_LIST key they stand for: (action name, the key it presses).
# Keys left out (E, V, L, S) work in the window only.
BUTTONS = {
    "1-7": [(f"players_{n}", str(n)) for n in range(1, MAX_PLAYERS + 1)],
    "G": [("grey_boxes", "g")],
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


def build_snapshot(reader: Reader, *, fps: float, stage_ms: dict[str, float], no_signal: bool,
                   capture_mode: bool, showing_raw: bool, flash: tuple[str, float] | None) -> dict:
    """The dashboard's view of reader (its tracker, layout, count, and modes) this frame.

    stage_ms: smoothed milliseconds per stage; "detect" and "preprocess" are shown.
    showing_raw: the window shows the raw camera frame rather than the preprocessed one.
    flash: Hud.pop_flash(), a message (text, seconds) flashed since the last snapshot, or None.
    """
    count = reader.count
    dealt = DealtCards.from_tracker(reader.tracker)
    hands = reader.table.hands(reader.tracker.round)
    dealer_labels = next((hand.labels for hand in hands if hand.seat == DEALER), [])
    return {
        "count": {
            "headline": f"{count.true:+.1f}",
            "cue": count_cue(count.true)[0],
            "running": f"{count.running:+d}",
            "cards_left": count.cards_left,
            "deck_size": CARDS_PER_DECK,
        },
        "grid": [_grid_row(rank, dealt) for rank in GRID_RANKS],
        "unrevealed": dealt.unrevealed,
        "hands": [_hand(hand, dealer_labels) for hand in hands],
        "players": reader.layout.players,
        "edit_help": EDIT_HELP if reader.editor.active else [],
        "keys": [_key(key, label) for key, label in KEY_LIST],
        "preprocess": reader.preprocess,
        "showing_raw": showing_raw,
        "capture_mode": capture_mode,
        "no_signal": no_signal,
        "flash": {"text": flash[0], "seconds": flash[1]} if flash else None,
        "timing": {
            "fps": round(fps, 1),
            "detect_ms": round(stage_ms.get("detect", 0.0), 1),
            "preprocess_ms": round(stage_ms.get("preprocess", 0.0), 1),
        },
    }


def _key(key: str, label: str) -> dict:
    """One line of the keys list, with its page buttons (none for a window-only key)."""
    buttons = [{"action": action, "text": text} for action, text in BUTTONS.get(key, [])]
    return {"key": key, "label": label, "buttons": buttons}


def _grid_row(rank: str, dealt: DealtCards) -> dict:
    """One rank's row of the Cards Left grid: each suit's state, and how many are left."""
    cards = []
    for suit in SUITS:
        name = rank + suit
        cards.append({"name": name, "rank": rank, "suit": suit, "state": dealt.state(name),
                      "duplicate": name in dealt.duplicates})
    left = sum(card["state"] == "unseen" for card in cards)
    return {"rank": rank, "hi_lo": hi_lo(rank + SUITS[0]), "cards": cards, "left": left}


def _hand(hand: Hand, dealer_labels: list[str]) -> dict:
    """One seat's block. Only player seats get a strategy hint."""
    kind = {DEALER: "dealer", OTHER: "other"}.get(hand.seat, "player")
    cards = [{"order": card.order, "label": card.label, "hidden": card.hidden} for card in hand.cards]
    return {
        "seat": seat_title(hand.seat),
        "kind": kind,
        "cards": cards,
        "total": hand.total,
        "hint": hint(hand.labels, dealer_labels) if kind == "player" else None,
    }
