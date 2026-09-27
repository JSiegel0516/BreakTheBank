"""The 53 class labels the detector is trained on, and helpers to read them.

A label is rank + suit, e.g. "10H" (ten of hearts) or "AS" (ace of spades).
"BACK" is a face-down card. The position of a label in CLASSES is its numeric
class id in every label file and in the trained model.
"""

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
SUITS = ["S", "H", "D", "C"]  # spades, hearts, diamonds, clubs
BACK = "BACK"

CLASSES = [rank + suit for suit in SUITS for rank in RANKS] + [BACK]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}

# Suit glyphs, written as escape codes so this file stays plain ASCII.
# Only the synthetic data generator uses these, to draw placeholder cards.
SUIT_SYMBOL = {"S": "\u2660", "H": "\u2665", "D": "\u2666", "C": "\u2663"}
RED_SUITS = {"H", "D"}


def is_back(label: str) -> bool:
    return label == BACK


def rank_of(label: str) -> str:
    return label[:-1]


def suit_of(label: str) -> str:
    return label[-1]


def display(label: str) -> str:
    """Short text for on-screen overlays. OpenCV cannot draw suit glyphs, so keep it ASCII."""
    return "??" if is_back(label) else label
