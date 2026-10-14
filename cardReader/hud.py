"""Draw the reader's overlay on top of the camera feed.

    * seat outlines: the dealer strip and the player arc, with their names
    * every corner index the detector sees: thin grey box until the tracker
      confirms the card, then a box in the card's colour
    * a tag on each card of the round: its placement number in a yellow badge,
      then its name. A hidden (covered) card keeps a grey tag where it was last seen.
    * top left: each seat's cards in placement order and its blackjack total
    * top right: the Hi-Lo running count, true count, and cards left in the deck
    * bottom bar: the keys, fps, and short messages such as "new round"

Colours follow the cards: red for hearts and diamonds, white for spades and
clubs, amber for face-down cards. Text and line sizes scale with the frame
width so the overlay stays readable from 720p to 4K.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from .blackjack import ShoeCount, describe
from .deck import BACK, RED_SUITS, suit_of
from .seats import DEALER, SeatLayout
from .tracker import Card, Detection

FONT = cv2.FONT_HERSHEY_DUPLEX
PANEL = (25, 25, 25)
WHITE = (255, 255, 255)
GREY = (150, 150, 150)
DIM = (95, 95, 95)
RED = (70, 70, 255)
GREEN = (90, 220, 90)
AMBER = (0, 190, 255)
BADGE = (0, 220, 255)
SEAT_LINE = (255, 220, 120)
KEYS = "[1-7] players    [R] new round    [Shift+R] new shoe    [S] save frame    [Q] quit"


def card_colour(label: str):
    if label == BACK:
        return AMBER
    return RED if suit_of(label) in RED_SUITS else WHITE


def seat_title(name: str) -> str:
    return "DEALER" if name == DEALER else name


class Hud:
    def __init__(self):
        self._message = ""
        self._message_until = 0.0

    def flash(self, text: str, seconds: float = 1.5):
        """Show a short message above the bottom bar."""
        self._message, self._message_until = text, time.time() + seconds

    def draw(self, frame: np.ndarray, dets: list[Detection], cards: list[Card], fps: float,
             layout: SeatLayout, count: ShoeCount, capture_mode: bool = False) -> np.ndarray:
        """cards: every card of the round, hidden ones included, in placement order."""
        view = frame.copy()
        W = view.shape[1]
        self.s = min(max(W / 1600, 0.6), 2.5)  # 1.2 at 1080p, 2.4 at 4K
        self.thick = max(1, round(2 * self.s))
        self.pad = round(10 * self.s)
        self.r = round(15 * self.s)

        self._seat_outlines(view, layout)
        visible = {c.label for c in cards if not c.hidden}
        for d in dets:
            x1, y1, x2, y2 = map(int, d.box)
            if d.label in visible:
                cv2.rectangle(view, (x1, y1), (x2, y2), card_colour(d.label), self.thick)
            else:
                cv2.rectangle(view, (x1, y1), (x2, y2), GREY, 1)

        seat_of = {c.order: layout.assign(c.x, c.y) for c in cards}
        panels = [self._seat_panel(view, cards, seat_of, layout), self._count_panel(view, count)]
        placed: list[tuple] = []
        for c in cards:
            placed.append(self._card_tag(view, c, None if c.hidden else self._card_box(c, dets), panels, placed))
        self._bottom_bar(view, fps, capture_mode)
        return view

    # ------------------------------------------------------------ drawing helpers
    def _text(self, img, text, org, scale, colour):
        cv2.putText(img, text, (int(org[0]), int(org[1])), FONT, scale * self.s, colour, self.thick, cv2.LINE_AA)

    def _size(self, text, scale):
        (w, h), base = cv2.getTextSize(text, FONT, scale * self.s, self.thick)
        return w, h, base

    def _panel(self, img, x1, y1, x2, y2, alpha=0.7):
        H, W = img.shape[:2]
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(W, int(x2)), min(H, int(y2))
        if x2 <= x1 or y2 <= y1:
            return
        roi = img[y1:y2, x1:x2]
        img[y1:y2, x1:x2] = (roi * (1 - alpha) + np.array(PANEL) * alpha).astype(np.uint8)

    def _badge(self, img, number, cx, cy, colour=BADGE, scale=0.6):
        cv2.circle(img, (int(cx), int(cy)), self.r, colour, -1, cv2.LINE_AA)
        w, h, _ = self._size(str(number), scale)
        self._text(img, str(number), (cx - w // 2, cy + h // 2), scale, (0, 0, 0))

    def _chip(self, img, card: Card, x, cy):
        """Badge + card name at (x, centre y). Hidden cards are drawn greyed out. Returns its width."""
        badge, colour = (DIM, GREY) if card.hidden else (BADGE, card_colour(card.label))
        self._badge(img, card.order, x + self.r, cy, badge)
        w, h, _ = self._size(card.label, 0.8)
        self._text(img, card.label, (x + 2 * self.r + self.pad // 2, cy + h // 2), 0.8, colour)
        return 2 * self.r + self.pad // 2 + w

    # ------------------------------------------------------------ pieces
    def _seat_outlines(self, img, layout: SeatLayout):
        for seat in layout.seats:
            colour = AMBER if seat.name == DEALER else SEAT_LINE
            cv2.polylines(img, [seat.polygon], True, colour, max(1, self.thick // 2), cv2.LINE_AA)
            title = seat_title(seat.name)
            w, h, _ = self._size(title, 0.6)
            top = seat.polygon[np.argmin(seat.polygon[:, 1])]
            x, y = top[0] - w // 2, top[1] + h + self.pad
            self._panel(img, x - self.pad // 2, y - h - self.pad // 2, x + w + self.pad // 2, y + self.pad // 2, 0.6)
            self._text(img, title, (x, y), 0.6, colour)

    def _seat_panel(self, img, cards, seat_of, layout: SeatLayout):
        """One row per seat: name, its cards in order, its total. Returns the panel's rectangle."""
        rows = [(name, [c for c in cards if seat_of[c.order] == name]) for name in layout.names()]
        loose = [c for c in cards if seat_of[c.order] is None]
        if loose:
            rows.append(("OTHER", loose))
        name_w = max(self._size(seat_title(n), 0.6)[0] for n, _ in rows) + 2 * self.pad
        item_h = 2 * self.r + self.pad
        widths = []
        for name, hand in rows:
            w = sum(2 * self.r + self.pad // 2 + self._size(c.label, 0.8)[0] + self.pad for c in hand)
            total = describe([c.label for c in hand]) if name != "OTHER" else ""
            widths.append(w + self._size(total, 0.7)[0] + self.pad)
        x0, y0 = self.pad, self.pad
        x1, y1 = x0 + 2 * self.pad + name_w + max(widths), y0 + self.pad + len(rows) * item_h
        self._panel(img, x0, y0, x1, y1)
        y = y0 + self.pad // 2 + item_h // 2
        for name, hand in rows:
            _, nh, _ = self._size(seat_title(name), 0.6)
            self._text(img, seat_title(name), (x0 + self.pad, y + nh // 2), 0.6,
                       AMBER if name == DEALER else SEAT_LINE if name != "OTHER" else GREY)
            x = x0 + self.pad + name_w
            for c in hand:
                x += self._chip(img, c, x, y) + self.pad
            if name != "OTHER":
                total = describe([c.label for c in hand])
                _, th, _ = self._size(total, 0.7)
                self._text(img, total, (x + self.pad // 2, y + th // 2), 0.7, WHITE if hand else DIM)
            y += item_h
        return x0, y0, x1, y1

    def _count_panel(self, img, count: ShoeCount):
        """Hi-Lo running count, true count, cards left. Returns the panel's rectangle."""
        W = img.shape[1]
        colour = GREEN if count.running > 0 else RED if count.running < 0 else WHITE
        parts = [("RUNNING", f"{count.running:+d}", colour), ("TRUE", f"{count.true:+.1f}", colour),
                 ("LEFT", str(count.cards_left), WHITE)]
        gap = 2 * self.pad
        sizes = [(self._size(k, 0.5), self._size(v, 0.9)) for k, v, _ in parts]
        inner_w = sum(max(a[0], b[0]) for a, b in sizes) + gap * (len(parts) - 1)
        kh = max(a[1] for a, _ in sizes)
        vh = max(b[1] for _, b in sizes)
        panel_w, panel_h = inner_w + 2 * self.pad, kh + vh + 3 * self.pad
        x0, y0 = W - self.pad - panel_w, self.pad
        self._panel(img, x0, y0, x0 + panel_w, y0 + panel_h)
        x = x0 + self.pad
        for (k, v, c), (a, b) in zip(parts, sizes):
            self._text(img, k, (x, y0 + self.pad + kh), 0.5, GREY)
            self._text(img, v, (x, y0 + 2 * self.pad + kh + vh), 0.9, c)
            x += max(a[0], b[0]) + gap
        return x0, y0, x0 + panel_w, y0 + panel_h

    @staticmethod
    def _card_box(card: Card, dets: list[Detection]):
        """The topmost corner index of this card in this frame, or None if none was seen.

        A face card is any detection with its name (one deck, so they are all
        this card). A face-down card is the BACK detection nearest its position."""
        if card.label == BACK:
            mine = [d for d in dets if d.label == BACK]
            return min(mine, key=lambda d: (d.cx - card.x) ** 2 + (d.cy - card.y) ** 2).box if mine else None
        mine = [d for d in dets if d.label == card.label]
        return min(mine, key=lambda d: d.box[1]).box if mine else None

    def _card_tag(self, img, card: Card, box, panels, tags):
        """Badge + name just above the card's topmost corner index (below it if there is no room).
        panels: rectangles a tag must not cover; tags: tags already drawn, which it stacks above.
        Returns the tag's rectangle."""
        H, W = img.shape[:2]
        w, h, base = self._size(card.label, 0.8)
        tag_w, tag_h = 2 * self.r + self.pad + w + self.pad, max(2 * self.r, h + base) + self.pad
        if box is None:  # hidden, or not detected this frame: use the last known position
            box = (card.x, card.y, card.x, card.y)
        x = int(min(max(box[0], 0), W - tag_w))
        y = int(box[1] - self.pad // 2 - tag_h)
        if y < 0:
            y = int(box[3] + self.pad // 2)
        def hits(rects, y):
            return next(((ay1, ay2) for ax1, ay1, ax2, ay2 in rects
                         if x < ax2 and x + tag_w > ax1 and y < ay2 and y + tag_h > ay1), None)

        for _ in range(len(tags) + 1):  # stack above any tag in the way, as long as there is room
            hit = hits(tags, y)
            if hit is None or hit[0] - tag_h - self.pad // 2 < 0:
                break
            y = hit[0] - tag_h - self.pad // 2
        hit = hits(panels, y)
        if hit is not None:
            y = hit[1] + self.pad // 2
        y = int(min(max(y, 0), H - tag_h))
        self._panel(img, x, y, x + tag_w, y + tag_h, alpha=0.8)
        self._chip(img, card, x + self.pad // 2, y + tag_h // 2)
        return x, y, x + tag_w, y + tag_h

    def _bottom_bar(self, img, fps: float, capture_mode: bool):
        H, W = img.shape[:2]
        _, kh, base = self._size(KEYS, 0.5)
        bar_h = kh + base + 2 * self.pad
        self._panel(img, 0, H - bar_h, W, H, alpha=0.75)
        ty = H - self.pad - base
        self._text(img, KEYS, (self.pad, ty), 0.5, WHITE)
        status = f"{fps:4.1f} fps"
        if capture_mode:
            status = "CAPTURE MODE (no model)    " + status
        sw = self._size(status, 0.5)[0]
        self._text(img, status, (W - sw - self.pad, ty), 0.5, AMBER if capture_mode else GREY)

        if self._message and time.time() < self._message_until:
            mw, mh, mb = self._size(self._message, 0.9)
            x = (W - mw) // 2
            y2 = H - bar_h - self.pad
            self._panel(img, x - 2 * self.pad, y2 - mh - mb - 2 * self.pad, x + mw + 2 * self.pad, y2, alpha=0.85)
            self._text(img, self._message, (x, y2 - self.pad - mb), 0.9, BADGE)
