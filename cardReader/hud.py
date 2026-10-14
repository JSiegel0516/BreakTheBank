"""Draw the reader's overlay on top of the camera feed.

    * every corner index the detector sees: thin grey box until the tracker
      confirms the card, then a box in the card's colour
    * a tag on each confirmed card: its placement number in a yellow badge,
      then its name
    * top panel: the cards on the table in the order they were placed
    * bottom bar: the keys, fps, and short messages such as "frame saved"

Colours follow the cards: red for hearts and diamonds, white for spades and
clubs, amber for face-down cards. Text and line sizes scale with the frame
width so the overlay stays readable from 720p to 4K.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from .deck import BACK, RED_SUITS, suit_of
from .tracker import Card, Detection

FONT = cv2.FONT_HERSHEY_DUPLEX
PANEL = (25, 25, 25)
WHITE = (255, 255, 255)
GREY = (150, 150, 150)
RED = (70, 70, 255)
AMBER = (0, 190, 255)
BADGE = (0, 220, 255)
KEYS = "[Q] quit      [S] save frame      [R] reset count"


def card_colour(label: str):
    if label == BACK:
        return AMBER
    return RED if suit_of(label) in RED_SUITS else WHITE


class Hud:
    def __init__(self):
        self._message = ""
        self._message_until = 0.0

    def flash(self, text: str, seconds: float = 1.5):
        """Show a short message above the bottom bar."""
        self._message, self._message_until = text, time.time() + seconds

    def draw(self, frame: np.ndarray, dets: list[Detection], cards: list[Card], fps: float,
             capture_mode: bool = False) -> np.ndarray:
        view = frame.copy()
        H, W = view.shape[:2]
        self.s = min(max(W / 1600, 0.6), 2.5)  # 1.2 at 1080p, 2.4 at 4K
        self.thick = max(1, round(2 * self.s))
        self.pad = round(10 * self.s)

        on_table = {c.label for c in cards}
        for d in dets:
            x1, y1, x2, y2 = map(int, d.box)
            if d.label in on_table:
                cv2.rectangle(view, (x1, y1), (x2, y2), card_colour(d.label), self.thick)
            else:
                cv2.rectangle(view, (x1, y1), (x2, y2), GREY, 1)
        panel = self._order_panel(view, cards)
        for c in cards:
            self._card_tag(view, c, self._card_box(c, dets), panel)
        self._bottom_bar(view, fps, capture_mode)
        return view

    # ------------------------------------------------------------ pieces
    def _text(self, img, text, org, scale, colour):
        cv2.putText(img, text, org, FONT, scale * self.s, colour, self.thick, cv2.LINE_AA)

    def _size(self, text, scale):
        (w, h), base = cv2.getTextSize(text, FONT, scale * self.s, self.thick)
        return w, h, base

    def _panel(self, img, x1, y1, x2, y2, alpha=0.7):
        H, W = img.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            return
        roi = img[y1:y2, x1:x2]
        img[y1:y2, x1:x2] = (roi * (1 - alpha) + np.array(PANEL) * alpha).astype(np.uint8)

    def _badge(self, img, number, cx, cy, scale=0.6):
        r = round(15 * self.s)
        cv2.circle(img, (cx, cy), r, BADGE, -1, cv2.LINE_AA)
        w, h, _ = self._size(str(number), scale)
        self._text(img, str(number), (cx - w // 2, cy + h // 2), scale, (0, 0, 0))
        return r

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

    def _card_tag(self, img, card: Card, box, avoid):
        """Badge + name just above the card's topmost corner index (below it if there is no room).
        avoid: the order panel's rectangle, which a tag must not cover."""
        H, W = img.shape[:2]
        r = round(15 * self.s)
        w, h, base = self._size(card.label, 0.8)
        tag_w, tag_h = 2 * r + self.pad + w + self.pad, max(2 * r, h + base) + self.pad
        if box is None:  # hidden this frame: use the last known position
            box = (card.x, card.y, card.x, card.y)
        x = int(min(max(box[0], 0), W - tag_w))
        y = int(box[1] - self.pad // 2 - tag_h)
        if y < 0:
            y = int(box[3] + self.pad // 2)
        ax1, ay1, ax2, ay2 = avoid
        if x < ax2 and x + tag_w > ax1 and y < ay2 and y + tag_h > ay1:
            y = ay2 + self.pad // 2
        y = int(min(max(y, 0), H - tag_h))
        self._panel(img, x, y, x + tag_w, y + tag_h, alpha=0.8)
        cy = y + tag_h // 2
        self._badge(img, card.order, x + r + self.pad // 2, cy)
        self._text(img, card.label, (x + 2 * r + self.pad, cy + h // 2), 0.8, card_colour(card.label))

    def _order_panel(self, img, cards: list[Card]):
        """The placement order, wrapped onto as many lines as it needs. Returns its rectangle."""
        W = img.shape[1]
        title = f"CARDS IN ORDER ({len(cards)})" if cards else "CARDS IN ORDER: none yet"
        tw, th, _ = self._size(title, 0.55)
        max_w = W - 4 * self.pad
        r = round(15 * self.s)
        item_h = max(2 * r, self._size("10H", 0.8)[1]) + self.pad
        # lay items out in rows first, then size the panel to fit
        rows, row, row_w = [], [], 0
        for c in cards:
            w = 2 * r + self.pad // 2 + self._size(c.label, 0.8)[0] + 2 * self.pad
            if row and row_w + w > max_w:
                rows.append(row)
                row, row_w = [], 0
            row.append((c, w))
            row_w += w
        if row:
            rows.append(row)
        panel_w = max(tw, max((sum(w for _, w in rw) for rw in rows), default=0)) + 2 * self.pad
        panel_h = self.pad + th + self.pad + len(rows) * item_h + self.pad // 2
        x0, y0 = self.pad, self.pad
        self._panel(img, x0, y0, x0 + panel_w, y0 + panel_h)
        self._text(img, title, (x0 + self.pad, y0 + self.pad + th), 0.55, GREY)
        y = y0 + 2 * self.pad + th + item_h // 2
        for rw in rows:
            x = x0 + self.pad
            for c, w in rw:
                self._badge(img, c.order, x + r, y)
                lh = self._size(c.label, 0.8)[1]
                self._text(img, c.label, (x + 2 * r + self.pad // 2, y + lh // 2), 0.8, card_colour(c.label))
                x += w
            y += item_h
        return x0, y0, x0 + panel_w, y0 + panel_h

    def _bottom_bar(self, img, fps: float, capture_mode: bool):
        H, W = img.shape[:2]
        _, kh, base = self._size(KEYS, 0.55)
        bar_h = kh + base + 2 * self.pad
        self._panel(img, 0, H - bar_h, W, H, alpha=0.75)
        ty = H - self.pad - base
        self._text(img, KEYS, (self.pad, ty), 0.55, WHITE)
        status = f"{fps:4.1f} fps"
        if capture_mode:
            status = "CAPTURE MODE (no model)    " + status
        sw = self._size(status, 0.55)[0]
        self._text(img, status, (W - sw - self.pad, ty), 0.55, AMBER if capture_mode else GREY)

        if self._message and time.time() < self._message_until:
            mw, mh, mb = self._size(self._message, 0.9)
            x = (W - mw) // 2
            y2 = H - bar_h - self.pad
            self._panel(img, x - 2 * self.pad, y2 - mh - mb - 2 * self.pad, x + mw + 2 * self.pad, y2, alpha=0.85)
            self._text(img, self._message, (x, y2 - self.pad - mb), 0.9, BADGE)
