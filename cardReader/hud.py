"""Draw the reader's overlay on top of the camera feed.

    * seat outlines: the dealer strip and the player arc, with their names. In
      edit mode (key E) every corner gets a handle, the zone last clicked is
      tinted, and the bottom of the sidebar explains the mouse controls.
    * every corner index the detector sees: thin grey box until the tracker
      confirms the card, then a box in the card's colour
    * a tag on each card of the round: its placement number in a yellow badge,
      then its name. A hidden (covered) card keeps a grey tag where it was last seen.
    * a sidebar to the right of the video, so nothing covers the table: the
      Hi-Lo running count, true count, and cards left, then each seat's total
      and its cards in placement order
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
from .seats import DEALER, SeatLayout, ZoneEditor
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
SIDEBAR = 0.30  # sidebar width as a fraction of the video width
KEYS = "[1-7] players    [E] edit zones    [R] new round    [Shift+R] new shoe    [S] save frame    [Q] quit"


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
             layout: SeatLayout, count: ShoeCount, capture_mode: bool = False,
             editor: ZoneEditor | None = None) -> np.ndarray:
        """cards: every card of the round, hidden ones included, in placement order.
        editor: the zone editor, drawn with handles while it is active."""
        view = frame.copy()
        W = view.shape[1]
        self.s = min(max(W / 1600, 0.6), 2.5)  # 1.2 at 1080p, 2.4 at 4K
        self.thick = max(1, round(2 * self.s))
        self.pad = round(10 * self.s)
        self.r = round(15 * self.s)

        editing = editor is not None and editor.active
        self._seat_outlines(view, layout, editor if editing else None)
        visible = {c.label for c in cards if not c.hidden}
        for d in dets:
            x1, y1, x2, y2 = map(int, d.box)
            if d.label in visible:
                cv2.rectangle(view, (x1, y1), (x2, y2), card_colour(d.label), self.thick)
            else:
                cv2.rectangle(view, (x1, y1), (x2, y2), GREY, 1)

        placed: list[tuple] = []
        for c in cards:
            placed.append(self._card_tag(view, c, None if c.hidden else self._card_box(c, dets), placed))

        side = np.full((view.shape[0], round(SIDEBAR * W), 3), PANEL, np.uint8)
        seat_of = {c.order: layout.assign(c.x, c.y) for c in cards}
        y = self._count_block(side, count)
        self._seat_blocks(side, cards, seat_of, layout, y)
        if editing:
            self._edit_help(side)
        canvas = np.hstack([view, side])
        self._bottom_bar(canvas, fps, capture_mode, W)
        return canvas

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
    def _seat_outlines(self, img, layout: SeatLayout, editor: ZoneEditor | None):
        for i, seat in enumerate(layout.seats):
            colour = AMBER if seat.name == DEALER else SEAT_LINE
            if editor is not None and editor.selected == i:
                tint = img.copy()
                cv2.fillPoly(tint, [seat.polygon], BADGE)
                cv2.addWeighted(tint, 0.18, img, 0.82, 0, dst=img)
                colour = BADGE
            line = self.thick if editor is not None else max(1, self.thick // 2)
            cv2.polylines(img, [seat.polygon], True, colour, line, cv2.LINE_AA)
            title = seat_title(seat.name)
            w, h, _ = self._size(title, 0.6)
            top = seat.polygon[np.argmin(seat.polygon[:, 1])]
            x, y = top[0] - w // 2, top[1] + h + self.pad
            self._panel(img, x - self.pad // 2, y - h - self.pad // 2, x + w + self.pad // 2, y + self.pad // 2, 0.6)
            self._text(img, title, (x, y), 0.6, colour)
        if editor is None:
            return
        r = max(6, editor.handle_radius() * 2 // 3)
        for seat in layout.seats:
            for cx, cy in seat.polygon:
                cv2.circle(img, (int(cx), int(cy)), r, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(img, (int(cx), int(cy)), r - 2, WHITE, -1, cv2.LINE_AA)

    def _edit_help(self, side):
        """Mouse controls for edit mode, at the bottom of the sidebar so no handle is ever covered."""
        lines = [("EDIT ZONES", BADGE), ("drag a corner: reshape", WHITE), ("drag inside: move the zone", WHITE),
                 ("right-click: reset the zone", WHITE), ("E: save and lock", WHITE)]
        lh = self._size("Ag", 0.6)[1] + self.pad
        bar = self._size(KEYS, 0.5)[1] + self._size(KEYS, 0.5)[2] + 2 * self.pad
        y = side.shape[0] - bar - self.pad - len(lines) * lh
        cv2.rectangle(side, (0, y - self.pad), (side.shape[1], side.shape[0] - bar), (60, 50, 0), -1)
        for text, colour in lines:
            y += lh
            self._text(side, text, (self.pad, y - self.pad // 2), 0.6, colour)

    def _count_block(self, side, count: ShoeCount) -> int:
        """Hi-Lo running count, true count, cards left, at the top of the sidebar. Returns the y below it."""
        colour = GREEN if count.running > 0 else RED if count.running < 0 else WHITE
        parts = [("RUNNING", f"{count.running:+d}", colour), ("TRUE", f"{count.true:+.1f}", colour),
                 ("LEFT", str(count.cards_left), WHITE)]
        col_w = (side.shape[1] - 2 * self.pad) // len(parts)
        kh = self._size("RUNNING", 0.5)[1]
        vh = self._size("+0", 1.1)[1]
        for i, (k, v, c) in enumerate(parts):
            x = self.pad + i * col_w
            self._text(side, k, (x, 2 * self.pad + kh), 0.5, GREY)
            self._text(side, v, (x, 3 * self.pad + kh + vh), 1.1, c)
        y = 5 * self.pad + kh + vh
        cv2.line(side, (self.pad, y - self.pad), (side.shape[1] - self.pad, y - self.pad), DIM, 1)
        return y

    def _seat_blocks(self, side, cards, seat_of, layout: SeatLayout, y: int):
        """For each seat: its name and total on one line, its cards in placement order below,
        wrapping onto more lines for long hands. Cards outside every seat are listed as OTHER."""
        H, SW = side.shape[:2]
        rows = [(name, [c for c in cards if seat_of[c.order] == name]) for name in layout.names()]
        loose = [c for c in cards if seat_of[c.order] is None]
        if loose:
            rows.append(("OTHER", loose))
        item_h = 2 * self.r + self.pad
        for name, hand in rows:
            title = seat_title(name)
            _, th, _ = self._size(title, 0.65)
            if y + th + item_h > H:
                self._text(side, "...", (self.pad, H - self.pad), 0.65, GREY)
                return
            self._text(side, title, (self.pad, y + th), 0.65,
                       AMBER if name == DEALER else SEAT_LINE if name != "OTHER" else GREY)
            if name != "OTHER":
                total = describe([c.label for c in hand])
                tw = self._size(total, 0.65)[0]
                self._text(side, total, (SW - self.pad - tw, y + th), 0.65, WHITE if hand else DIM)
            y += th + self.pad
            x = self.pad
            cy = y + item_h // 2
            for c in hand:
                w = 2 * self.r + self.pad // 2 + self._size(c.label, 0.8)[0]
                if x + w > SW - self.pad and x > self.pad:  # wrap
                    x, cy, y = self.pad, cy + item_h, y + item_h
                self._chip(side, c, x, cy)
                x += w + 2 * self.pad
            y += item_h + self.pad

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

    def _card_tag(self, img, card: Card, box, tags):
        """Badge + name just above the card's topmost corner index (below it if there is no room).
        tags: tags already drawn; this one stacks above any it would cover. Returns its rectangle."""
        H, W = img.shape[:2]
        w, h, base = self._size(card.label, 0.8)
        tag_w, tag_h = 2 * self.r + self.pad + w + self.pad, max(2 * self.r, h + base) + self.pad
        if box is None:  # hidden, or not detected this frame: use the last known position
            box = (card.x, card.y, card.x, card.y)
        x = int(min(max(box[0], 0), W - tag_w))
        y = int(box[1] - self.pad // 2 - tag_h)
        if y < 0:
            y = int(box[3] + self.pad // 2)
        for _ in range(len(tags) + 1):
            hit = next((ay1 for ax1, ay1, ax2, ay2 in tags
                        if x < ax2 and x + tag_w > ax1 and y < ay2 and y + tag_h > ay1), None)
            if hit is None or hit - tag_h - self.pad // 2 < 0:
                break
            y = hit - tag_h - self.pad // 2
        y = int(min(max(y, 0), H - tag_h))
        self._panel(img, x, y, x + tag_w, y + tag_h, alpha=0.8)
        self._chip(img, card, x + self.pad // 2, y + tag_h // 2)
        return x, y, x + tag_w, y + tag_h

    def _bottom_bar(self, img, fps: float, capture_mode: bool, video_w: int):
        """Keys and fps across the whole window; messages centred over the video."""
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
            x = (video_w - mw) // 2
            y2 = H - bar_h - self.pad
            self._panel(img, x - 2 * self.pad, y2 - mh - mb - 2 * self.pad, x + mw + 2 * self.pad, y2, alpha=0.85)
            self._text(img, self._message, (x, y2 - self.pad - mb), 0.9, BADGE)
