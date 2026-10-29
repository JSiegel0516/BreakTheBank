"""Draw the reader's overlay on top of the camera feed.

    * seat outlines: the dealer strip and the player arc, with their names. In
      edit mode (key E) every corner gets a white handle, every edge a '+' to add
      a corner, the zone last clicked is tinted, and the bottom of the sidebar
      explains the mouse controls.
    * every corner index the detector sees: thin grey box until the tracker
      confirms the card, then a box in the card's colour
    * a tag on each card of the round: its placement number in a yellow badge,
      then its name. A hidden (covered) card keeps a grey tag where it was last seen.
    * a sidebar to the right of the video, so nothing covers the table: the
      Hi-Lo running count, true count, and cards left, then each seat's total
      and its cards in placement order
    * a legend at the bottom of the sidebar explaining every colour (key L hides it)
    * bottom bar: the keys, fps, and short messages such as "new round"

Colours follow the cards: red for hearts and diamonds, white for spades and
clubs, amber for face-down cards. Text and line sizes scale with the frame
width so the overlay stays readable from 720p to 4K.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from .blackjack import ShoeCount
from .cards import Card, Detection
from .deck import BACK, RED_SUITS, suit_of
from .seats import DEALER, SeatLayout, ZoneEditor
from .table import OTHER, Hand, Table

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
KEYS = ("[1-7] players    [E] edit zones    [L] legend    [R] new round    "
        "[Shift+R] new shoe    [S] save frame    [Q] quit")


def card_colour(label: str):
    if label == BACK:
        return AMBER
    return RED if suit_of(label) in RED_SUITS else WHITE


def seat_title(name: str) -> str:
    return "DEALER" if name == DEALER else name


def _top_of_overlap(tags: list[tuple], rect: tuple) -> int | None:
    """Top edge of the first tag rectangle that overlaps rect, or None. Rectangles are (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = rect
    for left, top, right, bottom in tags:
        if x1 < right and x2 > left and y1 < bottom and y2 > top:
            return top
    return None


class Hud:
    def __init__(self):
        self._message = ""
        self._message_until = 0.0
        self.show_legend = True

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
        for det in dets:
            x1, y1, x2, y2 = map(int, det.box)
            if det.label in visible:
                cv2.rectangle(view, (x1, y1), (x2, y2), card_colour(det.label), self.thick)
            else:
                cv2.rectangle(view, (x1, y1), (x2, y2), GREY, 1)

        placed: list[tuple] = []
        for card in cards:
            box = None if card.hidden else self._card_box(card, dets)
            placed.append(self._card_tag(view, card, box, placed))

        side = np.full((view.shape[0], round(SIDEBAR * W), 3), PANEL, np.uint8)
        y = self._count_block(side, count)
        bottom = side.shape[0] - self._bar_height()
        if editing:
            bottom = self._edit_help(side, bottom)
        if self.show_legend:
            bottom = self._legend(side, bottom)
        self._seat_blocks(side, Table(layout).hands(cards), y, bottom)
        canvas = np.hstack([view, side])
        self._bottom_bar(canvas, fps, capture_mode, W)
        return canvas

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
            for cx, cy in ZoneEditor.midpoints(seat.polygon):  # '+': drag to add a corner
                plus_r = max(4, r * 2 // 3)
                cv2.circle(img, (int(cx), int(cy)), plus_r, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.line(img, (int(cx) - plus_r + 3, int(cy)), (int(cx) + plus_r - 3, int(cy)), BADGE, 2, cv2.LINE_AA)
                cv2.line(img, (int(cx), int(cy) - plus_r + 3), (int(cx), int(cy) + plus_r - 3), BADGE, 2, cv2.LINE_AA)
            for cx, cy in seat.polygon:  # corners: drag to move, right-click to remove
                cv2.circle(img, (int(cx), int(cy)), r, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(img, (int(cx), int(cy)), r - 2, WHITE, -1, cv2.LINE_AA)

    def _bar_height(self) -> int:
        _, kh, base = self._size(KEYS, 0.5)
        return kh + base + 2 * self.pad

    def _edit_help(self, side, bottom: int) -> int:
        """Mouse controls for edit mode, in the sidebar so no handle is ever covered. Returns its top."""
        lines = [("EDIT ZONES", BADGE), ("drag a corner: reshape", WHITE), ("drag a +: add a corner", WHITE),
                 ("right-click a corner: remove it", WHITE), ("drag inside: move the zone", WHITE),
                 ("right-click inside: reset the zone", WHITE), ("E: save and lock", WHITE)]
        lh = self._size("Ag", 0.6)[1] + self.pad
        top = bottom - len(lines) * lh - self.pad
        cv2.rectangle(side, (0, top), (side.shape[1], bottom), (60, 50, 0), -1)
        y = top
        for text, colour in lines:
            y += lh
            self._text(side, text, (self.pad, y - self.pad // 2), 0.6, colour)
        return top

    def _legend(self, side, bottom: int) -> int:
        """What every colour means, in two columns above `bottom`. Returns its top."""
        entries = [("box", RED, "hearts, diamonds"), ("box", WHITE, "spades, clubs"),
                   ("box", AMBER, "face down"), ("thin", GREY, "not confirmed yet"),
                   ("badge", BADGE, "placement number"), ("badge", DIM, "covered, still counted"),
                   ("line", AMBER, "dealer zone"), ("line", SEAT_LINE, "player zone"),
                   ("count", GREEN, "count favours player"), ("count", RED, "count favours dealer")]
        scale = 0.45
        _, th, _ = self._size("Ag", scale)
        row_h = max(th, 2 * self.r // 2) + self.pad
        rows = (len(entries) + 1) // 2
        header_h = self._size("LEGEND", 0.5)[1] + self.pad
        top = bottom - self.pad - header_h - rows * row_h
        cv2.line(side, (self.pad, top), (side.shape[1] - self.pad, top), DIM, 1)
        self._text(side, "LEGEND", (self.pad, top + header_h), 0.5, GREY)
        hide = "L hides"
        self._text(side, hide, (side.shape[1] - self.pad - self._size(hide, 0.45)[0], top + header_h), 0.45, DIM)
        col_w = (side.shape[1] - 2 * self.pad) // 2
        sw = round(22 * self.s)  # swatch width
        for i, (kind, colour, text) in enumerate(entries):
            x = self.pad + (i % 2) * col_w
            cy = top + header_h + self.pad // 2 + (i // 2) * row_h + row_h // 2
            if kind == "box":
                cv2.rectangle(side, (x, cy - th // 2 - 2), (x + sw, cy + th // 2 + 2), colour, self.thick)
            elif kind == "thin":
                cv2.rectangle(side, (x, cy - th // 2 - 2), (x + sw, cy + th // 2 + 2), colour, 1)
            elif kind == "badge":
                cv2.circle(side, (x + sw // 2, cy), sw // 2, colour, -1, cv2.LINE_AA)
            elif kind == "line":
                cv2.line(side, (x, cy), (x + sw, cy), colour, self.thick, cv2.LINE_AA)
            else:
                sign = "+" if colour == GREEN else "-"
                self._text(side, sign, (x + sw // 4, cy + th // 2), 0.6, colour)
            self._text(side, text, (x + sw + self.pad, cy + th // 2), scale, WHITE)
        return top - self.pad

    def _count_block(self, side, count: ShoeCount) -> int:
        """Hi-Lo running count, true count, cards left, at the top of the sidebar. Returns the y below it."""
        colour = GREEN if count.running > 0 else RED if count.running < 0 else WHITE
        parts = [("RUNNING", f"{count.running:+d}", colour), ("TRUE", f"{count.true:+.1f}", colour),
                 ("LEFT", str(count.cards_left), WHITE)]
        col_w = (side.shape[1] - 2 * self.pad) // len(parts)
        kh = self._size("RUNNING", 0.5)[1]
        vh = self._size("+0", 1.1)[1]
        for i, (title, value, value_colour) in enumerate(parts):
            x = self.pad + i * col_w
            self._text(side, title, (x, 2 * self.pad + kh), 0.5, GREY)
            self._text(side, value, (x, 3 * self.pad + kh + vh), 1.1, value_colour)
        y = 5 * self.pad + kh + vh
        cv2.line(side, (self.pad, y - self.pad), (side.shape[1] - self.pad, y - self.pad), DIM, 1)
        return y

    def _seat_blocks(self, side, hands: list[Hand], y: int, bottom: int):
        """One line per hand: the seat's name, its cards in placement order, and its total on the
        right. A long hand wraps onto more lines. Loose cards come last, as OTHER."""
        side_w = side.shape[1]
        name_w = max(self._size(seat_title(hand.seat), 0.65)[0] for hand in hands) + 2 * self.pad
        line_h = 2 * self.r + self.pad
        for hand in hands:
            total = hand.total
            total_w, total_h, _ = self._size(total, 0.65) if total else (0, 0, 0)
            # lay the cards out first, so a hand that does not fit can be cut off cleanly
            x0 = self.pad + name_w
            right = side_w - self.pad
            if total:
                right -= total_w + 2 * self.pad  # the first line leaves room for the total
            spots, x, line = [], x0, 0
            for card in hand.cards:
                chip_w = 2 * self.r + self.pad // 2 + self._size(card.label, 0.8)[0]
                if x + chip_w > right and x > x0:  # wrap; later lines may use the full width
                    x, line, right = x0, line + 1, side_w - self.pad
                spots.append((card, x, line))
                x += chip_w + 2 * self.pad
            height = (line + 1) * line_h
            if y + height > bottom:
                self._text(side, "...", (self.pad, bottom - self.pad), 0.65, GREY)
                return
            cy = y + line_h // 2
            title = seat_title(hand.seat)
            title_colour = {DEALER: AMBER, OTHER: GREY}.get(hand.seat, SEAT_LINE)
            _, title_h, _ = self._size(title, 0.65)
            self._text(side, title, (self.pad, cy + title_h // 2), 0.65, title_colour)
            if total:
                total_colour = WHITE if hand.cards else DIM
                self._text(side, total, (side_w - self.pad - total_w, cy + total_h // 2), 0.65, total_colour)
            for card, chip_x, chip_line in spots:
                self._chip(side, card, chip_x, cy + chip_line * line_h)
            y += height

    @staticmethod
    def _card_box(card: Card, dets: list[Detection]):
        """The topmost corner index of this card in this frame, or None if none was seen.

        A face card is any detection with its name (one deck, so they are all
        this card). A face-down card is the BACK detection nearest its position."""
        if card.label == BACK:
            mine = [det for det in dets if det.label == BACK]
            if not mine:
                return None
            return min(mine, key=lambda det: (det.cx - card.x) ** 2 + (det.cy - card.y) ** 2).box
        mine = [det for det in dets if det.label == card.label]
        if not mine:
            return None
        return min(mine, key=lambda det: det.box[1]).box

    def _card_tag(self, img, card: Card, box, tags):
        """Badge + name just above the card's topmost corner index (below it if there is no room).
        tags: tags already drawn; this one stacks above any it would cover. Returns its rectangle."""
        H, W = img.shape[:2]
        w, h, base = self._size(card.label, 0.8)
        tag_w = 2 * self.r + self.pad + w + self.pad
        tag_h = max(2 * self.r, h + base) + self.pad
        gap = self.pad // 2
        if box is None:  # hidden, or not detected this frame: use the last known position
            box = (card.x, card.y, card.x, card.y)
        x = int(min(max(box[0], 0), W - tag_w))
        y = int(box[1] - gap - tag_h)
        if y < 0:
            y = int(box[3] + gap)
        # Moving above one tag can cover another, so repeat once per tag at most.
        for _ in range(len(tags) + 1):
            covered_top = _top_of_overlap(tags, (x, y, x + tag_w, y + tag_h))
            if covered_top is None:
                break
            above = covered_top - tag_h - gap
            if above < 0:
                break  # no room above it: stay put
            y = above
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
        status_w = self._size(status, 0.5)[0]
        self._text(img, status, (W - status_w - self.pad, ty), 0.5, AMBER if capture_mode else GREY)

        if self._message and time.time() < self._message_until:
            mw, mh, mb = self._size(self._message, 0.9)
            x = (video_w - mw) // 2
            y2 = H - bar_h - self.pad
            self._panel(img, x - 2 * self.pad, y2 - mh - mb - 2 * self.pad, x + mw + 2 * self.pad, y2, alpha=0.85)
            self._text(img, self._message, (x, y2 - self.pad - mb), 0.9, BADGE)
