# Reading cards off the table

Part of [BreakTheBank](../README.md). Training the detector is in [training.md](training.md).

## Reading

`python read.py` opens a window with the live feed, an overlay on it, and a
sidebar on the right so nothing covers the table (drawn by `cardReader/hud.py`):

| On screen | Meaning |
|---|---|
| Seat outlines | the dealer strip at the top and the player seats on an arc below it (`cardReader/seats.py`), each labelled with the seat's display name (`P1` or `DEALER` until renamed), centred just below the middle of the zone's top-most edge |
| Sidebar, top | tabs naming the two views, `CARDS LEFT` and `GAME`: the current one is filled in yellow, and `[V]` beside them is the key that switches (see below) |
| Sidebar, under the tabs | `TRUE COUNT` followed by the counting system's name (`Hi-Lo` unless [switched on the page](#browser-dashboard)), the true count in large type, beside it what the count means and whom it favours (`high cards left` over `favours player`, `low cards left` over `favours dealer`, or `even deck`), then the running count and cards left out of the shoe (`38/52`, or `90/104` with two decks) |
| Sidebar, below | the Cards Left view or the Game view (see below) |
| Name on a card | that card's name: the rank followed by the suit's shape (`BACK` for a face-down card). `N` adds its placement number in a yellow badge before the name |
| Grey name | a hidden card: covered right now, but still in its hand until it is removed (see [Rules the tracker assumes](#rules-the-tracker-assumes)), and in the count |
| Coloured boxes | corner indices the tracker has matched to a confirmed card, in that card's colour: red for hearts and diamonds, white for spades and clubs, amber for face-down. A frame that misreads the corner (a 6H read as 9H) keeps the card's colour |
| Thin grey boxes | detections not matched to a confirmed card: a card not confirmed yet, or a stray box away from every card. `G` hides them |
| Sidebar, bottom | the legend for the current view, then every key (`L` hides it) |
| `camera turned to 90 degrees: CAMERA_ROTATION = 90 keeps it` | `<` or `>` turned the view a quarter turn (see [Setting up the camera](#setting-up-the-camera)) |
| Bottom bar | the main keys, frames per second, the milliseconds per frame spent detecting, and the preprocess mode with its milliseconds (`showing raw` when `Shift+V` shows the raw camera frame) |
| `no signal: press C to reconnect` | the camera has sent no new frame for two seconds (see [Camera disconnects](#camera-disconnects)) |

The number is the placement order within the round: 1 is the first card
confirmed (seen for `CONFIRM_SECONDS`, about half a second), 2 the next, and so on. A
card that is covered keeps its number and comes back with it. The tags on the
video leave the number out until `N` is pressed, so the table stays readable;
the hands in the Game view and on the page always show it.

### Cards Left view and Game view

The tabs at the top of the sidebar always show which view is on. The reader
starts in the **Cards Left view**: the whole deck as a grid, with one
row per rank (2 to 10, J, Q, K, A) and one column per suit (spades, hearts,
diamonds, clubs). Each cell is a small white card with its rank above its suit
shape, red for hearts and diamonds and black for spades and clubs; when the
window is small the rank is left out and the suit shape stays. It shows at a
glance which cards have been dealt this shoe:

| Cell | Meaning |
|---|---|
| White card | not seen this shoe |
| Yellow slash and border | pending: counted this round, so a corrected misread can still change it |
| Grey, crossed off | locked: counted in an earlier round of this shoe |
| Magenta ring | a `duplicate` was reported for this card this shoe (a misread, or a reshuffle without `Shift+R`) |
| Partly filled from the left | two or three decks: the share of this card's copies seen so far, dim yellow if one was counted this round, grey otherwise. One of two copies fills half the card; it is crossed off only once every copy is seen |

Rows are tinted and labelled with the rank's value in the counting system:
green above 0, grey at 0, red below, and deeper for bigger values, so -3 is
darker than -1. In Hi-Lo that is green for 2 to 6 (+1), grey for 7 to 9 (0),
and red for 10 to A (-1). The legend lists each value with its ranks, so Hi-Opt
II and Uston APC have longer legends. The column on the right gives
how many cards of that rank are left, counting every copy. The line under the
grid counts the face-down cards dealt and never turned over. `Shift+R` clears
the grid. The grid shows the shoe of `DECKS` decks (see
[Decks in the shoe](#decks-in-the-shoe)). In the Cards Left view the grid comes first: when
the window is too small for both, the legend is left out.

The **Game view** lists each seat's total (`17`, `soft 18`, `BLACKJACK`,
`BUST (24)`, `9 +1 hidden`) and its cards in placement order. Cards outside
every seat are listed as OTHER. A split seat has two lines, its name followed
by `1` and `2` (see [Split hands](#seats-hands-and-the-count)). The video, the card tags, and the zone editor
look the same in both views. A card hidden longer than `HIDDEN_SECONDS` (5 s)
leaves its seat's hand in the Game view, and its total, but stays in the count
and in the Cards Left grid (see [Rules the tracker assumes](#rules-the-tracker-assumes)).

### What the colours mean

The legend at the bottom of the sidebar explains the colours of the current
view. The Cards Left view's colours are in the table above.

| Colour | Where | Meaning |
|---|---|---|
| Red box or name | corner index, tags, sidebar | a confirmed heart or diamond (the suit colour, not an error) |
| White box or name | same | a confirmed spade or club |
| Amber box or name | same | a confirmed face-down card |
| Thin grey box | corner index | a detection not matched to a confirmed card: it has not passed the [confirmation checks](#rules-the-tracker-assumes) yet, or it is away from every card (`G` hides these) |
| Yellow badge | sidebar, and tags after `N` | the card's placement number; the Game legend lists it only while `N` shows it on the tags |
| Grey badge and name | tags, sidebar | a hidden card: covered right now, still in its hand until it is removed, and in the count (the tag's badge only after `N`) |
| Amber outline | zones | the dealer's zone |
| Light blue outline | zones | a player's zone |
| Yellow tint | zones, edit mode | the zone you last clicked |
| Green or red count | sidebar | a positive count (high cards left, favours the player) or a negative one (low cards left, favours the dealer) |

### Keys

The bottom bar lists the main keys; the legend lists them all.

| Key | Action | Page button |
|---|---|---|
| `1` to `5` | number of players; loads your saved layout for that count, or the template | yes, one per count |
| `Shift+1` to `Shift+5` | split the pair of P1 to P5 into two hands, or join a split seat back into one (see [Split hands](#seats-hands-and-the-count)). On a US keyboard these send `!` `@` `#` `$` `%` | yes, `Split` or `Join` on the player's line under Hands |
| `E` | edit zones with the mouse; press `E` again to save them and lock | no |
| `V` | switch the sidebar between the Cards Left view and the Game view; the tabs and a message name the new one | no |
| `L` | show or hide the legend and key list | no |
| `G` | show or hide the thin grey boxes, the detections not matched to a confirmed card | yes |
| `N` | show or hide the placement numbers on the video tags (hidden at start); the hands keep theirs | yes |
| `R` | new round: forget the table, numbering restarts at 1, the count is kept | yes |
| `X` | rescan: forget this round's cards and read the table again; the count keeps only earlier rounds | yes |
| `Shift+R` | new shoe: after a reshuffle, the count goes back to 0 | yes |
| `S` | save the current frame to `data/raw/frames/` for labelling | no |
| `C` | reconnect the camera: reopen the same source after a cable or power interruption | yes |
| `P` | next preprocess mode: off, gamma, clahe, auto (see [Dim or washed-out tables](#dim-or-washed-out-tables)) | yes |
| `Shift+V` | show the raw camera frame, or the preprocessed frame the detector gets | yes |
| `<` / `>` | turn the view a quarter turn anticlockwise / clockwise; the message names the `CAMERA_ROTATION` that keeps it (see [Setting up the camera](#setting-up-the-camera)) | yes, one each |
| `Q` | quit (or Ctrl+C in the terminal) | yes, after a confirmation |

Keys work when the reader window is selected. The window can be resized.
With `--web`, the page's buttons (see [Browser dashboard](#browser-dashboard))
do exactly what their keys do, with the same message.
Every event is also printed in the terminal, with the seat and the count:

```
[   412] placed   #1   10H (P1) count -1
[   530] placed   #2   6C  (P2) count +0
[   901] hidden   #2   6C  (P2)
[   966] returned #2   6C  (P2)
[  1402] placed   #3   BACK (dealer) count +0
[  1870] renamed  #3   KS (was BACK) (dealer) count -1
[  2210] round_over
[  2650] placed   #1   KS (P3) count -1
[  2650] duplicate #1   KS (P3)
```

and on quitting: `counted this shoe: 10H 6C ...` with the final running and true
count, then the average milliseconds per frame for each stage of the loop:

```
average time per frame: read 0.4 ms, preprocess 0.1 ms, detect 31.2 ms, draw 6.8 ms
```

When the frames per second averaged over the run are more than a fifth away
from `FPS` in `settings.py`, a last line gives both:

```
measured 22.3 fps, but FPS in settings.py is 12.4: set FPS to about 22.3 so the *_SECONDS settings take the time they say
```

The tracker's times are set in seconds and turned into frames at `FPS`, so a
wrong `FPS` makes every one of them run fast or slow (see [Tuning the reader](#tuning-the-reader)).

`read` is taking the newest camera frame, `preprocess` the contrast help (see
[Dim or washed-out tables](#dim-or-washed-out-tables)), `detect` the model, and
`draw` the overlay and the window. The bottom bar shows the same detect and
preprocess times, smoothed, while it runs.

At start-up the model line names the device the detector runs on, as in
`model models/cards/weights/best.pt on mps`: `mps` (Apple GPU), `cuda` (NVIDIA
GPU), or `cpu`. `cpu` on a Mac with Apple silicon makes detect several times
slower; try `--device mps`.

A `removed` line means a hidden card stayed out of sight long enough to count
as picked up (see [Rules the tracker assumes](#rules-the-tracker-assumes)). It
leaves its hand; the count is not changed.

A `duplicate` line means a face card was placed (or renamed) under a name whose
every copy was already counted this shoe. One deck holds one of each card, so a
second King of Hearts is a misread or a reshuffle nobody marked with `Shift+R`.
With two decks the third one is. The count is not changed.

To fix a misread while the round is still open, press `X` to rescan the round.
The reader forgets every card of this round and takes this round's names back
out of the count, then reads the cards on the table again, numbered from 1.
Names counted in earlier rounds stay counted. Pressing `R` instead locks the
wrong name into the count. A rescan cannot undo a round that has already ended,
whether by `R` or by clearing the table. A card left on the table from an
earlier round is placed again after a rescan, so it shows as a `duplicate`
(face up) or is taken off cards left a second time (face down).

- `--source 0` picks the camera. With an iPhone as a Continuity Camera, OpenCV
  usually numbers it 0 and the built-in camera 1; try the other number if the
  wrong one opens.
- `--source round1.mp4` reads a recorded video instead.
- `--players 5` sets the starting number of seats (`PLAYERS` in `settings.py`).
- `--log round.csv` appends every event to a CSV, with its seat and the running count.
- `--preprocess auto` picks the starting preprocess mode (`PREPROCESS` in `settings.py`): `off`, `gamma`, `clahe`, or `auto`.
- `--no-display` runs without the window; stop it with Ctrl+C. With `--web` as well,
  the page is the whole interface, and its `Q` button quits.
- `--web` moves the sidebar to a browser page (see [Browser dashboard](#browser-dashboard)).
- `--web-host 0.0.0.0` serves that page to other devices on the network too.

macOS asks for camera permission the first time OpenCV opens a camera from a
terminal or VS Code. Grant it under System Settings > Privacy & Security > Camera.

### Browser dashboard

```
python read.py --web                        # page at http://127.0.0.1:8765/
python read.py --web --web-host 0.0.0.0     # a phone or tablet on the same Wi-Fi can open it too
python read.py --no-display --web           # no window at all: the page is the whole interface
```

The window's sidebar is 30% of the video's width, and its text scales with the
video, so on a laptop screen it is cramped. With `--web` the window shows only
the video, its tags and zones, and the bottom bar. The reader prints
`dashboard at http://127.0.0.1:8765/`; open that address in any browser. The
page holds what the sidebar did, with real fonts, and reflows to fit the browser
window without scrolling sideways. A wide window puts Cards Left on the left and
True Count, Hands, and Keys on the right. Under 900 pixels wide the sections
stack in one column: True Count, Hands, Cards Left, Keys.

The page is set in Inter, a font under the SIL Open Font License. The reader's
own server serves it (`cardReader/sidebar/Inter-Regular.woff2`, licence in
`Inter-OFL.txt`), so the page needs no internet connection. Only the regular
weight is bundled; the browser thickens it for bold text. Until it loads, or if
it cannot, the system font shows. Section headings (`TRUE COUNT`, `HANDS`,
`CARDS LEFT`, `KEYS`) and tile labels (`BLACKJACK`, `STAND`, `WIN`) are in
capitals. Every other label and message starts with a capital and continues in
lower case, such as `Running`, `New round`, and `Split`. The window keeps its
own lower-case labels. The page is drawn on dark
casino-felt greens, so its colours are paler than the window's: red text is
pink, green text is mint, and the grid rows are tinted emerald above 0,
grey-green at 0, and brown below, deeper for bigger values:

| On the page | Meaning |
|---|---|
| True count | the title names the counting system (`TRUE COUNT · Hi-Lo`); below it the true count in large type with what it means and whom it favours (`High cards left · favours player`, `Low cards left · favours dealer`, or `Even deck`), then the running count, cards left (`Left 39/52`, or `90/104` with two decks), tens and aces left (`Tens & aces 17/20`, out of 40 with two decks), and aces left (`Aces 3/4`, out of 8 with two decks). Tens and aces are 10, J, Q, K, and A; both figures match the Cards Left grid |
| Cards Left | the largest part of the page: one row per rank with its value in the counting system, a mini card per suit, and how many of that rank are left. Unseen cards are white. A card counted this round has a yellow slash and border. A card counted in an earlier round has a red cross on a darkened tile. A duplicate is ringed in magenta. With two or three decks a card with copies left is dimmed from the left by the share seen, and its hover text says so (`KH: 1 of 2 seen, locked`); the column on the right counts copies. Below it: the face-down cards never revealed |
| Hands | one line per seat: its name (click it to rename the seat, see below), its cards in placement order (a numbered badge, then the rank and suit), its total, and for a player seat the [basic strategy hint](#basic-strategy-hints). A covered card is greyed out. A player holding a pair has a `Split` button; a split seat has two lines, `<name> 1` and `<name> 2`, each with a `Join` button. Lines glow and gain tags as hands finish (see Outcome cues below). A row of tiles under a hand gives its odds from the unseen cards (see Odds row below) |
| Edit zones | while `E` is on, the mouse controls for the zones, which the window no longer has room for |
| Keys | first a `Model` dropdown and a `Count` dropdown (see below), then every key, as a button where the page has one (see the Page button column under [Keys](#keys)); the current number of players is filled in. The keys sit in five rows: table (`1-5`, `R`, `X`, `Shift+R`, `Shift+1-5`, then the decks button, see [Decks in the shoe](#decks-in-the-shoe)), overlay (`G`, `N`), image (`P`, `Shift+V`, `< >`), camera (`C`, `Q`), and window only (`E`, `V`, `L`, `S`). Each key sits above what it does; `1-5` and `< >` have their buttons below. A row wraps onto more lines when the page is narrow. `E`, `V`, `L`, and `S` are grey: they work in the window only. `Shift+1-5` is grey too, since its buttons are on the Hands lines instead. The arrow after `KEYS` folds the section to its title (see below) |
| Status line | the model running, frames per second, detect and preprocess milliseconds, the preprocess mode, `Showing raw`, `No signal`, and the messages the window flashes, with a capital first letter (`New round`, `Camera turned ...`) |

The page needs no reload. It changes within a frame of the reader, and a message
shorter than a frame still appears once. The reader sends the page an update
only when something on it changed, plus one every half second so the frames
per second stay current. If the reader stops, the status line says
`Not connected`, and the page catches up by itself once the reader is back.

**Buttons.** A button does exactly what its key does, with the same message on
the page and in the window: new round, new shoe, players, reconnect, and so on.
The reader acts on it at its next frame. Several presses at once, or a button
and a key together, are taken one per frame in order, so each sees what the one
before did (a second turn turns the already-turned view) and each message
shows. `Q` asks for confirmation first, since
quitting cannot be undone from the page. Zone editing (`E`) and saving frames
(`S`) need the window.

**Outcome cues.** The Hands lines show how each hand stands. A player
blackjack sets off confetti over the whole page for about 3.25 seconds, once
per hand, and its line glows gold. A player 21, or a hand that beat the dealer,
gets a gold outline and a gold total; a win also gets a `Win` tag. A bust has a
red total and a red glow. A hand that lost turns grey with a `Lost` tag, and a
push gets a plain `Push` tag. The dealer's line pulses amber when the dealer
has blackjack or 21, or when any player hand lost, and turns red when the
dealer busts. Win, push, and lose wait until the dealer is finished: no card
face down and a total of 17 or more, or bust. The dealer stands on soft 17. A
blackjack pushes only against a dealer blackjack. When the operating system
asks for reduced motion, the page skips the confetti and holds the dealer's
amber outline still; the glows and tags stay. The cues are on the page only:
the window, the terminal, and the `--log` CSV are unchanged.

**Odds row.** A row of tiles under a hand gives its chances as whole
percentages. Each tile has a short bold label over a large percentage, so the
row reads from across the table; it wraps onto more lines when the page is
narrow. The percentages are exact against the unseen cards: the shoe minus every face
seen this shoe, with each copy of a card counted, so two decks hold eight aces. A face-down card never turned up is one of those unseen cards,
so it counts as a random draw from them. The dealer's tiles play the dealer
out card by card from those unseen cards, standing on 17 (soft 17 included),
so they are exact too.

| Hand | Odds row |
|---|---|
| Player with two or more cards, still playing | five tiles, `21 8%`, `20 8%`, `19 8%`, `18 6%`, and `BUST 60%`: the chance the next card brings the total to each of 21, 20, 19, and 18, or busts. A soft hand counts its ace as 1 when 11 would bust. Under it, a second row headed `STAND` with three tiles, `WIN 42%`, `PUSH 0%`, and `LOSE 58%`: the chance the hand wins, pushes, or loses if it stands now. A dealer bust or a lower dealer total wins; a dealer blackjack loses |
| Empty player seat | one tile, `BLACKJACK 5%`: the chance of a two-card blackjack, 2 × aces/N × tens/(N − 1) for N unseen cards |
| Player with one card | one tile, `BLACKJACK 31%`: the chance the second card makes a blackjack; 0% unless the card is an ace or a ten-value card |
| Dealer with one card up and one face down | `BLACKJACK 31%`, the chance the hole card makes a blackjack (0% unless the up card is an ace or a ten-value card), then the dealer's final totals `21 5%`, `20 13%`, `19 13%`, `18 13%`, `17 13%`, and `BUST 12%`. Here `21` leaves blackjack out, so every tile is a separate outcome. With an ace up, an `INSURANCE` tile follows: `TAKE` when more than a third of the unseen cards are ten-value, else `SKIP`. Insurance pays 2 to 1, so a third is the break-even point |
| Dealer still drawing with no face-down card | the six final-total tiles, `21` to `BUST`, without `BLACKJACK` or `INSURANCE` |
| Finished player hand (bust, blackjack, 21, or a win, push, or loss) | a row headed `Odds were`. Its first tile, such as `21 8%` or `BUST 31%`, is the chance the last card made the hand's result: the hand without that card drawing from the unseen cards plus that card. The tile is the hand's total for a hand that stood. A blackjack shows `BLACKJACK 5%`, the empty seat's chance of a two-card blackjack with both its cards back among the unseen cards. Once the dealer has finished, a second tile, `WIN 42%`, `PUSH 6%`, or `LOSE 52%`, is the chance of that result when the hand stood: the `STAND` figure against the dealer's up card alone, with the hole card and the dealer's later cards back among the unseen cards |
| Finished dealer | a row headed `Odds were` with one tile, such as `19 13%`, `BUST 31%`, or `BLACKJACK 5%`: the chance of the dealer's final from its up card alone, with its other cards back among the unseen cards. `21` leaves blackjack out, as above |

Each tile is rounded to a whole percent on its own, so a row may sum to 99% or
101%. The `Odds were` figures take the dealer's first card placed as its up card.
Cards dealt to other seats since stay seen, so the figures are close rather than a
replay of the shoe. There is no live odds row for loose cards (OTHER), for a finished hand
(blackjack, 21, bust, or a win, push, or loss), for a hand with a face-down
card, or for a dealer with no card face up or that has finished drawing. Loose
cards and hands still playing get no `Odds were` row, and neither does a finished
hand when nothing is unseen. The
dealer's tiles and every `STAND` row are also left out when so few cards are
unseen that the dealer could run out before finishing, since the chances would
then be incomplete. There is no `STAND` row for an empty seat or a one-card
hand. Ten-value cards are 10,
J, Q, and K. The odds are on the page only.

**Folding the keys.** Click `KEYS` or its arrow, or press `K` on the page, to
fold the Keys section to its title and give Hands the room. Do the same to
open it again. The arrow points down when the section is open and right when
it is folded. `K` does nothing while a seat name or a dropdown has the
focus. The browser remembers the choice, so a reload keeps it. `K` is a page
key only: the reader window does not use it.

**Renaming players.** Click a seat's name under Hands (`P1`, `DEALER`) to type
the player's name, up to 16 characters. Enter saves it; Escape, or clicking away
without a change, cancels. The new name shows on the page and in the window's
Game view within a frame, with a message such as `P1 is now Ann`. Clear the
field and press Enter to go back to `P1`. A name longer than 16 characters, or a
seat not at the table, is refused with a message. Names are saved in
`data/seats.json`, so they come back when the reader restarts, and they belong to
the seat, whatever the player count. The page, the Game view, and the zone
labels on the video use them; the terminal and the `--log` CSV keep the seat ids
(`P1`), so `evaluate.py` scores any run. In a small window the Game view
shortens a long name to fit, ending it in `..`, so the cards keep their room. A
zone label stays inside its zone: where the zone narrows towards its top edge, or
an edited zone has a notch there, the label moves down, at most to the zone's middle, and a name that still crosses the
outline is shortened the same way. A zone too small for any label shows none. The
window draws letters outside plain ASCII (such as `é`) as `?`. OTHER cannot be renamed, and there is no renaming
from the window. Both lines of a split seat rename the same seat.

**Splitting a pair.** A `Split` button appears on a player's line while it holds
exactly two face-up cards of equal value (a jack and a queen are a pair of tens).
Clicking it does what `Shift+1` to `Shift+5` do in the window, with the same
message; a split line's `Join` button undoes it. The dealer and OTHER lines never
have either button.

**Switching models.** The `Model` dropdown lists every trained run in `models/`
that has `weights/best.pt`, by folder name, with the live model first as
`Cards (live: <run>)`. The one running is selected, and the status line names it.
Choose another to try it without restarting: the status line says
`Loading <run> ...`, the video pauses while the model loads (a few seconds), then
the terminal prints `model <run> on mps` (or `cuda`, `cpu`) and the page's status
line shows `Model <run>`. The cards on the table, their numbers, and the count are kept. In
capture mode (no model at start-up) choosing a model starts detection. The model
is loaded with the same `--imgsz`, `--conf`, and `--device` as at start-up, and
the next start uses `--model` (or `MODEL_PATH`) again; promote a run with
`python train.py --promote` to make it the default. A run trained while the reader is open
appears after the next switch. A path outside `models/`, a run without weights,
or a file that fails to load is refused with a message, and the model running
stays.

**Switching counting systems.** The `Count` dropdown lists Hi-Lo, Hi-Opt II,
and Uston APC (see [Counting systems](#seats-hands-and-the-count)). The reader starts on
`COUNT_SYSTEM` in `settings.py`. Choose another and the reader recounts every
card seen this shoe in it at once. The terminal prints
`--- counting with Uston APC: running count +3 ---`, and both the page and the
window flash `counting with Uston APC`. The count title, the running and true
counts, the grid's value column and shading, and the `--log` running count all
follow. The window has no key for it. The next start uses `COUNT_SYSTEM` again.

**Changing the decks.** The decks button ends the table row of the Keys
section; see [Decks in the shoe](#decks-in-the-shoe).

**Without a window.** `--no-display --web` runs the reader with no window: the
page shows everything and its buttons drive it. Quit with the page's `Q` button,
or Ctrl+C in the terminal. Use it when the computer at the table has no screen
to spare, or to watch from a phone across the room.

`V` and `L` change only the window's sidebar, so with `--web` they do nothing
you can see. Without `--web` the window looks exactly as before.

**Port.** The page is served on `WEB_PORT` in `settings.py` (8765). If another
program has it, the reader stops with
`could not serve the page on 127.0.0.1:8765 (...): set WEB_PORT in settings.py to a free port`.

**On a phone or tablet.** By default only this computer can open the page.
`--web-host 0.0.0.0` lets any device on the same network open it, with no
password, so use it only on a network you trust. Then:

1. Find this computer's address. On macOS, run `ipconfig getifaddr en0` (Wi-Fi),
   or see System Settings > Wi-Fi > Details. It looks like `192.168.1.23`.
2. On the phone, open `http://192.168.1.23:8765/` (your address, then the port).
3. If it does not load, macOS may be blocking incoming connections: allow
   Python under System Settings > Network > Firewall > Options.

The page shows no video, so the phone needs little bandwidth. The phone's
buttons work like the keys on the computer, so anyone who can open the page can
also start a new shoe or quit the reader.

### Decks in the shoe

The reader starts with `DECKS` decks in the shoe, from `settings.py`: 1, 2, or
3. The browser page's decks button, at the end of the table row of the Keys
section, shows the current number and cycles 1, 2, 3, 1 on each click. The
terminal prints `--- 2 decks: running count +1, 103 cards left ---`, and both
the page and the window flash `2 decks`. The window has no key for it.

Changing the number recounts the cards already seen against the new shoe.
Nothing is cleared, so press `Shift+R` as well when the decks were reshuffled.
Lowering the number keeps every copy seen, but counts at most one per deck:
two Kings of Hearts seen in a 2-deck shoe count once at 1 deck, and twice
again at 2 or 3.
Cards left counts down from 52, 104, or 156. Tens and aces are out of 20 per
deck, and aces out of 4 per deck. The odds row draws from every copy left.

With more than one deck a card can really appear twice. Each card may be
counted once per deck; one more read of it is a `duplicate`. Two corners of
one card still join into one card when they are close: up to four times
`MATCH_DISTANCE` apart, about a card's diagonal. A card of the same name
farther away is a second copy. A card that is picked up and comes back
elsewhere looks like a new copy too, so it is counted again while copies are
left.

### Setting up the camera

**Placement**

- Mount the camera directly overhead, pointing straight down, so every card is
  the same size and roughly upright. A phone arm, a ring-light stand with a phone
  clamp, or a tripod with a horizontal arm all work.
- Fill the frame with the playing area. The corner indices need to be at least
  about 20 pixels tall in the image; at 1080p that means the camera roughly 50 to
  80 cm above a small table.
- Light the table evenly. Avoid a bright lamp reflecting off the cards (glare
  hides the index, and no preprocessing brings it back) and strong shadows from
  your hands. For a dim room see [Dim or washed-out tables](#dim-or-washed-out-tables).
- Keep the camera still. Bumping it moves every card at once, and the tracker
  may count them as new cards.
- The dealer's side of the table should be at the top of the picture. If the
  camera is mounted sideways or upside down, press `>` (clockwise) or `<`
  (anticlockwise) to turn the view a quarter turn at a time, instead of
  remounting it. Detection, zones, tags, and frames saved with `S` all use the
  turned picture. The message names the value to keep it, for example
  `CAMERA_ROTATION = 90`; set that in `settings.py` (0, 90, 180, or 270) so the
  reader starts there. Turn before dealing: cards already on the table turn
  with the view and keep their numbers, but the zones are reloaded for the new
  shape. Saved zones are stored as fractions of the picture, so after a
  90-degree turn a saved layout may need a quick re-edit with `E`.

**iPhone as a wireless webcam (Continuity Camera)**

1. iPhone XR or newer on iOS 16+, signed into the same Apple ID as the Mac, with
   Wi-Fi and Bluetooth on for both.
2. On the iPhone: Settings > General > AirPlay & Continuity > Continuity Camera on.
3. Mount the phone with the rear camera facing the table, lock its screen, and
   keep it still and near the Mac. It switches into webcam mode on its own.
4. Run `python read.py --list-cameras` to see a snapshot from every camera with
   its number, then `python read.py --source <number>`. OpenCV usually numbers the
   iPhone 0 and the built-in camera 1, but this can change when the phone
   disconnects (unlocked, picked up, or idle) and reconnects. Then set
   `CAMERA_SOURCE` in `settings.py` so plain `python read.py` uses it.
5. Keep the phone on a charger. If the wireless feed stutters, plug it into the
   Mac with a cable; it works the same way.

**Camera permission (macOS).** The first time, macOS asks whether Terminal or
VS Code may use the camera. If you missed it, the window stays black: allow it
under System Settings > Privacy & Security > Camera, then restart the terminal.

**Sony a6700 (for a full table)**

- Connect it through an HDMI capture card. Turn on clean HDMI output (no
  on-screen menus) and turn off auto power-off.
- Set exposure, white balance, and focus to manual so the image does not drift
  when hands move through the frame.
- Pass the capture card's number as `--source`, and set `CAMERA_WIDTH` and
  `CAMERA_HEIGHT` in `settings.py` if it does not default to the resolution you want.
- At 4K over a whole table, raise `INPUT_SIZE` to 1600 or more so the corner
  indices stay readable. It is slower; check the fps in the bottom bar.

#### Camera disconnects

If a live camera sends no new frame for two seconds (a cable pulled, a capture
card losing power, the iPhone unlocked or picked up), the window keeps the last
frame and shows `no signal: press C to reconnect`. The terminal prints
`no signal from source ...`. While there is no signal nothing is detected, so
the cards, the round, and the count stay as they were.

Fix the cable or wake the phone, then press `C`. The reader releases the camera
and opens the same `--source` again at the same width and height. The terminal
and the window say whether it worked; if not, press `C` again. A camera can
open and still send no picture: the no-signal message then stays until a new
frame arrives, and `C` can be pressed again. `C` never
switches to another camera number. If the numbers have changed (an iPhone
joining or leaving can swap them), quit and run `python read.py --list-cameras`.
A recorded video never shows no signal; it simply ends.

#### Dim or washed-out tables

The reader can adjust each frame's contrast before the detector sees it
(`cardReader/preprocess.py`). Press `P` to try each mode live; the bottom bar
names the mode and its cost in milliseconds. The reader starts in `auto`
(`PREPROCESS` in `settings.py`); `--preprocess off` starts it in another mode.

| Mode | What it does | Benchmark verdict ([results](results.md#preprocessing-for-dim-and-bright-light)) |
|---|---|---|
| `off` | nothing; the frame goes to the detector as the camera sent it | best on a brightened picture, but loses many corners in a dim room |
| `gamma` | lifts the shadows with `PREPROCESS_BRIGHTEN_GAMMA` (0.6), through a lookup table | best in a dim room; costs corners on a bright picture |
| `clahe` | stretches contrast locally, tile by tile (contrast-limited adaptive histogram equalisation), on lightness only so colours do not shift | worse than `off` at every brightness except a very hazy picture; kept only to try by eye |
| `auto` (default) | measures the frame's mean grey level (0 to 255): below `PREPROCESS_DARK_BELOW` (110) it brightens with `PREPROCESS_BRIGHTEN_GAMMA`, above `PREPROCESS_BRIGHT_ABOVE` (190) it darkens with `PREPROCESS_DARKEN_GAMMA` (1.4), in between it leaves the frame alone | close to `gamma` in the dark, close to `off` in good light, and slightly ahead of `off` on an overlit table |

On an overlit table no mode helps much. Where the light clips the white of
the card, the index fades into it and no filter brings it back. A percentile
contrast stretch and a highlight-only tone curve were benchmarked and removed:
both were worse than `auto` overall (see
[Bright, overlit light](results.md#bright-overlit-light)).

At 1920×1080 on the development Mac (Apple silicon), `gamma` costs about
0.1 ms per frame, `clahe` about 2 ms, and `auto` 0.05 ms when it leaves the
frame alone, about 0.2 ms when it applies a gamma. The detector takes far longer.

To check a mode against the real validation photos under several kinds of
light, run `python benchmark.py` (a quick run: `python benchmark.py --limit 20`).
It needs `data/dataset/realVal.yaml` from `python train.py --rebuild`, or
`--data` pointing at one. It prints, per light level and mode, how many
labelled corner indices were found and named right, the extra boxes, and the
milliseconds spent preprocessing and detecting. A level is a gamma (`3.5` very
dark, `1` unchanged, `0.6` brighter), `gain1.6` or `gain2.2` (overexposed, with
the highlights clipped to white), or `flat0.55` or `flat0.35` (hazy, low contrast).
`--levels 1 gain2.2 flat0.35`, `--modes off auto`, `--imgsz 1024` and
`--csv bench.csv` narrow or record a run. The full run takes about 11 minutes
on a CPU; the results are in [results.md](results.md#preprocessing-for-dim-and-bright-light).

The window shows the preprocessed frame, so you see what the detector sees.
Press `Shift+V` to compare with the raw camera frame; the bottom bar then says
`showing raw`. The detector always gets the preprocessed frame, and `S` always
saves the raw one, so saved frames stay true to the camera for labelling.

**Glare cannot be fixed in software.** Where a lamp reflects off a card, the
sensor is saturated: those pixels are pure white and hold no trace of the
index. Darkening them only gives grey. Move the light or angle the camera
instead (see Placement above).

### Seats, hands, and the count

**Seats.** The layout is fixed and only depends on the number of players: the
dealer strip across the top of the image and the player seats on an arc below
it, P1 on the left. Press `1` to `5` to change the number of seats at any time.
Each card belongs to the seat its corner index lies in; a card outside every
seat is listed as OTHER. Point the camera so the dealer's side of the table is
at the top of the image.

**Adjusting the zones.** If the template does not fit your table, press `E`.
Every zone gets a white handle on each corner and a yellow `+` at the middle of
each edge:

| Mouse | Effect |
|---|---|
| drag a corner handle | moves that corner anywhere; the others stay put |
| drag a `+` | adds a corner there and moves it, so a zone can follow a curved or angled seat (a click without dragging adds nothing) |
| right-click a corner | removes it (a zone keeps at least 3 corners) |
| drag inside a zone | moves the whole zone, keeping its shape |
| right-click inside a zone | resets it to the template shape, removing added corners |

A click grabs the nearest handle within reach, which is a little wider than the
drawn handle. A drag past the frame edge stops at the edge; a handle left on the
right or bottom edge can be grabbed from just outside the video.

Press `E` again to save and lock. Layouts are saved to `data/seats.json`, **one
per player count**, and work at any camera resolution:

- Pressing `1` to `5` brings back your saved layout for that count, or the
  template if you never saved one. Layouts saved for 6 or 7 players by an
  older version stay in the file but are no longer loaded.
- At start-up the reader opens the player count you saved last, with its layout.
  `--players 3` starts with 3 players instead (their saved layout, or the template).
- With nothing saved yet, it starts with `PLAYERS` from `settings.py` (5).
- Delete `data/seats.json` to forget every saved layout.

Cards keep being read while you edit, and each card's seat follows the zones as you move them.
While editing, the detector runs on one frame in three (`EDIT_DETECT_EVERY` in
`read.py`), so the window keeps handling the mouse even when detection is slow.

**Hands.** Each seat's cards make one hand. Aces count 11 unless that busts,
then 1. Face-down cards are left out of the total and shown as `+1 hidden`.

**Split hands.** A player holding a pair can split it into two hands with
`Shift+1` to `Shift+5` (P1 to P5) or the page's `Split` button. The seat must
hold exactly two cards, both face up and of equal value; anything else is refused
with a message and nothing changes. The seat then shows two lines, its name
followed by `1` and `2`; hand 1 starts from the card further left in the image.
Every later card in the seat joins the hand whose first card lies nearest. Each
hand has its own total and hint.

- The same key or the `Join` button makes one hand again.
- `R`, `X`, `Shift+R`, a new player count, and the end of a round clear every split.
- Taking either of the two first cards off the table makes the seat one hand again.
- The terminal prints one line per split or join, such as
  `--- P1 split: 8S | 8H ---`. The `--log` CSV is unchanged: every card keeps
  the seat id (`P1`), so `evaluate.py` scores the seat as one hand.
- A split hand that receives another pair cannot split again.

**Counting systems.** Every face card adds its value to the running count once
per shoe. The reader starts on `COUNT_SYSTEM` in `settings.py` (`"hi_lo"`,
`"hi_opt_ii"`, or `"uston_apc"`), and the page's `Count` dropdown switches it
live. Each system sums to zero over a deck:

| Cards | Hi-Lo | Hi-Opt II | Uston APC |
|---|---|---|---|
| 2 | +1 | +1 | +1 |
| 3 | +1 | +1 | +2 |
| 4 | +1 | +2 | +2 |
| 5 | +1 | +2 | +3 |
| 6 | +1 | +1 | +2 |
| 7 | 0 | +1 | +2 |
| 8 | 0 | 0 | +1 |
| 9 | 0 | 0 | -1 |
| 10, J, Q, K | -1 | -2 | -3 |
| A | -1 | 0 | 0 |

The true count is the running count divided by the decks still to be dealt
(Hi-Lo, Hi-Opt II) or the half-decks (Uston APC). In Hi-Lo a running count of
+4 with half the deck left is a true count of +8; in Uston APC the same +4 is a
true count of +4, since half a deck is one half-deck. Hi-Opt II and Uston APC
count aces as 0, and the reader does not adjust for them; the page shows the
aces left beside the count instead. A positive
true count means the remaining cards are rich in high cards (tens and aces),
which favours the player. A negative one means low cards are left, which
favours the dealer.

- **Cards left** is 52 per deck in the shoe (see [Decks in the shoe](#decks-in-the-shoe))
  minus the face cards seen, minus the face-down cards that were dealt and never
  turned up (swept away face down, for example). Those are gone from the shoe,
  but nobody saw them, so they do not change the running count.
- Face-down cards are counted when they are turned up, and then stop being
  "never turned up". A face-down card still on the table when you press `R`
  (new round, lowercase r) gets a new number, but is not taken off cards left a
  second time. `Shift+R` (new shoe) resets all of this.
- **Decks left never drops below a quarter deck** (13 cards, which is half a
  half-deck for Uston APC). Near the end of the deck, dividing by a sliver of a
  deck would make the true count huge: in Hi-Lo, with one card left and a
  running count of -1 it reads -4, not -52.

### Basic strategy hints

`cardReader/strategy.py` suggests a play for each player hand against the
dealer's up card: `hit`, `stand`, `double` or `split`. The
[browser dashboard](#browser-dashboard) shows it at the end of each player seat's line. The up card is the
dealer's first face-up card. There is no hint when the hand has a face-down
card, fewer than two cards, is bust or is a blackjack, or when the dealer has no
face-up card. After a split each hand gets its own hint. Resplitting is not
allowed, so a split hand never shows `split`: a second pair is played as its hard
or soft total (8, 8 is hard 16, A, A is soft 12, always hit). Split aces get the
ordinary tables, with no stop after one card.

The table is the common multi-deck one, with these rules:

- The dealer stands on soft 17.
- Double on any first two cards. With three or more cards, `D` becomes hit and
  `Ds` becomes stand.
- Double after a split is allowed.
- No surrender.

`H` hit, `S` stand, `P` split, `D` double (else hit), `Ds` double (else stand).

**Hard totals**

| Player | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | A |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 or less | H | H | H | H | H | H | H | H | H | H |
| 9 | H | D | D | D | D | H | H | H | H | H |
| 10 | D | D | D | D | D | D | D | D | H | H |
| 11 | D | D | D | D | D | D | D | D | D | H |
| 12 | H | H | S | S | S | H | H | H | H | H |
| 13 to 16 | S | S | S | S | S | H | H | H | H | H |
| 17 or more | S | S | S | S | S | S | S | S | S | S |

**Soft totals** (an ace counted as 11)

| Player | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | A |
|---|---|---|---|---|---|---|---|---|---|---|
| A,2 and A,3 (13, 14) | H | H | H | D | D | H | H | H | H | H |
| A,4 and A,5 (15, 16) | H | H | D | D | D | H | H | H | H | H |
| A,6 (17) | H | D | D | D | D | H | H | H | H | H |
| A,7 (18) | S | Ds | Ds | Ds | Ds | S | S | H | H | H |
| A,8 or more (19 to 21) | S | S | S | S | S | S | S | S | S | S |

**Pairs** (two cards of the same value, so J,Q is a pair of tens)

| Player | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | A |
|---|---|---|---|---|---|---|---|---|---|---|
| 2,2 and 3,3 | P | P | P | P | P | P | H | H | H | H |
| 4,4 | H | H | H | P | P | H | H | H | H | H |
| 5,5 | D | D | D | D | D | D | D | D | H | H |
| 6,6 | P | P | P | P | P | H | H | H | H | H |
| 7,7 | P | P | P | P | P | P | H | H | H | H |
| 8,8 and A,A | P | P | P | P | P | P | P | P | P | P |
| 9,9 | P | P | P | P | P | S | P | P | S | S |
| 10,10 | S | S | S | S | S | S | S | S | S | S |

The count can hold one to three decks, and this table is the multi-deck one. The two
differ in a handful of close plays, such as hard 11 against an ace, which
doubles in single deck.

### Rules the tracker assumes

- **A card is only confirmed when it passes every check** (all in `settings.py`):

  | Check | Setting | Stops |
  |---|---|---|
  | seen for `CONFIRM_SECONDS` of the last `CONFIRM_WINDOW_SECONDS` | 0.5 s of 0.7 s (6 of 9 frames) | flickering detections adding up over time |
  | the rank of its most common name has `CONFIRM_AGREEMENT` of all reads | 80% | a card being counted under a disputed rank |
  | the model's average confidence is at least `CONFIRM_CONFIDENCE` | 0.7 | weak, borderline detections |
  | the corner moved less than `STILL_DISTANCE` over those frames | 0.6% of the width | a card being counted while it is still being slid |

  With the defaults a real card is confirmed about 0.5 s after it settles.
  Frames are given at `FPS` = 12.4, the speed the reader runs at.
  Reads split between suits of one rank (10H and 10D) agree, since the suit
  changes neither the hand nor the count; the card shows its most common name.
- **One deck (`DECKS` = 1).** Two sightings of the same face are the same card. This merges a
  card's corner indices, stops a stray misread from adding a duplicate, and means
  a card is counted once per shoe even if it leaves and comes back. A new card
  under a name counted in an earlier round prints as `duplicate`.
- **Two or three decks.** Two sightings of the same face are the same card only
  within four `MATCH_DISTANCE`s of each other; farther apart they are two copies.
  A card has two corner indices, so it joins at most two such sightings: a third
  one close by is a second copy, and two of a card fanned in one hand stay two
  cards. A hole card turned over, or a covered card coming back, still joins its
  own card. The cost: beside a card already showing both corners, a stray misread
  of its name can place a phantom copy, once it passes the confirmation checks.
  Each card is counted once per deck, and one more copy prints as `duplicate`
  (see [Decks in the shoe](#decks-in-the-shoe)).
- **Covered is not removed, at first.** A card that disappears for
  `DROP_SECONDS` (1 s, 12 frames) becomes hidden: it keeps its number, stays in its hand, and is not
  counted again when it reappears. A face card is recognised by its name anywhere
  on the table, a face-down card by reappearing near where it was.
- **Hidden too long is removed.** A card that stays hidden for
  `HIDDEN_SECONDS` (5 s, 62 frames) was picked up, like a hand the dealer collects mid-round. It
  leaves the round and its hand, and prints as `removed`, but stays in the count.
  Its number is never given to another card. If it comes back later it is placed
  under a new number and is not counted again. A face-down card that comes back
  this way is taken off cards left a second time. `HIDDEN_SECONDS = 0` removes a
  card as soon as it is hidden.
- **The round ends when the table is cleared.** If no card is visible for
  `ROUND_END_SECONDS` (3 s, 37 frames), the round is over and numbering
  restarts. This holds when every card was removed first. A hand briefly
  covering the whole table does not end it. `R` ends a round immediately.
- **Every face-down card is a separate card.**
- **A card's name can change, its number cannot.** The name is a vote over the
  reads in the last `LABEL_WINDOW_SECONDS` (1.5 s, 19 frames). A misread that corrects itself is
  renamed in place, and prints as `renamed`. The count follows the new name, and
  a name stays counted while any card of the round still holds it.
- **A hole card turned face up keeps its number.** The face-down box is drawn
  around the whole card, but the face-up boxes are its corner indices, away from
  the centre. A new face-up corner that is not near any card yet joins the nearest
  face-down card within 3 match distances that is out of sight this frame, and
  that card is renamed from `BACK` to its name. A face-down card still in view is
  never taken, so two face-down cards side by side keep their own numbers when
  they are turned over one after the other. No second phantom card is added.

## Scoring a recorded run

`evaluate.py` checks the whole reader, not only the detector: placement order,
names, and the count. Record a video, write down the cards dealt, then:

```bash
python read.py --source round1.mp4 --log round1.csv --no-display
python evaluate.py --log round1.csv --expected round1Expected.csv
```

The expected file is a CSV with one row per card, in the order placed. `round`
counts from 1 and `order` restarts at 1 each round. `seat` is optional. A card
left face down is written as `BACK`.

```
round,order,card,seat
1,1,10H,P1
1,2,6C,P2
1,3,KS,dealer
2,1,4D,P1
```

The report:

```
cards: 4 expected, 4 placed
matched: 4 (3 right name, 1 wrong name: 1 wrong suit, 0 wrong rank)
missed: 0
extra: 0
wrong seat: 0 (listed, not counted as errors)
running count: read +0, expected +0, difference +0

round 1 #3: expected KS, read KH (wrong suit)
```

- Rounds in the log end at each `round_over`. Pressing `R` logs nothing, so a
  card placed under a number already used in the round also starts a new one.
- Pressing `X` logs a `rescan` row. The cards of the round in progress are
  dropped, and the cards read after it are scored as that same round. Before
  the round's first card, `X` logs nothing, so the round `R` just ended is kept.
- A renamed card is scored under its last name.
- Cards are matched by round and number. A missed card shifts the numbers after
  it, so the rest of that round shows as wrong names.
- Use one shoe per log. `Shift+R` resets the logged running count.
- The expected running count is Hi-Lo. The log records the running count in
  the system the reader counts in, so record a run to score with Hi-Lo.
- The exit code is 1 only when a file is missing or has a bad row, not when cards disagree.

## Tuning the reader

All of these are in `settings.py`. Change one at a time and watch the effect.
A value out of range (a fraction above 1, `CONFIRM_SECONDS` larger than
`CONFIRM_WINDOW_SECONDS`, and so on) stops `read.py` at start-up with a message naming it.

| If you see this | Change | Why |
|---|---|---|
| Cards are missed or take a long time to appear | lower `CONFIDENCE` (e.g. 0.6 to 0.5) | the model accepts less certain boxes |
| Wrong or phantom cards appear | raise `CONFIDENCE` (e.g. 0.6 to 0.7) | only confident boxes count |
| Phantom or wrong cards get confirmed | raise `CONFIRM_CONFIDENCE` (0.7 to 0.8) or `CONFIRM_AGREEMENT` (0.8 to 0.9) | only consistent, confident reads count |
| A flickering detection still gets confirmed | raise `CONFIRM_SECONDS` closer to `CONFIRM_WINDOW_SECONDS` (e.g. 0.6 s of 0.7 s, 7 of 9 frames) | it must be seen almost every frame |
| A card is confirmed while still being slid | lower `STILL_DISTANCE` (0.006 to 0.004) | it must be stiller |
| Real cards never get a number | lower `CONFIRM_CONFIDENCE` (to 0.6) or raise `STILL_DISTANCE` (to 0.01, a shaky camera) | a check is too strict for your setup |
| Cards take too long to get their number | lower `CONFIRM_SECONDS` and `CONFIRM_WINDOW_SECONDS` together (e.g. 0.3 s of 0.5 s, 4 of 6 frames) | confirms sooner, with more risk of flicker |
| A covered card turns grey too quickly | raise `DROP_SECONDS` (e.g. 1 s to 2 s, 12 to 25 frames) | it can stay out of sight longer before it is marked hidden |
| A covered card is removed while still on the table | raise `HIDDEN_SECONDS` (e.g. 5 s to 10 s, 62 to 124 frames) | it can stay hidden longer before it counts as picked up |
| A collected hand stays in the seats too long | lower `HIDDEN_SECONDS` (e.g. 5 s to 3 s, 62 to 37 frames); 0 removes at once | a hidden card counts as picked up sooner |
| The round ends while cards are only covered | raise `ROUND_END_SECONDS` (e.g. 3 s to 5 s, 37 to 62 frames) | the table must stay empty longer |
| The next round starts too slowly after a sweep | lower `ROUND_END_SECONDS` | the empty table is noticed sooner |
| Cards land in the wrong seat | press `1` to `5` for the right player count, then `E` to drag the zones over the real seats | the template only approximates a table |
| One card gets two numbers when nudged | raise `MATCH_DISTANCE` (e.g. 0.03 to 0.05) | a moved corner is still matched to the same card |
| A turned-over hole card keeps saying BACK too long | lower `LABEL_WINDOW_SECONDS` (1.5 s to 1 s, 19 to 12 frames) | the name vote forgets old reads sooner |
| A card's name flips back and forth | raise `LABEL_WINDOW_SECONDS` (1.5 s to 3 s, 19 to 37 frames) | one bad second of reads cannot outvote the rest |
| Small or distant cards are not read | raise `INPUT_SIZE` (e.g. 1280 to 1600) | the model sees more detail; costs speed |
| The video is choppy (low fps) | lower `INPUT_SIZE` (e.g. 1280 to 960) | less work per frame |
| Cards are missed in a very dim room | `PREPROCESS = "gamma"` (try with `P` first) | the shadows are always lifted, not only below `PREPROCESS_DARK_BELOW` |
| `auto` switches on too late or too early | `PREPROCESS_DARK_BELOW` or `PREPROCESS_BRIGHT_ABOVE` | the mean grey level (0 to 255) at which it brightens or darkens |

The `*_SECONDS` settings are turned into frames once, at start-up:
`max(1, round(seconds * FPS))`, so every one is at least a frame, except
`HIDDEN_SECONDS = 0`. The frames in brackets above are at `FPS = 12.4`, the
speed the reader runs at on the development Mac. If your reader runs at another
speed, set `FPS` to it and every time keeps its meaning; the quit summary prints
the measured fps when it is more than a fifth away from `FPS`.

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'cv2'` | another virtual environment is active | `deactivate`, then `source .venv/bin/activate` in the project folder; check `which python` |
| Reader window is black | camera permission not granted | System Settings > Privacy & Security > Camera, allow Terminal or VS Code, restart it |
| Wrong camera opens | camera numbers change when an iPhone connects or disconnects | `python read.py --list-cameras` shows a snapshot from each number |
| The phone's view is plain grey | it points at a plain surface, the lens is blocked, or it is too close to focus | aim it at the table from 50 cm or more; `--list-cameras` marks such a camera as nearly blank |
| `Could not open video source` | the camera is in use or not connected | close other apps using it (FaceTime, Zoom, Photo Booth); wake the iPhone camera |
| `no signal: press C to reconnect` | the camera stopped sending frames | reconnect the cable or wake the phone, then press `C` (see [Camera disconnects](#camera-disconnects)) |
| Bottom bar says CAPTURE MODE | no model at `MODEL_PATH` | check `models/cards/weights/best.pt` exists, or train one |
| Cards on screen but none get a number | detections too weak, or the camera is too far | see [Tuning the reader](#tuning-the-reader); move the camera closer |
| Thin grey boxes clutter the table | detections not matched to a confirmed card: cards still settling, or stray boxes | press `G` to hide them; a one-frame misread of a confirmed card's corner keeps the card's colour, not grey |
| Keys do nothing | the terminal has focus, not the window | click the reader window first |
| The window close button does not quit | OpenCV reopens the window | press `Q`, or Ctrl+C in the terminal |
| Training stops with an out-of-memory error | batch too large for the GPU | `BATCH_SIZE = 4` in `settings.py` |
| Training reports `device=cpu` and is very slow | wrong environment, or no GPU support | activate this project's `.venv`; on a Mac check `python -c "import torch; print(torch.backends.mps.is_available())"` |
| An export says `skip ... boxes are whole cards` | it is labelled around whole cards, not corners | relabel the corners in Roboflow and export again |
| Zones in odd places after starting | a layout saved earlier was loaded | press `E`, then right-click inside each zone to reset it, or delete `data/seats.json` |
| Dragging misses the handles in a resized or Retina window | the window is scaled, so handles are drawn smaller than they are in the video | press on the handle's centre; the grab area is wider than the drawn handle |
| A drag does nothing, or starts away from where you pressed | on macOS the window reads the pointer when it handles the click, which waits for the detector | hold still for a moment after pressing; if it persists, raise `EDIT_DETECT_EVERY` in `read.py` |
| `yolo val` results end up in `runs/detect/` | relative `project=` path | use `project="$PWD/models"` as in [Validate a model on its own](training.md#5-validate-a-model-on-its-own) |

## Known limitations

- **At most three decks.** `DECKS` and the page's decks button allow 1 to 3.
  With more than one deck the reader tells a second copy from the same card by
  distance only, so a card picked up and put down far away counts as another copy
  while copies are left, and two copies dealt touching may merge into one.
- **Splits** must be marked by hand (`Shift+1` to `Shift+5` or the page), at most
  two hands per seat. The `--log` CSV records the seat, not the hand, so
  `evaluate.py` scores a split seat as one hand.
- **Face-down cards.** All `BACK` training examples are synthetic, drawn with a
  placeholder design. Photograph your deck's back (see
  [Using your own deck](training.md#using-your-own-deck-for-synthetic-images)) before relying on it.
- **A misread that never corrects itself** stays in the count under the wrong
  name. The tracker can only follow what the model reads.
- **A face card dealt right beside a face-down card** can be taken for its reveal
  if the face-down box happens to be missed on the very frame the new card is
  confirmed. The new card then gets no number of its own: its corners join the
  face-down card, whose name is a vote between its `BACK` box and those corners,
  so it can be renamed to the new card's name and counted under it.
- **Same-colour suits** are the most common confusion (see [Results](results.md)).
- **No frame tiling.** A 4K frame is shrunk to `INPUT_SIZE` before detection, so
  very small indices at a large table may be lost.
- **The validation set includes synthetic images**, which makes the full score
  optimistic; report the real-photo score.
