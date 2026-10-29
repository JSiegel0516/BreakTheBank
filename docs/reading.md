# Reading cards off the table

Part of [BreakTheBank](../README.md). Training the detector is in [training.md](training.md).

## Reading

`python read.py` opens a window with the live feed, an overlay on it, and a
sidebar on the right so nothing covers the table (drawn by `cardReader/hud.py`):

| On screen | Meaning |
|---|---|
| Seat outlines | the dealer strip at the top and the player seats on an arc below it (`cardReader/seats.py`) |
| Sidebar, top | the Hi-Lo true count in large type with whom it favours (`favours player`, `favours dealer`, `neutral`), then the running count and cards left (`38/52`) |
| Sidebar, below | the counter view or the seats view; `V` switches between them (see below) |
| Yellow badge and name on a card | that card's placement number and name |
| Grey badge and name | a hidden card: covered right now, but still in the hand and in the count |
| Coloured boxes | corner indices the tracker has matched to a confirmed card, in that card's colour: red for hearts and diamonds, white for spades and clubs, amber for face-down. A frame that misreads the corner (a 6H read as 9H) keeps the card's colour |
| Thin grey boxes | detections not matched to a confirmed card: a card not confirmed yet, or a stray box away from every card. `G` hides them |
| Sidebar, bottom | the legend for the current view, then every key (`L` hides it) |
| Bottom bar | the main keys, frames per second, the milliseconds per frame spent detecting, and the preprocess mode with its milliseconds (`showing raw` when `Shift+V` shows the raw camera frame) |
| `no signal: press C to reconnect` | the camera has sent no new frame for two seconds (see [Camera disconnects](#camera-disconnects)) |

The number is the placement order within the round: 1 is the first card
confirmed (seen for `CONFIRM_FRAMES` frames in a row), 2 the next, and so on. A
card that is covered keeps its number and comes back with it.

### Counter view and seats view

The reader starts in the **counter view**: the whole deck as a grid, with one
row per rank (2 to 10, J, Q, K, A) and one column per suit (spades, hearts,
diamonds, clubs). It shows at a glance which cards have been dealt this shoe:

| Cell | Meaning |
|---|---|
| Plain | not seen this shoe |
| Yellow slash | pending: counted this round, so a corrected misread can still change it |
| Grey, crossed off | locked: counted in an earlier round of this shoe |
| Magenta ring | a `duplicate` was reported for this card this shoe (a misread, or a reshuffle without `Shift+R`) |

Rows are tinted by Hi-Lo group and labelled with its value: green for 2 to 6
(+1), grey for 7 to 9 (0), red for 10 to A (-1). The column on the right gives
how many cards of that rank are left. The line under the grid counts the
face-down cards dealt and never turned over. `Shift+R` clears the grid. The
grid assumes one 52-card deck. In the counter view the grid comes first: when
the window is too small for both, the legend is left out.

The **seats view** lists each seat's total (`17`, `soft 18`, `BLACKJACK`,
`BUST (24)`, `9 +1 hidden`) and its cards in placement order. Cards outside
every seat are listed as OTHER. The video, the card tags, and the zone editor
look the same in both views.

### What the colours mean

The legend at the bottom of the sidebar explains the colours of the current
view. The counter view's colours are in the table above.

| Colour | Where | Meaning |
|---|---|---|
| Red box or name | corner index, tags, sidebar | a confirmed heart or diamond (the suit colour, not an error) |
| White box or name | same | a confirmed spade or club |
| Amber box or name | same | a confirmed face-down card |
| Thin grey box | corner index | a detection not matched to a confirmed card: it has not passed the [confirmation checks](#rules-the-tracker-assumes) yet, or it is away from every card (`G` hides these) |
| Yellow badge | tags, sidebar | the card's placement number |
| Grey badge and name | tags, sidebar | a hidden card: covered right now, still in its hand and the count |
| Amber outline | zones | the dealer's zone |
| Light blue outline | zones | a player's zone |
| Yellow tint | zones, edit mode | the zone you last clicked |
| Green or red count | sidebar | a positive (favours the player) or negative (favours the dealer) count |

### Keys

The bottom bar lists the main keys; the legend lists them all.

| Key | Action |
|---|---|
| `1` to `7` | number of players; loads your saved layout for that count, or the template |
| `E` | edit zones with the mouse; press `E` again to save them and lock |
| `V` | switch the sidebar between the counter view and the seats view |
| `L` | show or hide the legend and key list |
| `G` | show or hide the thin grey boxes, the detections not matched to a confirmed card |
| `R` | new round: forget the table, numbering restarts at 1, the count is kept |
| `Shift+R` | new shoe: after a reshuffle, the count goes back to 0 |
| `S` | save the current frame to `data/raw/frames/` for labelling |
| `C` | reconnect the camera: reopen the same source after a cable or power interruption |
| `P` | next preprocess mode: off, gamma, clahe, auto (see [Dim or washed-out tables](#dim-or-washed-out-tables)) |
| `Shift+V` | show the raw camera frame, or the preprocessed frame the detector gets |
| `Q` | quit (or Ctrl+C in the terminal) |

Keys work when the reader window is selected. The window can be resized.
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

`read` is taking the newest camera frame, `preprocess` the contrast help (see
[Dim or washed-out tables](#dim-or-washed-out-tables)), `detect` the model, and
`draw` the overlay and the window. The bottom bar shows the same detect and
preprocess times, smoothed, while it runs.

At start-up the model line names the device the detector runs on, as in
`model models/cards/weights/best.pt on mps`: `mps` (Apple GPU), `cuda` (NVIDIA
GPU), or `cpu`. `cpu` on a Mac with Apple silicon makes detect several times
slower; try `--device mps`.

A `duplicate` line means a face card was placed (or renamed) under a name already
counted this shoe in an earlier round. One deck holds one of each card, so it is
a misread or a reshuffle nobody marked with `Shift+R`. The count is not changed.

- `--source 0` picks the camera. With an iPhone as a Continuity Camera, OpenCV
  usually numbers it 0 and the built-in camera 1; try the other number if the
  wrong one opens.
- `--source round1.mp4` reads a recorded video instead.
- `--players 5` sets the starting number of seats (`PLAYERS` in `settings.py`).
- `--log round.csv` appends every event to a CSV, with its seat and the running count.
- `--preprocess auto` picks the starting preprocess mode (`PREPROCESS` in `settings.py`): `off`, `gamma`, `clahe`, or `auto`.
- `--no-display` runs without the window; stop it with Ctrl+C.

macOS asks for camera permission the first time OpenCV opens a camera from a
terminal or VS Code. Grant it under System Settings > Privacy & Security > Camera.

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
| `clahe` | stretches contrast locally, tile by tile (contrast-limited adaptive histogram equalisation), on lightness only so colours do not shift | worse than `off` at every brightness; kept only to try by eye |
| `auto` (default) | measures the frame's mean grey level (0 to 255): below `PREPROCESS_DARK_BELOW` (110) it brightens with `PREPROCESS_BRIGHTEN_GAMMA`, above `PREPROCESS_BRIGHT_ABOVE` (190) it darkens with `PREPROCESS_DARKEN_GAMMA` (1.4), in between it leaves the frame alone | close to `gamma` in the dark and close to `off` in good light |

At 1920×1080 on the development Mac (Apple silicon), `gamma` costs about
0.1 ms per frame, `clahe` about 2 ms, and `auto` 0.05 ms when it leaves the
frame alone, about 0.2 ms when it applies a gamma. The detector takes far longer.

To check a mode against the real validation photos at several brightness
levels, run `python benchmark.py` (a quick run: `python benchmark.py --limit 20`).
It needs `data/dataset/realVal.yaml` from `python train.py --rebuild`, or
`--data` pointing at one. It prints, per brightness level and mode, how many
labelled corner indices were found and named right, the extra boxes, and the
milliseconds spent preprocessing and detecting. `--levels 3.5 1.0`, `--modes off auto`,
`--imgsz 1024` and `--csv bench.csv` narrow or record a run. The full run takes
about 6 minutes on a CPU; the results are in [results.md](results.md#preprocessing-for-dim-and-bright-light).

The window shows the preprocessed frame, so you see what the detector sees.
Press `Shift+V` to compare with the raw camera frame; the bottom bar then says
`showing raw`. The detector always gets the preprocessed frame, and `S` always
saves the raw one, so saved frames stay true to the camera for labelling.

**Glare cannot be fixed in software.** Where a lamp reflects off a card, the
sensor is saturated: those pixels are pure white and hold no trace of the
index. Darkening them only gives grey. Move the light or angle the camera
instead (see Placement above).

### Seats, hands, and the Hi-Lo count

**Seats.** The layout is fixed and only depends on the number of players: the
dealer strip across the top of the image and the player seats on an arc below
it, P1 on the left. Press `1` to `7` to change the number of seats at any time.
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

- Pressing `1` to `7` brings back your saved layout for that count, or the
  template if you never saved one.
- At start-up the reader opens the player count you saved last, with its layout.
  `--players 3` starts with 3 players instead (their saved layout, or the template).
- With nothing saved yet, it starts with `PLAYERS` from `settings.py` (5).
- Delete `data/seats.json` to forget every saved layout.

Cards keep being read while you edit, and each card's seat follows the zones as you move them.
While editing, the detector runs on one frame in three (`EDIT_DETECT_EVERY` in
`read.py`), so the window keeps handling the mouse even when detection is slow.

**Hands.** Each seat's cards make one hand. Aces count 11 unless that busts,
then 1. Face-down cards are left out of the total and shown as `+1 hidden`.
Split hands are not separated yet.

**Hi-Lo count.** Every face card adds to the running count once per shoe:

| Cards | Value |
|---|---|
| 2, 3, 4, 5, 6 | +1 |
| 7, 8, 9 | 0 |
| 10, J, Q, K, A | -1 |

The true count is the running count divided by the decks still to be dealt, so a
running count of +4 with half the deck left is a true count of +8. A positive
true count means the remaining cards are rich in tens and aces.

- **Cards left** is 52 minus the face cards seen, minus the face-down cards that
  were dealt and never turned up (swept away face down, for example). Those are
  gone from the deck, but nobody saw them, so they do not change the running count.
- Face-down cards are counted when they are turned up, and then stop being
  "never turned up". A face-down card still on the table when you press `R`
  (new round, lowercase r) gets a new number, but is not taken off cards left a
  second time. `Shift+R` (new shoe) resets all of this.
- **Decks left never drops below a quarter deck** (13 cards). Near the end of the
  deck, dividing by a sliver of a deck would make the true count huge: with one
  card left and a running count of -1 it reads -4, not -52.

### Rules the tracker assumes

- **A card is only confirmed when it passes every check** (all in `settings.py`):

  | Check | Setting | Stops |
  |---|---|---|
  | seen in `CONFIRM_FRAMES` of the last `CONFIRM_WINDOW` frames | 15 of 20 | flickering detections adding up over time |
  | the rank of its most common name has `CONFIRM_AGREEMENT` of all reads | 80% | a card being counted under a disputed rank |
  | the model's average confidence is at least `CONFIRM_CONFIDENCE` | 0.7 | weak, borderline detections |
  | the corner moved less than `STILL_DISTANCE` over those frames | 0.6% of the width | a card being counted while it is still being slid |

  With the defaults a real card is confirmed about 0.7 s after it settles.
  Reads split between suits of one rank (10H and 10D) agree, since the suit
  changes neither the hand nor the count; the card shows its most common name.
- **One deck.** Two sightings of the same face are the same card. This merges a
  card's corner indices, stops a stray misread from adding a duplicate, and means
  a card is counted once per shoe even if it leaves and comes back. A new card
  under a name counted in an earlier round prints as `duplicate`.
- **Covered is not removed.** A card that disappears for `DROP_FRAMES` frames
  becomes hidden: it keeps its number, stays in its hand, and is not counted again
  when it reappears. A face card is recognised by its name anywhere on the table,
  a face-down card by reappearing near where it was.
- **The round ends when the table is cleared.** If no card is visible for
  `ROUND_END_FRAMES` frames (about 3 seconds), the round is over and numbering
  restarts. A hand briefly covering the whole table does not end it. `R` ends a
  round immediately.
- **Every face-down card is a separate card.**
- **A card's name can change, its number cannot.** The name is a vote over the
  last `LABEL_WINDOW` frames of reads (1.5 s). A misread that corrects itself is
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
- A renamed card is scored under its last name.
- Cards are matched by round and number. A missed card shifts the numbers after
  it, so the rest of that round shows as wrong names.
- Use one shoe per log. `Shift+R` resets the logged running count.
- The exit code is 1 only when a file is missing or has a bad row, not when cards disagree.

## Tuning the reader

All of these are in `settings.py`. Change one at a time and watch the effect.
A value out of range (a fraction above 1, `CONFIRM_FRAMES` larger than
`CONFIRM_WINDOW`, and so on) stops `read.py` at start-up with a message naming it.

| If you see this | Change | Why |
|---|---|---|
| Cards are missed or take a long time to appear | lower `CONFIDENCE` (e.g. 0.6 to 0.5) | the model accepts less certain boxes |
| Wrong or phantom cards appear | raise `CONFIDENCE` (e.g. 0.6 to 0.7) | only confident boxes count |
| Phantom or wrong cards get confirmed | raise `CONFIRM_CONFIDENCE` (0.7 to 0.8) or `CONFIRM_AGREEMENT` (0.8 to 0.9) | only consistent, confident reads count |
| A flickering detection still gets confirmed | raise `CONFIRM_FRAMES` closer to `CONFIRM_WINDOW` (e.g. 18 of 20) | it must be seen almost every frame |
| A card is confirmed while still being slid | lower `STILL_DISTANCE` (0.006 to 0.004) | it must be stiller |
| Real cards never get a number | lower `CONFIRM_CONFIDENCE` (to 0.6) or raise `STILL_DISTANCE` (to 0.01, a shaky camera) | a check is too strict for your setup |
| Cards take too long to get their number | lower `CONFIRM_FRAMES` and `CONFIRM_WINDOW` together (e.g. 10 of 14) | confirms sooner, with more risk of flicker |
| A covered card turns grey too quickly | raise `DROP_FRAMES` | it can stay hidden longer before it is marked hidden |
| The round ends while cards are only covered | raise `ROUND_END_FRAMES` (e.g. 90 to 150) | the table must stay empty longer |
| The next round starts too slowly after a sweep | lower `ROUND_END_FRAMES` | the empty table is noticed sooner |
| Cards land in the wrong seat | press `1` to `7` for the right player count, then `E` to drag the zones over the real seats | the template only approximates a table |
| One card gets two numbers when nudged | raise `MATCH_DISTANCE` (e.g. 0.03 to 0.05) | a moved corner is still matched to the same card |
| A turned-over hole card keeps saying BACK too long | lower `LABEL_WINDOW` (45 to 30) | the name vote forgets old reads sooner |
| A card's name flips back and forth | raise `LABEL_WINDOW` (45 to 90) | one bad second of reads cannot outvote the rest |
| Small or distant cards are not read | raise `INPUT_SIZE` (e.g. 1280 to 1600) | the model sees more detail; costs speed |
| The video is choppy (low fps) | lower `INPUT_SIZE` (e.g. 1280 to 960) | less work per frame |
| Cards are missed in a very dim room | `PREPROCESS = "gamma"` (try with `P` first) | the shadows are always lifted, not only below `PREPROCESS_DARK_BELOW` |
| `auto` switches on too late or too early | `PREPROCESS_DARK_BELOW` or `PREPROCESS_BRIGHT_ABOVE` | the mean grey level (0 to 255) at which it brightens or darkens |

`CONFIRM_FRAMES`, `DROP_FRAMES`, and `ROUND_END_FRAMES` are counted in frames, so at 30 fps, 30
frames is one second. If the fps is much lower, scale them down to match.

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

- **One deck.** Two sightings of the same face are treated as one card, so a
  shoe with several decks would merge real duplicates, and its count would be wrong.
- **Splits** are shown as one hand per seat.
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
