# Reading cards off the table

Part of [BreakTheBank](../README.md). Training the detector is in [training.md](training.md).

## Reading

`python read.py` opens a window with the live feed, an overlay on it, and a
sidebar on the right so nothing covers the table (drawn by `cardReader/hud.py`):

| On screen | Meaning |
|---|---|
| Seat outlines | the dealer strip at the top and the player seats on an arc below it (`cardReader/seats.py`) |
| Sidebar, top | the Hi-Lo running count, true count, and cards left in the deck |
| Sidebar, below | each seat's total (`17`, `soft 18`, `BLACKJACK`, `BUST (24)`, `9 +1 hidden`) and its cards in placement order; cards outside every seat are listed as OTHER |
| Yellow badge and name on a card | that card's placement number and name |
| Grey badge and name | a hidden card: covered right now, but still in the hand and in the count |
| Coloured boxes | the corner indices of confirmed cards: red for hearts and diamonds, white for spades and clubs, amber for face-down |
| Thin grey boxes | detections the tracker has not confirmed yet (a card still moving, or a one-frame misread) |
| Sidebar, bottom | the colour legend (`L` hides it) |
| Bottom bar | the keys, and frames per second |

The number is the placement order within the round: 1 is the first card
confirmed (seen for `CONFIRM_FRAMES` frames in a row), 2 the next, and so on. A
card that is covered keeps its number and comes back with it.

### What the colours mean

The same legend is shown at the bottom of the sidebar.

| Colour | Where | Meaning |
|---|---|---|
| Red box or name | corner index, tags, sidebar | a confirmed heart or diamond (the suit colour, not an error) |
| White box or name | same | a confirmed spade or club |
| Amber box or name | same | a confirmed face-down card |
| Thin grey box | corner index | something detected that has not passed the [confirmation checks](#rules-the-tracker-assumes) yet |
| Yellow badge | tags, sidebar | the card's placement number |
| Grey badge and name | tags, sidebar | a hidden card: covered right now, still in its hand and the count |
| Amber outline | zones | the dealer's zone |
| Light blue outline | zones | a player's zone |
| Yellow tint | zones, edit mode | the zone you last clicked |
| Green or red count | sidebar | a positive (favours the player) or negative (favours the dealer) count |

### Keys

| Key | Action |
|---|---|
| `1` to `7` | number of players; loads your saved layout for that count, or the template |
| `E` | edit zones with the mouse; press `E` again to save them and lock |
| `L` | show or hide the colour legend |
| `R` | new round: forget the table, numbering restarts at 1, the count is kept |
| `Shift+R` | new shoe: after a reshuffle, the count goes back to 0 |
| `S` | save the current frame to `data/raw/frames/` for labelling |
| `Q` | quit (or Ctrl+C in the terminal) |

Keys work when the reader window is selected. The window can be resized.
Every event is also printed in the terminal, with the seat and the count:

```
[   412] placed   #1   10H (P1) count -1
[   530] placed   #2   6C  (P2) count +0
[   901] hidden   #2   6C  (P2)
[   966] returned #2   6C  (P2)
[  2210] round_over
```

and on quitting: `counted this shoe: 10H 6C ...` with the final running and true count.

- `--source 0` picks the camera. With an iPhone as a Continuity Camera, OpenCV
  usually numbers it 0 and the built-in camera 1; try the other number if the
  wrong one opens.
- `--source round1.mp4` reads a recorded video instead.
- `--players 5` sets the starting number of seats (`PLAYERS` in `settings.py`).
- `--log round.csv` appends every event to a CSV, with its seat and the running count.
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
  hides the index) and strong shadows from your hands.
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

Press `E` again to save and lock. Layouts are saved to `data/seats.json`, **one
per player count**, and work at any camera resolution:

- Pressing `1` to `7` brings back your saved layout for that count, or the
  template if you never saved one.
- At start-up the reader opens the player count you saved last, with its layout.
  `--players 3` starts with 3 players instead (their saved layout, or the template).
- With nothing saved yet, it starts with `PLAYERS` from `settings.py` (5).
- Delete `data/seats.json` to forget every saved layout.

Cards keep being read while you edit, and each card's seat follows the zones as you move them.

**Hands.** Each seat's cards make one hand. Aces count 11 unless that busts,
then 1. Face-down cards are left out of the total and shown as `+1 hidden`.
Split hands are not separated yet.

**Hi-Lo count.** Every face card adds to the running count once per shoe:

| Cards | Value |
|---|---|
| 2, 3, 4, 5, 6 | +1 |
| 7, 8, 9 | 0 |
| 10, J, Q, K, A | -1 |

The true count is the running count divided by the decks not yet seen. With one
deck that is `(52 - cards seen) / 52`, so a running count of +4 with half the
deck left is a true count of +8. A positive true count means the remaining cards
are rich in tens and aces. Face-down cards are counted when they are turned up.

### Rules the tracker assumes

- **A card is only confirmed when it passes every check** (all in `settings.py`):

  | Check | Setting | Stops |
  |---|---|---|
  | seen in `CONFIRM_FRAMES` of the last `CONFIRM_WINDOW` frames | 15 of 20 | flickering detections adding up over time |
  | its most common name has `CONFIRM_AGREEMENT` of all reads | 80% | a card being counted under a disputed name |
  | the model's average confidence is at least `CONFIRM_CONFIDENCE` | 0.7 | weak, borderline detections |
  | the corner moved less than `STILL_DISTANCE` over those frames | 0.6% of the width | a card being counted while it is still being slid |

  With the defaults a real card is confirmed about 0.7 s after it settles.
- **One deck.** Two sightings of the same face are the same card. This merges a
  card's corner indices, stops a stray misread from adding a duplicate, and means
  a card is counted once per shoe even if it leaves and comes back.
- **Covered is not removed.** A card that disappears for `DROP_FRAMES` frames
  becomes hidden: it keeps its number, stays in its hand, and is not counted again
  when it reappears. A face card is recognised by its name anywhere on the table,
  a face-down card by reappearing near where it was.
- **The round ends when the table is cleared.** If no card is visible for
  `ROUND_END_FRAMES` frames (about 3 seconds), the round is over and numbering
  restarts. A hand briefly covering the whole table does not end it. `R` ends a
  round immediately.
- **Every face-down card is a separate card.**
- A card turned face up in place (a dealer's hole card) keeps its BACK label
  for a while, because the label is a vote over every frame it has been seen.

## Tuning the reader

All of these are in `settings.py`. Change one at a time and watch the effect.

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
| Small or distant cards are not read | raise `INPUT_SIZE` (e.g. 1280 to 1600) | the model sees more detail; costs speed |
| The video is choppy (low fps) | lower `INPUT_SIZE` (e.g. 1280 to 960) | less work per frame |

`CONFIRM_FRAMES`, `DROP_FRAMES`, and `ROUND_END_FRAMES` are counted in frames, so at 30 fps, 30
frames is one second. If the fps is much lower, scale them down to match.

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'cv2'` | another virtual environment is active | `deactivate`, then `source .venv/bin/activate` inside `Blackjack_Project`; check `which python` |
| Reader window is black | camera permission not granted | System Settings > Privacy & Security > Camera, allow Terminal or VS Code, restart it |
| Wrong camera opens | camera numbers change when an iPhone connects or disconnects | `python read.py --list-cameras` shows a snapshot from each number |
| The phone's view is plain grey | it points at a plain surface, the lens is blocked, or it is too close to focus | aim it at the table from 50 cm or more; `--list-cameras` marks such a camera as nearly blank |
| `Could not open video source` | the camera is in use or not connected | close other apps using it (FaceTime, Zoom, Photo Booth); wake the iPhone camera |
| Bottom bar says CAPTURE MODE | no model at `MODEL_PATH` | check `models/cards/weights/best.pt` exists, or train one |
| Cards on screen but none get a number | detections too weak, or the camera is too far | see [Tuning the reader](#tuning-the-reader); move the camera closer |
| Keys do nothing | the terminal has focus, not the window | click the reader window first |
| The window close button does not quit | OpenCV reopens the window | press `Q`, or Ctrl+C in the terminal |
| Training stops with an out-of-memory error | batch too large for the GPU | `BATCH_SIZE = 4` in `settings.py` |
| Training reports `device=cpu` and is very slow | wrong environment, or no GPU support | activate this project's `.venv`; on a Mac check `python -c "import torch; print(torch.backends.mps.is_available())"` |
| An export says `skip ... boxes are whole cards` | it is labelled around whole cards, not corners | relabel the corners in Roboflow and export again |
| Zones in odd places after starting | a layout saved earlier was loaded | press `E`, then right-click inside each zone to reset it, or delete `data/seats.json` |
| Dragging misses the handles after resizing the window | the window is scaled | make the window its original size, or grab slightly inside the handle |
| `yolo val` results end up in `runs/detect/` | relative `project=` path | use `project="$PWD/models"` as in [Validate a model on its own](training.md#5-validate-a-model-on-its-own) |

## Known limitations

- **One deck.** Two sightings of the same face are treated as one card, so a
  shoe with several decks would merge real duplicates, and its count would be wrong.
- **A misread at placement stays in the count.** If a card is first confirmed
  under the wrong name (a 9H read as 6H), that name is counted; the corrected name
  is then counted as a second card.
- **Splits** are shown as one hand per seat.
- **Face-down cards.** All `BACK` training examples are synthetic, drawn with a
  placeholder design. Photograph your deck's back (see
  [Using your own deck](training.md#using-your-own-deck-for-synthetic-images)) before relying on it.
- **A card flipped over in place** (a dealer's hole card) keeps its `BACK` label
  for a while, because the label is a vote over every frame it has been seen.
- **Same-colour suits** are the most common confusion (see [Results](results.md)).
- **No frame tiling.** A 4K frame is shrunk to `INPUT_SIZE` before detection, so
  very small indices at a large table may be lost.
- **The validation set includes synthetic images**, which makes the full score
  optimistic; report the real-photo score.
