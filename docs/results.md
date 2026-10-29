# Results

Part of [BreakTheBank](../README.md). How these numbers are produced is in [training.md](training.md). Each model's folder under `models/` also has a `notes.txt`.

## Results: first full training run (model 2)

Trained on 2026-10-13 with the settings in
[Training settings used](training.md#training-settings-used-for-reports): 100 epochs, 4.5
hours on an Apple M4 Pro. Early stopping never triggered; the validation score
kept improving to the end. The model is `models/cards/weights/best.pt`.

| Validation set | Images | Boxes | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| Full (real + synthetic) | 514 | 6,309 | 0.979 | 0.977 | 0.989 | 0.758 |
| **Real photos only** | 201 | 2,728 | **0.974** | **0.977** | **0.984** | **0.559** |

The real-photo row is the honest score. mAP50-95 is lower there because it
rewards pixel-exact boxes, and hand-drawn labels are looser than synthetic ones;
the card names themselves are right almost every time.

What happened to the 2,728 labelled corner indices in the real validation photos:

| Outcome | Count | Share |
|---|---|---|
| Found and named correctly | 2,676 | 98.1% |
| Found but named wrongly | 35 | 1.3% |
| Missed | 17 | 0.6% |

The model also drew 84 boxes where no corner was labelled. Some are real
corners that were missed during labelling rather than true mistakes.

Most wrong names keep the rank and swap the suit for one of the same colour:
10H read as 10D (4 times), QS as QC (3), and single cases of QD/QH, JS/JC, JD/JH.
The rest are similar-looking ranks: 9H as 6H (a card seen upside down), 7D as 9D,
3D as 2D. The weakest cards on real photos are KD, 10D, and 3H (mAP50 about 0.93);
more labelled photos of red cards would help most.

Speed: about 1.2 ms of model time per 1024-pixel image on the M4 Pro GPU, about
7 ms including pre- and post-processing.

To reproduce the real-photo numbers, see [Validate a model on its own](training.md#5-validate-a-model-on-its-own).
The charts are in `models/valReal/` after running it.

## Results: second training run (live model)

Model 3 (`models/model_3_2026-10-14_n150`, now `models/cards`): YOLO11n trained
on 1,854 real photos (the two earlier exports plus 852 composited Roboflow
photos whose missing corners were filled in with `--complete-corners`) and
2,000 synthetic images. Early stopping ended it at epoch 110 of 150, best epoch
90, after 6.1 hours.

Both models scored on the same real-photo validation set (320 images):

| Validation subset | Images | Model 2 (first run) | Model 3 (second run) |
|---|---|---|---|
| All real photos | 320 | mAP50 0.969, mAP50-95 0.501, 95.3% corners right | **mAP50 0.981, mAP50-95 0.626, 98.4% right** |
| Table photos (same 201 as the first run) | 201 | mAP50 0.984, mAP50-95 0.559, 98.1% right | **mAP50 0.986, mAP50-95 0.572, 98.5% right** |
| Composited photos (new style) | 119 | mAP50 0.845, mAP50-95 0.311, 84.7% right | **mAP50 0.975, mAP50-95 0.842, 98.1% right** |

On the table photos the gain is small but consistent (29 wrong names instead of
35, 11 missed corners instead of 17). On photos unlike its training data the new
model is far better, which suggests it copes much better with unfamiliar
backgrounds and lighting. About a third of the composited corner labels were
drawn by model 2, so model 3's mAP50-95 on that subset partly measures agreement
with those boxes. Remaining confusions are mostly same-rank, same-colour suits
(10H read as 10D) and similar faces (QH as JH). Each model's `notes.txt` has the details.

## Results: does a bigger model help?

Model 4 (`models/model_4_2026-10-15_s150`) is YOLO11s, 3.6 times the weights of
YOLO11n, trained on exactly the same images as model 3 (early-stopped at epoch
102, best epoch 82, 9.6 hours). Both scored on the same 320 real photos:

| | Model 3 (YOLO11n, live) | Model 4 (YOLO11s) |
|---|---|---|
| Corners read correctly | 98.4% | 98.4% |
| Wrong card names (of 3,449) | 37 | 31 |
| Missed corners | 17 | 23 |
| mAP50 / mAP50-95 | 0.981 / 0.626 | 0.981 / 0.637 |
| Table photos only (201) | 98.5% right, 29 wrong | 98.6% right, 25 wrong |
| Detection time per 1080p frame (M4 Pro, `INPUT_SIZE` 1280) | 8.0 ms, about 124 fps | 15.8 ms, about 63 fps |
| Model file | 5.6 MB | 19.3 MB |

The two are tied within noise: a handful fewer wrong names and a handful more
missed corners out of 3,449, at twice the time per frame. Both models still make
the same top mistake, 10H read as 10D. A bigger model did not fix it, so the
limit is the training data rather than model size: more labelled photos of red
cards would help more than a larger network. Model 3 stays live because it is
as accurate and twice as fast, which leaves headroom for a 4K camera or a higher
`INPUT_SIZE`.

## Preprocessing for dim and bright light

Does adjusting a frame's contrast before detection help the live model when
the light is poor? `benchmark.py` relit every real validation photo (320
photos, 3,449 labelled corner indices) to five brightness levels and ran
each preprocess mode before the live model (model 3, `INPUT_SIZE` 1280,
`CONFIDENCE` 0.6). A level is a gamma on the photo: 3.5 is very dark, 2.5 much
darker, 1.8 darker, 1 unchanged, 0.6 brighter. A corner is found when a box's
centre lies within half the corner's longer side of the labelled centre, and
right when that box also has the corner's name. Run on 2026-10-29 with the
settings shipped (`auto` thresholds 110 and 190):

```
python benchmark.py
```

| Level | Mode | Found | Right | Extra boxes | Preprocess ms |
|---|---|---|---|---|---|
| 3.5 (very dark) | off | 83.3% | 82.5% | 44 | 0.0 |
| | gamma | **94.8%** | **94.2%** | 60 | 0.2 |
| | clahe | 70.1% | 68.7% | 28 | 5.9 |
| | auto | 94.0% | 93.4% | 59 | 0.4 |
| 2.5 (much darker) | off | 92.3% | 91.6% | 57 | 0.0 |
| | gamma | **96.6%** | **96.1%** | 66 | 0.2 |
| | clahe | 81.9% | 80.9% | 38 | 5.5 |
| | auto | 96.4% | 95.9% | 65 | 0.4 |
| 1.8 (darker) | off | 96.0% | 95.4% | 63 | 0.0 |
| | gamma | **97.3%** | **96.7%** | 64 | 0.2 |
| | clahe | 89.7% | 88.7% | 44 | 5.4 |
| | auto | 97.2% | 96.6% | 64 | 0.3 |
| 1 (unchanged) | off | 97.3% | 96.7% | 67 | 0.0 |
| | gamma | 97.3% | 96.8% | 62 | 0.2 |
| | clahe | 94.5% | 93.8% | 55 | 5.3 |
| | auto | **97.5%** | **96.9%** | 69 | 0.3 |
| 0.6 (brighter) | off | **97.3%** | **96.8%** | 62 | 0.0 |
| | gamma | 96.1% | 95.5% | 60 | 0.2 |
| | clahe | 95.6% | 94.9% | 56 | 5.3 |
| | auto | 97.2% | 96.5% | 62 | 0.2 |

The detector took 54 ms per photo in every row. That run used the CPU (the GPU
was not available to it), so the preprocess times are also inflated by the
model competing for the same cores. Measured alone at 1920×1080, `gamma` takes
about 0.1 ms, `clahe` about 2 ms, and `auto` 0.05 to 0.2 ms.

What it shows:

- **Brightening helps a dim picture a lot.** At the darkest level `off` misses
  one corner in six; `gamma` and `auto` get back to about 94%. The gain shrinks
  as the light improves and is gone on the unchanged photos.
- **Brightening a picture that is already bright costs a little.** At level 0.6
  `gamma` gets 1.3 points fewer names right than `off`. `auto` measures the brightness first,
  so it stays within 0.3 points of `off` there.
- **CLAHE hurts at every level**: 1.9 to 13.8 points fewer names right than
  `off`. Likely cause: stretching local contrast amplifies noise and changes how
  the corner indices look, in a way the training photos never showed. It is
  kept as a mode only to compare by eye.
- **Darkening does not help.** `auto` darkening washed-out frames earlier
  (`PREPROCESS_BRIGHT_ABOVE` 160 instead of 190) changed the brighter level by 2
  corners out of 3,449. The model copes with bright, unclipped pictures as they are.

**Recommendation: `auto`, now the default (`PREPROCESS = "auto"`).** It is the
best or within 0.8 points of the best (names right) at every level, costs well under a
millisecond, and never needs switching by hand. `PREPROCESS_DARK_BELOW` was
raised from 70 to 110 on this benchmark: at 110, `auto` found 18 more corners
at the darkest level and 3 more on the unchanged photos than at 70, and 3
fewer at the brighter level. In a room that is always dim, `gamma` is slightly
better still (`--preprocess gamma`).

None of this helps with glare: a reflection that saturates the sensor leaves
pure white pixels with no index in them (see
[Dim or washed-out tables](reading.md#dim-or-washed-out-tables)).
