# data/

What belongs here, where the training data comes from, and how to check you
have the same data the live model was trained on.

https://drive.google.com/file/d/18ACxuf26ypKerWfg-vODjaORBfuLIBpC/view?usp=share_link 

## Layout

| Folder or file | In git? | What it is |
|---|---|---|
| `raw/labeled/<export>/` | no | YOLO-format exports of labelled table photos, one folder each. Training reads every one. |
| `raw/unused/<export>/` | no | exports kept for reference but never trained on |
| `raw/cardArt/` | yes | one PNG per card face, cut from photos of this project's deck, for synthetic images |
| `raw/backgrounds/` | no | photos of the empty table for synthetic images (none yet: plain felt is drawn) |
| `raw/frames/` | no | frames saved from `read.py` with `S`, waiting to be labelled |
| `dataset/` | no | the training set `train.py` builds from `raw/`. Safe to delete. |
| `manifest.json` | yes | what `raw/labeled` and `raw/unused` must hold, with a fingerprint per export |
| `seats.json` | no | zones edited with the mouse in `read.py`, one layout per player count |

## Getting the training data

The labelled photos are about 450 MB and are not in git. They are in the
team's shared folder:

    SHARED FOLDER: <link to be added>

Unpack it so that the exports sit directly under `data/raw/`, for example
`data/raw/labeled/blackjackCorners/data.yaml`. Then check them:

```bash
python train.py --check-data
```

Every export is hashed file by file and compared with `manifest.json`. `ok`
on every line means you have exactly what the live model was trained on. A
`DIFFERENT` or `MISSING` line names the export and why.

## The exports

| Folder | Roboflow project | Licence | Images | What it is | Used by |
|---|---|---|---|---|---|
| `labeled/playingCards` | `matthew-tu/playing-cards-bef7p-p2ou0` | CC BY 4.0 | 353 | real table photos, one background, cards spread out | models 2, 3, 4 |
| `labeled/blackjackCorners` | `matthew-tu/blackjack-2-jcplc` | CC BY 4.0 | 650 | real table photos, many backgrounds and lighting, overlapping hands; relabelled to corner indices | models 2, 3, 4 |
| `labeled/cardsComplete` | `matthew-tu/cards-noyri-8umiy` | Public Domain | 852 | composited photos. The export labelled only some corners; model 2 added the rest with `train.py --complete-corners` (its `preview/` sheets show which) | models 3, 4 |
| `unused/cards` | `matthew-tu/cards-noyri-8umiy` | Public Domain | 852 | the original of `cardsComplete`, before the corners were completed. Never trained on. | |

Model 1 trained on an earlier folder (`data/merged`) that no longer exists; its
`notes.txt` describes it. Each model's folder under `models/` says in its
`notes.txt` which exports it used and with what result; see also
[docs/results.md](../docs/results.md).

## Adding data

1. Save frames from your own table with `S` in `read.py`, label the corner
   indices in Roboflow, and export as YOLOv11 (no resize, no augmentation) into
   `data/raw/labeled/<newExport>/`. See [docs/training.md](../docs/training.md).
2. Train: `python train.py --rebuild --name model_N_<date>_<tag>`.
3. If the result is worth keeping: `python train.py --manifest`, commit
   `data/manifest.json` together with the run's record under `models/`, and
   upload the new export to the shared folder so the next person's
   `--check-data` passes.

`manifest.json` is written by code. Do not edit it by hand.
