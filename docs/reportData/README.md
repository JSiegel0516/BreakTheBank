# Report data

Tables behind the final report's Results section. Each is a CSV with one header row, ready for pandas and matplotlib.
The numbers come from each run's `models/<run>/results.csv` and `notes.txt`, and from [results.md](../results.md).

## accuracyByModel.csv

One row per trained model.

| Column | Meaning |
|---|---|
| `model` | model number, 1 to 5 |
| `run_name` | folder under `models/` |
| `model_size` | YOLO11 `nano` or `small` |
| `training_round` | data round: `prototype`, then 1 to 3. Models 3 and 4 share round 2's data |
| `real_photos` | labelled real photos, train and validation together. Blank for model 1 (not recorded) |
| `rotated_copies` | real train photos added again at 90 and 270 degrees |
| `synthetic_images` | computer-drawn images, train and validation together. Blank for model 1 (not recorded) |
| `train_images`, `val_images` | images in each split, synthetic and rotated included |
| `epochs_run`, `best_epoch` | epochs trained before stopping, and the epoch kept as `best.pt` |
| `train_hours` | wall-clock training time. Not comparable across `hardware` |
| `hardware`, `optimizer` | where it trained, and the optimizer Ultralytics chose |
| `real_precision`, `real_recall`, `real_map50`, `real_map50_95` | scores on the same 320 held-out real photos (3,449 corners) |
| `detect_ms_per_frame` | detection time per 1080p frame on the M4 Pro. Blank where not measured |
| `weights_mb` | size of `best.pt` |

The real-photo scores for models 1 to 4 were measured on 2026-11-24 with `yolo val` at `imgsz` 1024. Model 5's come from its `notes.txt`, measured the same way on the PC that trained it.

## trainingCurves.csv

One row per epoch for models 3, 4 and 5, which share one validation set: 633 images, synthetic included.
`val_*` columns are that set's scores, so they run higher than the real-photo scores above.
Losses are Ultralytics' box and class losses; lower is better. `elapsed_hours` is time since training started.
`is_best_epoch` marks the epoch saved as `best.pt` (highest `val_map50_95`).

## errorsModel3vs5.csv

Corner-level errors of model 3 and model 5 on the 320 real photos, from `benchmark.py --confusions`
at the live settings (`INPUT_SIZE` 1280, `CONFIDENCE` 0.6). One row per model, error type and corner angle.
`error_type` is `missed` (no box found) or `wrong_name` (found, wrong card). Summing `corners` per model and
error type gives the totals: model 3 missed 93 and misnamed 18, model 5 missed 63 and misnamed 17.
