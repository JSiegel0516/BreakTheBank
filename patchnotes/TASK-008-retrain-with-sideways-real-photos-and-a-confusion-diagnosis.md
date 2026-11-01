# Patch notes: Sideways training photos and model 5
Task TASK-008 · branch orch/task-008 · 2026-11-01

## For users
Adds a confusion report listing wrong card names and error rates by corner shape and size.
Adds sideways copies of real training photos while preserving validation photos.
Promotes model 5 with 17 wrong names and 63 missed corners, versus model 3’s 18 and 93.
Documents the Google Colab training round trip and the optimizer difference between models.

## For developers
Adds matching-based confusion reporting to `benchmark.py`.
Adds exact box rotation and training-only image copies to `cardReader/dataset.py`.
Adds validated rotation settings to `settings.py`.
Updates `cardReader/provenance.py` to recognize Windows base-weight paths.
Records both models’ real-photo scores in `docs/results.md`.
Adds tests covering rotation, split isolation, confusion reporting, settings validation, and promotion provenance.

## Checks
Independent Ruff checks pass.
Promoted weights match the run and recorded checksum.
Live and run record files match.
Supplied validation reports pytest passing.

## Known gaps
This read-only review does not rerun tests or model scoring.
The existing unnamed training walkthrough can overwrite live weights.
