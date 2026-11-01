# Patch notes: Bright-light preprocessing study
Task TASK-015 · branch orch/task-015 · 2026-11-01

## For users
Adds benchmark levels for clipped overexposure and hazy, low-contrast pictures.
Records results showing neither experimental filter meets the retention threshold.
Keeps existing preprocessing modes and automatic behavior unchanged.

## For developers
Extends benchmark.py with named gain and contrast-flattening levels, preserving numeric command-line gamma inputs.
Adds synthetic clipping, contrast, parsing, and output checks in tests/test_benchmark.py.
Records model details, accuracy comparisons, and rejected filters in docs/results.md.
Updates usage and mode guidance in docs/reading.md.
Updates the benchmark description in README.md.
Corrects overstated performance comments in settings.py to describe aggregate results and the haze exception.

## Checks
Independent Ruff checks pass.
Fourteen benchmark tests pass independently.
Supplied validation reports the full test suite passes.

## Known gaps
Read-only restrictions prevent two temporary-file tests from completing independently.
The full suite and model benchmark are not rerun during this review.
