# Patch notes: Bigger blackjack confetti
Task TASK-022 · branch orch/task-022 · 2026-11-01

## For users
Blackjack celebrations show more, larger pieces with a wider spread and last about 3.25 seconds.

## For developers
Updates particle count, size, launch speed, and duration in cardReader/sidebar/page.html.
Preserves existing triggers, reduced-motion handling, colours, physics, and the half-second fade.
Updates the duration wording in docs/reading.md.

## Checks
Independent JavaScript syntax validation passes.
Supplied validation reports the test suite passes.

## Known gaps
The read-only sandbox prevents an independent pytest rerun because pytest requires temporary files.
Browser appearance remains unverified.
