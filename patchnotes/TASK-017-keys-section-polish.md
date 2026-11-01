# Patch notes: Keys section polish
Task TASK-017 · branch orch/task-017 · 2026-11-01

## For users
Groups related keys into five wrapping rows and keeps the model dropdown first.
Adds a folding Keys title and a page-only K shortcut.
Remembers the folded state across reloads.
Enlarges section titles and adjusts Cards Left spacing.

## For developers
Adds key group metadata in cardReader/sidebar/snapshot.py without changing button actions.
Reorders the shared key list in cardReader/hud.py, also changing the window legend order.
Desktop groups previously wrapped; cardReader/sidebar/page.html now stacks key labels above actions to fit each group.
Documents grouping and folding in docs/reading.md.
Adds group coverage and page markup checks in tests/test_sidebar.py.

## Checks
Independent Ruff checks and two new tests pass.
Supplied validation reports the full test suite passing.
Recorded browser checks report single group rows at 1400 pixels and clickable controls without horizontal overflow at 360 pixels.
Recorded checks report no scrolling at 1400 by 1080 and 1920 by 960 pixels.

## Known gaps
The read-only sandbox prevents independent snapshot and full-suite testing because fixtures require temporary files.
Browser checks are not independently repeated.
