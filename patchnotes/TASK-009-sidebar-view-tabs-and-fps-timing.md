# Patch notes: Sidebar view tabs and FPS timing
Task TASK-009 · superseded by TASK-010 (commit d856ba9) · 2026-10-29

## For users
No new behavior. Everything this task asked for already ships with TASK-010.
The sidebar shows CARDS LEFT and GAME tabs with a V key hint.
Timing thresholds are set in seconds through one frames per second (FPS) setting.

## For developers
Changes nothing. The TASK-010 commit on branch orch/task-010 carries this task's scope.
See the TASK-010 patch notes for the files involved.
The orch/task-009 branch had no diff against its base and can be deleted.

## Checks
The full test suite (379 tests) and ruff pass on the TASK-010 commit.
