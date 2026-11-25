# Patch notes: Switch cameras while the reader runs

## For users
Shift+C switches to the next camera number that opens, from 0 to 5.
The page adds a Camera dropdown and a next camera button.
A switch starts a new round and keeps the count.
A camera that does not open leaves the current one running.
The bottom bar names the source, such as "cam 1".
The window lists C and Shift+C as one key entry.
A recorded video cannot switch cameras.

## For developers
cardReader/camera.py adds Camera.switch(), which opens the new device before releasing the old one.
It also adds Camera.label and MAX_CAMERA_INDEX.
read.py adds next_camera(), switch_camera(), and camera_snapshot().
press_key() drops its source parameter and reads cam.source instead.
page_keys() takes the camera and handles the "switch_camera" action.
A new frame size reloads the seat zones and rescales the tracker distances.
cardReader/reader.py adds Reader.new_round(), which the R key now uses.
build_snapshot() requires a camera argument.
status_text() and Hud.draw() accept a camera label.
A separate Shift+C entry pushed the legend off a 720p window, so C and Shift+C now share one entry.

## Checks
Ruff reports no issues.
The full test suite passes: 861 tests.
New tests cover switching, failed opens, wrapping, resizing, page actions, and video files.
The page script passes a syntax check.

## Known gaps
Real cameras are untested.
The page script has not run in a browser.