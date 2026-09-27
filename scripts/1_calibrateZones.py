"""Step 1: click out the table zones on a frozen frame and save them to data/zones.json.

    python scripts/1_calibrateZones.py                # from the camera in settings.py
    python scripts/1_calibrateZones.py --image data/raw/frames/table.jpg

Keys:  left-click add point | u undo point | n next zone | b previous zone
       r reset current zone | s save | q quit

Draw each polygon clockwise around the area where that seat's cards land.
Re-run any time the camera moves.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import settings  # noqa: E402
from cardVision.camera import Camera, parse_source  # noqa: E402
from cardVision.zones import DEFAULT_ZONE_NAMES, Zone, ZoneMap  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=settings.CAMERA_SOURCE, help="camera index or video path")
    ap.add_argument("--image", default=None, help="still image to calibrate on instead of the camera")
    ap.add_argument("--out", default=settings.ZONES_FILE)
    ap.add_argument("--names", default=",".join(DEFAULT_ZONE_NAMES))
    args = ap.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
    else:
        with Camera(parse_source(args.source)) as cam:
            ok, frame = cam.read()
        if not ok:
            sys.exit("no frame from camera")
    h, w = frame.shape[:2]

    names = args.names.split(",")
    zones = [Zone(n) for n in names]
    if Path(args.out).exists():
        existing = {z.name: z for z in ZoneMap.load(args.out).zones}
        zones = [existing.get(n, Zone(n)) for n in names]
    cur = 0

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            zones[cur].polygon.append((x / w, y / h))

    win = "calibrate zones"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        view = frame.copy()
        ZoneMap(zones).draw(view)
        cv2.putText(view, f"editing: {zones[cur].name}  (n next, u undo, s save, q quit)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(win, view)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("u") and zones[cur].polygon:
            zones[cur].polygon.pop()
        elif k == ord("r"):
            zones[cur].polygon.clear()
        elif k == ord("n"):
            cur = (cur + 1) % len(zones)
        elif k == ord("b"):
            cur = (cur - 1) % len(zones)
        elif k == ord("s"):
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            ZoneMap(zones).save(args.out)
            print("saved", args.out)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
