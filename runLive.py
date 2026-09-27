"""The live table reader. This is the script you run at the table.

    python runLive.py                                  # camera and model from settings.py
    python runLive.py --source 1 --model models/cards/weights/best.pt
    python runLive.py --source data/raw/videos/round1.mp4

Pipeline per frame:
    camera -> YOLO detector -> zone lookup -> tracker -> blackjack totals -> overlay

Without a trained model it still runs in capture mode: it shows the zones over
the live feed and lets you press 's' to save frames for labelling.

Keys: q quit | s save frame to data/raw/frames | z reload zones.json
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

import settings
from cardVision.blackjack import table_summary
from cardVision.camera import Camera, parse_source
from cardVision.deck import BACK, display
from cardVision.tracker import CardTracker, Detection
from cardVision.zones import ZoneMap, example_zones


def load_zones(path: str) -> ZoneMap:
    if Path(path).exists():
        return ZoneMap.load(path)
    print(f"{path} not found, using example zones. Run scripts/1_calibrateZones.py to set real ones.")
    return example_zones()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=settings.CAMERA_SOURCE, help="camera index, video file, or stream URL")
    ap.add_argument("--model", default=settings.MODEL_PATH, help="trained .pt weights; capture mode if the file is missing")
    ap.add_argument("--zones", default=settings.ZONES_FILE)
    ap.add_argument("--imgsz", type=int, default=settings.INPUT_SIZE)
    ap.add_argument("--conf", type=float, default=settings.CONFIDENCE)
    ap.add_argument("--device", default=None, help="mps, cuda, or cpu (auto if omitted)")
    ap.add_argument("--width", type=int, default=settings.CAMERA_WIDTH)
    ap.add_argument("--height", type=int, default=settings.CAMERA_HEIGHT)
    ap.add_argument("--confirm", type=int, default=settings.CONFIRM_FRAMES, help="frames before a card counts")
    ap.add_argument("--drop", type=int, default=settings.DROP_FRAMES, help="missed frames before a card is removed")
    ap.add_argument("--match-dist", type=float, default=settings.MATCH_DISTANCE, help="fraction of frame width")
    ap.add_argument("--save-dir", default=settings.FRAMES_DIR)
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    model = None
    if args.model and Path(args.model).exists():
        from ultralytics import YOLO
        model = YOLO(args.model)
    else:
        print(f"no model at {args.model}: running in capture mode (press s to save frames)")

    zones = load_zones(args.zones)
    cam = Camera(parse_source(args.source), args.width, args.height)
    W, H = cam.size
    print(f"source {args.source} at {W}x{H}")
    tracker = CardTracker(lambda x, y: zones.assign(x, y, W, H),
                          confirm_frames=args.confirm, drop_frames=args.drop,
                          match_dist=args.match_dist * W)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    t_last, fps = time.time(), 0.0
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                if cam.is_file:
                    break
                time.sleep(0.01)
                continue
            H, W = frame.shape[:2]

            dets: list[Detection] = []
            if model is not None:
                res = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
                for box in res.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    dets.append(Detection(model.names[int(box.cls)], float(box.conf),
                                          (x1 + x2) / 2, (y1 + y2) / 2, (x1, y1, x2, y2)))
                for e in tracker.update(dets):
                    print(e)

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_last, 1e-6)
            t_last = now

            if not args.no_display:
                view = frame.copy()
                zones.draw(view)
                for d in dets:  # raw detections, thin grey
                    x1, y1, x2, y2 = map(int, d.box)
                    cv2.rectangle(view, (x1, y1), (x2, y2), (160, 160, 160), 1)
                for t in tracker.confirmed_tracks():  # confirmed, bold
                    col = (0, 255, 0) if t.label != BACK else (255, 200, 0)
                    cv2.circle(view, (int(t.cx), int(t.cy)), 8, col, 2)
                    cv2.putText(view, display(t.label), (int(t.cx) + 10, int(t.cy) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
                y = 30
                for name, desc in table_summary(tracker.state(), zones.names()).items():
                    cards = " ".join(display(c) for c in tracker.state().get(name, []))
                    cv2.putText(view, f"{name:8s} {desc:16s} {cards}", (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    y += 28
                cv2.putText(view, f"{fps:4.1f} fps", (W - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("blackjack table", view)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                elif k == ord("s"):
                    p = Path(args.save_dir) / f"frame_{int(time.time() * 1000)}.jpg"
                    cv2.imwrite(str(p), frame)
                    print("saved", p)
                elif k == ord("z"):
                    zones = load_zones(args.zones)
                    tracker.zone_fn = lambda x, y: zones.assign(x, y, W, H)
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
