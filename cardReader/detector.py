"""Find corner indices in a frame with the trained YOLO model."""
from __future__ import annotations

import numpy as np

from .cards import Detection


class Detector:
    """The trained YOLO model, called once per frame.

    weights: path to a .pt file. imgsz: inference size in pixels; each frame is
    scaled, keeping its aspect ratio, so its longer side fits imgsz, then padded.
    conf: boxes scoring below this are dropped. device: mps, cuda, or cpu; None
    lets ultralytics choose.

    Overlapping boxes are merged whatever their names (agnostic NMS, non-maximum
    suppression), so one corner index gets one name: the most confident.
    Construction runs the model once on a black frame, so the first live frame
    is not slow, and sets device_used to the device it ran on: mps, cuda, or cpu."""

    def __init__(self, weights: str, imgsz: int = 1280, conf: float = 0.6, device: str | None = None):
        from ultralytics import YOLO  # slow import, only paid when a model is used
        self.model = YOLO(weights)
        self.imgsz, self.conf, self.device = imgsz, conf, device
        self._predict(np.zeros((imgsz, imgsz, 3), np.uint8))
        self.device_used: str = self.model.predictor.device.type

    def _predict(self, frame: np.ndarray):
        return self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, device=self.device,
                                  agnostic_nms=True, verbose=False)[0]

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        """Every box in a BGR frame, with its label, confidence, centre, and corners in frame pixels."""
        res = self._predict(frame)
        out = []
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            out.append(Detection(self.model.names[int(box.cls)], float(box.conf),
                                 (x1 + x2) / 2, (y1 + y2) / 2, (x1, y1, x2, y2)))
        return out
