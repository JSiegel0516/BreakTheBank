"""Find corner indices in a frame with the trained YOLO model."""
from __future__ import annotations

import numpy as np

from .tracker import Detection


class Detector:
    def __init__(self, weights: str, imgsz: int = 1280, conf: float = 0.6, device: str | None = None):
        from ultralytics import YOLO  # slow import, only paid when a model is used
        self.model = YOLO(weights)
        self.imgsz, self.conf, self.device = imgsz, conf, device

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        res = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)[0]
        out = []
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            out.append(Detection(self.model.names[int(box.cls)], float(box.conf),
                                 (x1 + x2) / 2, (y1 + y2) / 2, (x1, y1, x2, y2)))
        return out
