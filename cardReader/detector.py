"""Find corner indices in a frame with the trained YOLO model."""
from __future__ import annotations

import numpy as np

from .cards import Detection


class Detector:
    """The trained YOLO model, called once per frame.

    weights: path to a .pt file. imgsz: inference size in pixels; each frame is
    scaled, keeping its aspect ratio, so its longer side fits imgsz, then padded.
    conf: boxes scoring below this are dropped; read.py passes SUSTAIN_CONFIDENCE, below
    CONFIDENCE, and the tracker decides which boxes may start a track. device: mps, cuda, or cpu; None
    lets ultralytics choose.

    Overlapping boxes are merged whatever their names (agnostic NMS, non-maximum
    suppression), so one corner index gets one name: the most confident. iou is the
    share of their area two boxes must share to be merged (settings.NMS_IOU); lower
    merges more stacked boxes on one index.
    Construction runs the model once on a black frame, so the first live frame
    is not slow, and sets device_used to the device it ran on: mps, cuda, or cpu."""

    def __init__(self, weights: str, imgsz: int = 1280, conf: float = 0.6, device: str | None = None,
                 iou: float = 0.7):
        from ultralytics import YOLO  # slow import, only paid when a model is used
        self.model = YOLO(weights)
        self.imgsz, self.conf, self.device, self.iou = imgsz, conf, device, iou
        self._predict(np.zeros((imgsz, imgsz, 3), np.uint8))
        self.device_used: str = self.model.predictor.device.type

    def _predict(self, frame: np.ndarray):
        return self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou, device=self.device,
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
