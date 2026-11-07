"""cardReader/detector.py with ultralytics faked: the predict arguments, the warm-up, and the device."""
import sys
from types import ModuleType, SimpleNamespace

import numpy as np

from cardReader.detector import Detector


class FakeYOLO:
    """Records every predict call and returns one 9H box. The predictor exists only after a predict."""

    def __init__(self, weights):
        self.weights, self.calls, self.names = weights, [], {0: "6H", 1: "9H"}

    def predict(self, frame, **kw):
        self.calls.append((frame.shape, kw))
        self.predictor = SimpleNamespace(device=SimpleNamespace(type="mps"))
        box = SimpleNamespace(xyxy=[np.array([10.0, 20.0, 30.0, 60.0])], cls=1, conf=0.8)
        return [SimpleNamespace(boxes=[box])]


def test_detector_warms_up_reports_its_device_and_merges_boxes_across_names(monkeypatch):
    ultralytics = ModuleType("ultralytics")
    ultralytics.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)
    detector = Detector("x.pt", imgsz=640, conf=0.5)
    assert detector.device_used == "mps"
    assert detector.model.calls[0][0] == (640, 640, 3)    # the warm-up on a black frame

    dets = detector(np.zeros((360, 640, 3), np.uint8))
    assert [(d.label, d.conf, d.cx, d.cy, d.box) for d in dets] == [("9H", 0.8, 20.0, 40.0, (10.0, 20.0, 30.0, 60.0))]
    for _, kw in detector.model.calls:
        assert kw["agnostic_nms"] is True
        assert (kw["imgsz"], kw["conf"], kw["device"], kw["iou"]) == (640, 0.5, None, 0.7)


def test_the_nms_overlap_is_passed_to_every_predict(monkeypatch):
    ultralytics = ModuleType("ultralytics")
    ultralytics.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)
    detector = Detector("x.pt", imgsz=640, conf=0.5, iou=0.5)
    detector(np.zeros((360, 640, 3), np.uint8))
    assert [kw["iou"] for _, kw in detector.model.calls] == [0.5, 0.5]
