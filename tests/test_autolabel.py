"""autolabel.py: filling in corners an export left out, with the model faked."""
import sys
import types

import cv2
import numpy as np
import pytest

from cardReader import autolabel


def test_iou():
    a = (0, 0, 10, 10)
    assert autolabel.iou(a, a) == 1.0
    assert autolabel.iou(a, (20, 20, 30, 30)) == 0.0
    assert autolabel.iou(a, (5, 0, 15, 10)) == pytest.approx(50 / 150)
    assert autolabel.iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


class FakeBox:
    def __init__(self, cls, xyxy, conf=0.9):
        self.cls, self.xyxy, self.conf = cls, np.array([xyxy], dtype=float), conf


def fake_yolo(predictions, calls):
    """A stand-in for ultralytics.YOLO whose predict() returns the given boxes for every image."""
    class Result:
        boxes = [FakeBox(*p) for p in predictions]

    class YOLO:
        names = {0: "KH", 1: "QS", 2: "2C"}

        def __init__(self, weights):
            calls.append(("load", weights))

        def predict(self, img, **kw):
            calls.append(("predict", kw.get("conf")))
            return [Result()]
    return types.SimpleNamespace(YOLO=YOLO)


def test_complete_corners_adds_only_corners_of_labelled_cards_scaled_to_the_exports_boxes(tmp_path, monkeypatch):
    export = tmp_path / "cards"
    (export / "train" / "images").mkdir(parents=True)
    (export / "train" / "labels").mkdir(parents=True)
    cv2.imwrite(str(export / "train" / "images" / "p.jpg"), np.full((200, 200, 3), 255, np.uint8))
    # the export's class order differs from the model's; KH is 5 here
    (export / "data.yaml").write_text("train: ../train/images\nnc: 6\nnames: ['a', 'b', 'c', 'd', 'e', 'kh']\n")
    # a person labelled one KH corner: 20x20 px box at (10,10)-(30,30), in YOLO fractions
    (export / "train" / "labels" / "p.txt").write_text("5 0.1 0.1 0.1 0.1\n")

    calls = []
    predictions = [
        (0, (12, 12, 32, 32)),        # KH over the labelled corner: already done, skipped
        (0, (150, 150, 190, 190)),    # KH, the missing corner, 40x40: added, scaled to 20x20
        (1, (100, 100, 140, 140)),    # QS: no QS labelled in this photo, skipped
        (2, (50, 50, 90, 90)),        # 2C: not a class of this export, skipped
    ]
    monkeypatch.setitem(sys.modules, "ultralytics", fake_yolo(predictions, calls))
    out = autolabel.complete_corners(export, tmp_path / "labeled", "weights.pt", preview=6)

    assert out == tmp_path / "labeled" / "cardsComplete"
    assert calls == [("load", "weights.pt"), ("predict", autolabel.MIN_CONF)]
    lines = (out / "train" / "labels" / "p.txt").read_text().splitlines()
    assert lines[0] == "5 0.1 0.1 0.1 0.1"                       # original kept
    cls, cx, cy, w, h = lines[1].split()
    assert cls == "5"                                            # the export's own id for KH
    assert (float(cx), float(cy)) == pytest.approx((0.85, 0.85))  # same centre as the prediction
    assert (float(w), float(h)) == pytest.approx((0.1, 0.1))      # 40 px scaled by 0.5 to the export's 20 px
    assert len(lines) == 2
    assert (out / "train" / "images" / "p.jpg").exists() and (out / "data.yaml").exists()
    summary = (out / "preview" / "summary.txt").read_text()
    assert "images 1, original boxes 1, added 1 (in 1 images)" in summary and "scaled by 0.50 x 0.50" in summary
    assert list((out / "preview").glob("sheet_*.jpg"))


def test_complete_corners_with_nothing_to_add_keeps_labels_unchanged(tmp_path, monkeypatch):
    export = tmp_path / "ex"
    (export / "train" / "images").mkdir(parents=True)
    (export / "train" / "labels").mkdir(parents=True)
    cv2.imwrite(str(export / "train" / "images" / "p.jpg"), np.zeros((100, 100, 3), np.uint8))
    (export / "data.yaml").write_text("train: ../train/images\nnc: 1\nnames: ['KH']\n")
    (export / "train" / "labels" / "p.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    monkeypatch.setitem(sys.modules, "ultralytics", fake_yolo([], []))
    out = autolabel.complete_corners(export, tmp_path / "labeled", "w.pt")
    assert (out / "train" / "labels" / "p.txt").read_text() == "0 0.5 0.5 0.2 0.2\n"
    assert "added 0" in (out / "preview" / "summary.txt").read_text()
