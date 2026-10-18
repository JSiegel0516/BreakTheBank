"""synthetic.py: card art, geometry, and the images it composes."""
import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cardReader import synthetic
from cardReader.deck import BACK, CLASS_TO_ID, CLASSES

CARD_ART = Path(__file__).resolve().parents[1] / "data" / "raw" / "cardArt"


def test_placeholder_cards_have_the_card_size_and_an_alpha_channel():
    for label in ("AS", "10H", BACK):
        img = synthetic.draw_placeholder(label)
        assert img.size == (synthetic.CARD_W, synthetic.CARD_H) and img.mode == "RGBA"
    assert synthetic.draw_placeholder("KD").tobytes() != synthetic.draw_placeholder("KC").tobytes()


def test_the_projects_own_card_art_loads_with_its_index_rect():
    art, rect = synthetic.load_card_art(CARD_ART)
    assert set(art) == set(CLASSES) and all(len(v) == 1 for v in art.values())
    assert art["AS"][0].mode == "RGBA" and art[BACK][0].size == art["AS"][0].size  # placeholder back, same size
    x, y, w, h = rect
    assert 0 < x < 0.1 and 0 < y < 0.1 and 0.1 < w < 0.25 and 0.2 < h < 0.35  # the corner index, top left


def test_without_card_art_placeholders_and_their_rect_are_used(tmp_path):
    art, rect = synthetic.load_card_art(tmp_path)
    assert set(art) == set(CLASSES) and rect == synthetic.PLACEHOLDER_INDEX_RECT


def test_rotate_points_matches_pil_rotate_with_expand():
    cw, ch = 100, 200
    for angle in (0, 37, 90, 180, -120):
        rot = Image.new("RGBA", (cw, ch)).rotate(angle, expand=True)
        ow, oh = rot.size
        pts = synthetic.rotate_points([(cw / 2, ch / 2)] + synthetic.rect_corners(0, 0, cw, ch), angle, cw, ch, ow, oh)
        assert np.allclose(pts[0], (ow / 2, oh / 2))                      # the centre stays the centre
        xs, ys = zip(*pts[1:])
        assert np.allclose((min(xs), min(ys), max(xs), max(ys)), (0, 0, ow, oh), atol=1.0)  # corners span the output


def test_visible_fraction_counts_only_the_cards_own_pixels():
    canvas = np.zeros((100, 100), np.int32)
    canvas[:, :50] = 1
    canvas[:, 50:] = 2  # card 2 lies over the right half
    square = synthetic.rect_corners(25, 25, 50, 50)
    assert synthetic.visible_fraction(canvas, square, 1) == pytest.approx(0.5, abs=0.03)  # edge pixels are inclusive
    assert synthetic.visible_fraction(canvas, synthetic.rect_corners(0, 0, 40, 100), 1) == 1.0
    assert synthetic.visible_fraction(canvas, synthetic.rect_corners(200, 200, 10, 10), 1) == 0.0


def test_compose_makes_valid_corner_index_labels():
    W, H = 640, 480
    art, rect = synthetic.load_card_art(CARD_ART)
    rng = random.Random(3)
    seen_face = seen_back = False
    for _ in range(6):
        img, labels = synthetic.compose(rng, art, rect, [], W, H, max_cards=6, glare_p=0.5)
        assert img.size == (W, H) and img.mode == "RGB"
        for cls, x1, y1, x2, y2 in labels:
            assert 0 <= cls < len(CLASSES)
            assert 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H and x2 - x1 >= 6 and y2 - y1 >= 6
            if cls == CLASS_TO_ID[BACK]:
                seen_back = True
            else:
                seen_face = True
                assert x2 - x1 < 0.12 * W  # a corner index, not a whole card
    assert seen_face and seen_back


def test_generate_writes_images_and_yolo_label_files(tmp_path):
    for s in ("train", "val"):
        (tmp_path / "images" / s).mkdir(parents=True)
        (tmp_path / "labels" / s).mkdir(parents=True)
    synthetic.generate(tmp_path, 4, tmp_path / "noart", tmp_path / "nobg", 0.5, size=(320, 240), max_cards=3)
    images = sorted((p for s in ("train", "val") for p in (tmp_path / "images" / s).glob("*.jpg")), key=lambda p: p.name)
    assert [p.name for p in images] == [f"synth_{i:06d}.jpg" for i in range(4)]
    for img in images:
        lines = (tmp_path / "labels" / img.parent.name / (img.stem + ".txt")).read_text().split("\n")
        for line in filter(None, lines):
            cls, cx, cy, w, h = line.split()
            assert 0 <= int(cls) < len(CLASSES) and all(0 <= float(v) <= 1 for v in (cx, cy, w, h))
