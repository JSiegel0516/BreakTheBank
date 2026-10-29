"""benchmark.py: scoring and the level x mode loop, with the detector faked."""
import cv2
import numpy as np
import pytest

import settings
from benchmark import Label, Tally, relight, run, score, table, val_images
from cardReader.cards import Detection
from cardReader.preprocess import PreprocessConfig

CONFIG = PreprocessConfig.from_settings(settings)


def det(label, cx, cy):
    return Detection(label, 0.9, cx, cy, (cx - 10, cy - 15, cx + 10, cy + 15))


def test_score_counts_found_right_wrong_name_missed_and_extra():
    labels = [Label("KH", 100, 100, 20, 30), Label("QS", 300, 100, 20, 30), Label("2C", 500, 500, 20, 30)]
    dets = [det("KH", 104, 97),        # found, right name
            det("QD", 298, 105),       # found, wrong name
            det("9D", 50, 400)]        # nothing labelled there: extra
    # 2C has no box near it: missed
    assert score(labels, dets) == (2, 1, 1)


def test_one_box_counts_for_one_labelled_corner_only():
    labels = [Label("KH", 100, 100, 20, 30), Label("KH", 112, 100, 20, 30)]
    assert score(labels, [det("KH", 106, 100)]) == (1, 1, 0)
    assert score([], [det("KH", 106, 100)]) == (0, 0, 1)
    assert score(labels, []) == (0, 0, 0)


def test_a_box_too_far_from_the_corner_is_not_a_match():
    assert score([Label("KH", 100, 100, 20, 30)], [det("KH", 100, 120)]) == (0, 0, 1)


def test_relight_darkens_above_1_and_brightens_below():
    grey = np.full((4, 4, 3), 128, np.uint8)
    assert relight(grey, 1.0) is grey
    assert relight(grey, 2.5).max() < 128 < relight(grey, 0.6).min()


def test_run_scores_every_level_and_mode_with_a_faked_detector():
    image = np.full((200, 300, 3), 120, np.uint8)
    labels = [Label("KH", 100, 100, 20, 30)]

    def detector(frame):   # reads the corner only when the frame is bright enough, and always sees a ghost
        found = [det("KH", 100, 100)] if frame.mean() >= 80 else []
        return found + [det("BACK", 250, 150)]

    tallies = run([(image, labels)] * 2, detector, levels=(2.5, 1.0), modes=("off", "gamma"), config=CONFIG)
    assert list(tallies) == [(2.5, "off"), (2.5, "gamma"), (1.0, "off"), (1.0, "gamma")]
    dark_off, dark_gamma = tallies[2.5, "off"], tallies[2.5, "gamma"]
    assert (dark_off.images, dark_off.labelled, dark_off.found, dark_off.right, dark_off.extra) == (2, 2, 0, 0, 2)
    assert (dark_gamma.found, dark_gamma.right) == (2, 2)          # brightening brings the corner back
    assert tallies[1.0, "off"].found == 2
    assert all(t.detect_ms >= 0 and t.preprocess_ms >= 0 for t in tallies.values())


def test_table_has_a_row_per_level_and_mode_with_shares():
    tallies = {(1.8, "off"): Tally(images=2, labelled=40, found=30, right=27, extra=3,
                                   preprocess_total=0.0, detect_total=40.0),
               (1.8, "clahe"): Tally(images=2, labelled=40, found=36, right=35, extra=1,
                                     preprocess_total=4.0, detect_total=42.0)}
    lines = table(tallies).splitlines()
    assert len(lines) == 3
    assert lines[1].split() == ["1.8", "off", "40", "30", "75.0%", "27", "67.5%", "3", "0.0", "20.0"]
    assert lines[2].split() == ["1.8", "clahe", "40", "36", "90.0%", "35", "87.5%", "1", "2.0", "21.0"]


def test_val_images_reads_a_yolo_list_and_its_labels_in_pixels(tmp_path):
    (tmp_path / "images" / "val").mkdir(parents=True)
    (tmp_path / "labels" / "val").mkdir(parents=True)
    for name in ("a", "b", "c"):
        cv2.imwrite(str(tmp_path / "images" / "val" / f"{name}.jpg"), np.zeros((100, 200, 3), np.uint8))
    (tmp_path / "labels" / "val" / "a.txt").write_text("1 0.5 0.25 0.1 0.2\n")
    listing = tmp_path / "realVal.txt"
    listing.write_text("".join(f"{tmp_path / 'images' / 'val' / n}.jpg\n" for n in ("a", "b", "c")))
    data = tmp_path / "realVal.yaml"
    data.write_text(f"path: {tmp_path}\ntrain: images/train\nval: realVal.txt\nnames:\n  0: AS\n  1: 2S\n")

    samples = list(val_images(data, limit=2))
    assert len(samples) == 2
    image, labels = samples[0]
    assert image.shape == (100, 200, 3)
    assert labels == [Label("2S", 100, 25, 20, 20)]
    assert samples[1][1] == []                                       # an image with no label file has no corners


def test_val_images_says_where_to_get_the_dataset_when_it_is_missing(tmp_path):
    with pytest.raises(SystemExit, match="train.py --rebuild"):
        list(val_images(tmp_path / "realVal.yaml"))
