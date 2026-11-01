"""benchmark.py: scoring, the level x mode loop, and the confusion listing, with the detector faked."""
from pathlib import Path

import cv2
import numpy as np
import pytest

import settings
from benchmark import (
    Corner,
    Label,
    Tally,
    aspect_bucket,
    confusion_summary,
    default_confusions_csv,
    rank_swaps,
    read_corners,
    relight,
    run,
    score,
    table,
    val_images,
    val_samples,
    wrong_name_rows,
)
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
    assert [path.name for path, _, _ in val_samples(data)] == ["a.jpg", "b.jpg", "c.jpg"]


def test_val_images_says_where_to_get_the_dataset_when_it_is_missing(tmp_path):
    with pytest.raises(SystemExit, match="train.py --rebuild"):
        list(val_images(tmp_path / "realVal.yaml"))


def test_aspect_bucket_splits_upright_diagonal_and_sideways():
    assert aspect_bucket(Label("6H", 0, 0, 20, 30)) == "upright"       # 0.67
    assert aspect_bucket(Label("6H", 0, 0, 24, 30)) == "diagonal"      # 0.8 is the edge, still diagonal
    assert aspect_bucket(Label("6H", 0, 0, 30, 24)) == "diagonal"      # 1.25 likewise
    assert aspect_bucket(Label("6H", 0, 0, 30, 20)) == "sideways"      # 1.5


def test_read_corners_records_a_wrong_name_a_missed_corner_and_ignores_extras():
    image = np.zeros((200, 600, 3), np.uint8)
    labels = [Label("6H", 100, 100, 20, 30), Label("9S", 300, 100, 30, 20), Label("KD", 500, 100, 20, 30)]

    def detector(frame):
        assert frame is image                                  # the photo goes in unchanged
        return [Detection("6H", 0.95, 102, 98), Detection("6S", 0.7, 301, 101), Detection("BACK", 0.8, 50, 180)]

    corners = read_corners([("photos/e1_a.jpg", image, labels)], detector)
    assert [(c.label.name, c.predicted, c.conf) for c in corners] == [
        ("6H", "6H", 0.95), ("9S", "6S", 0.7), ("KD", None, None)]
    assert all(c.image == "photos/e1_a.jpg" for c in corners)
    assert wrong_name_rows(corners) == [["e1_a.jpg", "e1", "9S", "6S", "0.700", "30.0", "20.0", "1.50"]]


def corner(name, predicted, w=20.0, h=30.0):
    return Corner("e0_x.jpg", Label(name, 0, 0, w, h), predicted, None if predicted is None else 0.9)


def test_rank_swaps_count_each_pair_both_ways_and_skip_suit_only_mistakes():
    corners = [corner("6H", "9H"), corner("9S", "6C"), corner("7D", "9D"), corner("6H", "6S"),
               corner("6H", "6H"), corner("QH", None)]
    assert rank_swaps(corners) == [("6/9", 2), ("7/9", 1)]


def test_summary_gives_totals_swaps_and_wrong_name_rates_by_aspect_and_width():
    corners = [corner("6H", "9H", 20, 30), corner("6H", "6H", 20, 30),      # upright, one wrong of two found
               corner("9S", "9S", 30, 20), corner("9S", None, 30, 20),      # sideways, one found, one missed
               corner("KD", "KD", 40, 40), corner("KD", "KD", 60, 60)]      # diagonal
    text = confusion_summary(corners)
    assert "6 labelled, 5 found, 1 wrong names, 1 missed" in text
    assert "6/9  1" in text
    lines = {line.split()[0]: line.split()[1:] for line in text.splitlines() if line.strip()}
    assert lines["upright"] == ["2", "2", "1", "50.0%", "0"]               # labelled found wrong wrong% missed
    assert lines["sideways"] == ["2", "1", "0", "0.0%", "1"]
    assert lines["diagonal"] == ["2", "2", "0", "0.0%", "0"]
    assert lines["narrow"][0] == "20-20"                                    # widths 20 20 | 30 30 | 40 60
    assert lines["narrow"][1:] == ["2", "2", "1", "50.0%", "0"]
    assert lines["wide"][0] == "40-60"


def test_confusions_csv_defaults_to_the_models_run_folder():
    assert default_confusions_csv("models/cards/weights/best.pt") == Path("models/cards/confusions.csv")
    assert default_confusions_csv("yolo11n.pt") == Path("confusions.csv")
