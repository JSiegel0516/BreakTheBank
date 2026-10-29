"""dataset.py: combining YOLO exports into data/dataset. No model involved."""
import pytest

from card_reader import dataset
from card_reader.deck import CLASS_TO_ID, CLASSES


def export(root, name, names, split_labels, width=0.04):
    """A tiny YOLO export. split_labels: {"train": {"img0": [(class_id, cx, cy), ...]}, ...}.
    names may be a list or a dict, like Roboflow writes them."""
    e = root / name
    for split, images in split_labels.items():
        (e / split / "images").mkdir(parents=True)
        (e / split / "labels").mkdir(parents=True)
        for stem, boxes in images.items():
            (e / split / "images" / f"{stem}.jpg").write_bytes(b"\xff\xd8" + stem.encode())
            (e / split / "labels" / f"{stem}.txt").write_text(
                "".join(f"{c} {cx} {cy} {width} {width}\n" for c, cx, cy in boxes))
    if isinstance(names, dict):
        body = "names:\n" + "".join(f"  {k}: {v}\n" for k, v in names.items())
    else:
        body = "names: " + str(names) + "\n"
    (e / "data.yaml").write_text(f"train: ../train/images\nval: ../valid/images\nnc: {len(names)}\n{body}")
    return e


def read_labels(out, split):
    return {p.name: p.read_text().split() for p in (out / "labels" / split).iterdir()}


def test_class_ids_are_remapped_by_name_ignoring_case(tmp_path):
    e = export(tmp_path / "raw", "ex", ["10h", "Ks", "AS"],
               {"train": {"a": [(0, 0.1, 0.1), (1, 0.5, 0.5), (2, 0.9, 0.9)]}})
    out = tmp_path / "out"
    for s in ("train", "val"):
        (out / "images" / s).mkdir(parents=True)
        (out / "labels" / s).mkdir(parents=True)
    dataset.import_export(e, out, "e0_", 0.0)
    lines = (out / "labels" / "train" / "e0_a.txt").read_text().splitlines()
    assert [int(line.split()[0]) for line in lines] == [CLASS_TO_ID["10H"], CLASS_TO_ID["KS"], CLASS_TO_ID["AS"]]
    assert lines[0].split()[1:] == ["0.1", "0.1", "0.04", "0.04"]  # coordinates untouched
    assert (out / "images" / "train" / "e0_a.jpg").read_bytes() == b"\xff\xd8a"


def test_dict_names_and_the_exports_own_split_are_kept(tmp_path):
    export(tmp_path / "raw", "ex", {0: "AS", 1: "2S"},
           {"train": {"t": [(0, 0.5, 0.5)]}, "valid": {"v": [(1, 0.5, 0.5)]}, "test": {"x": [(0, 0.5, 0.5)]}})
    out = tmp_path / "out"
    dataset.build(out, tmp_path / "raw", tmp_path / "noart", tmp_path / "nobg", 0, 0.15)
    assert set(read_labels(out, "train")) == {"e0_t.txt"}
    assert set(read_labels(out, "val")) == {"e0_v.txt", "e0_x.txt"}  # test counts as val


def test_an_export_with_only_a_train_split_gets_a_stable_hash_split(tmp_path):
    images = {f"img{i:02d}": [(0, 0.5, 0.5)] for i in range(40)}
    export(tmp_path / "raw", "ex", ["AS"], {"train": images})
    out = tmp_path / "out"
    dataset.build(out, tmp_path / "raw", tmp_path / "noart", tmp_path / "nobg", 0, 0.25)
    first = (set(read_labels(out, "train")), set(read_labels(out, "val")))
    assert 4 <= len(first[1]) <= 16 and len(first[0]) + len(first[1]) == 40
    dataset.build(out, tmp_path / "raw", tmp_path / "noart", tmp_path / "nobg", 0, 0.25)
    assert (set(read_labels(out, "train")), set(read_labels(out, "val"))) == first  # same split every time


def test_exports_that_are_not_corner_labels_are_skipped(tmp_path, capsys):
    raw = tmp_path / "raw"
    export(raw, "cardArtExport", ["cards", "A", "2", "c", "d"], {"train": {"a": [(0, 0.5, 0.5)]}})  # rank/suit names
    export(raw, "wholeCards", ["AS", "KS"], {"train": {"a": [(0, 0.5, 0.5), (1, 0.2, 0.2)]}}, width=0.2)
    export(raw, "good", ["AS", "KS"], {"train": {"a": [(0, 0.5, 0.5)]}})
    out = tmp_path / "out"
    dataset.build(out, raw, tmp_path / "noart", tmp_path / "nobg", 0, 0.0)
    printed = capsys.readouterr().out
    assert "skip cardArtExport: its classes are not card names" in printed
    assert "skip wholeCards: boxes are whole cards" in printed
    assert set(read_labels(out, "train")) == {"e1_a.txt"}  # only the good export, keeping its index


def test_images_without_usable_boxes_are_left_out(tmp_path, capsys):
    export(tmp_path / "raw", "ex", ["AS", "joker"],
           {"train": {"a": [(0, 0.5, 0.5)], "b": [(1, 0.5, 0.5)], "c": []}})
    out = tmp_path / "out"
    dataset.build(out, tmp_path / "raw", tmp_path / "noart", tmp_path / "nobg", 0, 0.0)
    assert set(read_labels(out, "train")) == {"e0_a.txt"}
    printed = capsys.readouterr().out
    assert "dropping classes not in the deck: ['JOKER']" in printed and "skipped (no boxes) 2" in printed


def test_build_writes_data_yaml_with_every_class_and_a_real_only_val_list(tmp_path):
    export(tmp_path / "raw", "ex", ["AS"], {"train": {"t": [(0, 0.5, 0.5)]}, "valid": {"v": [(0, 0.5, 0.5)]}})
    out = tmp_path / "out"
    dataset.build(out, tmp_path / "raw", tmp_path / "noart", tmp_path / "nobg", 0, 0.15)
    yaml_text = (out / "data.yaml").read_text()
    assert f"path: {out.resolve()}" in yaml_text and "train: images/train" in yaml_text
    assert all(f"  {i}: {c}\n" in yaml_text for i, c in enumerate(CLASSES)) and "52: BACK" in yaml_text
    assert (out / "realVal.txt").read_text().strip().endswith("e0_v.jpg")
    assert "val: realVal.txt" in (out / "realVal.yaml").read_text()


def test_build_adds_synthetic_images_and_keeps_them_out_of_real_val(tmp_path):
    export(tmp_path / "raw", "ex", ["AS"], {"train": {"t": [(0, 0.5, 0.5)]}, "valid": {"v": [(0, 0.5, 0.5)]}})
    out = tmp_path / "out"
    dataset.build(out, tmp_path / "raw", tmp_path / "noart", tmp_path / "nobg", 3, 0.5)  # placeholder card art
    synth = sorted(p.name for s in ("train", "val") for p in (out / "images" / s).glob("synth_*.jpg"))
    assert synth == ["synth_000000.jpg", "synth_000001.jpg", "synth_000002.jpg"]
    for s in ("train", "val"):
        for img in (out / "images" / s).glob("synth_*.jpg"):
            assert (out / "labels" / s / (img.stem + ".txt")).exists()
    assert "synth_" not in (out / "realVal.txt").read_text()


def test_build_refuses_an_empty_dataset(tmp_path):
    with pytest.raises(SystemExit, match="dataset is empty"):
        dataset.build(tmp_path / "out", tmp_path / "nothing", tmp_path / "noart", tmp_path / "nobg", 0, 0.15)
