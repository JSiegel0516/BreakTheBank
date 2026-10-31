import hashlib
import json

from cardReader.provenance import check_manifest, describe_export, live_model, promote, write_manifest


def fake_export(raw, group, name, labels=("0 0.5 0.5 0.05 0.05",), roboflow=True):
    e = raw / group / name
    (e / "train" / "images").mkdir(parents=True)
    (e / "train" / "labels").mkdir(parents=True)
    for i in range(2):
        (e / "train" / "images" / f"img{i}.jpg").write_bytes(b"\xff\xd8 fake jpeg %d" % i)
        (e / "train" / "labels" / f"img{i}.txt").write_text("\n".join(labels) + "\n")
    rf = "roboflow:\n  workspace: w\n  project: p\n  url: https://x/p\n  license: CC BY 4.0\n" if roboflow else ""
    (e / "data.yaml").write_text(f"train: ../train/images\nval: ../valid/images\nnc: 53\nnames: ['AS']\n{rf}")
    (e / "README.roboflow.txt").write_text(
        "\nThis dataset was exported via roboflow.com on October 13, 2026 at 7:24 PM GMT\n")
    return e


def test_manifest_records_source_counts_and_fingerprint(tmp_path):
    raw = tmp_path / "raw"
    fake_export(raw, "labeled", "tableA", labels=("0 0.5 0.5 0.05 0.05", "1 0.2 0.2 0.05 0.05"))
    fake_export(raw, "unused", "oneCard", roboflow=False)
    (raw / "labeled" / "tableA" / "preview").mkdir()
    (raw / "labeled" / "tableA" / "preview" / "sheet.jpg").write_bytes(b"ignored")
    m = write_manifest(raw, tmp_path / "manifest.json")
    a = m["exports"]["labeled/tableA"]
    assert a["source"]["project"] == "p" and a["exported"].startswith("October 13, 2026")
    assert a["images"] == {"train": 2} and a["label_files"] == 2 and a["boxes"] == 4
    assert a["files"] == 6  # 2 images, 2 labels, data.yaml, README; the preview sheet is skipped
    assert "source" not in m["exports"]["unused/oneCard"]
    written = json.loads((tmp_path / "manifest.json").read_text())
    assert written["exports"].keys() == {"labeled/tableA", "unused/oneCard"}


def test_fingerprint_hashes_files_in_case_sensitive_order_on_every_os(tmp_path):
    e = fake_export(tmp_path / "raw", "labeled", "tableA")
    files = [p for p in e.rglob("*") if p.is_file()]
    expected = hashlib.sha256()
    for path in sorted(files, key=lambda p: p.relative_to(e).parts):
        rel = path.relative_to(e).as_posix()
        expected.update(f"{rel}\0{hashlib.sha256(path.read_bytes()).hexdigest()}\n".encode())
    assert describe_export(e)["sha256"] == expected.hexdigest()


def test_check_passes_then_catches_a_changed_label_and_a_missing_export(tmp_path):
    raw = tmp_path / "raw"
    a = fake_export(raw, "labeled", "tableA")
    fake_export(raw, "labeled", "tableB")
    path = tmp_path / "manifest.json"
    write_manifest(raw, path)
    assert check_manifest(raw, path) == []

    (a / "train" / "labels" / "img0.txt").write_text("0 0.5 0.5 0.06 0.05\n")  # one box nudged
    problems = check_manifest(raw, path)
    assert len(problems) == 1 and problems[0].startswith("labeled/tableA differs") and "same counts" in problems[0]

    (a / "train" / "labels" / "img0.txt").write_text("")  # a box removed
    assert "1 boxes" in check_manifest(raw, path)[0]

    import shutil
    shutil.rmtree(raw / "labeled" / "tableB")
    assert any("tableB is missing" in p for p in check_manifest(raw, path))


def test_check_without_a_manifest_says_how_to_make_one(tmp_path):
    assert "--manifest" in check_manifest(tmp_path, tmp_path / "none.json")[0]


def test_promote_copies_the_record_and_names_the_live_run(tmp_path):
    run = tmp_path / "models" / "model_9_2026-10-20_n150"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"weights")
    (run / "weights" / "last.pt").write_bytes(b"not copied")
    (run / "notes.txt").write_text("Model 9\n")
    (run / "args.yaml").write_text("model: /abs/path/yolo11n.pt\nepochs: 150\nimgsz: 1024\nbatch: 8\n")
    (run / "results.csv").write_text("epoch\n1\n")
    live = tmp_path / "models" / "cards"
    assert live_model(live) is None
    promote(run, live)
    assert (live / "weights" / "best.pt").read_bytes() == b"weights"
    assert not (live / "weights" / "last.pt").exists()
    assert (live / "notes.txt").read_text() == "Model 9\n" and (live / "results.csv").exists()
    text = (live / "LIVE.txt").read_text()
    assert live_model(live) == "model_9_2026-10-20_n150"
    assert "base yolo11n.pt, epochs 150, imgsz 1024, batch 8" in text and "sha256" in text


def test_promote_refuses_a_folder_without_weights(tmp_path):
    import pytest
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="no weights/best.pt"):
        promote(tmp_path / "empty", tmp_path / "cards")
