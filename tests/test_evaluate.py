import csv
import sys

import numpy as np
import pytest

from card_reader.cards import Detection
from card_reader.hud import Hud
from card_reader.reader import LOG_HEADER, Reader
from card_reader.seats import SeatLayout
from card_reader.tracker import CardTracker
from evaluate import main, read_expected, read_log, report, score

HEADER = "time,frame,event,order,card,seat,running_count,x,y\n"


def log_file(tmp_path, *events):
    """A read.py --log CSV with one row per (event, order, card, seat, running count)."""
    rows = [f"0.0,{frame},{kind},{order},{card},{seat},{count},0,0"
            for frame, (kind, order, card, seat, count) in enumerate(events, start=1)]
    path = tmp_path / "log.csv"
    path.write_text(HEADER + "\n".join(rows) + "\n")
    return path


def expected_file(tmp_path, text):
    path = tmp_path / "expected.csv"
    path.write_text(text)
    return path


def evaluate(tmp_path, expected_text, *events):
    read, count = read_log(log_file(tmp_path, *events))
    return score(read_expected(expected_file(tmp_path, expected_text)), read, count)


def kinds(result):
    return [(p.kind, p.round, p.order) for p in result.problems]


ROUND_1 = "round,order,card\n1,1,10H\n1,2,6C\n"


def test_perfect_match(tmp_path):
    result = evaluate(tmp_path, ROUND_1,
                      ("placed", 1, "10H", "P1", -1), ("placed", 2, "6C", "P2", 0))
    assert (result.matched, result.correct, result.problems) == (2, 2, [])
    assert (result.count_read, result.count_expected) == (0, 0)


def test_missed_card(tmp_path):
    result = evaluate(tmp_path, ROUND_1, ("placed", 1, "10H", "P1", -1))
    assert kinds(result) == [("missed", 1, 2)]
    assert result.count_read - result.count_expected == -1


def test_extra_card(tmp_path):
    result = evaluate(tmp_path, ROUND_1, ("placed", 1, "10H", "P1", -1), ("placed", 2, "6C", "P2", 0),
                      ("placed", 3, "9D", "P2", 0))
    assert kinds(result) == [("extra", 1, 3)]
    assert result.matched == 2


def test_wrong_suit_and_wrong_rank_are_told_apart(tmp_path):
    result = evaluate(tmp_path, ROUND_1, ("placed", 1, "10D", "P1", -1), ("placed", 2, "9C", "P2", -1))
    assert kinds(result) == [("wrong suit", 1, 1), ("wrong rank", 1, 2)]
    assert (result.matched, result.correct) == (2, 0)


def test_face_down_card_left_down_is_a_wrong_rank(tmp_path):
    result = evaluate(tmp_path, "round,order,card\n1,1,KS\n", ("placed", 1, "BACK", "dealer", 0))
    assert kinds(result) == [("wrong rank", 1, 1)]


def test_rename_fixes_the_name(tmp_path):
    result = evaluate(tmp_path, ROUND_1, ("placed", 1, "10H", "P1", -1), ("placed", 2, "9C", "P2", -1),
                      ("renamed", 2, "6C", "P2", 0), ("hidden", 2, "6C", "P2", 0))
    assert (result.correct, result.problems) == (2, [])


def test_multiple_rounds_split_at_round_over(tmp_path):
    expected = "round,order,card\n1,1,10H\n1,2,6C\n2,1,4D\n2,2,AS\n"
    result = evaluate(tmp_path, expected, ("placed", 1, "10H", "P1", -1), ("placed", 2, "6C", "P2", 0),
                      ("round_over", "", "", "", 0),
                      ("placed", 1, "4D", "P1", 1), ("placed", 2, "AS", "P2", 0))
    assert (result.cards_expected, result.cards_read, result.correct) == (4, 4, 4)
    assert (result.count_read, result.count_expected) == (0, 0)


def test_a_reused_number_starts_a_new_round(tmp_path):
    read, _ = read_log(log_file(tmp_path, ("placed", 1, "10H", "P1", -1), ("placed", 1, "4D", "P1", 0)))
    assert {r: [p.card for p in placements.values()] for r, placements in read.items()} == {1: ["10H"], 2: ["4D"]}


def test_wrong_seat_is_listed_but_not_an_error(tmp_path):
    result = evaluate(tmp_path, "round,order,card,seat\n1,1,10H,P2\n", ("placed", 1, "10H", "P1", -1))
    assert kinds(result) == [("wrong seat", 1, 1)]
    assert result.correct == 1


def test_report_shows_the_counts_and_each_problem(tmp_path):
    result = evaluate(tmp_path, ROUND_1, ("placed", 1, "10D", "P1", -1))
    text = report(result)
    assert "matched: 1 (0 right name, 1 wrong name: 1 wrong suit, 0 wrong rank)" in text
    assert "missed: 1" in text
    assert "running count: read -1, expected +0, difference -1" in text
    assert "round 1 #1: expected 10H, read 10D (wrong suit)" in text
    assert "round 1 #2: 6C missed" in text


@pytest.mark.parametrize("text, message", [
    ("round,card\n1,10H\n", "missing column order"),
    ("round,order,card\n1,x,10H\n", "'x' is not a whole number"),
    ("round,order,card\n1,1,1H\n", "'1H' is not a card name"),
    ("round,order,card\n1,1,10H\n1,1,6C\n", "given twice"),
])
def test_bad_expected_file_is_an_error(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        read_expected(expected_file(tmp_path, text))


def test_exit_code_is_non_zero_only_on_input_errors(tmp_path, monkeypatch, capsys):
    log = log_file(tmp_path, ("placed", 1, "9C", "P1", 0))
    good = expected_file(tmp_path, ROUND_1)
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--log", str(log), "--expected", str(good)])
    main()                                               # mismatches alone: returns normally
    assert "wrong rank" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--log", str(log), "--expected", str(tmp_path / "nope.csv")])
    with pytest.raises(SystemExit) as stopped:
        main()
    assert "nope.csv" in str(stopped.value.code)         # a message, which exits with status 1


def test_a_log_written_by_the_reader_is_scored(tmp_path):
    path = tmp_path / "log.csv"
    with open(path, "w", newline="") as file:
        log = csv.writer(file)
        log.writerow(LOG_HEADER)
        tracker = CardTracker(confirm_frames=2, drop_frames=3, round_end_frames=5, match_dist=20)
        reader = Reader(tracker, SeatLayout(3, 640, 360), Hud(), tmp_path / "seats.json", tmp_path / "frames", log=log)
        frame = np.zeros((360, 640, 3), np.uint8)
        p1, p2 = (reader.layout.seats[i].polygon.mean(axis=0) for i in (1, 2))
        for _ in range(3):
            reader.process(frame, [Detection("10H", 0.9, *map(float, p1)), Detection("6C", 0.9, *map(float, p2))])
        for _ in range(10):
            reader.process(frame, [])                    # swept: round over
        for _ in range(3):
            reader.process(frame, [Detection("10H", 0.9, *map(float, p1))])  # a misread 4D, flagged duplicate
    read, count = read_log(path)
    expected = read_expected(expected_file(tmp_path, "round,order,card,seat\n1,1,10H,P1\n1,2,6C,P2\n2,1,4D,P1\n"))
    result = score(expected, read, count)
    assert kinds(result) == [("wrong rank", 2, 1)]
    assert (result.correct, result.count_read, result.count_expected) == (2, 0, 1)
