"""Score the whole reader against the cards that were really dealt.

    python read.py --source round1.mp4 --log round1.csv --no-display
    python evaluate.py --log round1.csv --expected round1Expected.csv

The expected file is a CSV with one row per card placed, in the order placed:

    round,order,card,seat
    1,1,10H,P1
    1,2,6C,P2
    1,3,KS,dealer
    2,1,4D,P1

round counts from 1 and order restarts at 1 every round. seat is optional; a
wrong seat is listed but not counted as an error. Write a face-down card that
was never turned over as BACK.

In the log, a round ends at each round_over event. A placed event whose number
was already used in the round also starts a new one, since pressing r and
appending a second read.py run to the same log write no event of their own.
A rescan row (key x) drops the cards of the round in progress, and the cards
placed after it fill that same round again.
Each card's name is the one it was placed with, or its last renamed name.
Cards are matched by round and placement number, so a missed card shifts the
numbers of the cards after it in that round.

The report lists cards matched, missed, extra, and wrong names (wrong suit and
wrong rank separately), and the final running count against the Hi-Lo count of
the expected cards. Use one shoe per log: Shift+R resets the logged count.
The exit code is 1 only when a file cannot be read or has a bad row.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path

from cardReader.blackjack import hi_lo
from cardReader.deck import BACK, CLASSES, rank_of
from cardReader.reader import LOG_HEADER


@dataclass
class Placement:
    card: str
    seat: str = ""


Rounds = dict[int, dict[int, Placement]]  # round -> placement number -> card


@dataclass
class Problem:
    kind: str  # "missed" | "extra" | "wrong suit" | "wrong rank" | "wrong seat"
    round: int
    order: int
    expected: str = ""
    read: str = ""


@dataclass
class Score:
    cards_expected: int
    cards_read: int
    count_expected: int  # Hi-Lo running count of the expected cards
    count_read: int      # the last running count in the log
    correct: int = 0     # right name at the right number
    problems: list[Problem] = field(default_factory=list)

    def tally(self, kind: str) -> int:
        return sum(1 for problem in self.problems if problem.kind == kind)

    @property
    def matched(self) -> int:
        """Cards placed under a number the expected file also has, whatever their name."""
        return self.correct + self.tally("wrong suit") + self.tally("wrong rank")


def read_log(path: Path) -> tuple[Rounds, int]:
    """Cards placed in each round of a read.py --log CSV, and the last running count logged.

    Raises ValueError for a row with the wrong number of columns or a bad number."""
    rounds: Rounds = {}
    current: dict[int, Placement] = {}
    running_count = 0
    with open(path, newline="") as file:
        for line, row in enumerate(csv.reader(file), start=1):
            if not row or row == LOG_HEADER:
                continue
            if len(row) != len(LOG_HEADER):
                raise ValueError(f"{path} line {line}: {len(row)} columns, expected {len(LOG_HEADER)}")
            event = dict(zip(LOG_HEADER, row, strict=True))
            running_count = _whole_number(event["running_count"], path, line)
            kind = event["event"]
            if kind == "round_over":
                current = {}
                continue
            if kind == "rescan":  # the round in progress is read again under the same number
                if current:
                    del rounds[len(rounds)]
                current = {}
                continue
            if kind not in ("placed", "renamed"):
                continue
            order = _whole_number(event["order"], path, line)
            if kind == "placed" and order in current:
                current = {}
            if not current:  # the round's first card: store the round, then keep filling it in place
                rounds[len(rounds) + 1] = current
            current[order] = Placement(event["card"], event["seat"])
    return rounds, running_count


def read_expected(path: Path) -> Rounds:
    """The expected-cards CSV (round,order,card[,seat]) as rounds of placements.

    Raises ValueError for a missing column, a bad number or card name, or a
    round and placement number given twice."""
    rounds: Rounds = {}
    with open(path, newline="") as file:
        rows = csv.DictReader(file, restval="")
        missing = {"round", "order", "card"} - set(rows.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing column {', '.join(sorted(missing))}")
        for row in rows:
            line = rows.line_num
            round_number = _whole_number(row["round"], path, line)
            order = _whole_number(row["order"], path, line)
            card = row["card"].strip().upper()
            if card not in CLASSES:
                raise ValueError(f"{path} line {line}: {row['card']!r} is not a card name such as 10H or BACK")
            placements = rounds.setdefault(round_number, {})
            if order in placements:
                raise ValueError(f"{path} line {line}: round {round_number} card {order} is given twice")
            placements[order] = Placement(card, row.get("seat", "").strip())
    return rounds


def score(expected: Rounds, read: Rounds, count_read: int) -> Score:
    """Compare the read cards with the expected ones, round by round and number by number."""
    count_expected = sum(hi_lo(p.card) for placements in expected.values() for p in placements.values())
    result = Score(cards_expected=sum(len(placements) for placements in expected.values()),
                   cards_read=sum(len(placements) for placements in read.values()),
                   count_expected=count_expected, count_read=count_read)
    for round_number in sorted(expected.keys() | read.keys()):
        want, got = expected.get(round_number, {}), read.get(round_number, {})
        for order in sorted(want.keys() | got.keys()):
            want_card = want[order].card if order in want else ""
            got_card = got[order].card if order in got else ""
            kind = _name_problem(want_card, got_card)
            if kind:
                result.problems.append(Problem(kind, round_number, order, want_card, got_card))
            else:
                result.correct += 1
            if not (want_card and got_card):
                continue
            want_seat, got_seat = want[order].seat, got[order].seat
            if want_seat and want_seat != got_seat:
                result.problems.append(Problem("wrong seat", round_number, order, want_seat, got_seat))
    return result


def report(result: Score) -> str:
    """The score as a few lines of text, then one line per problem."""
    wrong_suit, wrong_rank = result.tally("wrong suit"), result.tally("wrong rank")
    difference = result.count_read - result.count_expected
    lines = [
        f"cards: {result.cards_expected} expected, {result.cards_read} placed",
        f"matched: {result.matched} ({result.correct} right name, {wrong_suit + wrong_rank} wrong name: "
        f"{wrong_suit} wrong suit, {wrong_rank} wrong rank)",
        f"missed: {result.tally('missed')}",
        f"extra: {result.tally('extra')}",
        f"wrong seat: {result.tally('wrong seat')} (listed, not counted as errors)",
        f"running count: read {result.count_read:+d}, expected {result.count_expected:+d}, "
        f"difference {difference:+d}",
    ]
    if result.problems:
        lines.append("")
    for problem in result.problems:
        where = f"round {problem.round} #{problem.order}"
        if problem.kind == "missed":
            lines.append(f"{where}: {problem.expected} missed")
        elif problem.kind == "extra":
            lines.append(f"{where}: {problem.read} extra")
        else:
            lines.append(f"{where}: expected {problem.expected}, read {problem.read} ({problem.kind})")
    return "\n".join(lines)


def _name_problem(expected: str, read: str) -> str:
    """The kind of name mismatch between two cards at the same number, or "" when they agree."""
    if not read:
        return "missed"
    if not expected:
        return "extra"
    if expected == read:
        return ""
    same_rank = BACK not in (expected, read) and rank_of(expected) == rank_of(read)
    return "wrong suit" if same_rank else "wrong rank"


def _whole_number(text: str, path: Path, line: int) -> int:
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"{path} line {line}: {text!r} is not a whole number") from None


def main():
    ap = argparse.ArgumentParser(description="Score a read.py --log CSV against the cards really dealt.")
    ap.add_argument("--log", required=True, type=Path, help="CSV written by read.py --log")
    ap.add_argument("--expected", required=True, type=Path, help="CSV of the cards dealt: round,order,card[,seat]")
    args = ap.parse_args()
    try:
        read, count_read = read_log(args.log)
        expected = read_expected(args.expected)
    except (OSError, ValueError) as error:
        raise SystemExit(error) from None
    print(report(score(expected, read, count_read)))


if __name__ == "__main__":
    main()
