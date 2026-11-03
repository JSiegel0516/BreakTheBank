import http.client
import json
import queue
import re
import urllib.error
import urllib.request

import numpy as np
import pytest

from cardReader.blackjack import _dealer_finals, dealer_final_chance, stood_odds
from cardReader.cards import Detection
from cardReader.hud import CUE_SEPARATOR, EDIT_HELP, KEY_LIST, DealtCards, Hud, count_cue
from cardReader.reader import Reader
from cardReader.seats import SeatLayout
from cardReader.sidebar import server as server_module
from cardReader.sidebar.server import SidebarServer
from cardReader.sidebar.snapshot import ACTION_KEYS, BUTTONS, KEY_GROUPS, build_snapshot
from cardReader.tracker import CardTracker

W, H = 1280, 720


def place(reader, *cards):
    """Show the (label, seat index, x offset) cards for two frames, enough to confirm them."""
    dets = []
    for label, seat, dx in cards:
        cx, cy = reader.layout.seats[seat].polygon.mean(axis=0)
        dets.append(Detection(label, 0.9, float(cx) + dx, float(cy)))
    frame = np.zeros((H, W, 3), np.uint8)
    reader.process(frame, dets)
    reader.process(frame, dets)


@pytest.fixture
def reader(tmp_path):
    """Round 1 dealt a 2S, which is locked. Round 2: dealer 6C, P1 KS 6H, and a 2S again in P2 (a duplicate)."""
    tracker = CardTracker(confirm_frames=2, drop_frames=5, match_dist=20)
    reader = Reader(tracker, SeatLayout(3, W, H), Hud(), tmp_path / "seats.json", tmp_path / "frames")
    empty = np.zeros((H, W, 3), np.uint8)
    place(reader, ("2S", 1, 0))
    for _ in range(5):
        reader.process(empty, [])                  # the 2S is swept away
    reader.handle_key(ord("r"), empty)
    place(reader, ("6C", 0, 0), ("KS", 1, -30), ("6H", 1, 30), ("2S", 2, 0))
    return reader


MODEL = {"current": "models/cards/weights/best.pt",
         "choices": [{"path": "models/cards/weights/best.pt", "label": "cards (live: model_5)"}]}


def snapshot_of(reader, flash=None):
    return build_snapshot(reader, fps=12.345, stage_ms={"detect": 31.26, "preprocess": 0.44}, no_signal=False,
                          capture_mode=False, showing_raw=True, flash=flash, model=MODEL)


def test_snapshot_holds_the_count_grid_hands_and_status(reader):
    snapshot = snapshot_of(reader, flash=("new round: numbering restarts at 1", 1.5))
    assert snapshot["count"] == {"headline": "+2.2", "cue": "high cards left \u00b7 favours player", "running": "+2",
                                 "cards_left": 48, "deck_size": 52, "high_left": 19, "high_total": 20,
                                 "aces_left": 4, "aces_total": 4}
    rows = {row["rank"]: row for row in snapshot["grid"]}
    assert [row["rank"] for row in snapshot["grid"]] == ["2", "3", "4", "5", "6", "7", "8", "9", "10",
                                                         "J", "Q", "K", "A"]
    assert rows["2"]["cards"][0] == {"name": "2S", "rank": "2", "suit": "S", "state": "locked", "duplicate": True,
                                     "seen": 1, "copies": 1}
    assert snapshot["decks"] == 1
    assert [card["state"] for card in rows["6"]["cards"]] == ["unseen", "pending", "unseen", "pending"]
    assert (rows["2"]["left"], rows["6"]["left"], rows["K"]["left"], rows["A"]["left"]) == (3, 2, 3, 4)
    assert (rows["2"]["value"], rows["8"]["value"], rows["A"]["value"]) == (1, 0, -1)
    assert snapshot["count_system"] == {"current": "hi_lo", "choices": [
        {"id": "hi_lo", "name": "Hi-Lo"}, {"id": "hi_opt_ii", "name": "Hi-Opt II"},
        {"id": "uston_apc", "name": "Uston APC"}]}
    assert snapshot["unrevealed"] == 0

    hands = {hand["seat"]: hand for hand in snapshot["hands"]}
    assert list(hands) == ["dealer", "P1", "P2", "P3"]
    assert [hand["name"] for hand in snapshot["hands"]] == ["DEALER", "P1", "P2", "P3"]
    assert hands["dealer"]["kind"] == "dealer" and hands["dealer"]["hint"] is None
    assert [card["label"] for card in hands["P1"]["cards"]] == ["KS", "6H"]
    assert hands["P1"]["total"] == "16" and hands["P1"]["hint"] == "stand"   # hard 16 against a 6
    assert hands["P2"]["hint"] is None                                       # one card: no hint
    assert hands["P3"] == {"seat": "P3", "name": "P3", "part": None, "kind": "player", "cards": [], "total": "-",
                           "hint": None, "can_split": False, "outcome": None,
                           "odds": {"blackjack": 5}, "stand": None, "insurance": None,  # 2 * 4/48 aces * 15/47 tens
                           "was": None}
    # 48 unseen: 16 needs a 5, 4, 3, or 2 (three 2s left) for 21 to 18; a 6 (two left), 7, 8, 9, or ten busts
    assert hands["P1"]["odds"] == {"21": 8, "20": 8, "19": 8, "18": 6, "bust": 60}
    assert hands["P2"]["odds"] == {"blackjack": 0}                           # a 2 cannot make a blackjack
    # a 6 up and no hole card yet: the dealer draws on, with no blackjack entry
    assert hands["dealer"]["odds"] == {"21": 10, "20": 10, "19": 11, "18": 10, "17": 17, "bust": 42}
    assert hands["dealer"]["insurance"] is None and hands["dealer"]["stand"] is None
    assert hands["P1"]["stand"] == {"win": 42, "push": 0, "lose": 58}       # 16 wins only on a dealer bust
    assert hands["P2"]["stand"] is None and hands["P3"]["stand"] is None and hands["P1"]["insurance"] is None
    assert hands["dealer"]["outcome"] is None and hands["P1"]["outcome"] is None and hands["P2"]["outcome"] is None

    assert snapshot["players"] == 3 and snapshot["edit_help"] == []
    keys = {entry["key"]: entry for entry in snapshot["keys"]}
    assert [(entry["key"], entry["label"]) for entry in snapshot["keys"]] == KEY_LIST
    assert keys["R"]["buttons"] == [{"action": "new_round", "text": "r"}]
    assert keys["X"]["buttons"] == [{"action": "rescan", "text": "x"}]
    assert [button["text"] for button in keys["1-5"]["buttons"]] == ["1", "2", "3", "4", "5"]
    assert keys["N"]["buttons"] == [{"action": "placement_numbers", "text": "n"}]
    assert keys["E"]["buttons"] == keys["S"]["buttons"] == keys["V"]["buttons"] == keys["L"]["buttons"] == []
    assert (keys["1-5"]["group"], keys["G"]["group"], keys["E"]["group"]) == ("table", "overlay", "window")
    assert (snapshot["preprocess"], snapshot["showing_raw"], snapshot["no_signal"]) == ("off", True, False)
    assert snapshot["model"] == MODEL
    assert snapshot["flash"] == {"text": "new round: numbering restarts at 1", "seconds": 1.5}
    assert snapshot["timing"] == {"fps": 12.3, "detect_ms": 31.3, "preprocess_ms": 0.4}
    assert json.loads(json.dumps(snapshot)) == snapshot


def test_snapshot_shows_the_edit_help_while_editing(reader):
    reader.editor.active = True
    assert snapshot_of(reader)["edit_help"] == EDIT_HELP
    assert snapshot_of(reader)["flash"] is None


def test_snapshot_hands_carry_the_seat_id_and_its_display_name(reader):
    reader.rename_seat("P1", "Ann")
    reader.rename_seat("dealer", "House")
    hands = {hand["seat"]: hand["name"] for hand in snapshot_of(reader)["hands"]}
    assert hands == {"dealer": "House", "P1": "Ann", "P2": "P2", "P3": "P3"}


def new_reader(tmp_path, players):
    tracker = CardTracker(confirm_frames=2, drop_frames=5, match_dist=20)
    return Reader(tracker, SeatLayout(players, W, H), Hud(), tmp_path / "seats.json", tmp_path / "frames")


def outcomes_of(reader):
    return [(hand["seat"], hand["part"], hand["outcome"]) for hand in snapshot_of(reader)["hands"]]


def test_snapshot_gives_each_player_hand_its_outcome_and_the_dealer_its_own_mark(tmp_path):
    reader = new_reader(tmp_path, 3)
    place(reader, ("KC", 0, -30), ("7D", 0, 30), ("AS", 1, -30), ("KH", 1, 30), ("9S", 2, -30), ("8H", 2, 30),
          ("5C", 3, -30), ("6H", 3, 30))
    frame = np.zeros((H, W, 3), np.uint8)
    corner = Detection("2C", 0.9, 5.0, H - 5.0)                         # in no seat
    reader.process(frame, [corner])
    reader.process(frame, [corner])
    assert outcomes_of(reader) == [("dealer", None, None), ("P1", None, "blackjack"), ("P2", None, "push"),
                                   ("P3", None, "lose"), ("OTHER", None, None)]


def test_snapshot_marks_a_dealer_blackjack_and_a_player_loss(tmp_path):
    reader = new_reader(tmp_path, 1)
    place(reader, ("AC", 0, -30), ("QS", 0, 30), ("KS", 1, -30), ("8H", 1, 30))
    assert outcomes_of(reader) == [("dealer", None, "blackjack"), ("P1", None, "lose")]


def test_each_hand_of_a_split_seat_gets_its_own_outcome(tmp_path):
    reader = new_reader(tmp_path, 1)
    place(reader, ("8S", 1, -30), ("8H", 1, 30))
    reader.split_seat("P1")
    table = [("KC", 0, -30), ("8D", 0, 30), ("8S", 1, -30), ("8H", 1, 30), ("KS", 1, -70), ("3C", 1, 70)]
    place(reader, *table)
    assert outcomes_of(reader) == [("dealer", None, None), ("P1", 1, "push"), ("P1", 2, "lose")]   # 18, 11 v 18

    place(reader, *table, ("QD", 1, 110))
    assert outcomes_of(reader) == [("dealer", None, None), ("P1", 1, "push"), ("P1", 2, "win")]    # 21 v 18


def hand_rows(snapshot):
    return [(hand["seat"], hand["name"], hand["part"], [card["label"] for card in hand["cards"]], hand["hint"],
             hand["can_split"]) for hand in snapshot["hands"]]


def test_snapshot_offers_a_split_only_on_a_players_pair_and_shows_split_hands_by_part(tmp_path):
    tracker = CardTracker(confirm_frames=2, drop_frames=5, match_dist=20)
    reader = Reader(tracker, SeatLayout(3, W, H), Hud(), tmp_path / "seats.json", tmp_path / "frames")
    reader.rename_seat("P1", "Ann")
    table = [("6C", 0, -30), ("6D", 0, 30), ("8S", 1, -30), ("8H", 1, 30), ("9S", 2, -30), ("9H", 2, 30)]
    place(reader, *table)
    assert hand_rows(snapshot_of(reader)) == [
        ("dealer", "DEALER", None, ["6C", "6D"], None, False),         # the dealer's pair is never split
        ("P1", "Ann", None, ["8S", "8H"], "split", True),
        ("P2", "P2", None, ["9S", "9H"], "split", True),
        ("P3", "P3", None, [], None, False)]

    reader.split_seat("P1")
    assert hand_rows(snapshot_of(reader))[1:3] == [
        ("P1", "Ann", 1, ["8S"], None, False), ("P1", "Ann", 2, ["8H"], None, False)]

    place(reader, *table, ("8D", 1, -70), ("3C", 1, 70))
    assert hand_rows(snapshot_of(reader))[1:3] == [
        ("P1", "Ann", 1, ["8S", "8D"], "stand", False),                # 8,8 again: hard 16, no resplit
        ("P1", "Ann", 2, ["8H", "3C"], "double", False)]


@pytest.fixture
def server():
    server = SidebarServer("127.0.0.1", 0)
    yield server
    server.close()


def test_server_serves_the_page_and_streams_the_last_snapshot(server):
    page = urllib.request.urlopen(server.url, timeout=5).read().decode()
    assert "new EventSource(\"/events\")" in page
    with pytest.raises(urllib.error.HTTPError, match="404"):
        urllib.request.urlopen(server.url + "missing", timeout=5)
    with urllib.request.urlopen(server.url + "Inter-Regular.woff2", timeout=5) as font:
        assert font.headers.get_content_type() == "font/woff2"
        assert font.read() == server_module.FONT.read_bytes() and server_module.FONT.read_bytes()[:4] == b"wOF2"

    snapshot = {"count": {"headline": "+1.0"}, "timing": {"fps": 12.0}}
    server.publish(snapshot)
    port = int(server.url.rsplit(":", 1)[1].strip("/"))
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    connection.request("GET", "/events")
    response = connection.getresponse()
    assert response.getheader("Content-Type") == "text/event-stream"
    line = response.readline().decode()
    assert line.startswith("data: ") and json.loads(line[len("data: "):]) == snapshot
    connection.close()


def post(server, body: bytes, content_type="application/json") -> int:
    """POST body to /action; returns the status code."""
    request = urllib.request.Request(server.url + "action", data=body, headers={"Content-Type": content_type})
    try:
        return urllib.request.urlopen(request, timeout=5).status
    except urllib.error.HTTPError as error:
        return error.code


def test_posted_actions_are_drained_in_order_and_bad_posts_are_refused(server):
    assert server.drain_actions() == []
    assert post(server, b'{"action": "new_round"}') == 204
    assert post(server, b'{"action": "quit"}') == 204
    assert post(server, b'{"action": "rename_seat", "seat": "P1", "name": "Ann"}') == 204
    assert server.drain_actions() == [{"action": "new_round"}, {"action": "quit"},
                                      {"action": "rename_seat", "seat": "P1", "name": "Ann"}]
    assert server.drain_actions() == []

    assert post(server, b'{"action": "quit"}', content_type="text/plain") == 415   # what another site could send
    assert post(server, b"not json") == 400
    assert post(server, b'["new_round"]') == 400
    assert post(server, b'"new_round"') == 400
    assert post(server, b'{"action": 7}') == 400
    assert post(server, b'{"button": "quit"}') == 400
    assert post(server, b'{"action": "rename_seat", "seat": "P1", "name": 5}') == 400
    assert post(server, b'{"action": "rename_seat", "seat": ["P1"], "name": "Ann"}') == 400
    assert post(server, b'{"action": "' + b"x" * 2000 + b'"}') == 400
    assert server.drain_actions() == []


def test_every_button_presses_a_key_the_reader_handles():
    assert ACTION_KEYS["players_3"] == ord("3") and ACTION_KEYS["new_shoe"] == ord("R")
    assert set(ACTION_KEYS) == {f"players_{n}" for n in range(1, 6)} | {
        "placement_numbers", "grey_boxes", "new_round", "rescan", "new_shoe", "reconnect", "preprocess",
        "raw_or_enhanced", "turn_left", "turn_right", "quit"}
    assert set(BUTTONS) <= {key for key, _ in KEY_LIST}


def test_every_key_has_one_page_group_and_each_group_is_one_run_of_the_list():
    assert set(KEY_GROUPS) == {key for key, _ in KEY_LIST}
    groups = [KEY_GROUPS[key] for key, _ in KEY_LIST]
    rows = [group for i, group in enumerate(groups) if i == 0 or group != groups[i - 1]]
    assert rows == ["table", "overlay", "image", "camera", "window"]   # each group's keys are consecutive


def test_the_page_has_a_keys_fold_button_and_a_note_under_the_title():
    page = server_module.PAGE.read_text()
    assert '<button id="keysToggle" class="toggle" aria-expanded="true"' in page
    assert '<p id="keysNote" class="note">' in page
    assert 'localStorage.getItem("keysCollapsed")' in page


def test_the_page_colours_every_count_cue_and_stacks_hands_above_cards_left_when_narrow():
    page = server_module.PAGE.read_text()
    for true, tone in [(1.0, "favours-player"), (-1.0, "favours-dealer"), (0.0, "favours-neither")]:
        cue = count_cue(true)[0].replace(CUE_SEPARATOR, " \\u00b7 ")   # the page writes the dot as an escape
        assert f'"{cue}": "{tone}"' in page
    assert 'grid-template-areas: "count" "seats" "deck" "keys" "status";' in page
    assert 'grid-template-areas: "deck count" "deck seats" "deck keys" "status status";' in page
    assert "${count.high_left}/${count.high_total}" in page and "${count.aces_left}/${count.aces_total}" in page


def test_the_count_facts_match_the_grid_after_a_shoe_of_high_cards(tmp_path):
    reader = new_reader(tmp_path, 3)
    place(reader, ("AS", 1, -30), ("KH", 1, 30), ("10D", 2, -30), ("AC", 2, 30))
    snapshot = snapshot_of(reader)
    rows = {row["rank"]: row for row in snapshot["grid"]}
    high_in_grid = sum(rows[rank]["left"] for rank in ["10", "J", "Q", "K", "A"])
    assert (snapshot["count"]["high_left"], snapshot["count"]["aces_left"]) == (16, 2)
    assert (high_in_grid, rows["A"]["left"]) == (16, 2)
    assert snapshot["count"]["cue"] == "low cards left \u00b7 favours dealer"


def test_with_two_decks_the_count_grid_and_odds_use_every_copy(tmp_path):
    tracker = CardTracker(confirm_frames=2, drop_frames=5, match_dist=20, decks=2)
    reader = Reader(tracker, SeatLayout(3, W, H), Hud(), tmp_path / "seats.json", tmp_path / "frames")
    place(reader, ("AS", 1, -30), ("KH", 1, 30), ("10D", 2, 0))
    snapshot = snapshot_of(reader)
    count = snapshot["count"]
    assert snapshot["decks"] == 2
    assert (count["cards_left"], count["deck_size"]) == (101, 104)
    assert (count["high_left"], count["high_total"], count["aces_left"], count["aces_total"]) == (37, 40, 7, 8)
    rows = {row["rank"]: row for row in snapshot["grid"]}
    ace_of_spades = rows["A"]["cards"][0]
    assert (ace_of_spades["state"], ace_of_spades["seen"], ace_of_spades["copies"]) == ("pending", 1, 2)
    assert (rows["A"]["cards"][1]["state"], rows["A"]["cards"][1]["seen"]) == ("unseen", 0)
    assert (rows["A"]["left"], rows["K"]["left"], rows["10"]["left"], rows["2"]["left"]) == (7, 7, 7, 8)
    hands = {hand["seat"]: hand for hand in snapshot["hands"]}
    assert hands["P2"]["odds"] == {"blackjack": 7}                # 7 aces among 101 unseen copies (one deck: 6%)


def test_the_page_offers_the_decks_button_in_the_table_row_and_dims_tiles_by_copies_seen():
    page = server_module.PAGE.read_text()
    assert 'data-action="cycle_decks"' in page
    assert "renderKeys(latest.keys, latest.players, latest.decks)" in page
    assert 'if (entry.group === "table" && groupEnds) rows[rows.length - 1].push(renderDecks(decks));' in page
    assert "linear-gradient(to right, var(--seen-tint) var(--seen), #fff var(--seen))" in page
    assert "`${card.seen} of ${card.copies} seen`" in page
    assert "--seen: ${(100 * card.seen / card.copies).toFixed(1)}%" in page


def test_the_snapshot_follows_the_counting_system(reader):
    reader.set_count_system("uston_apc")          # seen: 2S (locked), 6C, KS, 6H
    snapshot = snapshot_of(reader)
    rows = {row["rank"]: row for row in snapshot["grid"]}
    assert snapshot["count_system"]["current"] == "uston_apc"
    assert [rows[rank]["value"] for rank in ["2", "5", "9", "K", "A"]] == [1, 3, -1, -3, 0]
    assert snapshot["count"]["running"] == "+2"     # 1 + 2 - 3 + 2
    assert snapshot["count"]["headline"] == "+1.1"  # +2 per 48/26 half-decks left


def test_the_dealer_with_a_hole_card_gets_its_blackjack_chance_and_loose_cards_none(tmp_path):
    reader = new_reader(tmp_path, 2)
    place(reader, ("AS", 0, -30), ("BACK", 0, 30))
    loose = [Detection("7C", 0.9, 5.0, 5.0)]                       # a corner of the frame, in no seat
    frame = np.zeros((H, W, 3), np.uint8)
    reader.process(frame, loose)
    reader.process(frame, loose)
    hands = {hand["seat"]: hand for hand in snapshot_of(reader)["hands"]}
    # 16 tens among 50 unseen (AS, 7C seen); "21" leaves blackjack out
    assert hands["dealer"]["odds"] == {"blackjack": 32, "21": 5, "20": 13, "19": 13, "18": 11, "17": 13, "bust": 12}
    assert hands["dealer"]["insurance"] == "skip"                  # 16 of 50 is under a third
    assert hands["OTHER"]["odds"] is None and hands["OTHER"]["insurance"] is None and hands["OTHER"]["stand"] is None
    assert hands["P1"]["odds"] == {"blackjack": 4}                 # 2 * 3/50 aces * 16/49 tens = 3.9%


def test_the_page_renders_each_hands_odds_as_a_row_of_stat_tiles():
    page = server_module.PAGE.read_text()
    assert "${renderOdds(hand)}" in page
    assert 'const NEXT_CARD = ["21", "20", "19", "18", "bust"];' in page
    assert 'statTile("BLACKJACK", hand.odds.blackjack)' in page
    assert '<span class="stat"><span class="label">${esc(label)}</span><span class="value">' in page
    assert ".hand .odds { grid-column: 1 / -1; display: flex; flex-wrap: wrap;" in page
    assert "font-size: 1.4rem; font-weight: 700;" in page
    assert "next card:" not in page and "blackjack chance" not in page


def test_the_page_renders_the_dealers_finals_insurance_and_each_players_stand_row():
    page = server_module.PAGE.read_text()
    assert 'const DEALER_FINALS = ["blackjack", "21", "20", "19", "18", "17", "bust"];' in page
    assert 'if (hand.kind === "dealer") return renderDealerOdds(hand);' in page
    assert 'textTile("INSURANCE", hand.insurance.toUpperCase())' in page
    assert 'const STAND = ["win", "push", "lose"];' in page
    assert '<div class="odds"><span class="group">STAND</span>${percentTiles(hand.stand, STAND)}</div>' in page
    assert ".hand .odds .group {" in page


def test_a_snapshot_plays_the_dealer_out_once_for_every_hand(tmp_path):
    reader = new_reader(tmp_path, 3)
    place(reader, ("9C", 0, 0), ("KS", 1, -30), ("6H", 1, 30), ("10D", 2, -30), ("2H", 2, 30),
          ("5S", 3, -30), ("7C", 3, 30))
    _dealer_finals.cache_clear()
    hands = {hand["seat"]: hand for hand in snapshot_of(reader)["hands"]}
    assert _dealer_finals.cache_info().misses == 1
    for seat in ("P1", "P2", "P3"):
        stand = hands[seat]["stand"]
        assert set(stand) == {"win", "push", "lose"} and 99 <= sum(stand.values()) <= 101
    assert "blackjack" not in hands["dealer"]["odds"] and hands["dealer"]["insurance"] is None
    assert 99 <= sum(hands["dealer"]["odds"].values()) <= 101


def test_the_page_shows_a_finished_hand_how_likely_its_result_was_in_place_of_its_live_odds():
    page = server_module.PAGE.read_text()
    render_odds = page[page.index("function renderOdds(hand) {"):page.index("function renderDealerOdds(hand)")]
    first_line = render_odds.splitlines()[1].strip()
    assert first_line == "if (hand.was) return renderWas(hand.was);"     # before the live rows and the dealer's
    assert "let tiles = statTile(was.result.toUpperCase(), was.chance);" in page
    assert "if (was.stand) tiles += statTile(was.stand.result.toUpperCase(), was.stand.chance);" in page
    assert '<div class="odds"><span class="group">Odds were</span>${tiles}</div>' in page


def test_a_finished_dealer_has_no_finals_and_its_players_no_stand_row(tmp_path):
    reader = new_reader(tmp_path, 1)
    place(reader, ("KC", 0, -30), ("7D", 0, 30), ("9S", 1, -30), ("8H", 1, 30))
    hands = {hand["seat"]: hand for hand in snapshot_of(reader)["hands"]}
    assert hands["dealer"]["odds"] is None and hands["dealer"]["insurance"] is None
    assert hands["P1"]["outcome"] == "push" and hands["P1"]["stand"] is None


def test_finished_hands_carry_how_likely_their_results_were(tmp_path):
    reader = new_reader(tmp_path, 2)
    place(reader, ("KC", 0, -30), ("7D", 0, 30), ("9S", 1, -30), ("8H", 1, 30), ("5C", 2, 0))
    loose = [Detection("7C", 0.9, 5.0, 5.0)]                       # a corner of the frame, in no seat
    frame = np.zeros((H, W, 3), np.uint8)
    reader.process(frame, loose)
    reader.process(frame, loose)
    unseen = DealtCards.from_tracker(reader.tracker).unseen()
    hands = {hand["seat"]: hand for hand in snapshot_of(reader)["hands"]}
    # P1 pushed with 17: 4 eights among the 46 unseen and its 8H; the stand figure replays the dealer from its KC
    assert hands["P1"]["was"] == {"result": "17", "chance": 9,
                                  "stand": stood_odds(["9S", "8H"], ["KC", "7D"], unseen)}
    assert hands["P1"]["was"]["stand"]["result"] == "push"
    assert hands["dealer"]["was"] == dealer_final_chance(["KC", "7D"], unseen)
    assert hands["dealer"]["was"]["result"] == "17"
    assert hands["P2"]["was"] is None and hands["P2"]["odds"] is not None   # one card: still playing
    assert hands["OTHER"]["was"] is None
    for hand in (hands["P1"], hands["dealer"]):
        assert hand["odds"] is None and hand["stand"] is None and hand["insurance"] is None


def test_a_hand_finished_before_the_dealer_has_no_stand_figure_yet(tmp_path):
    reader = new_reader(tmp_path, 1)
    place(reader, ("9C", 0, 0), ("KS", 1, -30), ("AH", 1, 30))
    hands = {hand["seat"]: hand for hand in snapshot_of(reader)["hands"]}
    # a blackjack over 49 unseen plus the KS and AH: 2 * 4/51 aces * 16/50 tens
    assert hands["P1"]["was"] == {"result": "blackjack", "chance": 5, "stand": None}
    assert hands["dealer"]["was"] is None


def test_the_page_has_a_counting_system_dropdown_and_scales_the_row_shading():
    page = server_module.PAGE.read_text()
    assert '<select id="countSelect">' in page
    assert 'postAction({ action: "switch_count", system: event.target.value });' in page
    assert "TRUE COUNT \\u00b7 ${systemName(latest.count_system)}" in page
    assert "background: ${shade(row.value)}" in page and "color: ${valueColour(row.value)}" in page
    assert "row.hi_lo" not in page


def test_the_page_marks_hand_outcomes_and_holds_a_confetti_canvas():
    page = server_module.PAGE.read_text()
    assert '<canvas id="confetti" aria-hidden="true"></canvas>' in page
    assert "outcome-${hand.outcome}" in page
    for cue in ["blackjack", "twenty-one", "win", "bust", "lose"]:
        assert f".outcome-{cue}" in page
    assert 'const OUTCOME_TAG = { win: "Win", push: "Push", lose: "Lost" };' in page
    assert '<span class="tag ${hand.outcome}">' in page
    assert ".hand.dealer.pulse" in page and ".hand.dealer.outcome-bust" in page


def test_the_page_writes_labels_in_sentence_case_and_keeps_headings_and_tile_labels_uppercase():
    page = server_module.PAGE.read_text()
    for label in ["<span>Running</span>", "<span>Left</span>", "<span>Tens &amp; aces</span>", "<span>Aces</span>",
                  'title="Split into two hands">Split</button>', 'title="Make one hand again">Join</button>',
                  "Face down, never revealed: ${unrevealed}", '<div id="countBody">Waiting for read.py ...</div>',
                  "Not connected: is read.py --web running?", "No signal: press C to reconnect",
                  "<span>Model ${esc(modelName(snapshot.model))}</span>",
                  "<span>Detect ${t.detect_ms.toFixed(1)} ms</span>",
                  "<span>Prep ${esc(snapshot.preprocess)} ${t.preprocess_ms.toFixed(1)} ms</span>",
                  "<span>Showing raw</span>", "${t.fps.toFixed(1)} fps", 'title="Click to rename"',
                  '<span class="action">Deck${decks > 1 ? "s" : ""}</span>',
                  "<span>Model</span>", "<span>Count</span>"]:
        assert label in page
    # labels the snapshot sends in lower case, and that the window shares, get a capital on the page
    assert '<span class="action">${esc(sentence(entry.label))}</span>' in page
    assert '<span class="cue">${esc(sentence(count.cue))}</span>' in page
    assert "flash = sentence(message.text);" in page
    assert "${esc(sentence(hand.total))}" in page                                       # "Soft 18"
    assert "snapshot.edit_help.map((line) => esc(sentence(line)))" in page
    for heading in ["TRUE COUNT \\u00b7 ${systemName(latest.count_system)}", "<h2>CARDS LEFT</h2>",
                    '<h2 id="seatsTitle">HANDS</h2>', ">KEYS</button></h2>"]:
        assert heading in page
    assert 'statTile("BLACKJACK", hand.odds.blackjack)' in page and "<span class=\"group\">STAND</span>" in page
    assert ".tag {" in page and "text-transform: uppercase; vertical-align" not in page   # tags keep their case


def _colours(page: str) -> dict[str, str]:
    root = page[page.index(":root {"):page.index("}", page.index(":root {"))]
    return dict(re.findall(r"--([a-z-]+):\s*(#[0-9A-Fa-f]{6})", root))


def _contrast(foreground: str, background: str) -> float:
    def luminance(colour: str) -> float:
        channels = [int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
    lighter, darker = sorted((luminance(foreground), luminance(background)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def test_the_page_uses_the_darker_greens_and_every_text_colour_stays_readable_on_a_section():
    colours = _colours(server_module.PAGE.read_text())
    assert (colours["panel"], colours["card"], colours["well"]) == ("#003F22", "#2A523E", "#002E19")
    for text in ["text", "grey", "dim", "badge", "amber", "red", "green", "seat"]:
        assert _contrast(colours[text], colours["card"]) >= 4.5, text


def test_the_page_sets_its_text_in_the_bundled_inter_with_the_system_fonts_behind_it():
    page = server_module.PAGE.read_text()
    assert '@font-face {\n    font-family: "Inter";' in page
    assert 'src: url("/Inter-Regular.woff2") format("woff2");' in page
    assert 'body { font: 16px/1.3 "Inter", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }' in page
    assert "fonts.googleapis" not in page and "https://" not in page                  # works offline
    assert server_module.FONT.name == "Inter-Regular.woff2" and server_module.FONT.with_name("Inter-OFL.txt").exists()
    assert 'matchMedia("(prefers-reduced-motion: reduce)")' in page
    assert "requestAnimationFrame(drawConfetti)" in page


def received(client: queue.Queue) -> list[dict]:
    messages = []
    while not client.empty():
        messages.append(json.loads(client.get_nowait()))
    return messages


def test_publish_sends_changes_at_once_and_unchanged_state_only_on_the_heartbeat(server, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(server_module.time, "monotonic", lambda: clock[0])
    client = server._subscribe()
    for seconds, count, fps in [(0.0, 1, 12.0), (0.1, 1, 12.5), (0.1, 2, 12.5), (0.1, 2, 12.6), (0.5, 2, 13.0)]:
        clock[0] += seconds
        server.publish({"count": count, "timing": {"fps": fps}})
    assert received(client) == [{"count": 1, "timing": {"fps": 12.0}},   # first
                                {"count": 2, "timing": {"fps": 12.5}},   # changed
                                {"count": 2, "timing": {"fps": 13.0}}]   # heartbeat; fps alone waits for it


def test_a_flash_is_sent_once(server):
    client = server._subscribe()
    flash = {"text": "new round", "seconds": 1.5}
    for snapshot in [{"flash": None}, {"flash": flash}, {"flash": None}, {"flash": None}]:
        server.publish(snapshot)
    assert [message["flash"] for message in received(client)] == [None, flash, None]


def test_a_page_that_falls_behind_skips_events_instead_of_growing_its_queue(server, monkeypatch):
    monkeypatch.setattr(server_module, "QUEUE_SIZE", 2)
    client = server._subscribe()
    for count in range(5):
        server.publish({"count": count})
    assert received(client) == [{"count": 0}, {"count": 1}]
