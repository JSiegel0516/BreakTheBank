import sys
from types import SimpleNamespace

import pytest

import read
import settings
import train


def shipped(**changes):
    """A copy of the values in settings.py, with some replaced."""
    values = {name: value for name, value in vars(settings).items() if name.isupper()}
    return SimpleNamespace(**{**values, **changes})


def test_shipped_settings_are_valid():
    settings.validate(settings)


@pytest.mark.parametrize("decks", [1, 2, 3])
def test_one_to_three_decks_are_valid_and_one_is_the_default(decks):
    assert settings.DECKS == 1
    settings.validate(shipped(DECKS=decks))


def test_no_rotated_copies_is_valid():
    settings.validate(shipped(TRAIN_ROTATED_COPIES=()))


def test_focus_defaults_are_the_eight_misread_cards_with_two_extra_copies_and_none_is_valid():
    assert settings.FOCUS_CLASSES == ("AD", "2D", "6D", "BACK", "9S", "4C", "4H", "2H")
    assert settings.FOCUS_EXTRA_COPIES == 2
    settings.validate(shipped(FOCUS_CLASSES=(), FOCUS_EXTRA_COPIES=0))


@pytest.mark.parametrize("name, value, allowed", [
    ("CONFIRM_SECONDS", 0, "number above 0"),
    ("CONFIRM_WINDOW_SECONDS", -0.7, "number above 0"),
    ("DROP_SECONDS", "1", "number above 0"),
    ("LABEL_WINDOW_SECONDS", 0, "number above 0"),
    ("ROUND_END_SECONDS", -3.0, "number above 0"),
    ("HIDDEN_SECONDS", -1, "number of 0 or more"),
    ("COPY_HOLD_SECONDS", -0.5, "number of 0 or more"),
    ("PLAYERS", 6, "whole number from 1 to 5"),
    ("CONFIDENCE", 1.5, "number from 0 to 1"),
    ("CONFIRM_CONFIDENCE", -0.1, "number from 0 to 1"),
    ("SUSTAIN_CONFIDENCE", -0.1, "number from 0 to 1"),
    ("SUSTAIN_CONFIDENCE", "0.4", "number from 0 to 1"),
    ("MATCH_DISTANCE", 0, "fraction above 0 and at most 1"),
    ("VAL_FRACTION", 1.0, "fraction above 0 and below 1"),
    ("CAMERA_WIDTH", -1, "None or a whole number of 1 or more"),
    ("SYNTHETIC_IMAGES", -5, "whole number of 0 or more"),
    ("BATCH_SIZE", True, "whole number of 1 or more"),
    ("PREPROCESS", "sharpen", "one of 'off', 'gamma', 'clahe', 'auto'"),
    ("PREPROCESS_BRIGHTEN_GAMMA", 0, "number above 0"),
    ("PREPROCESS_DARKEN_GAMMA", -1.4, "number above 0"),
    ("PREPROCESS_DARK_BELOW", -1, "number from 0 to 255"),
    ("PREPROCESS_BRIGHT_ABOVE", 300, "number from 0 to 255"),
    ("PREPROCESS_CLAHE_CLIP", 0, "number above 0"),
    ("PREPROCESS_CLAHE_TILES", 0, "whole number of 1 or more"),
    ("TRAIN_ROTATED_COPIES", (45,), "tuple of different angles from 90, 180, 270"),
    ("TRAIN_ROTATED_COPIES", (90, 90), "tuple of different angles from 90, 180, 270"),
    ("TRAIN_ROTATED_COPIES", [90], "tuple of different angles from 90, 180, 270"),
    ("TRAIN_ROTATED_COPIES", (0,), "tuple of different angles from 90, 180, 270"),
    ("TRAIN_ROTATED_COPIES", ([90],), "tuple of different angles from 90, 180, 270"),
    ("CAMERA_ROTATION", 45, "one of 0, 90, 180, 270"),
    ("CAMERA_ROTATION", 90.0, "one of 0, 90, 180, 270"),
    ("WEB_PORT", 0, "whole number from 1 to 65535"),
    ("WEB_PORT", 70000, "whole number from 1 to 65535"),
    ("WEB_PORT", "8765", "whole number from 1 to 65535"),
    ("COUNT_SYSTEM", "ko", "one of 'hi_lo', 'hi_opt_ii', 'uston_apc'"),
    ("COUNT_SYSTEM", ["hi_lo"], "one of 'hi_lo', 'hi_opt_ii', 'uston_apc'"),
    ("DECKS", 0, "whole number from 1 to 3"),
    ("DECKS", 4, "whole number from 1 to 3"),
    ("DECKS", 1.5, "whole number from 1 to 3"),
    ("DECKS", 2.0, "whole number from 1 to 3"),
    ("DECKS", True, "whole number from 1 to 3"),
    ("FOCUS_CLASSES", ("AD", "ZZ"), "tuple of different card names from deck.CLASSES"),
    ("FOCUS_CLASSES", ("ad",), "tuple of different card names from deck.CLASSES"),
    ("FOCUS_CLASSES", ("AD", "AD"), "tuple of different card names from deck.CLASSES"),
    ("FOCUS_CLASSES", ["AD"], "tuple of different card names from deck.CLASSES"),
    ("FOCUS_EXTRA_COPIES", -1, "whole number of 0 or more"),
    ("FOCUS_EXTRA_COPIES", 1.5, "whole number of 0 or more"),
    ("FOCUS_EXTRA_COPIES", True, "whole number of 0 or more"),
])
def test_invalid_value_is_reported_with_its_name_and_allowed_range(name, value, allowed):
    with pytest.raises(ValueError) as error:
        settings.validate(shipped(**{name: value}))
    assert f"{name} = {value!r}" in str(error.value)
    assert allowed in str(error.value)


def test_confirm_time_cannot_exceed_the_window_it_is_counted_in():
    with pytest.raises(ValueError, match="CONFIRM_SECONDS = 0.5 must be at most CONFIRM_WINDOW_SECONDS = 0.4"):
        settings.validate(shipped(CONFIRM_WINDOW_SECONDS=0.4))


def test_sustain_confidence_cannot_exceed_the_confidence_that_starts_a_track():
    with pytest.raises(ValueError, match="SUSTAIN_CONFIDENCE = 0.7 must be at most CONFIDENCE = 0.6"):
        settings.validate(shipped(SUSTAIN_CONFIDENCE=0.7))
    with pytest.raises(ValueError, match="SUSTAIN_CONFIDENCE = 1.5: must be a number from 0 to 1"):
        settings.validate(shipped(SUSTAIN_CONFIDENCE=1.5, CONFIDENCE=1.5))
    settings.validate(shipped(SUSTAIN_CONFIDENCE=0.6))                  # equal: weak boxes are off


def test_hidden_seconds_zero_is_allowed():
    settings.validate(shipped(HIDDEN_SECONDS=0))


def test_copy_hold_seconds_zero_is_allowed():
    settings.validate(shipped(COPY_HOLD_SECONDS=0))


def test_auto_dark_threshold_must_be_below_the_bright_one():
    with pytest.raises(ValueError, match="PREPROCESS_DARK_BELOW = 200 must be below PREPROCESS_BRIGHT_ABOVE = 190"):
        settings.validate(shipped(PREPROCESS_DARK_BELOW=200))


def test_every_invalid_value_is_listed_at_once():
    with pytest.raises(ValueError) as error:
        settings.validate(shipped(PLAYERS=0, INPUT_SIZE=0, STILL_DISTANCE=2))
    message = str(error.value)
    assert "PLAYERS = 0" in message and "INPUT_SIZE = 0" in message and "STILL_DISTANCE = 2" in message


def test_read_stops_on_invalid_settings_before_opening_the_camera(monkeypatch):
    monkeypatch.setattr(settings, "PLAYERS", 9)
    monkeypatch.setattr(read, "Camera", lambda *a, **k: pytest.fail("camera opened despite invalid settings"))
    monkeypatch.setattr(sys, "argv", ["read.py", "--no-display"])
    with pytest.raises(SystemExit) as stop:
        read.main()
    assert "PLAYERS = 9" in str(stop.value)


def test_train_stops_on_invalid_settings_before_building_data(monkeypatch):
    monkeypatch.setattr(settings, "VAL_FRACTION", 0)
    monkeypatch.setattr(train.dataset, "build", lambda *a, **k: pytest.fail("dataset built despite invalid settings"))
    monkeypatch.setattr(sys, "argv", ["train.py", "--rebuild"])
    with pytest.raises(SystemExit) as stop:
        train.main()
    assert "VAL_FRACTION = 0" in str(stop.value)
