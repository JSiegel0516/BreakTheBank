"""Brighten, darken, or locally stretch a frame before detection, for dim or washed-out tables.

    config = PreprocessConfig.from_settings(settings)
    enhanced = enhance(frame, "auto", config)    # a BGR frame of the same shape and dtype

Modes (settings.PREPROCESS, read.py --preprocess, key P live):
    off    the frame unchanged, not copied
    gamma  the brightening gamma, through a lookup table: lifts the shadows of a dim table
    clahe  contrast-limited adaptive histogram equalisation (CLAHE) on the lightness
           channel of LAB colour only, so hues are not shifted
    auto   the mean grey of a sample about 160 pixels wide decides: below dark_below the
           brightening gamma, above bright_above the darkening gamma, otherwise unchanged

Gamma here maps a level v (0 to 255) to 255 * (v / 255) ** gamma: below 1 brightens,
above 1 darkens. Glare that saturates the sensor is 255 in every channel and carries
no detail, so no mode can bring a glared index back.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

MODES = ("off", "gamma", "clahe", "auto")
AUTO_SAMPLE_WIDTH = 160  # auto measures brightness on a sample this wide: plenty for a mean, and cheap


@dataclass(frozen=True)
class PreprocessConfig:
    """The numbers the modes use; the values come from settings.py."""
    brighten_gamma: float  # "gamma", and "auto" on a dim frame
    darken_gamma: float    # "auto" on a washed-out frame
    dark_below: float      # "auto": mean grey below this is dim
    bright_above: float    # "auto": mean grey above this is washed out
    clahe_clip: float      # "clahe": contrast limit per tile
    clahe_tiles: int       # "clahe": tiles across and down

    @classmethod
    def from_settings(cls, s) -> PreprocessConfig:
        return cls(brighten_gamma=s.PREPROCESS_BRIGHTEN_GAMMA, darken_gamma=s.PREPROCESS_DARKEN_GAMMA,
                   dark_below=s.PREPROCESS_DARK_BELOW, bright_above=s.PREPROCESS_BRIGHT_ABOVE,
                   clahe_clip=s.PREPROCESS_CLAHE_CLIP, clahe_tiles=s.PREPROCESS_CLAHE_TILES)


@lru_cache
def _gamma_table(gamma: float) -> np.ndarray:
    levels = np.arange(256) / 255
    return np.round(255 * levels ** gamma).astype(np.uint8)


@lru_cache
def _clahe(clip: float, tiles: int):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))


def _apply_gamma(frame: np.ndarray, gamma: float) -> np.ndarray:
    return cv2.LUT(frame, _gamma_table(gamma))


def _apply_clahe(frame: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    lightness = _clahe(config.clahe_clip, config.clahe_tiles).apply(lightness)
    return cv2.cvtColor(cv2.merge((lightness, a, b)), cv2.COLOR_LAB2BGR)


def _mean_grey(frame: np.ndarray) -> float:
    """Mean grey level (0 to 255) of a BGR frame, measured on every n-th pixel across and down so the
    sample is about AUTO_SAMPLE_WIDTH pixels wide (far cheaper than resizing a 1080p frame)."""
    step = max(1, frame.shape[1] // AUTO_SAMPLE_WIDTH)
    small = np.ascontiguousarray(frame[::step, ::step])
    return float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())


def enhance(frame: np.ndarray, mode: str, config: PreprocessConfig) -> np.ndarray:
    """Return the BGR frame preprocessed by `mode` (one of MODES), same shape and dtype.

    The input is never changed. "off", and "auto" on a frame of middling brightness,
    return the input itself; every other result is a new array. Raises ValueError
    for an unknown mode."""
    if mode == "off":
        return frame
    if mode == "gamma":
        return _apply_gamma(frame, config.brighten_gamma)
    if mode == "clahe":
        return _apply_clahe(frame, config)
    if mode == "auto":
        grey = _mean_grey(frame)
        if grey < config.dark_below:
            return _apply_gamma(frame, config.brighten_gamma)
        if grey > config.bright_above:
            return _apply_gamma(frame, config.darken_gamma)
        return frame
    raise ValueError(f"unknown preprocess mode {mode!r}: use one of {', '.join(MODES)}")
