import numpy as np
import pytest

import settings
from cardReader.preprocess import MODES, PreprocessConfig, enhance

CONFIG = PreprocessConfig.from_settings(settings)


def flat(value, H=90, W=160):
    return np.full((H, W, 3), value, np.uint8)


def random_frame(H=90, W=160):
    return np.random.default_rng(0).integers(0, 256, (H, W, 3), dtype=np.uint8)


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_keeps_shape_and_dtype_and_leaves_the_input_alone(mode):
    frame = random_frame()
    before = frame.copy()
    out = enhance(frame, mode, CONFIG)
    assert out.shape == frame.shape and out.dtype == np.uint8
    assert (frame == before).all()


def test_off_returns_the_input_itself():
    frame = random_frame()
    assert enhance(frame, "off", CONFIG) is frame


def test_gamma_brightens_with_the_brightening_gamma():
    out = enhance(flat(50), "gamma", CONFIG)
    expected = round(255 * (50 / 255) ** CONFIG.brighten_gamma)
    assert out.min() == out.max() == expected and expected > 50


def test_auto_brightens_a_dim_frame_darkens_a_washed_out_one_and_leaves_the_rest():
    dark, mid, bright = flat(CONFIG.dark_below - 20), flat(128), flat(CONFIG.bright_above + 20)
    assert (enhance(dark, "auto", CONFIG) == enhance(dark, "gamma", CONFIG)).all()
    assert enhance(dark, "auto", CONFIG).mean() > dark.mean()
    assert enhance(mid, "auto", CONFIG) is mid
    darker = enhance(bright, "auto", CONFIG)
    assert darker.mean() < bright.mean()
    assert darker[0, 0, 0] == round(255 * ((CONFIG.bright_above + 20) / 255) ** CONFIG.darken_gamma)


def test_clahe_spreads_contrast_without_shifting_colour():
    ramp = np.tile(np.linspace(100, 140, 160).astype(np.uint8), (90, 1))
    grey = np.dstack([ramp, ramp, ramp])                       # B = G = R: no colour at all
    out = enhance(grey, "clahe", CONFIG).astype(int)
    assert np.ptp(out) > np.ptp(grey)                          # the contrast is stretched ...
    assert np.abs(out[..., 0] - out[..., 1]).max() <= 2        # ... but the pixels stay grey
    assert np.abs(out[..., 1] - out[..., 2]).max() <= 2


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="sharpen"):
        enhance(random_frame(), "sharpen", CONFIG)
