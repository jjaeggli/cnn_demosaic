import numpy as np
import pytest

from cnn_demosaic import exposure, transform
from cnn_demosaic.exposure import Exposure, ExposureParameters

# The capture settings of a real raw file (Fujifilm X-E2S, assets/DSCF5657.raf).
ISO_SPEED = 400.0
SHUTTER_SPEED = 0.002222222276031971
APERTURE = 8.0


class FakeRaw:
    """A stand-in for rawpy.RawPy which exposes the capture metadata attributes."""

    def __init__(self, other=None):
        self.other = other


def fake_raw():
    from types import SimpleNamespace
    return FakeRaw(
        SimpleNamespace(
            iso_speed=ISO_SPEED,
            shutter_speed=SHUTTER_SPEED,
            aperture=APERTURE,
            focal_length=35.0,
            timestamp=None,
            shot_order=0,
            artist="",
        )
    )


def test_gain_defaults_is_unit_gain():
    # The default headroom of 2.47 EV places a metered 18% gray at 0.18, so
    # the default gain must be 1.0.
    gain = exposure.compute_gain(0.18, 2.47, 0.0)
    assert gain == pytest.approx(1.0, rel=0.01)


def test_gain_offset_doubles_per_stop():
    gain = exposure.compute_gain(0.18, 2.47, 1.0)
    assert gain == pytest.approx(2.0, rel=0.01)
    assert gain == pytest.approx(2.0 * exposure.compute_gain(0.18, 2.47, 0.0))


def test_gain_headroom():
    # With 2.0 EV of headroom, a metered gray sits at 0.25 and must be scaled
    # down by 0.72 to reach the 0.18 target.
    gain = exposure.compute_gain(0.18, 2.0, 0.0)
    assert gain == pytest.approx(0.18 * 4.0)


def test_compute_ev100():
    # log2(8^2 / (1/450)) + log2(4) = log2(28800) + 2.
    ev100 = exposure.compute_ev100(ISO_SPEED, SHUTTER_SPEED, APERTURE)
    assert ev100 == pytest.approx(np.log2(28800.0) + 2.0, abs=1e-4)


def test_compute_ev100_reference():
    # EV100 is zero for 1s, f/1.0 at ISO 100.
    assert exposure.compute_ev100(100.0, 1.0, 1.0) == pytest.approx(0.0)


def test_compute_ev100_missing_metadata():
    assert exposure.compute_ev100(None, SHUTTER_SPEED, APERTURE) is None
    assert exposure.compute_ev100(ISO_SPEED, 0.0, APERTURE) is None
    assert exposure.compute_ev100(ISO_SPEED, SHUTTER_SPEED, None) is None


def test_get_exposure_params_from_metadata():
    params = exposure.get_exposure_params(fake_raw())
    assert params.iso_speed == ISO_SPEED
    assert params.shutter_speed == SHUTTER_SPEED
    assert params.aperture == APERTURE
    assert params.ev100 == pytest.approx(exposure.compute_ev100(ISO_SPEED, SHUTTER_SPEED, APERTURE))
    assert params.mid_gray_target == exposure.DEFAULT_MID_GRAY
    assert params.metering_headroom_ev == exposure.DEFAULT_METERING_HEADROOM_EV
    assert params.exposure_offset_ev == 0.0
    assert params.gain == pytest.approx(1.0, rel=0.01)


def test_get_exposure_params_offset():
    params = exposure.get_exposure_params(fake_raw(), exposure_offset_ev=-1.5)
    assert params.exposure_offset_ev == -1.5
    assert params.gain == pytest.approx(exposure.compute_gain(0.18, 2.47, -1.5))


def test_get_exposure_params_without_metadata():
    params = exposure.get_exposure_params(FakeRaw(other=None))
    assert params.iso_speed is None
    assert params.ev100 is None
    assert params.gain == pytest.approx(1.0, rel=0.01)


def test_process_unit_gain_preserves_values():
    img_arr = np.array([[-0.2, 0.25, 1.0], [0.5, 1.5, 2.5]], dtype=np.float32)
    output_arr = Exposure().process(img_arr)
    # Negative values are clipped to zero; positive values, including values
    # above 1.0, are preserved without upper clipping.
    expected = np.array([[0.0, 0.25, 1.0], [0.5, 1.5, 2.5]], dtype=np.float32)
    assert np.allclose(output_arr, expected)


def test_process_gain():
    img_arr = np.array([[0.1, 0.2, 0.4], [0.8, 1.6, 3.2]], dtype=np.float32)
    params = ExposureParameters(gain=2.0)
    output_arr = Exposure(params).process(img_arr)
    assert np.allclose(output_arr, img_arr * 2.0)


def test_process_black_white_levels():
    # A non-trivial black/white point scales the values before the gain.
    params = ExposureParameters(gain=1.0, black_level=0.5, white_level=1.5)
    img_arr = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    output_arr = Exposure(params).process(img_arr)
    assert np.allclose(output_arr, [0.0, 0.5, 1.0])


def test_process_gamma_stage():
    # The gamma stage is only active when the parameter is set, e.g. by a
    # future trained model.
    img_arr = np.array([0.25, 0.5, 1.0], dtype=np.float32)
    params = ExposureParameters(gain=1.0, gamma=2.0)
    output_arr = Exposure(params).process(img_arr)
    assert np.allclose(output_arr, transform.gamma(img_arr, 2.0))


def test_process_s_curve_stage():
    img_arr = np.linspace(0.0, 1.0, 11, dtype=np.float32)
    params = ExposureParameters(gain=1.0, use_s_curve=True, contrast=2.0, slope=5.0, shift=0.5)
    output_arr = Exposure(params).process(img_arr)
    assert np.allclose(output_arr, transform.s_curve(img_arr, midpoint=0.5, slope=5.0, contrast=2.0))


def test_json_roundtrip():
    params = exposure.get_exposure_params(fake_raw(), exposure_offset_ev=0.3)
    restored = ExposureParameters.from_json(params.to_json())
    assert restored == params


def test_encode_output_gamma_linear():
    img_arr = np.array([[0.18, 0.5, 1.0], [-0.1, 1.5, 0.0]], dtype=np.float32)
    assert transform.encode_output_gamma(img_arr, "linear") is img_arr


def test_encode_output_gamma_srgb():
    img_arr = np.array([[-0.1, 0.18, 1.5]], dtype=np.float32)
    output_arr = transform.encode_output_gamma(img_arr, "srgb")
    assert np.allclose(output_arr, transform.srgb_oetf(np.maximum(img_arr, 0.0)))
    # A linear 0.18 mid-gray encodes to approximately 0.46.
    assert output_arr[0, 1] == pytest.approx(0.46, abs=0.01)


def test_encode_output_gamma_rec2020():
    img_arr = np.array([[-0.1, 0.18, 1.5]], dtype=np.float32)
    output_arr = transform.encode_output_gamma(img_arr, "rec2020")
    assert np.allclose(output_arr, transform.rec2020_oetf(np.maximum(img_arr, 0.0)))
    # A linear 0.18 mid-gray encodes to approximately 0.41.
    assert output_arr[0, 1] == pytest.approx(0.41, abs=0.01)


def test_encode_output_gamma_prophoto():
    img_arr = np.array([[-0.1, 0.18, 1.5]], dtype=np.float32)
    output_arr = transform.encode_output_gamma(img_arr, "prophoto")
    assert np.allclose(output_arr, transform.prophoto_oetf(np.maximum(img_arr, 0.0)))
    # A linear 0.18 mid-gray encodes to approximately 0.39.
    assert output_arr[0, 1] == pytest.approx(0.39, abs=0.01)


def test_encode_output_gamma_invalid():
    with pytest.raises(ValueError, match="Invalid output gamma"):
        transform.encode_output_gamma(np.ones(3), "rec709")
