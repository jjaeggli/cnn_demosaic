import numpy as np
import pytest

from cnn_demosaic import color_convert, transform

# A real camera color matrix (Sony ILCE-6300) from the numbercrunch.de raw
# processing reference. Despite the rawpy docstring, this matrix converts XYZ
# to camera RGB.
XYZ_TO_CAM = np.array(
    [
        [0.5973, -0.1695, -0.0419],
        [-0.3826, 1.1797, 0.2293],
        [-0.0639, 0.1398, 0.5789],
    ],
    dtype=np.float32,
)

# The green-normalized daylight white balance multipliers implied by the
# matrix above (1 / row sums of xyz_to_cam @ srgb_to_xyz).
DAYLIGHT_WB = np.array([3.0225, 1.0, 1.5023], dtype=np.float32)

CAMERA_WHITEBALANCE = [1.895, 1.0, 1.558, 1.0]
BLACK_LEVEL_PER_CHANNEL = [1024, 1024, 1024, 1024]
WHITE_LEVEL = 16383


class FakeRaw:
    """A stand-in for rawpy.RawPy which exposes the color metadata attributes."""

    def __init__(self):
        self.rgb_xyz_matrix = np.zeros((4, 3), dtype=np.float32)
        self.rgb_xyz_matrix[:3] = XYZ_TO_CAM
        self.camera_whitebalance = CAMERA_WHITEBALANCE
        self.black_level_per_channel = BLACK_LEVEL_PER_CHANNEL
        self.white_level = WHITE_LEVEL


@pytest.fixture()
def color_params():
    return color_convert.get_color_params(FakeRaw())


@pytest.fixture()
def macbeth():
    """A set of linear sRGB reference colors (Macbeth chart approximations)."""
    return np.array(
        [
            [0.441, 0.276, 0.132],
            [0.727, 0.522, 0.377],
            [0.623, 0.598, 0.542],
            [0.550, 0.750, 0.350],
            [0.350, 0.420, 0.650],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )


def test_matrix_constants_roundtrip():
    identity = color_convert.xyz_to_srgb @ color_convert.srgb_to_xyz
    assert np.allclose(identity, np.eye(3), atol=1e-5)


def test_get_color_params(color_params):
    assert np.allclose(color_params.xyz_to_cam, XYZ_TO_CAM)
    assert color_params.white_level == WHITE_LEVEL
    assert np.allclose(color_params.black_level_per_channel, [1024, 1024, 1024])
    # The white balance multipliers should be normalized so that green is 1.0.
    assert color_params.wb_multipliers[1] == pytest.approx(1.0)
    assert np.allclose(
        color_params.wb_multipliers,
        np.asarray(CAMERA_WHITEBALANCE[:3]) / CAMERA_WHITEBALANCE[1],
    )


def test_cam_to_srgb_composition(color_params):
    # The camera to sRGB matrix must be the row-normalized dcraw composition:
    # inv(normalize_rows(xyz_to_cam @ srgb_to_xyz)).
    srgb_to_cam = XYZ_TO_CAM @ color_convert.srgb_to_xyz
    srgb_to_cam = srgb_to_cam / srgb_to_cam.sum(axis=1)[:, np.newaxis]
    assert np.allclose(color_params.cam_to_srgb, np.linalg.inv(srgb_to_cam), atol=1e-6)


def test_cam_to_srgb_white_preservation(color_params):
    # The row normalization ensures that sRGB white maps to camera white, so
    # the inverse maps camera white back to sRGB white.
    white = np.ones(3, dtype=np.float32)
    assert np.allclose(color_params.cam_to_srgb @ white, white, atol=1e-5)


def test_matrix_implies_daylight_whitebalance(color_params):
    # The reciprocal of the un-normalized composition row sums are the
    # daylight white balance multipliers, as in dcraw's cam_xyz_coeff.
    srgb_to_cam = XYZ_TO_CAM @ color_convert.srgb_to_xyz
    pre_mul = 1.0 / srgb_to_cam.sum(axis=1)
    wb_daylight = pre_mul / pre_mul[1]
    assert np.allclose(wb_daylight, DAYLIGHT_WB, atol=1e-3)


def test_get_color_params_missing_matrix():
    raw = FakeRaw()
    raw.rgb_xyz_matrix = np.zeros((4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="camera color matrix"):
        color_convert.get_color_params(raw)


def test_get_color_params_degenerate_matrix():
    raw = FakeRaw()
    raw.rgb_xyz_matrix[1] = 0.0
    with pytest.raises(ValueError, match="degenerate channel row"):
        color_convert.get_color_params(raw)


def test_get_color_params_zero_green():
    raw = FakeRaw()
    raw.camera_whitebalance = [1.9, 0.0, 1.5, 0.0]
    with pytest.raises(ValueError, match="green white balance multiplier is zero"):
        color_convert.get_color_params(raw)


def test_camera_to_srgb_matrix_convention(color_params):
    # The composite matrix must be applied as M @ v per pixel, and not as the
    # transpose, which is what a naive np.dot(arr, M) would produce.
    img_arr = np.array(
        [
            [0.4, 0.3, 0.2],
            [0.9, 0.5, 0.1],
        ],
        dtype=np.float32,
    )
    expected = np.maximum(np.stack([color_params.cam_to_srgb @ pixel for pixel in img_arr]), 0.0)
    output = color_convert.camera_to_srgb(img_arr, color_params)
    assert np.allclose(output, expected)


def test_synthetic_roundtrip(color_params, macbeth):
    # Camera values constructed as the inverse of the full pipeline must be
    # recovered exactly by white balancing and color conversion.
    wb = color_params.wb_multipliers
    camera_raw = (macbeth @ np.linalg.inv(color_params.cam_to_srgb).T) / wb

    output = color_convert.camera_to_srgb(
        color_convert.white_balance(camera_raw, wb),
        color_params,
    )
    assert np.allclose(output, macbeth, atol=1e-5)


def test_daylight_colors_recover_srgb(color_params, macbeth):
    # Camera values of colors under daylight, white balanced with the
    # multipliers implied by the color matrix, must convert back to the
    # original sRGB colors. This is the dcraw daylight processing chain.
    srgb_to_cam = XYZ_TO_CAM @ color_convert.srgb_to_xyz
    row_sums = srgb_to_cam.sum(axis=1)
    wb_daylight = row_sums[1] / row_sums
    camera_raw = (macbeth @ srgb_to_cam.T) / row_sums[1]

    output = color_convert.camera_to_srgb(
        color_convert.white_balance(camera_raw, wb_daylight),
        color_params,
    )
    assert np.allclose(output, macbeth, atol=1e-4)


def test_neutral_stays_neutral(color_params):
    # A neutral patch photographed under the as-shot illuminant produces
    # camera RGB values which are inversely proportional to the white balance
    # multipliers. White balancing and converting should keep it neutral.
    neutral_value = 0.7
    camera_response = (neutral_value / color_params.wb_multipliers).astype(np.float32)

    output = color_convert.camera_to_srgb(
        color_convert.white_balance(camera_response, color_params.wb_multipliers),
        color_params,
    )
    assert output[0] == pytest.approx(output[1], abs=1e-4)
    assert output[1] == pytest.approx(output[2], abs=1e-4)


def test_white_balance_shapes(color_params):
    wb = color_params.wb_multipliers
    img_hwc = np.random.random((6, 7, 3)).astype(np.float32)
    output_hwc = color_convert.white_balance(img_hwc, wb)
    assert output_hwc.shape == (6, 7, 3)
    assert np.allclose(output_hwc[2, 3], img_hwc[2, 3] * wb)

    img_n3 = img_hwc.reshape(-1, 3)
    output_n3 = color_convert.white_balance(img_n3, wb)
    assert np.allclose(output_hwc.reshape(-1, 3), output_n3)


def test_linearize(color_params):
    norm_min, norm_max = 256.0, 16000.0
    img_arr = np.random.random((4, 5, 3)).astype(np.float32)

    output = color_convert.linearize(img_arr, color_params, norm_min, norm_max)
    raw_arr = img_arr * (norm_max - norm_min) + norm_min
    expected = (raw_arr - 1024.0) / (WHITE_LEVEL - 1024.0)
    assert np.allclose(output, expected)


def test_linearize_identity_range(color_params):
    # When the normalization range matches the sensor range exactly, the
    # linearization is an identity mapping.
    norm_min, norm_max = 1024.0, float(WHITE_LEVEL)
    img_arr = np.random.random((4, 5, 3)).astype(np.float32)
    output = color_convert.linearize(img_arr, color_params, norm_min, norm_max)
    assert np.allclose(output, img_arr)


def test_linearize_invalid_range(color_params):
    img_arr = np.random.random((4, 5, 3)).astype(np.float32)
    with pytest.raises(ValueError, match="normalization range"):
        color_convert.linearize(img_arr, color_params, 1.0, 1.0)


def test_camera_to_srgb_clips_negative():
    # Out of gamut colors produce negative values which must be clipped to
    # zero, while positive values are preserved without upper clipping.
    params = color_convert.ColorTransformParameters(
        xyz_to_cam=np.eye(3, dtype=np.float32),
        cam_to_srgb=color_convert.xyz_to_srgb,
        wb_multipliers=np.ones(3, dtype=np.float32),
        black_level_per_channel=np.zeros(3, dtype=np.float32),
        white_level=1.0,
    )
    img_arr = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    output = color_convert.camera_to_srgb(img_arr, params)
    raw_output = color_convert.xyz_to_srgb @ img_arr[0]
    assert output[0, 0] == pytest.approx(raw_output[0])
    assert output[0, 0] > 1.0
    assert output[0, 1] == 0.0
    assert raw_output[1] < 0.0


def test_srgb_oetf():
    assert transform.srgb_oetf(0.0) == pytest.approx(0.0)
    assert transform.srgb_oetf(1.0) == pytest.approx(1.0)
    # The linear segment breakpoint. The official 0.04045 value is rounded;
    # 12.92 * 0.0031308 differs from it by less than 1e-7.
    assert transform.srgb_oetf(0.0031308) == pytest.approx(0.04045, abs=1e-6)
    assert transform.srgb_oetf(0.5) == pytest.approx(0.7353569830524495)

    # The function must be monotonically increasing over the full range.
    x = np.linspace(0.0, 1.0, 1000)
    y = transform.srgb_oetf(x)
    assert np.all(np.diff(y) >= 0.0)
