import io
import struct

import numpy as np
import pytest
import tifffile
from PIL import Image, ImageCms

from cnn_demosaic import icc, output

ICC_COMBINATIONS = sorted(
    (output_linear, output_colorspace)
    for output_linear in (False, True)
    for output_colorspace in (
        icc.OUTPUT_COLORSPACE_SRGB,
        icc.OUTPUT_COLORSPACE_REC2020,
        icc.OUTPUT_COLORSPACE_PROPHOTO,
    )
)

# Native white points of the color spaces in XYZ (Y = 1.0).
SPACE_WHITE_POINTS = {
    icc.OUTPUT_COLORSPACE_SRGB: (0.9505, 1.0, 1.0891),
    icc.OUTPUT_COLORSPACE_REC2020: (0.9505, 1.0, 1.0891),
    icc.OUTPUT_COLORSPACE_PROPHOTO: (0.9642, 1.0, 0.8249),
}

# The ICC PCS illuminant (D50) that the Bradford-adapted colorant columns sum to.
PCS_ILLUMINANT = (0.9642, 1.0, 0.8249)


def parse_icc_tags(profile_bytes):
    count = struct.unpack(">I", profile_bytes[128:132])[0]
    tags = {}
    for i in range(count):
        offset = 132 + 12 * i
        signature = profile_bytes[offset : offset + 4].decode("ascii")
        tag_offset, tag_size = struct.unpack(">II", profile_bytes[offset + 4 : offset + 12])
        tags[signature] = profile_bytes[tag_offset : tag_offset + tag_size]
    return tags


def tag_xyz_values(tag_data):
    return tuple(value / 65536 for value in struct.unpack(">3i", tag_data[8:20]))


def test_write_tiff_clamps(tmp_path):
    # Values outside of the 16-bit integer range must be clamped, not wrapped.
    path = tmp_path / "out.tiff"
    arr = np.array([[[-0.5, 0.5, 1.5]]], dtype=np.float32)
    output.write_tiff(arr, path)

    loaded = tifffile.imread(path)
    assert loaded[0, 0, 0] == 0
    assert loaded[0, 0, 1] == pytest.approx(32767)
    assert loaded[0, 0, 2] == 65535


def test_write_tiff_monochrome(tmp_path):
    path = tmp_path / "out.tiff"
    arr = np.array([[0.0, 0.25, 1.0]], dtype=np.float32)
    output.write_tiff(arr, path)

    loaded = tifffile.imread(path)
    assert loaded.shape == (1, 3)
    assert loaded[0, 2] == 65535


@pytest.mark.parametrize(("output_linear", "output_colorspace"), ICC_COMBINATIONS)
def test_write_tiff_embeds_icc_profile(tmp_path, output_linear, output_colorspace):
    profile = icc.get_icc_profile(output_linear, output_colorspace)
    path = tmp_path / "out.tiff"
    arr = np.array([[[-0.5, 0.5, 1.5]]], dtype=np.float32)
    output.write_tiff(arr, path, icc_profile=profile)

    with tifffile.TiffFile(path) as tif:
        assert tif.pages[0].tags["InterColorProfile"].value == profile
    with Image.open(path) as img:
        assert img.info.get("icc_profile") == profile


def test_write_tiff_monochrome_ignores_icc_profile(tmp_path):
    profile = icc.get_icc_profile(False, icc.OUTPUT_COLORSPACE_SRGB)
    path = tmp_path / "out.tiff"
    arr = np.array([[0.0, 0.25, 1.0]], dtype=np.float32)
    output.write_tiff(arr, path, icc_profile=profile)

    with tifffile.TiffFile(path) as tif:
        assert "InterColorProfile" not in tif.pages[0].tags


def test_write_exr_ignores_icc_profile(tmp_path):
    # The EXR writer must accept the output handler signature, even though
    # the format has no standard ICC profile support.
    path = tmp_path / "out.exr"
    arr = np.array([[[-0.5, 0.5, 1.5]]], dtype=np.float32)
    output.write_exr(arr, path, icc_profile=b"\x00\x01\x02\x03")

    import pyexr

    loaded = pyexr.read(str(path))[:, :, :3]
    assert np.allclose(loaded, arr)


@pytest.mark.parametrize(("output_linear", "output_colorspace"), ICC_COMBINATIONS)
def test_icc_profile_matches_output_encoding(output_linear, output_colorspace):
    profile = icc.get_icc_profile(output_linear, output_colorspace)
    tags = parse_icc_tags(profile)

    white_point = tag_xyz_values(tags["wtpt"])
    for value, expected in zip(white_point, SPACE_WHITE_POINTS[output_colorspace]):
        assert value == pytest.approx(expected, abs=0.002)

    # The linear profiles must describe the same primaries as the
    # display-referred profiles bundled from the Compact-ICC-Profiles
    # collection, which provides an independent reference for the colorant
    # matrices.
    reference_tags = parse_icc_tags(icc.get_icc_profile(False, output_colorspace))
    for channel in "rgb":
        colorant = tag_xyz_values(tags[f"{channel}XYZ"])
        reference = tag_xyz_values(reference_tags[f"{channel}XYZ"])
        for value, expected in zip(colorant, reference):
            assert value == pytest.approx(expected, abs=0.001)

    # The colorant columns must sum to the D50 PCS illuminant.
    colorants = [tag_xyz_values(tags[f"{channel}XYZ"]) for channel in "rgb"]
    sums = tuple(sum(colorant[i] for colorant in colorants) for i in range(3))
    for value, expected in zip(sums, PCS_ILLUMINANT):
        assert value == pytest.approx(expected, abs=0.002)

    # A linear output encoding requires an identity TRC (curv count 0).
    # A display-referred encoding requires a monotonic sampled curve.
    for channel in "rgb":
        count = struct.unpack(">I", tags[f"{channel}TRC"][8:12])[0]
        if output_linear:
            assert count == 0
        else:
            assert count > 1
            curve = struct.unpack(f">{count}H", tags[f"{channel}TRC"][12 : 12 + 2 * count])
            assert curve[0] == 0
            assert curve[-1] == 65535
            assert list(curve) == sorted(curve)


@pytest.mark.parametrize(("output_linear", "output_colorspace"), ICC_COMBINATIONS)
def test_icc_profile_loads_in_littlecms(output_linear, output_colorspace):
    profile = icc.get_icc_profile(output_linear, output_colorspace)
    assert profile is not None
    loaded = ImageCms.ImageCmsProfile(io.BytesIO(profile))
    assert ImageCms.getProfileDescription(loaded).strip()


@pytest.mark.parametrize(
    "output_colorspace",
    (
        icc.OUTPUT_COLORSPACE_SRGB,
        icc.OUTPUT_COLORSPACE_REC2020,
        icc.OUTPUT_COLORSPACE_PROPHOTO,
    ),
)
def test_icc_profile_colorimetry_in_littlecms(output_colorspace):
    # Transforming through the CMM from the linear profile to the
    # display-referred profile must reproduce the color space OETF: the
    # linear profile supplies identity decoding and the primaries must
    # round-trip exactly.
    from cnn_demosaic import transform

    oetf = {
        icc.OUTPUT_COLORSPACE_SRGB: transform.srgb_oetf,
        icc.OUTPUT_COLORSPACE_REC2020: transform.rec2020_oetf,
        icc.OUTPUT_COLORSPACE_PROPHOTO: transform.prophoto_oetf,
    }[output_colorspace]

    linear_profile = icc.get_icc_profile(True, output_colorspace)
    display_profile = icc.get_icc_profile(False, output_colorspace)
    assert linear_profile is not None
    assert display_profile is not None
    transform_cmm = ImageCms.buildTransform(
        ImageCms.ImageCmsProfile(io.BytesIO(linear_profile)),
        ImageCms.ImageCmsProfile(io.BytesIO(display_profile)),
        "RGB",
        "RGB",
        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
    )

    for channel_index, channel in enumerate("rgb"):
        image = Image.new("RGB", (1, 1))
        image.putpixel((0, 0), tuple(255 if i == channel_index else 0 for i in range(3)))
        result = ImageCms.applyTransform(image, transform_cmm).getpixel((0, 0))
        assert result == tuple(255 if i == channel_index else 0 for i in range(3)), channel

    ramp = Image.new("RGB", (256, 1))
    ramp.putdata([(v, v, v) for v in range(256)])
    gray = np.asarray(ImageCms.applyTransform(ramp, transform_cmm), dtype=np.int16)[0]
    expected = np.round(np.clip(oetf(np.arange(256) / 255.0), 0, 1) * 255)
    assert (gray.max(axis=1) - gray.min(axis=1)).max() <= 1
    assert np.abs(gray[:, 1] - expected).max() <= 2


def test_get_icc_profile_unknown_colorspace():
    assert icc.get_icc_profile(False, "adobergb") is None


def test_get_icc_profile_known_combinations():
    for _, colorspace in ICC_COMBINATIONS:
        for output_linear in (False, True):
            profile = icc.get_icc_profile(output_linear, colorspace)
            assert isinstance(profile, bytes)
            assert profile[36:40] == b"acsp"


def test_write_exr_preserves_range(tmp_path):
    # The EXR container is floating point, so out of range values, including
    # clipped highlights above 1.0, must be preserved.
    path = tmp_path / "out.exr"
    arr = np.array([[[-0.5, 0.5, 1.5]]], dtype=np.float32)
    output.write_exr(arr, path)

    import pyexr

    loaded = pyexr.read(str(path))[:, :, :3]
    assert np.allclose(loaded, arr)
