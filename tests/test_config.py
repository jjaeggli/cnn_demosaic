import numpy as np
import pytest
import tifffile

from cnn_demosaic import config, icc


@pytest.fixture()
def raw_file(tmp_path):
    raw_path = tmp_path / "input.raf"
    raw_path.write_bytes(b"fake raw data")
    return raw_path


def make_config(raw_path, **kwargs):
    return config.Config(str(raw_path), **kwargs)


def test_valid_output_colorspaces(raw_file):
    for colorspace in (
        config.OUTPUT_COLORSPACE_SRGB,
        config.OUTPUT_COLORSPACE_REC2020,
        config.OUTPUT_COLORSPACE_PROPHOTO,
    ):
        cfg = make_config(raw_file, output_colorspace=colorspace)
        assert cfg.output_colorspace in cfg.VALID_OUTPUT_COLORSPACES


def test_invalid_output_colorspace(raw_file):
    cfg = make_config(raw_file, output_colorspace="adobergb")
    with pytest.raises(ValueError, match="Invalid output color space"):
        cfg.validate_config()


def test_model_pipeline_requires_srgb_output(raw_file):
    cfg = make_config(
        raw_file,
        color_pipeline=config.COLOR_PIPELINE_MODEL,
        output_colorspace=config.OUTPUT_COLORSPACE_REC2020,
    )
    with pytest.raises(ValueError, match="only supports 'srgb'"):
        cfg.validate_config()


def test_model_pipeline_allows_srgb_output(raw_file):
    cfg = make_config(
        raw_file,
        color_pipeline=config.COLOR_PIPELINE_MODEL,
        output_colorspace=config.OUTPUT_COLORSPACE_SRGB,
    )
    assert cfg.output_colorspace == config.OUTPUT_COLORSPACE_SRGB


def test_default_output_colorspace(raw_file):
    cfg = make_config(raw_file)
    assert cfg.output_colorspace == config.DEFAULT_OUTPUT_COLORSPACE


def test_default_output_encoding_is_display_referred(raw_file):
    cfg = make_config(raw_file)
    assert cfg.output_linear is False
    assert cfg.output_gamma == cfg.output_colorspace


def test_output_gamma_linear_flag(raw_file):
    cfg = make_config(raw_file, output_linear=True)
    assert cfg.output_gamma == config.OUTPUT_GAMMA_LINEAR


@pytest.mark.parametrize(
    "colorspace",
    (
        config.OUTPUT_COLORSPACE_SRGB,
        config.OUTPUT_COLORSPACE_REC2020,
        config.OUTPUT_COLORSPACE_PROPHOTO,
    ),
)
def test_output_gamma_matches_colorspace(raw_file, colorspace):
    cfg = make_config(raw_file, output_colorspace=colorspace)
    assert cfg.output_gamma == colorspace


def test_output_handler_embeds_icc_profile(raw_file, tmp_path):
    output_path = tmp_path / "out.tiff"
    cfg = make_config(raw_file, format="tiff", output_filename=str(output_path))
    cfg.output_handler(np.full((2, 2, 3), 0.5, dtype=np.float32))

    expected_profile = icc.get_icc_profile(False, config.DEFAULT_OUTPUT_COLORSPACE)
    with tifffile.TiffFile(output_path) as tif:
        assert tif.pages[0].tags["InterColorProfile"].value == expected_profile
