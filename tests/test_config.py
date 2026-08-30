import pytest

from cnn_demosaic import config


@pytest.fixture()
def raw_file(tmp_path):
    raw_path = tmp_path / "input.raf"
    raw_path.write_bytes(b"fake raw data")
    return raw_path


def make_config(raw_path, **kwargs):
    return config.Config(str(raw_path), **kwargs)


def test_valid_output_colorspaces(raw_file):
    for colorspace in (config.OUTPUT_COLORSPACE_SRGB, config.OUTPUT_COLORSPACE_REC2020,
                       config.OUTPUT_COLORSPACE_PROPHOTO):
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
