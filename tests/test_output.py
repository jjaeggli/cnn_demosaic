import numpy as np
import pytest
import tifffile

from cnn_demosaic import output


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


def test_write_exr_preserves_range(tmp_path):
    # The EXR container is floating point, so out of range values, including
    # clipped highlights above 1.0, must be preserved.
    path = tmp_path / "out.exr"
    arr = np.array([[[-0.5, 0.5, 1.5]]], dtype=np.float32)
    output.write_exr(arr, path)

    import pyexr
    loaded = pyexr.read(str(path))[:, :, :3]
    assert np.allclose(loaded, arr)
