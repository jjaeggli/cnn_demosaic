import numpy as np
import pyexr
import tifffile

from typing import Callable


EXR_FORMAT = "exr"
TIFF_FORMAT = "tiff"
EXR_SUFFIX = ".exr"
TIFF_SUFFIX = ".tiff"

# Type aliases for image format types.
# type OutputHandler = Callable[[np.ndarray, str]]
# type Format = tuple[OutputHandler, pathlib.Path]


def write_tiff(output_arr: np.ndarray, output_path):
    output_arr = (output_arr * 65535).astype(np.uint16)
    tifffile.imwrite(output_path, output_arr, photometric="rgb", compression="zlib")


def write_exr(output_arr: np.ndarray, output_path):
    pyexr.write(
        str(output_path),
        output_arr,
        precision=pyexr.HALF,
    )


IMG_FORMATS = {EXR_FORMAT: (EXR_SUFFIX, write_exr), TIFF_FORMAT: (TIFF_SUFFIX, write_tiff)}
