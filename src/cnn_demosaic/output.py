import numpy as np
import pyexr
import tifffile


EXR_FORMAT = "exr"
TIFF_FORMAT = "tiff"
EXR_SUFFIX = ".exr"
TIFF_SUFFIX = ".tiff"

# Type aliases for image format types.
# type OutputHandler = Callable[[np.ndarray, str]]
# type Format = tuple[OutputHandler, pathlib.Path]


def write_tiff(output_arr: np.ndarray, output_path):
    output_arr = (output_arr * 65535).astype(np.uint16)
    # Determine the number of channels based on the array shape.
    # The array is expected to be a 2D image, either (H, W, 3) for color
    # or (H, W) for monochrome.
    if output_arr.ndim == 3 and output_arr.shape[-1] == 3:
        # 3-channel color image (H, W, 3)
        photometric_mode = "rgb"
    elif output_arr.ndim == 2:
        photometric_mode = "minisblack"
    else:
        raise ValueError(f"Unsupported array shape for TIFF export: {output_arr.shape}. Expected (H, W, 3) or (H, W).")

    tifffile.imwrite(output_path, output_arr, photometric=photometric_mode, compression="zlib")


def write_exr(output_arr: np.ndarray, output_path):
    # Determine the number of channels based on the array shape.
    # The array is expected to be a 2D image, either (H, W, 3) for color
    # or (H, W) for monochrome.
    if output_arr.ndim == 3 and output_arr.shape[-1] == 3:
        channels_spec = ['R', 'G', 'B']
    elif output_arr.ndim == 2:
        # The default channel name for a monochrome channel in the library is `Z` but Gimp always
        # complains about this.
        channels_spec = ['Y']
    else:
        raise ValueError(f"Unsupported array shape for EXR export: {output_arr.shape}. Expected (H, W, 3) or (H, W).")

    pyexr.write(
        str(output_path),
        output_arr,
        precision=pyexr.HALF,
        channel_names=channels_spec,
    )


IMG_FORMATS = {EXR_FORMAT: (EXR_SUFFIX, write_exr), TIFF_FORMAT: (TIFF_SUFFIX, write_tiff)}
