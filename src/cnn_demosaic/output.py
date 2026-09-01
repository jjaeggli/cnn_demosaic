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


def write_tiff(output_arr: np.ndarray, output_path, icc_profile: bytes | None = None):
    output_arr = np.clip(output_arr, 0.0, 1.0)
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
        raise ValueError(
            f"Unsupported array shape for TIFF export: {output_arr.shape}. Expected (H, W, 3) or (H, W)."
        )

    writer_kwargs = {}
    # A color profile only applies to the RGB photometric mode. The grayscale
    # mode would require a matching grayscale profile.
    if photometric_mode == "rgb" and icc_profile is not None:
        writer_kwargs["iccprofile"] = icc_profile

    tifffile.imwrite(
        output_path,
        output_arr,
        photometric=photometric_mode,
        compression="zlib",
        **writer_kwargs,
    )


def write_exr(output_arr: np.ndarray, output_path, icc_profile: bytes | None = None):
    # The icc_profile argument is accepted for output handler compatibility
    # but ignored: the EXR format has no standard ICC profile support. The
    # equivalent addition would be a chromaticities header attribute.
    # Determine the number of channels based on the array shape.
    # The array is expected to be a 2D image, either (H, W, 3) for color
    # or (H, W) for monochrome.
    if output_arr.ndim == 3 and output_arr.shape[-1] == 3:
        channels_spec = ["R", "G", "B"]
    elif output_arr.ndim == 2:
        # The default channel name for a monochrome channel in the library is `Z` but Gimp always
        # complains about this.
        channels_spec = ["Y"]
    else:
        raise ValueError(
            f"Unsupported array shape for EXR export: {output_arr.shape}. Expected (H, W, 3) or (H, W)."
        )

    pyexr.write(
        str(output_path),
        output_arr,
        precision=pyexr.HALF,
        channel_names=channels_spec,
    )


IMG_FORMATS = {EXR_FORMAT: (EXR_SUFFIX, write_exr), TIFF_FORMAT: (TIFF_SUFFIX, write_tiff)}
