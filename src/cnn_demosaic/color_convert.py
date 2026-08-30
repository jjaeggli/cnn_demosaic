# Module for color space conversion from camera RGB to an RGB output color
# space (sRGB, Rec.2020, or ProPhoto).
#
# This implements the standard dcraw-style conversion chain:
#
#   linear camera RGB -> white balance -> camera RGB -> linear output RGB
#
# Color matrix semantics (this matters!):
#
# The rawpy rgb_xyz_matrix is equivalent to LibRaw cam_xyz field. It is the
# camera specific matrix that turns XYZ color into camera primaries
# (XYZ -> camera RGB). See LibRaw's cam_xyz_coeff:
#
#   rgb_to_cam = cam_xyz @ rgb_to_xyz
#   rgb_to_cam is row-normalized so that output space white (1, 1, 1) maps to
#   camera white (1, 1, 1). The reciprocal of the un-normalized row sums are
#   the daylight white balance multipliers (pre_mul).
#   cam_to_rgb = inv(rgb_to_cam)
#
# White balance multipliers are applied to the camera RGB values before the
# color matrix, exactly as dcraw's scale_colors precedes convert_to_rgb.
#
# All matrices use the D65 white point convention, matching the XYZ (D65)
# camera matrices found in raw files. The sRGB and Rec.2020 matrices are
# computed directly from their published primaries with a D65 white. ProPhoto
# is natively referenced to D50, so the matrix below reuses the ProPhoto
# primary chromaticities recomputed against a D65 white to remain consistent
# with the D65 pipeline convention.
#
# Reference: https://www.numbercrunch.de/blog/2020/12/from-numbers-to-images-raw-image-processing-with-python/
# Reference: http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html

from dataclasses import dataclass

import numpy as np
import rawpy

srgb_to_xyz = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float32,
)

# ITU-R BT.2020 primaries with a D65 white.
rec2020_to_xyz = np.array(
    [
        [0.6369580, 0.1446169, 0.1688809],
        [0.2627002, 0.6779981, 0.0593017],
        [0.0000000, 0.0280727, 1.0609851],
    ],
    dtype=np.float32,
)

# ProPhoto (ROMM RGB) primaries with a D65 white (natively D50, see above).
prophoto_to_xyz = np.array(
    [
        [0.7721473, 0.1369327, 0.0413759],
        [0.2788236, 0.7210577, 0.0001187],
        [0.0000000, 0.0000000, 1.0890578],
    ],
    dtype=np.float32,
)

# This roughly matches the sRGB D65 matrix shown here:
# http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
xyz_to_srgb = np.linalg.inv(srgb_to_xyz)

OUTPUT_COLORSPACE_SRGB = "srgb"
OUTPUT_COLORSPACE_REC2020 = "rec2020"
OUTPUT_COLORSPACE_PROPHOTO = "prophoto"
DEFAULT_OUTPUT_COLORSPACE = OUTPUT_COLORSPACE_SRGB

# Registry of output color space names to their RGB -> XYZ (D65) matrices.
RGB_TO_XYZ = {
    OUTPUT_COLORSPACE_SRGB: srgb_to_xyz,
    OUTPUT_COLORSPACE_REC2020: rec2020_to_xyz,
    OUTPUT_COLORSPACE_PROPHOTO: prophoto_to_xyz,
}
VALID_OUTPUT_COLORSPACES = tuple(RGB_TO_XYZ)


@dataclass
class ColorTransformParameters:
    """Color conversion parameters extracted from a raw image.

    Attributes:
        xyz_to_cam: The 3x3 matrix from the raw file which converts XYZ (D65)
            values to camera RGB values.
        cam_to_rgb: The composite 3x3 matrix which converts white balanced
            camera RGB values to linear RGB values in the output color space.
        wb_multipliers: Per-channel white balance multipliers, normalized so
            that the green multiplier is 1.0.
        black_level_per_channel: Per-channel sensor black level in raw units.
        white_level: Sensor white level in raw units.
    """

    xyz_to_cam: np.ndarray
    cam_to_rgb: np.ndarray
    wb_multipliers: np.ndarray
    black_level_per_channel: np.ndarray
    white_level: float


def get_color_params(
    raw: rawpy.RawPy, output_colorspace: str = DEFAULT_OUTPUT_COLORSPACE
) -> ColorTransformParameters:
    """Extracts the color conversion parameters from an opened raw image.

    Args:
        raw: An opened rawpy.RawPy instance (or any object exposing the same
            color metadata attributes).
        output_colorspace: The target output color space, one of
            VALID_OUTPUT_COLORSPACES.

    Returns:
        A ColorTransformParameters instance.
    """
    if output_colorspace not in RGB_TO_XYZ:
        raise ValueError(
            f"Invalid output color space [{output_colorspace}], " +
            f"must be one of {VALID_OUTPUT_COLORSPACES}"
        )
    rgb_to_xyz = RGB_TO_XYZ[output_colorspace]

    # Despite the rawpy docstring, this matrix converts XYZ to camera RGB.
    xyz_to_cam = np.asarray(raw.rgb_xyz_matrix[:3], dtype=np.float32)
    if not np.any(xyz_to_cam):
        raise ValueError("The raw image does not contain a camera color matrix!")

    # Compose the output color space to camera RGB matrix and normalize each
    # camera channel row so that output space white maps to camera white, as
    # dcraw does in cam_xyz_coeff. The camera to output space matrix is the
    # inverse of this.
    rgb_to_cam = xyz_to_cam @ rgb_to_xyz
    row_sums = rgb_to_cam.sum(axis=1)
    if np.any(row_sums <= 1e-5):
        raise ValueError("The camera color matrix contains a degenerate channel row!")
    rgb_to_cam = rgb_to_cam / row_sums[:, np.newaxis]
    cam_to_rgb = np.linalg.inv(rgb_to_cam)

    camera_whitebalance = np.asarray(raw.camera_whitebalance, dtype=np.float32)[:3]
    if camera_whitebalance[1] == 0:
        raise ValueError("The camera green white balance multiplier is zero!")

    # Normalize the white balance multipliers so that green is 1.0.
    wb_multipliers = camera_whitebalance / camera_whitebalance[1]

    black_level_per_channel = np.asarray(raw.black_level_per_channel, dtype=np.float32)[:3]

    return ColorTransformParameters(
        xyz_to_cam=xyz_to_cam,
        cam_to_rgb=cam_to_rgb,
        wb_multipliers=wb_multipliers,
        black_level_per_channel=black_level_per_channel,
        white_level=float(raw.white_level),
    )


def linearize(img_arr, params: ColorTransformParameters, norm_min: float, norm_max: float):
    """Remaps min/max normalized image values to black/white level aware linear values.

    The demosaic model operates on image values normalized to (0.0, 1.0) using
    the image minimum and maximum. This function inverts that mapping and
    rescales the result using the sensor black and white levels, producing
    linear sensor values in the range (0.0, 1.0).

    Args:
        img_arr: An (H, W, 3) or (N, 3) array of normalized RGB values.
        params: The ColorTransformParameters for the image.
        norm_min: The image minimum which was used for normalization.
        norm_max: The image maximum which was used for normalization.

    Returns:
        An array of linear RGB values in the same shape as the input.
    """
    norm_range = norm_max - norm_min
    if norm_range <= 0.0:
        raise ValueError(f"Invalid normalization range: [{norm_min}, {norm_max}]")

    raw_arr = img_arr * norm_range + norm_min
    return (raw_arr - params.black_level_per_channel) / (
        params.white_level - params.black_level_per_channel
    )


def white_balance(img_arr, wb_multipliers):
    """Applies per-channel white balance multipliers to an RGB array."""
    return img_arr * np.asarray(wb_multipliers)


def camera_to_rgb(img_arr, params: ColorTransformParameters):
    """Converts white-balanced linear camera RGB values to the output color space.

    The input values are expected to be linear camera RGB values which have
    already been white balanced and linearized. The composite camera RGB to
    output color space matrix is applied per pixel. As with dcraw, any
    resulting negative values are clipped to zero. Upper values are not
    clipped, so that the full dynamic range is preserved for floating point
    output formats.

    Args:
        img_arr: An (H, W, 3) or (N, 3) array of white balanced camera RGB values.
        params: The ColorTransformParameters for the image.

    Returns:
        An array of linear RGB values in the output color space, in the same
        shape as the input.
    """
    output_arr = np.einsum("ij,...j->...i", params.cam_to_rgb, np.asarray(img_arr))
    return np.maximum(output_arr, 0.0)
