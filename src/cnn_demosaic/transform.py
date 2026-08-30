import numpy as np
import tensorflow as tf

OUTPUT_GAMMA_LINEAR = "linear"
OUTPUT_GAMMA_SRGB = "srgb"
OUTPUT_GAMMA_REC2020 = "rec2020"
OUTPUT_GAMMA_PROPHOTO = "prophoto"


def normalize_arr(array_input):
    """Fits the image to the range (0.0, 1.0)."""
    arr_min = np.min(array_input)
    arr_max = np.max(array_input)
    return (array_input - arr_min) / (arr_max - arr_min)


def adj_levels(image_array, in_min, in_max, out_min=0.0, out_max=1.0, clip=False):
    """Scales the input levels to the output levels."""
    dyn_range = in_max - in_min
    dyn_range = dyn_range if dyn_range != 0.0 else 0.00001

    scale_ratio = (out_max - out_min) / dyn_range
    adjusted_image = (image_array - in_min) * scale_ratio + out_min

    if clip:
        adjusted_image = np.clip(adjusted_image, out_min, out_max)
    return adjusted_image


def adj_levels_per_tile_fn(image_array):
    """Per-tile function normalizes the tile to a standard range for processing.

    The resulting post-processing function denormalizes the tile back to the
    original levels. Performing this simple process greatly reduces banding and
    striping artifacts in low value areas of the image. This does not appear to
    require specialized model training.
    """
    process_min, process_max = 0.1, 0.9
    orig_min = image_array.min()
    orig_max = image_array.max()
    output_array = adj_levels(image_array, orig_min, orig_max, process_min, process_max)

    def post_fn(input_array):
        return adj_levels(input_array, process_min, process_max, orig_min, orig_max)

    return output_array, post_fn


def s_curve(image_array: np.ndarray, midpoint=0.5, slope=1.0, contrast=1.0):
    """Applies an s-curve function the the array."""
    if not isinstance(image_array, np.ndarray):
        raise ValueError("Input must be a NumPy array.")
    output_array = (image_array - midpoint) * contrast
    output_array = 1 / (1 + np.exp(-slope * output_array))
    output_array = np.clip(output_array, 0, 1)
    return output_array


def gamma(image, gamma=1.0):
    if gamma <= 0:
        raise ValueError("Gamma must be greater than zero")
    return np.power(image, gamma)


def srgb_oetf(img_arr):
    """Applies the sRGB opto-electronic transfer function (linear to encoded).

    This converts linear sRGB values to encoded (display referred) sRGB values
    using the standard piecewise transfer function defined in IEC 61966-2-1.
    """
    img_arr = np.asarray(img_arr)
    # The power branch requires non-negative input values.
    power = 1.055 * np.power(np.maximum(img_arr, 0.0), 1 / 2.4) - 0.055
    return np.where(img_arr <= 0.0031308, 12.92 * img_arr, power)


def rec2020_oetf(img_arr):
    """Applies the Rec.2020 (ITU-R BT.2020) transfer function (linear to encoded).

    This converts linear Rec.2020 values to encoded (display referred) values
    using the piecewise transfer function defined in ITU-R BT.2020.
    """
    img_arr = np.asarray(img_arr)
    alpha = 1.09929682680944
    beta = 0.018053968510807
    # The power branch requires non-negative input values.
    power = alpha * np.power(np.maximum(img_arr, 0.0), 0.45) - (alpha - 1.0)
    return np.where(img_arr < beta, 4.5 * img_arr, power)


def prophoto_oetf(img_arr):
    """Applies the ProPhoto RGB (ROMM) transfer characteristic (linear to encoded).

    This converts linear ProPhoto values to encoded values using the ROMM
    transfer characteristic: a pure 1/1.8 power function with a linear segment
    below 1/512 to avoid an infinite slope at zero.
    """
    img_arr = np.asarray(img_arr)
    # The power branch requires non-negative input values.
    power = np.power(np.maximum(img_arr, 0.0), 1 / 1.8)
    return np.where(img_arr < 1 / 512, 16.0 * img_arr, power)


GAMMA_ENCODERS = {
    OUTPUT_GAMMA_SRGB: srgb_oetf,
    OUTPUT_GAMMA_REC2020: rec2020_oetf,
    OUTPUT_GAMMA_PROPHOTO: prophoto_oetf,
}


def encode_output_gamma(img_arr, output_gamma=OUTPUT_GAMMA_LINEAR):
    """Encodes linear image values with the specified output transfer function.

    OUTPUT_GAMMA_LINEAR returns the scene-referred values unchanged, which is
    the appropriate encoding for floating point formats such as EXR. The
    remaining encodings apply the transfer function matching the output color
    space (sRGB, Rec.2020, or ProPhoto), producing display-referred encoded
    values.

    Args:
        img_arr: An array of linear values.
        output_gamma: The output encoding, one of the OUTPUT_GAMMA_* constants.

    Returns:
        An array of encoded values in the same shape as the input.
    """
    if output_gamma == OUTPUT_GAMMA_LINEAR:
        return img_arr
    encoder = GAMMA_ENCODERS.get(output_gamma)
    if encoder is None:
        raise ValueError(
            f"Invalid output gamma [{output_gamma}], must be " +
            f"one of {(OUTPUT_GAMMA_LINEAR,) + tuple(GAMMA_ENCODERS)}"
        )
    return encoder(np.maximum(np.asarray(img_arr), 0.0))


@tf.function()
def tf_rgb_luma_fn(rgb_ch):
    # Convert the image to a luma channel using the following ratios:
    # 0.299R + 0.587G + 0.114B
    return rgb_ch[:, 0] * 0.299 + rgb_ch[:, 1] * 0.587 + rgb_ch[:, 2] * 0.114


@tf.function()
def tf_s_curve_fn(img_arr, offset, contrast, slope):
    """Applies the s-curve function in a TensorFlow context."""
    output = (img_arr - offset) * contrast
    return 1 / (1 + tf.math.exp(-1 * slope * output))


@tf.function()
def tf_levels_fn(img_arr, in_min, in_max):
    """Applies the levels function in a TensorFlow context."""
    dyn_range = in_max - in_min
    dyn_range = dyn_range + 1e-6

    scale_ratio = 1.0 / dyn_range
    return (img_arr - in_min) * scale_ratio


@tf.function()
def tf_clip_fn(img_arr):
    return tf.clip_by_value(img_arr, clip_value_min=0.0, clip_value_max=1.0)
