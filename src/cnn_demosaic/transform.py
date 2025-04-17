import numpy as np


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
