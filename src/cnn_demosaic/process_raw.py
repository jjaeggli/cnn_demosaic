# Module which performs raw processing and provides an entry point for user actions.

import argparse
import logging
import math
import numpy as np
import pathlib
import rawpy
import tensorflow as tf

from cnn_demosaic import color_model
from cnn_demosaic import exposure_model
from cnn_demosaic.color import Color
from cnn_demosaic.demosaic import Demosaic
from cnn_demosaic.exposure import Exposure
from cnn_demosaic import model
from cnn_demosaic import output
from cnn_demosaic import transform
from importlib import resources
from tensorflow import keras


WEIGHTS_MODULE = "cnn_demosaic.weights"
DEFAULT_WEIGHTS = "x-trans.weights.h5"
EXPOSURE_WEIGHTS = "exposure.weights.h5"
COLOR_WEIGHTS = "color.weights.h5"
RAF_SUFFIX = ".raf"

logger = logging.getLogger()


def process_raw(
    raw_path: pathlib.Path,
    weights_path: pathlib.Path,
    exposure_weights_path: pathlib.Path,
    color_weights_path: pathlib.Path,
    output_handler,
    fake=False,
    crop=False,
    post_process=True,
):
    """Performs raw image processing on the specified file."""
    with rawpy.imread(str(raw_path)) as raw_img:
        raw_img_arr = raw_img.raw_image.astype(np.float32).copy()
        raw_img_sizes = raw_img.sizes
        camera_whitebalance = raw_img.camera_whitebalance
        daylight_whitebalance = np.asarray(raw_img.daylight_whitebalance)

    loaded_model = None

    is_xtrans = raw_path.suffix.lower() == RAF_SUFFIX

    wb_scale = math.fsum(daylight_whitebalance) / math.fsum(camera_whitebalance)
    wb_matrix = np.asarray(camera_whitebalance)[:3] * wb_scale

    if is_xtrans:
        if fake:
            loaded_model = model.create_fake_xtrans_model(weights_path)
        else:
            loaded_model = model.create_xtrans_model(weights_path)
    else:
        loaded_model = model.create_32_64_32_model(weights_path)

    exp_model = exposure_model.create_exposure_model(exposure_weights_path)
    exposure = Exposure(exp_model)

    c_model = color_model.build_color_model(color_weights_path)
    color = Color(c_model)

    per_tile_fn = transform.adj_levels_per_tile_fn

    processor = Demosaic(loaded_model, per_tile_fn=per_tile_fn, xtrans=is_xtrans)
    raw_img_arr = transform.normalize_arr(raw_img_arr)
    output_arr = processor.demosaic(raw_img_arr)

    if post_process:
        logger.info(
            "Performing post-processing."
        )
        output_arr = exposure.process(output_arr)
        output_arr = color.process(output_arr, wb_matrix)

    output_arr = np.asarray(output_arr, dtype=np.float32)

    if crop:
        output_arr = crop_image(output_arr, raw_img_sizes)

    output_handler(output_arr)


def crop_image(img_arr, img_sizes: rawpy.ImageSizes):
    # TODO(jjaeggli): Move these parameters to an argument.
    # Add offsets to match JPG output image. 4896 x 3264
    col_offset = 19
    row_offset = 16
    width_override = 4896
    height_override = 3264
    col_start = img_sizes.left_margin + col_offset
    col_end = col_start + width_override
    row_start = img_sizes.top_margin + row_offset
    row_end = row_start + height_override
    # col_start = img_sizes.left_margin
    # col_end = col_start + img_sizes.width
    # row_start = img_sizes.top_margin
    # row_end = row_start + img_sizes.height
    logger.debug(
        "Cropping image array to dimensions [%s:%s,%s:%s]", (row_start, row_end, col_start, col_end)
    )
    return img_arr[row_start:row_end, col_start:col_end]


def get_format_from_str(input_str):
    format = output.IMG_FORMATS.get(input_str.lower().strip("."))
    if format is None:
        raise ValueError(f"Invalid format or unsupported file type specified [{input_str}]")
    return format


def get_format(format_arg, output_arg):
    use_format = False
    format_suffix = None
    format_writer = None
    if format_arg is not None:
        use_format = True
        format_suffix, format_writer = get_format_from_str(format_arg)
    if output_arg is not None:
        output_path = pathlib.Path(output_arg)
        output_suffix = output_path.suffix.lower()
        if not use_format:
            format_suffix, format_writer = get_format_from_str(output_suffix)
        elif output_suffix != format_suffix:
            raise ValueError(
                f"Conflict between specified format and output file suffix: [{output_suffix}]"
            )
    if format_suffix is None:
        return output.IMG_FORMATS[output.EXR_FORMAT]
    return format_suffix, format_writer


def main():
    default_weights_path = resources.files(WEIGHTS_MODULE).joinpath(DEFAULT_WEIGHTS)

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", default=default_weights_path)
    parser.add_argument("-o", "--output", required=False)
    parser.add_argument("-f", "--format", required=False)
    parser.add_argument("-k", "--fake", required=False, default=False, action="store_true")
    parser.add_argument("-c", "--crop", required=False, default=False, action="store_true")
    parser.add_argument("-n", "--nopost", required=False, default=False, action="store_true")
    parser.add_argument("raw_filename")
    args = parser.parse_args()

    # Prevent TensorFlow from allocating all GPU memory.
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    keras.utils.disable_interactive_logging()

    raw_path = pathlib.Path(args.raw_filename)
    if not raw_path.is_file():
        raise ValueError(f"The input filename {raw_path} is not a file!")

    weights_path = pathlib.Path(args.weights)
    if not weights_path.is_file():
        raise ValueError(f"The weights filename {weights_path} is not a file!")
    if not "".join(weights_path.suffixes) == ".weights.h5":
        raise ValueError("The weights filename must have the suffix .weights.h5")

    exposure_weights_path = pathlib.Path(resources.files(WEIGHTS_MODULE).joinpath(EXPOSURE_WEIGHTS))
    if not exposure_weights_path.is_file():
        raise ValueError(f"The exposure weights filename {exposure_weights_path} is not a file!")

    color_weights_path = pathlib.Path(resources.files(WEIGHTS_MODULE).joinpath(COLOR_WEIGHTS))
    if not color_weights_path.is_file():
        raise ValueError(f"The color weights filename {color_weights_path} is not a file!")

    # Determine the suffix from arguments.
    format_suffix, format_writer = get_format(args.format, args.output)

    # Either use the default or specified output path.
    output_path = pathlib.Path(raw_path.with_suffix(format_suffix))
    if args.output is not None:
        output_path = pathlib.Path(args.output)
    if output_path.exists():
        raise ValueError(f"The output filename {output_path.absolute()} already exists!")

    logger.debug(f'Processing: ["{raw_path.absolute()}"] => "{output_path.absolute()}"')

    def output_handler(output_arr):
        if format_writer is not None:
            format_writer(output_arr, output_path)

    post_process = not args.nopost

    process_raw(
        raw_path,
        weights_path,
        exposure_weights_path,
        color_weights_path,
        output_handler,
        fake=args.fake,
        crop=args.crop,
        post_process=post_process
    )


if __name__ == "__main__":
    logger.setLevel(logging.WARN)
    main()
