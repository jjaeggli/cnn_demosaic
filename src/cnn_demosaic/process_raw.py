# Module which performs raw processing and provides an entry point for user actions.

import argparse
import logging
import math
import numpy as np
import rawpy
import tensorflow as tf

from cnn_demosaic import color_model
from cnn_demosaic import config
from cnn_demosaic import exposure_model
from cnn_demosaic.color import WhiteBalance, ColorTransform, MonochromeTransform
from cnn_demosaic.demosaic import Demosaic
from cnn_demosaic.exposure import Exposure
from cnn_demosaic.types import MonochromeParameters
from cnn_demosaic import model
from cnn_demosaic import output
from importlib import resources
from cnn_demosaic import transform
from cnn_demosaic.util import path_parser
from tensorflow import keras


WEIGHTS_MODULE = "cnn_demosaic.weights"
DEFAULT_WEIGHTS = "x-trans.weights.h5"
EXPOSURE_WEIGHTS = "exposure.weights.h5"
COLOR_WEIGHTS = "color.weights.h5"
RAF_SUFFIX = ".raf"


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


def process_raw(cfg: config.Config):
    """Performs raw image processing on the specified file."""
    raw_path = cfg.raw_path

    with rawpy.imread(str(raw_path)) as raw_img:
        raw_img_arr = raw_img.raw_image.astype(np.float32).copy()
        raw_img_sizes = raw_img.sizes
        camera_whitebalance = raw_img.camera_whitebalance
        daylight_whitebalance = np.asarray(raw_img.daylight_whitebalance)

    loaded_model = None

    # TODO(jjaeggli): Update this value to match the training input value range.
    wb_scale = math.fsum(daylight_whitebalance) / math.fsum(camera_whitebalance)
    wb_matrix = np.asarray(camera_whitebalance)[:3] * wb_scale

    demosaic_weights_path = cfg.demosaic_weights_path

    # TODO(jjaeggli): Actually look at the sensor format to determine which sensor
    #   format should be used.
    is_xtrans = raw_path.suffix.lower() == RAF_SUFFIX

    if is_xtrans:
        if cfg.fake:
            loaded_model = model.create_fake_xtrans_model(demosaic_weights_path)
        else:
            loaded_model = model.create_xtrans_model(demosaic_weights_path)
    else:
        loaded_model = model.create_32_64_32_model(demosaic_weights_path)

    per_tile_fn = transform.adj_levels_per_tile_fn

    processor = Demosaic(loaded_model, per_tile_fn=per_tile_fn, xtrans=is_xtrans)
    raw_img_arr = transform.normalize_arr(raw_img_arr)
    output_arr = processor.demosaic(raw_img_arr)

    if cfg.post_process:
        if cfg.monochrome:
            output_arr = post_process_bw(output_arr, cfg)
        else:
            output_arr = post_process(output_arr, wb_matrix, cfg)

    output_arr = np.asarray(output_arr, dtype=np.float32)

    if cfg.crop:
        output_arr = crop_image(output_arr, raw_img_sizes)

    cfg.output_handler(output_arr)


def post_process(img_arr, wb_matrix, cfg: config.Config):
    exp_model = exposure_model.create_exposure_model(cfg.exposure_weights_path)
    exposure = Exposure(exp_model)

    wb_model = color_model.create_white_balance_model(cfg.white_balance_weights_path)
    white_balance = WhiteBalance(wb_model)

    ct_model = color_model.create_color_transform_model(cfg.color_weights_path)
    color_transform = ColorTransform(ct_model)

    logger.info("Performing post-processing.")
    output_arr = exposure.process(img_arr)
    output_arr = white_balance.process(output_arr, wb_matrix)
    output_arr = color_transform.process(output_arr)

    return output_arr


def post_process_bw(img_arr, cfg: config.Config):
    # This was causing inversion, so it is disabled for now.
    # exp_model = exposure_model.create_exposure_model(cfg.exposure_weights_path)
    # exposure = Exposure(exp_model)
    # output_arr = exposure.process(img_arr)

    logger.info("Performing BW post-processing.")
    # RGB Weights for the deep orange monochrome filter.
    deep_orange_weights = MonochromeParameters(0.8, 0.6, 0.1)
    monochrome_transform = MonochromeTransform(deep_orange_weights)
    output_arr = monochrome_transform.process(img_arr)

    return output_arr


def crop_image(img_arr, img_sizes: rawpy.ImageSizes):
    height = img_sizes.height
    width = img_sizes.width
    col_start = img_sizes.left_margin
    col_end = col_start + width
    row_start = img_sizes.top_margin
    row_end = row_start + height
    logger.debug(
        "Cropping image array to dimensions: [%s+%s,%s+%s]", row_start, width, col_start, height
    )
    return img_arr[row_start:row_end, col_start:col_end]


def crop_image_xe2_jpeg(img_arr, img_sizes: rawpy.ImageSizes):
    col_offset = 19
    row_offset = 16
    width = 4896
    height = 3264
    col_start = img_sizes.left_margin + col_offset
    col_end = col_start + width
    row_start = img_sizes.top_margin + row_offset
    row_end = row_start + height
    logger.debug(
        "Cropping image array to dimensions: [%s+%s,%s+%s]", row_start, width, col_start, height
    )
    return img_arr[row_start:row_end, col_start:col_end]


def main():
    default_weights_path = resources.files(WEIGHTS_MODULE).joinpath(DEFAULT_WEIGHTS)

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", default=default_weights_path)
    parser.add_argument("-o", "--output", required=False)
    parser.add_argument("-f", "--format", required=False)
    parser.add_argument("-k", "--fake", required=False, default=False, action="store_true")
    parser.add_argument("-c", "--crop", required=False, default=False, action="store_true")
    parser.add_argument("-n", "--nopost", required=False, default=False, action="store_true")
    parser.add_argument("-m", "--monochrome", required=False, default=False, action="store_true")
    parser.add_argument("raw_filename")
    args = parser.parse_args()

    # Prevent TensorFlow from allocating all GPU memory.
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    keras.utils.disable_interactive_logging()

    post_process = not args.nopost

    processing_paths = [args.raw_filename]
    output_path = args.output

    # If the path starts with an '@' symbol, this indicates that it contains a range
    # expression and should be expanded.
    if args.raw_filename.startswith('@'):
        # If a path expression is supplied, args.output will be ignored.
        output_path = None
        processing_paths = path_parser.parse_path_statement(args.raw_filename[1:])

    for raw_file_path_str in processing_paths:
        # Create a new configuration for each processed raw file.
        # This ensures that raw_path and output_path are correctly set for the current iteration.
        current_processing_config = config.Config(
            raw_file_path_str, # Use the string path for the current file
            output_filename=output_path,
            format=args.format,
            demosaic_weights=args.weights,
            crop=args.crop,
            post_process=post_process,
            monochrome=args.monochrome,
            fake=args.fake,
        )

        current_processing_config.validate_config()

        raw_path_obj = current_processing_config.raw_path
        output_path_obj = current_processing_config.output_path

        logger.debug(f'Processing: ["{raw_path_obj.absolute()}"] => "{output_path_obj.absolute()}"')

        process_raw(current_processing_config)


if __name__ == "__main__":
    main()
