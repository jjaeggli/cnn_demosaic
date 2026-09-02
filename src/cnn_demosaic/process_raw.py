# Module which performs raw processing and provides an entry point for user actions.

import argparse
import logging
import math
from importlib import resources
from typing import Protocol

import numpy as np
import rawpy
import tensorflow as tf
from tensorflow import keras

from cnn_demosaic import color_convert, color_model, config, exposure, model, transform
from cnn_demosaic.color import ColorTransform, MonochromeTransform, WhiteBalance
from cnn_demosaic.demosaic import Demosaic
from cnn_demosaic.types import MonochromeParameters
from cnn_demosaic.util import path_parser

WEIGHTS_MODULE = "cnn_demosaic.weights"
DEFAULT_WEIGHTS = "x-trans.weights.h5"
COLOR_WEIGHTS = "color.weights.h5"
RAF_SUFFIX = ".raf"


class RawImageSizes(Protocol):
    """The attributes of rawpy.ImageSizes used for cropping.

    A structural type so that rawpy does not need to be imported for typing,
    since rawpy does not export ImageSizes.
    """

    @property
    def height(self) -> int: ...

    @property
    def width(self) -> int: ...

    @property
    def left_margin(self) -> int: ...

    @property
    def top_margin(self) -> int: ...


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
        color_params = color_convert.get_color_params(raw_img, cfg.output_colorspace)
        exposure_params = exposure.get_exposure_params(
            raw_img,
            exposure_offset_ev=cfg.exposure_offset_ev,
            metering_headroom_ev=cfg.metering_headroom_ev,
            mid_gray_target=cfg.mid_gray_target,
        )
    exposure_processor = exposure.Exposure(exposure_params)

    logger.debug("Exposure parameters: %s", exposure_params)

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

    # The demosaic model operates on image values normalized to (0.0, 1.0) using
    # the image minimum and maximum. These values are retained so that the color
    # conversion stage can recover the linear sensor values.
    norm_min = float(np.min(raw_img_arr))
    norm_max = float(np.max(raw_img_arr))
    raw_img_arr = transform.normalize_arr(raw_img_arr)
    output_arr = processor.demosaic(raw_img_arr)

    if cfg.post_process:
        if cfg.monochrome:
            output_arr = post_process_bw(output_arr, exposure_processor, cfg)
        elif cfg.color_pipeline == config.COLOR_PIPELINE_MODEL:
            output_arr = post_process(output_arr, wb_matrix, cfg)
        else:
            output_arr = post_process_default(
                output_arr, color_params, exposure_processor, norm_min, norm_max, cfg
            )

    output_arr = np.asarray(output_arr, dtype=np.float32)
    output_arr = transform.encode_output_gamma(output_arr, cfg.output_gamma)

    if cfg.crop:
        output_arr = crop_image(output_arr, raw_img_sizes)

    cfg.output_handler(output_arr)


def post_process(img_arr, wb_matrix, cfg: config.Config):
    wb_model = color_model.create_white_balance_model(cfg.white_balance_weights_path)
    white_balance = WhiteBalance(wb_model)

    ct_model = color_model.create_color_transform_model(cfg.color_weights_path)
    color_transform = ColorTransform(ct_model)

    logger.info("Performing model color post-processing.")
    output_arr = white_balance.process(img_arr, wb_matrix)
    output_arr = color_transform.process(output_arr)

    return output_arr


def post_process_default(
    img_arr,
    color_params,
    exposure_processor: exposure.Exposure,
    norm_min,
    norm_max,
    cfg: config.Config,
):
    """Performs color conversion using the camera color matrix and white balance.

    Args:
        img_arr: The (H, W, 3) demosaiced image in normalized camera RGB.
        color_params: The ColorTransformParameters for the image.
        exposure_processor: The exposure processor with the parameters derived
            from the camera metering metadata.
        norm_min: The image minimum which was used for demosaic normalization.
        norm_max: The image maximum which was used for demosaic normalization.
        cfg: The processing configuration.
    """
    logger.info("Performing default color post-processing.")
    output_arr = color_convert.linearize(img_arr, color_params, norm_min, norm_max)
    output_arr = color_convert.white_balance(output_arr, color_params.wb_multipliers)
    output_arr = color_convert.camera_to_rgb(output_arr, color_params)
    output_arr = exposure_processor.process(output_arr)

    return output_arr


def post_process_bw(img_arr, exposure_processor: exposure.Exposure, cfg: config.Config):
    logger.info("Performing BW post-processing.")
    # RGB weights for the deep orange monochrome filter, normalized to sum to
    # 1.0 so that the metered exposure gain is preserved.
    deep_orange_weights = MonochromeParameters(0.8 / 1.5, 0.6 / 1.5, 0.1 / 1.5)
    monochrome_transform = MonochromeTransform(deep_orange_weights)
    output_arr = monochrome_transform.process(img_arr)
    output_arr = exposure_processor.process(output_arr)

    return output_arr


def crop_image(img_arr, img_sizes: RawImageSizes):
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


def crop_image_xe2_jpeg(img_arr, img_sizes: RawImageSizes):
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
    parser.add_argument(
        "-x",
        "--color",
        required=False,
        default=config.COLOR_PIPELINE_DEFAULT,
        choices=[config.COLOR_PIPELINE_DEFAULT, config.COLOR_PIPELINE_MODEL],
        help=(
            "The color processing pipeline to use. 'default' uses the camera "
            + "color matrix from the raw file. 'model' uses a white balance and "
            + "color transform models."
        ),
    )
    parser.add_argument(
        "-l",
        "--linear",
        required=False,
        default=False,
        action="store_true",
        help=(
            "Emit scene-referred linear values instead of encoding them "
            "with the transfer function matching the output color space."
        ),
    )
    parser.add_argument(
        "-s",
        "--colorspace",
        required=False,
        default=color_convert.DEFAULT_OUTPUT_COLORSPACE,
        choices=list(color_convert.VALID_OUTPUT_COLORSPACES),
        help=(
            "The output color space for the 'default' color pipeline. "
            "'srgb' (default), 'rec2020', or 'prophoto'. The values are "
            "encoded with the matching transfer function unless --linear "
            "is set. The 'model' color pipeline only supports 'srgb'."
        ),
    )
    parser.add_argument(
        "--exposure-offset",
        required=False,
        type=float,
        default=exposure.DEFAULT_EXPOSURE_OFFSET_EV,
        help="Exposure correction applied to the metered exposure, in stops.",
    )
    parser.add_argument(
        "--meter-headroom",
        required=False,
        type=float,
        default=exposure.DEFAULT_METERING_HEADROOM_EV,
        help=(
            "The number of stops between the metered mid-gray and sensor "
            + "clipping, used to determine the standard exposure."
        ),
    )
    parser.add_argument(
        "--mid-gray",
        required=False,
        type=float,
        default=exposure.DEFAULT_MID_GRAY,
        help="The target linear value for a metered 18%% gray.",
    )
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
    if args.raw_filename.startswith("@"):
        # If a path expression is supplied, args.output will be ignored.
        output_path = None
        processing_paths = path_parser.parse_path_statement(args.raw_filename[1:])

    for raw_file_path_str in processing_paths:
        # Create a new configuration for each processed raw file.
        # This ensures that raw_path and output_path are correctly set for the current iteration.
        current_processing_config = config.Config(
            raw_file_path_str,  # Use the string path for the current file
            output_filename=output_path,
            format=args.format,
            demosaic_weights=args.weights,
            crop=args.crop,
            post_process=post_process,
            monochrome=args.monochrome,
            fake=args.fake,
            color_pipeline=args.color,
            output_linear=args.linear,
            output_colorspace=args.colorspace,
            exposure_offset_ev=args.exposure_offset,
            metering_headroom_ev=args.meter_headroom,
            mid_gray_target=args.mid_gray,
        )

        current_processing_config.validate_config()

        raw_path_obj = current_processing_config.raw_path
        output_path_obj = current_processing_config.output_path

        logger.debug(f'Processing: ["{raw_path_obj.absolute()}"] => "{output_path_obj.absolute()}"')

        process_raw(current_processing_config)


if __name__ == "__main__":
    main()
