# Module which performs raw processing and provides an entry point for user actions.

import argparse
import logging
import numpy as np
import pathlib
import pyexr
import tifffile
import rawpy
import tensorflow as tf

from cnn_demosaic.demosaic import Demosaic
from cnn_demosaic import model
from cnn_demosaic import transform
from importlib import resources
from tensorflow import keras


DEFAULT_WEIGHTS = "x-trans.weights.h5"
EXR_FORMAT = "exr"
TIFF_FORMAT = "tiff"
EXR_SUFFIX = ".exr"
TIFF_SUFFIX = ".tiff"
IMG_FORMATS = {EXR_FORMAT: EXR_SUFFIX, TIFF_FORMAT: TIFF_SUFFIX}
RAF_SUFFIX = ".raf"

logger = logging.getLogger()


def process_raw(
    raw_path: pathlib.Path, weights_path: pathlib.Path, output_path: pathlib.Path, fake=False
):
    """Performs raw image processing on the specified file."""
    with rawpy.imread(str(raw_path)) as raw_img:
        raw_img_array = raw_img.raw_image.astype(np.float32).copy()

    loaded_model = None

    is_xtrans = raw_path.suffix.lower() == RAF_SUFFIX

    if is_xtrans:
        if fake:
            loaded_model = model.create_fake_xtrans_model(weights_path)
        else:
            loaded_model = model.create_xtrans_model(weights_path)
    else:
        loaded_model = model.create_32_64_32_model(weights_path)

    per_tile_fn = transform.adj_levels_per_tile_fn

    processor = Demosaic(loaded_model, per_tile_fn=per_tile_fn, xtrans=is_xtrans)
    raw_img_array = transform.normalize_arr(raw_img_array)
    output_arr = processor.demosaic(raw_img_array)

    # TODO(jjaeggli): Perform color space conversion and other output image operations.

    if output_path.suffix.lower() == EXR_SUFFIX:
        write_exr(output_arr, output_path)
    elif output_path.suffix.lower() == TIFF_SUFFIX:
        write_tiff(output_arr, output_path)
    else:
        raise ValueError(f"The output file type ['{output_path.suffix}'] is not supported.")


def write_tiff(output_arr, output_path):
    output_arr = (output_arr * 65535).astype(np.uint16)
    tifffile.imwrite(output_path, output_arr, photometric="rgb", compression="zlib")


def write_exr(output_arr, output_path):
    pyexr.write(
        str(output_path),
        output_arr,
        precision=pyexr.HALF,
    )


def get_format_from_str(input_str):
    format_suffix = IMG_FORMATS.get(input_str.lower().strip("."))
    if format_suffix is None:
        raise ValueError(f"Invalid format or unsupported file type specified [{input_str}]")
    return format_suffix


def get_format(format_arg, output_arg):
    use_format = False
    format_suffix = None
    if format_arg is not None:
        use_format = True
        format_suffix = get_format_from_str(format_arg)
    if output_arg is not None:
        output_path = pathlib.Path(output_arg)
        output_suffix = output_path.suffix.lower()
        if not use_format:
            get_format_from_str(output_suffix)
        elif output_suffix != format_suffix:
            raise ValueError(
                f"Conflict between specified format and output file suffix: [{output_suffix}]"
            )
    if format_suffix is None:
        return IMG_FORMATS[EXR_FORMAT]
    return format_suffix


def main():
    default_weights_path = resources.files("cnn_demosaic.weights").joinpath(DEFAULT_WEIGHTS)

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", default=default_weights_path)
    parser.add_argument("-o", "--output", required=False)
    parser.add_argument("-f", "--format", required=False)
    parser.add_argument("-k", "--fake", required=False, default=False, action="store_true")
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

    # Determine the suffix from arguments.
    format_suffix = get_format(args.format, args.output)

    # Either use the default or specified output path.
    output_path = pathlib.Path(raw_path.with_suffix(format_suffix))
    if args.output is not None:
        output_path = pathlib.Path(args.output)
    if output_path.exists():
        raise ValueError(f"The output filename {output_path.absolute()} already exists!")

    logger.debug(f'Processing: ["{raw_path.absolute()}"] => "{output_path.absolute()}"')

    process_raw(raw_path, weights_path, output_path, fake=args.fake)


if __name__ == "__main__":
    logger.setLevel(logging.WARN)
    main()
