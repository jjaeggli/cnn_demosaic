# Module which performs raw processing and provides an entry point for user actions.

import argparse
import logging
import numpy as np
import pathlib
import pyexr
import rawpy
import tensorflow as tf

from cnn_demosaic.demosaic import Demosaic
from cnn_demosaic import model
from cnn_demosaic import transform
from importlib import resources
from tensorflow import keras
from PIL import Image


DEFAULT_WEIGHTS = "x-trans.weights.h5"
IMG_FORMATS = [".exr", ".png"]
EXR_SUFFIX = ".exr"
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

    if output_path.suffix.lower() == ".png":
        output_arr = (output_arr * 65535).astype(np.uint16)
        Image.fromarray(output_arr).save(str(output_path), "PNG")
    elif output_path.suffix.lower() == ".exr":
        pyexr.write(
            str(output_path),
            output_arr,
            precision=pyexr.HALF,
        )
    else:
        raise ValueError(f"The output file type ['{output_path.suffix}'] is not supported.")


def main():
    default_weights_path = resources.files("cnn_demosaic.weights").joinpath(DEFAULT_WEIGHTS)

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", default=default_weights_path)
    parser.add_argument("-o", "--output", required=False)
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

    output_path = pathlib.Path(raw_path.with_suffix(EXR_SUFFIX))
    if args.output is not None:
        output_path = pathlib.Path(args.output)
        if output_path.suffix not in IMG_FORMATS:
            raise ValueError(
                f"The image filename must be a supported format: [{output_path.suffix}]"
            )
    if output_path.exists():
        raise ValueError(f"The output filename {output_path.absolute()} already exists!")

    logger.debug(f'Processing: ["{raw_path.absolute()}"] => "{output_path.absolute()}"')

    process_raw(raw_path, weights_path, output_path, fake=args.fake)


if __name__ == "__main__":
    logger.setLevel(logging.WARN)
    main()
