from cnn_demosaic import output
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

WEIGHTS_MODULE = "cnn_demosaic.weights"
DEMOSAIC_WEIGHTS = "x-trans.weights.h5"
EXPOSURE_WEIGHTS = "exposure.weights.h5"
COLOR_WEIGHTS = "color.weights.h5"
COLOR_TRANSFORM_WEIGHTS = "color_transform.weights.h5"
WHITE_BALANCE_WEIGHTS = "white_balance.weights.h5"


@dataclass
class Config:
    raw_filename: str
    output_filename: str | None = None
    format: str | None = None
    color_weights: str | None = None
    demosaic_weights: str | None = None
    exposure_weights: str | None = None
    white_balance_weights: str | None = None
    crop: bool = True
    jpeg_crop: bool = False
    post_process: bool = True
    monochrome: bool = False
    fake: bool = False

    def __post_init__(self):
        self.raw_path = Path(self.raw_filename)

        self._extension, self._format_writer = get_format(self.format, self.output_filename)

    @property
    def demosaic_weights_path(self):
        return get_weights_resource_path(self.demosaic_weights, DEMOSAIC_WEIGHTS)

    @property
    def color_weights_path(self):
        return get_weights_resource_path(self.color_weights, COLOR_TRANSFORM_WEIGHTS)

    @property
    def exposure_weights_path(self):
        return get_weights_resource_path(self.exposure_weights, EXPOSURE_WEIGHTS)

    @property
    def white_balance_weights_path(self):
        return get_weights_resource_path(self.white_balance_weights, WHITE_BALANCE_WEIGHTS)

    @property
    def output_suffix(self):
        return self._extension

    @property
    def output_path(self) -> Path:
        # Either use the default or specified output path.
        output_path = Path(self.raw_path.with_suffix(self.output_suffix))
        if self.output_filename is not None:
            output_path = Path(self.output_filename)
        return output_path

    def validate_config(self):
        if not self.raw_path.is_file():
            raise ValueError(f"The input filename {self.raw_path.absolute()} is not a file!")
        if self.output_path.exists():
            raise ValueError(f"The output filename {self.output_path.absolute()} already exists!")

        validate_weights_path(self.demosaic_weights_path)

        # Post processing implies that models other than the demosaicing model
        # will be used.
        if self.post_process:
            validate_weights_path(self.color_weights_path)
            validate_weights_path(self.exposure_weights_path)
            validate_weights_path(self.white_balance_weights_path)

    @property
    def output_handler(self):
        # TODO(jjaeggli): Output handler should be able to encompas a number of different
        #   things, such as a bitstream output handler.

        def output_handler(output_arr):
            if self._format_writer is not None:
                self._format_writer(output_arr, self.output_path)

        return output_handler


def get_weights_resource_path(resource: str | None, default: str):
    resource_or_default = default if resource is None else resource
    resource_path = Path(resources.files(WEIGHTS_MODULE).joinpath(resource_or_default))
    return resource_path


def validate_weights_path(resource_path):
    if not resource_path.exists():
        raise ValueError(f"The weights filename {resource_path} is not a file!")
    if not "".join(resource_path.suffixes) == ".weights.h5":
        raise ValueError("The weights filename must have the suffix .weights.h5")


# TODO(jjaeggli): Extract building the handler to the output module.


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
        output_path = Path(output_arg)
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
