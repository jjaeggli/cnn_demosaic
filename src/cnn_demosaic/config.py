from cnn_demosaic import exposure
from cnn_demosaic import icc
from cnn_demosaic import output
from cnn_demosaic.color_convert import (
    DEFAULT_OUTPUT_COLORSPACE,
    OUTPUT_COLORSPACE_PROPHOTO,
    OUTPUT_COLORSPACE_REC2020,
    OUTPUT_COLORSPACE_SRGB,
)
from cnn_demosaic.transform import OUTPUT_GAMMA_LINEAR
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

WEIGHTS_MODULE = "cnn_demosaic.weights"
DEMOSAIC_WEIGHTS = "x-trans.weights.h5"
COLOR_WEIGHTS = "color.weights.h5"
COLOR_TRANSFORM_WEIGHTS = "color_transform.weights.h5"
WHITE_BALANCE_WEIGHTS = "white_balance.weights.h5"
COLOR_PIPELINE_DEFAULT = "default"
COLOR_PIPELINE_MODEL = "model"


@dataclass
class Config:
    raw_filename: str
    output_filename: str | None = None
    format: str | None = None
    color_pipeline: str = COLOR_PIPELINE_DEFAULT
    color_weights: str | None = None
    demosaic_weights: str | None = None
    white_balance_weights: str | None = None
    output_linear: bool = False
    output_colorspace: str = DEFAULT_OUTPUT_COLORSPACE
    exposure_offset_ev: float = exposure.DEFAULT_EXPOSURE_OFFSET_EV
    metering_headroom_ev: float = exposure.DEFAULT_METERING_HEADROOM_EV
    mid_gray_target: float = exposure.DEFAULT_MID_GRAY
    crop: bool = True
    jpeg_crop: bool = False
    post_process: bool = True
    monochrome: bool = False
    fake: bool = False

    VALID_COLOR_PIPELINES = (COLOR_PIPELINE_DEFAULT, COLOR_PIPELINE_MODEL)
    VALID_OUTPUT_COLORSPACES = (
        OUTPUT_COLORSPACE_SRGB,
        OUTPUT_COLORSPACE_REC2020,
        OUTPUT_COLORSPACE_PROPHOTO,
    )

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
        if self.color_pipeline not in self.VALID_COLOR_PIPELINES:
            raise ValueError(
                f"Invalid color pipeline [{self.color_pipeline}], "
                + f"must be one of {self.VALID_COLOR_PIPELINES}"
            )
        if self.output_colorspace not in self.VALID_OUTPUT_COLORSPACES:
            raise ValueError(
                f"Invalid output color space [{self.output_colorspace}], "
                + f"must be one of {self.VALID_OUTPUT_COLORSPACES}"
            )
        # The learned color transform model was trained to output sRGB values.
        if (
            self.color_pipeline == COLOR_PIPELINE_MODEL
            and self.output_colorspace != OUTPUT_COLORSPACE_SRGB
        ):
            raise ValueError(
                f"The '{COLOR_PIPELINE_MODEL}' color pipeline only supports "
                + f"'{OUTPUT_COLORSPACE_SRGB}' output color space, "
                + f"got '{self.output_colorspace}'"
            )

        validate_weights_path(self.demosaic_weights_path)

        # The learned color pipeline requires the white balance and color
        # transform models.
        if self.color_pipeline == COLOR_PIPELINE_MODEL:
            validate_weights_path(self.color_weights_path)
            validate_weights_path(self.white_balance_weights_path)

    @property
    def output_gamma(self):
        """Returns the output transfer function for the current encoding.

        Linear output preserves the scene-referred values. Display-referred
        output uses the transfer function matching the output color space,
        so the gamma names double as the output color space names.
        """
        if self.output_linear:
            return OUTPUT_GAMMA_LINEAR
        return self.output_colorspace

    @property
    def output_handler(self):
        # TODO(jjaeggli): Output handler should be able to encompas a number of different
        #   things, such as a bitstream output handler.
        icc_profile = icc.get_icc_profile(self.output_linear, self.output_colorspace)

        def output_handler(output_arr):
            if self._format_writer is not None:
                self._format_writer(output_arr, self.output_path, icc_profile=icc_profile)

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
