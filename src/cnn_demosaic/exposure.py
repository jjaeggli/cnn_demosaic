# Module for correcting exposure / luminance signal in the image.
#
# The default exposure processing is scene-referred and deterministic. The
# linearized image values are scaled by a linear gain so that a metered 18%
# gray maps to a standard linear mid-gray value (0.18). The metering headroom
# is the number of stops between the metered mid-gray and sensor clipping.
# With the default headroom of 2.47 EV the gain is 1.0, meaning that the
# camera's own metering combined with the white-level normalization already
# produces the standard exposure.
#
# The processing parameters are exposed as a serializable ExposureParameters
# object. A trained model can produce this object in the future, in which case
# the gamma and S-curve stages of apply_parameters will become active.

import json
import math
from dataclasses import asdict, dataclass

import numpy as np

from cnn_demosaic import transform
from cnn_demosaic.profile import profile

DEFAULT_MID_GRAY = 0.18
DEFAULT_METERING_HEADROOM_EV = 2.47
DEFAULT_EXPOSURE_OFFSET_EV = 0.0
DEFAULT_BLACK_LEVEL = 0.0
DEFAULT_WHITE_LEVEL = 1.0
DEFAULT_GAMMA = 1.0


@dataclass
class ExposureParameters:
    """Parameters for exposure processing.

    Attributes:
        gain: The linear gain applied to the image values.
        exposure_offset_ev: The user exposure correction in stops.
        metering_headroom_ev: The number of stops between the metered mid-gray
            and sensor clipping.
        mid_gray_target: The target linear value for a metered 18% gray.
        black_level: The input black level. For the default pipeline this is
            0.0, since black levels are handled by the linearization stage.
        white_level: The input white level. For the default pipeline this is
            1.0, since the white level is handled by the linearization stage.
        gamma: The gamma exponent. 1.0 (no-op) for the scene-referred default.
        use_s_curve: Whether to apply the S-curve. Disabled for the
            scene-referred default; retained for a future trained model.
        contrast, slope, shift: S-curve parameters for a future trained model.
        iso_speed, shutter_speed, aperture: Capture metadata used to determine
            the metered exposure.
        ev100: The metered exposure value normalized to ISO 100.
    """

    gain: float = 1.0
    exposure_offset_ev: float = DEFAULT_EXPOSURE_OFFSET_EV
    metering_headroom_ev: float = DEFAULT_METERING_HEADROOM_EV
    mid_gray_target: float = DEFAULT_MID_GRAY
    black_level: float = DEFAULT_BLACK_LEVEL
    white_level: float = DEFAULT_WHITE_LEVEL
    gamma: float = DEFAULT_GAMMA
    use_s_curve: bool = False
    contrast: float | None = None
    slope: float | None = None
    shift: float | None = None
    iso_speed: float | None = None
    shutter_speed: float | None = None
    aperture: float | None = None
    ev100: float | None = None

    def to_json(self) -> str:
        """
        Exports the ExposureParameters instance as a JSON string.
        """
        return json.dumps(asdict(self), indent=4)

    @classmethod
    def from_json(cls, json_str: str) -> "ExposureParameters":
        """
        Creates an ExposureParameters instance from a JSON string.
        """
        data = json.loads(json_str)
        return cls(**data)


def compute_ev100(iso_speed, shutter_speed, aperture):
    """Computes the metered exposure value normalized to ISO 100.

    This uses the reflected light metering convention:

        EV100 = log2(N^2 / t) + log2(ISO / 100)

    Args:
        iso_speed: The ISO speed setting of the capture.
        shutter_speed: The shutter speed in seconds.
        aperture: The aperture f-number.

    Returns:
        The EV100 value, or None if any of the capture settings are missing.
    """
    if not iso_speed or not shutter_speed or not aperture:
        return None
    return math.log2(aperture ** 2 / shutter_speed) + math.log2(iso_speed / 100.0)


def compute_gain(mid_gray_target, metering_headroom_ev, exposure_offset_ev):
    """Computes the linear gain which produces the standard exposure.

    A metered 18% gray sits metering_headroom_ev stops below sensor clipping,
    which places it at 2^-metering_headroom_ev after white level
    normalization. The gain maps that value to mid_gray_target, scaled by the
    user exposure offset.
    """
    return mid_gray_target * 2.0 ** (metering_headroom_ev + exposure_offset_ev)


def get_exposure_params(
    raw,
    exposure_offset_ev: float = DEFAULT_EXPOSURE_OFFSET_EV,
    metering_headroom_ev: float = DEFAULT_METERING_HEADROOM_EV,
    mid_gray_target: float = DEFAULT_MID_GRAY,
) -> ExposureParameters:
    """Builds ExposureParameters from the capture metadata of a raw image.

    Args:
        raw: An opened rawpy.RawPy instance (or any object exposing the `other`
            capture metadata attribute).
        exposure_offset_ev: The user exposure correction in stops.
        metering_headroom_ev: The number of stops between the metered mid-gray
            and sensor clipping.
        mid_gray_target: The target linear value for a metered 18% gray.

    Returns:
        An ExposureParameters instance.
    """
    iso_speed = shutter_speed = aperture = ev100 = None
    other = getattr(raw, "other", None)
    if other is not None:
        iso_speed = float(other.iso_speed) if other.iso_speed else None
        shutter_speed = float(other.shutter_speed) if other.shutter_speed else None
        aperture = float(other.aperture) if other.aperture else None
        ev100 = compute_ev100(iso_speed, shutter_speed, aperture)

    return ExposureParameters(
        gain=compute_gain(mid_gray_target, metering_headroom_ev, exposure_offset_ev),
        exposure_offset_ev=exposure_offset_ev,
        metering_headroom_ev=metering_headroom_ev,
        mid_gray_target=mid_gray_target,
        iso_speed=iso_speed,
        shutter_speed=shutter_speed,
        aperture=aperture,
        ev100=ev100,
    )


class Exposure:
    """Applies exposure parameters to an image array."""

    def __init__(self, params: ExposureParameters | None = None):
        self.params = params if params is not None else ExposureParameters()

    @profile()
    def process(self, img_arr):
        return self.apply_parameters(img_arr, self.params)

    @profile()
    def apply_parameters(self, img_arr, params: ExposureParameters):
        """Applies the parameters to the image array.

        The default parameters apply a pure linear gain without upper
        clipping, so the full dynamic range is retained for floating point
        output formats. The gamma and S-curve stages are only applied when the
        corresponding parameters are set, e.g. by a trained model.

        Args:
            params: An instance of ExposureParameters.
        """
        output_arr = transform.adj_levels(
            img_arr, params.black_level, params.white_level, clip=False
        )
        output_arr = np.asarray(output_arr, dtype=np.float32) * params.gain
        output_arr = np.maximum(output_arr, 0.0)

        if params.gamma is not None and params.gamma != 1.0:
            output_arr = transform.gamma(output_arr, params.gamma)

        if params.use_s_curve:
            midpoint = params.shift if params.shift is not None else 0.5
            slope = params.slope if params.slope is not None else 1.0
            contrast = params.contrast if params.contrast is not None else 1.0
            output_arr = transform.s_curve(
                output_arr, midpoint=midpoint, slope=slope, contrast=contrast
            )

        return output_arr
