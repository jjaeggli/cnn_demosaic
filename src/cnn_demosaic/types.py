import json
from dataclasses import asdict, dataclass


@dataclass
class MonochromeParameters:
    ch_r: float
    ch_g: float
    ch_b: float


@dataclass
class ExposureParameters:
    black_level: float
    white_level: float
    gamma: float
    use_s_curve: bool
    contrast: float | None
    slope: float | None
    shift: float | None


@dataclass
class ProcessingParameters:
    exposure: ExposureParameters | None
    monochrome: MonochromeParameters | None

    def to_json(self) -> str:
        """
        Exports the ProcessingParameters instance as a JSON string.
        """
        return json.dumps(asdict(self), indent=4)

    @classmethod
    def from_json(cls, json_str: str) -> "ProcessingParameters":
        """
        Creates an ProcessingParameters instance from a JSON string.
        """
        data = json.loads(json_str)
        return cls(**data)
