from __future__ import annotations
from datetime import datetime
import typing

from ..config import datetime_range


def get_times(cfg: dict[str, typing.Any]) -> list[datetime]:
    """
    vector of times to generate precipitation
    in general, the precipitation time step may be distinct from the simulation time step
    """

    return datetime_range(cfg["time"][0], cfg["time"][0] + cfg["tdur"], cfg["dtprec"])
