"""Constant parameters that are used throughout the module."""
from typing import Any

import statictypes


class Constants:
    """Readonly document constants."""

    scanning_resolution: float = 21e-6  # The scanning resolution of the images in meters (21 micrometers)
    crs_epsg: str = "EPSG::21781"  # CH1903 / LV03
    max_point_cloud_threads: int = 10  # How many threads to cap the program on to not run out of memory
    dense_cloud_min_confidence: int = 3  # The minimum proprietary Metashape confidence value to accept.
    dem_resolution: float = 5.0  # The resolution in metres to generate DEMs with.

    @statictypes.enforce
    def __getitem__(self, key: str) -> Any:
        """
        Get an item like a dict.

        param: key: The attribute name.

        return: attribute: The value of the attribute.
        """
        attribute = self.__getattribute__(key)
        return attribute

    @staticmethod
    def raise_readonly_error(key, value):
        """Raise a readonly error if a value is trying to be set."""
        raise ValueError(f"Trying to change a constant. Key: {key}, value: {value}")

    def __setattr__(self, key, value):
        """Override the Constants.key = value action."""
        self.raise_readonly_error(key, value)

    def __setitem__(self, key, value):
        """Override the Constants['key'] = value action."""
        self.raise_readonly_error(key, value)


CONSTANTS = Constants()
