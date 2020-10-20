"""Constant parameters that are used throughout the module."""

from terra.utilities import ConstantType


class Constants(ConstantType):  # pylint: disable=R0903
    """Readonly constants."""

    scanning_resolution: float = 21e-6  # The scanning resolution of the images in meters (21 micrometers)
    crs_epsg: str = "EPSG::21781"  # CH1903 / LV03
    max_point_cloud_threads: int = 10  # How many threads to cap the program on to not run out of memory
    dense_cloud_min_confidence: int = 3  # The minimum proprietary Metashape confidence value to accept.
    dem_resolution: float = 5.0  # The resolution in metres to generate DEMs with.


CONSTANTS = Constants()
