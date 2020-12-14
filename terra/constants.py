"""Constant parameters that are used throughout the module."""

from terra.utilities import ConstantType

WILD_FIDUCIAL_LOCATIONS: dict[str, tuple[int, int]] = {
    "top": (250, 3500),
    "left": (2450, 250),
    "right": (2350, 6749),
    "bottom": (4370, 3500)
}

ZEISS_FIDUCIAL_LOCATIONS: dict[str, tuple[int, int]] = {
    "top": (500, 4270),
    "left": (500, 1400),
    "right": (500, 7100),
    "bottom": (5700, 4270)
}


class Constants(ConstantType):  # pylint: disable=R0903
    """Readonly constants."""

    scanning_resolution: float = 21e-6  # The scanning resolution of the images in meters (21 micrometers)
    crs_epsg: str = "EPSG::21781"  # CH1903 / LV03
    max_point_cloud_threads: int = 10  # How many threads to cap the program on to not run out of memory
    dense_cloud_min_confidence: int = 3  # The minimum proprietary Metashape confidence value to accept.
    dem_resolution: float = 5.0  # The resolution in metres to generate DEMs with.
    max_threads: int = 20  # The maximum amount of threads to process in.
    wild_fiducial_locations = WILD_FIDUCIAL_LOCATIONS
    zeiss_fiducial_locations = ZEISS_FIDUCIAL_LOCATIONS
    transform_outlier_threshold: float = 10
    tripod_height: float = 1.2


CONSTANTS = Constants()
