"""Constant parameters that are used throughout the module."""
from __future__ import annotations

from datetime import datetime

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


class Constants(ConstantType):  # pylint: disable=too-few-public-methods
    """Readonly constants."""

    scanning_resolution: float = 21e-6  # The scanning resolution of the images in meters (21 micrometers)
    crs_epsg: str = "EPSG::21781"  # CH1903 / LV03
    max_point_cloud_threads: int = 10  # How many threads to cap the program on to not run out of memory
    dense_cloud_min_confidence: int = 2  # The minimum proprietary Metashape confidence value to accept.
    dem_resolution: float = 5.0  # The resolution in metres to generate DEMs with.
    max_threads: int = 20  # The maximum amount of threads to process in.
    wild_fiducial_locations = WILD_FIDUCIAL_LOCATIONS  # The approximate locations of the wild instrument fiducials
    zeiss_fiducial_locations = ZEISS_FIDUCIAL_LOCATIONS  # The approximate locations of the zeiss instrument fiducials
    transform_outlier_threshold: float = 10  # Pixel threshold for the image internal coordinate transforms
    tripod_height: float = 1.2  # The approximate height of the tripods that were used (according to SwissTopo)
    position_accuracy: float = 2.0  # The assumed position accuracy in m after accounting for spatial systematic shifts
    rotation_accuracy: float = 1.0  # The assumed rotational accuracy in degrees.
    max_height: float = 5000.0  # The highest mountain in the Alps is just less than 5000 m.
    base_dem_date: datetime = datetime(year=2018, month=8, day=1)  # The date of the base DEM
    ice_density: float = 0.85  # The m to m w.e. conversion factor for geodetic elevation change.
    stable_ground_accuracy: float = 1.0  # The metashape accuracy to assign the GCPs


CONSTANTS = Constants()
