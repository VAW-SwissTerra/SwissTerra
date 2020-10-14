"""Constant parameters that are used throughout the module."""


CONSTANTS = {

    "scanning_resolution": 21e-6,  # The scanning resolution of the images in meters (21 micrometers)
    "crs_epsg": "EPSG::21781",  # CH1903 / LV03
    "max_point_cloud_threads": 10,  # How many threads to cap the program on to not run out of memory
    "dense_cloud_min_confidence": 3,  # The minimum proprietary Metashape confidence value for a dense cloud to accept.
}
