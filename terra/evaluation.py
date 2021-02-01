"""Result evaluation functions."""
from __future__ import annotations

import datetime
import json
import os
import pickle
import subprocess
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.widgets
import numpy as np
import pandas as pd
import pytransform3d
import pytransform3d.rotations
import pytransform3d.transformations
import rasterio as rio
import rasterio.warp  # pylint: disable=unused-import
import sklearn.linear_model
import sklearn.pipeline
import sklearn.utils
from tqdm import tqdm

from terra import base_dem, dem_tools, files
from terra.constants import CONSTANTS
from terra.preprocessing import outlines
from terra.processing import inputs

TEMP_DIRECTORY = os.path.join(files.TEMP_DIRECTORY, "evaluation")

CACHE_FILES = {
    "metashape_dems_dir": os.path.join(inputs.TEMP_DIRECTORY, "output/dems"),
    "ddems_dir": os.path.join(TEMP_DIRECTORY, "ddems/"),
    "ddems_coreg_dir": os.path.join(TEMP_DIRECTORY, "coreg_ddems/"),
    "ddems_yearly_coreg_dir": os.path.join(TEMP_DIRECTORY, "yearly_coreg_ddems/"),
    "dem_coreg_dir": os.path.join(TEMP_DIRECTORY, "coreg/"),
    "dem_coreg_meta_dir": os.path.join(TEMP_DIRECTORY, "coreg_meta/"),
    "ddem_quality": os.path.join(files.MANUAL_INPUT_DIR, "ddem_quality.csv"),
    "ddem_stats": os.path.join(TEMP_DIRECTORY, "ddem_stats.pkl"),
    "merged_ddem": os.path.join(TEMP_DIRECTORY, "merged_ddem.tif"),
    "merged_yearly_ddem": os.path.join(TEMP_DIRECTORY, "merged_yearly_ddem.tif"),
    "elevation_vs_change_data": os.path.join(TEMP_DIRECTORY, "elevation_vs_change_data.csv"),
}


def read_icp_stats(station_name: str) -> dict[str, float]:
    """
    Read the ICP coregistration statistics from a specified station name.

    :param station_name: The name of the station, e.g. station_3500.
    :returns: A dictionary of coregistration statistics.
    """
    with open(os.path.join(CACHE_FILES["dem_coreg_meta_dir"], f"{station_name}_coregistration.json")) as infile:
        coreg_stats = json.load(infile)

    # Calculate the transformation angle from the transformation matrix
    matrix = np.array(coreg_stats["transform"].replace("\n", " ").split(" ")).astype(float).reshape((4, 4))
    quaternion = pytransform3d.transformations.pq_from_transform(matrix)[3:]
    angle = np.rad2eg(pytransform3d.rotations.axis_angle_from_quaternion(quaternion)[3])

    stats = {"fitness": coreg_stats["fitness"], "overlap": coreg_stats["overlap"], "angle": angle}

    return stats


def examine_ddem(filepath: str) -> bool:
    """
    Examine a dDEM and rate its validity (good, bad or unclear).

    :param filepath: The path to the dDEM.
    :returns: A flag of whether or not the user has pressed the stop button.
    """
    def log_quality(quality: str):
        """Log the given quality value."""
        print(f"Registered {filepath} as {quality}")

        station_name = dem_tools.extract_station_name(filepath)
        stats = read_icp_stats(station_name)
        date = datetime.datetime.now()

        with open(CACHE_FILES["ddem_quality"], "a+") as outfile:
            line = f"{filepath},{quality},{stats['fitness']},{stats['overlap']},{stats['angle']},{date}\n"
            outfile.write(line)

    # Read the dDEM
    ddem = rio.open(filepath)
    ddem_values = ddem.read(1)

    # Load and crop/resample the glacier mask and reference DEM
    glacier_mask = dem_tools.read_and_crop_glacier_mask(ddem)

    # Stop button, signalling that the examination is done.
    should_stop = False

    def stop(*_):
        nonlocal should_stop
        should_stop = True

    plt.subplot(111)
    plt.imshow(glacier_mask, cmap="Greys_r")
    plt.imshow(ddem_values, cmap="coolwarm_r", vmin=-100, vmax=100)

    good_button = matplotlib.widgets.Button(plt.axes([0.9, 0.0, 0.1, 0.075]), "Good", )
    good_button.on_clicked(lambda _: log_quality("good"))

    bad_button = matplotlib.widgets.Button(plt.axes([0.8, 0.0, 0.1, 0.075]), "Bad")
    bad_button.on_clicked(lambda _: log_quality("bad"))

    unclear_button = matplotlib.widgets.Button(plt.axes([0.7, 0.0, 0.1, 0.075]), "Unclear")
    unclear_button.on_clicked(lambda _: log_quality("unclear"))

    stop_button = matplotlib.widgets.Button(plt.axes([0.5, 0.0, 0.1, 0.075]), "Stop")
    stop_button.on_clicked(stop)
    plt.show()

    return should_stop


def model_df(data: pd.DataFrame, sample_cols: list[str],
             target_col: str) -> tuple[sklearn.linear_model.LogisticRegression, float]:
    """
    Train a logistic model on a dataframe and return the model + model score.

    The target_col must be boolean.

    :param data: The dataframe to analyse.
    :param sample_cols: The columns of X values (predictors).
    :param target_col: The column of y values (targets).
    :returns: The linear model and its model score.
    """
    # Make sure that the target_col is a boolean field
    data[target_col].astype(bool)

    # Ugly implementation of outlier detection. Kind of works like RANSAC:
    # Try to estimate the model at most ten times and exclude 10% of the data each time.
    # Possible outliers therefore have a chance of being excluded, possibly leading to a better model.
    # Finally, the model with the most valid data is returned.
    models = {}
    for _ in range(10):
        shuffled_data = sklearn.utils.shuffle(data)

        # Exclude the last 10% of the shuffled data.
        exclusion_border = int(shuffled_data.shape[0] * 0.9)
        shuffled_data = shuffled_data.iloc[:exclusion_border]

        # Warnings may arise here which should just be repressed.
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            model = sklearn.linear_model.LogisticRegression(
                random_state=0, solver='lbfgs', multi_class='ovr', max_iter=300)
            # Try to fit the model.
            try:
                model.fit(shuffled_data[sample_cols], shuffled_data[target_col])
            # ValueErrors may be raised if the data contain only True's or False's (then try again with other data)
            except ValueError:
                continue
            score = model.score(shuffled_data[sample_cols], shuffled_data[target_col])

            n_good = np.count_nonzero(model.predict(shuffled_data[sample_cols]))
            models[n_good] = model, score

    return models[max(models.keys())]


def get_ddem_stats(directory: str, redo=False) -> pd.DataFrame:
    """
    Calculate statistics from each dDEM in a directory.

    The pixel count, median, 95th percentile, and 5th percentile is calculated for on- and off-glacier surfaces.

    :param directory: The directory of the dDEMs to calculate statistics from.
    :param redo: Whether to redo the analysis if it has already been made (it takes a while to run)
    :returns: A dataframe with each of the dDEM's statistics in one row.
    """
    ddems = dem_tools.find_dems(directory)

    if os.path.isfile(CACHE_FILES["ddem_stats"]) and not redo:
        return pd.read_pickle(CACHE_FILES["ddem_stats"])

    stats = pd.DataFrame(data=ddems, columns=["filepath"])
    stats["station_name"] = stats["filepath"].apply(dem_tools.extract_station_name)

    for i, filepath in tqdm(enumerate(ddems), total=len(ddems), desc="Calculating dDEM statistics"):
        ddem = rio.open(filepath)
        glacier_mask = dem_tools.read_and_crop_glacier_mask(ddem)
        ddem_values = ddem.read(1)

        for on_off, values in zip(["on", "off"], [ddem_values[glacier_mask], ddem_values[~glacier_mask]]):
            col_names = [f"{on_off}_{stat}" for stat in ["count", "median", "upper", "lower"]]
            # Return a zero count and 200 as a "nodata" value if there is no on- / off-glacier pixel
            # The 200 is because these stats are used for regression, and 200 is a bad value
            # TODO: Handle the "bad values" in a better fashion?
            if np.all(np.isnan(values)):
                stats.loc[i, col_names] = [0] + [200] * (len(col_names) - 1)
                continue
            count = np.count_nonzero(~np.isnan(values))
            median = np.nanmedian(values)
            upper_percentile = np.nanpercentile(values, 95)
            lower_percentile = np.nanpercentile(values, 5)
            stats.loc[i, col_names] = count, median, upper_percentile, lower_percentile

    stats.to_pickle(CACHE_FILES["ddem_stats"])
    return stats


def evaluate_ddems(improve: bool = False) -> list[str]:
    """
    Train and or run a model to evaluate the quality of the dDEMs, and return a list of the ones deemed good.

    :param improve: Improve the model?
    :returns: A list of filepaths to the dDEMs that are deemed good.
    """
    if improve:
        coreg_ddems = np.array(dem_tools.find_dems(CACHE_FILES["ddems_coreg_dir"]))

        # Shuffle the data so that dDEMs from different areas come up in succession.
        np.random.shuffle(coreg_ddems)

        # For the improvement part: Generate a list of dDEMs that have already been looked at.
        existing_filepaths: list[str] = []
        if os.path.exists(CACHE_FILES["ddem_quality"]):
            existing_filepaths += list(pd.read_csv(CACHE_FILES["ddem_quality"])["filepath"])

        stopped = False
        for filepath in coreg_ddems:
            if stopped:
                break
            if filepath in existing_filepaths:
                continue
            stopped = examine_ddem(filepath)
            existing_filepaths.append(filepath)

    # Read the log of the manually placed dDEM quality flags.
    log = pd.read_csv(CACHE_FILES["ddem_quality"]).drop_duplicates("filepath", keep="last")
    # Remove all the ones where the quality was deemed unclear.
    log = log[log["quality"] != "unclear"]
    # Extract the station name from the filepath.
    log["station_name"] = log["filepath"].apply(dem_tools.extract_station_name)
    # Make a boolean field of the quality
    log["good"] = log["quality"] == "good"

    # TODO: Why is it shuffled?
    log = sklearn.utils.shuffle(log)

    # Read the ICP coregistration statistics from each respective coregistered DEM
    # for i, row in log.iterrows():
    #    stats = read_icp_stats(row["station_name"])
    #    log.loc[i, list(stats.keys())] = list(stats.values())

    # Train a classifier to classify the dDEM quality based on the ICP coregistration statistics.
    # Try ten times to get a model with a score >= 0.9
    for _ in range(10):
        model, score = model_df(log, ["fitness", "overlap", "angle"], "good")
        if score > 0.9:
            break
    else:
        raise ValueError(f"Score too low: {score:.2f}. Try running again.")

    print(f"Modelled coregistered dDEM usability with an accuracy of: {score:.2f}")

    # Load all of the dDEMs (not just the ones with manuall quality flags)
    all_files = pd.DataFrame(data=dem_tools.find_dems(CACHE_FILES["ddems_coreg_dir"]), columns=["filepath"])
    all_files["station_name"] = all_files["filepath"].apply(dem_tools.extract_station_name)

    # Read the ICP coregistration statistics from each respective coregistered DEM
    for i, row in all_files.iterrows():
        try:
            stats = read_icp_stats(row["station_name"])
        # If coregistration failed, it will not have generated a stats file.
        except FileNotFoundError:
            all_files.drop(index=i, inplace=True)
            continue
        all_files.loc[i, list(stats.keys())] = list(stats.values())

    # Use the classifier to classify all dDEMs based on the coregistration quality.
    all_files["good"] = model.predict(all_files[["fitness", "overlap", "angle"]])

    # Find the filepaths corresponding to the dDEMs that were classified as good
    good_filepaths = list(all_files[all_files["good"]]["filepath"].values)

    return good_filepaths

    stopped = False
    if improve:
        for filepath in all_files[~all_files["good"]]["filepath"].values:
            if filepath in existing_filepaths:
                continue
            stopped = examine_ddem(filepath)
            existing_filepaths.append(filepath)

    non_coreg_log = pd.read_csv(CACHE_FILES["ddem_quality"], header=None, names=["filepath", "quality"])
    # .drop_duplicates("filepath", keep="last")
    # Keep only the filepaths containing /ddems/ (not /coreg_ddems/)
    non_coreg_log = non_coreg_log[non_coreg_log["filepath"].str.contains("/ddems/")]
    # Remove the "unclear" marks
    non_coreg_log = non_coreg_log[non_coreg_log["quality"] != "unclear"]
    # Convert the good/bad quality column into a boolean column
    non_coreg_log["good"] = non_coreg_log["quality"] == "good"
    non_coreg_log["station_name"] = non_coreg_log["filepath"].apply(dem_tools.extract_station_name)

    ddem_stats = get_ddem_stats(CACHE_FILES["ddems_dir"])
    training_cols = ddem_stats.columns[2:]
    for i, row in non_coreg_log.iterrows():
        stat_row = ddem_stats.loc[ddem_stats["filepath"] == row["filepath"]].iloc[0]

        non_coreg_log.loc[i, training_cols] = stat_row.loc[training_cols]

    model, score = model_df(non_coreg_log, training_cols, "good")
    print(f"Modelled normal dDEM usability with an accuracy of {score:.2f}")

    ddem_stats["good"] = model.predict(ddem_stats[training_cols])

    good_log = log[log["good"]]
    output_filepaths: list[str] = list(log[log["good"]]["filepath"].values)

    return output_filepaths

    usable_old_ddems = 0
    for i, row in ddem_stats.iterrows():
        if not row["good"] or row["station_name"] in good_log["station_name"].values:
            continue
        usable_old_ddems += 1
        output_filepaths.append(row["filepath"])

    return output_filepaths


def plot_periglacial_error(show: bool = False):
    """
    Compute and plot the periglacial error distribution from the merged dDEM.

    :param show: Open the result using xdg-open.
    """
    ddem_dataset = rio.open(CACHE_FILES["merged_ddem"])
    glacier_mask = dem_tools.read_and_crop_glacier_mask(ddem_dataset, resampling=rio.warp.Resampling.nearest)

    ddem_values = ddem_dataset.read(1)
    print("Read dDEM and glacier mask")
    periglacial_values = ddem_values[~glacier_mask]
    periglacial_values = periglacial_values[~np.isnan(periglacial_values)]

    periglacial_values = periglacial_values[np.abs(periglacial_values) < np.percentile(periglacial_values, 99)]

    median, std = np.nanmedian(periglacial_values), np.nanstd(periglacial_values)
    plt.figure(figsize=(5, 3))
    plt.hist(periglacial_values, bins=100, color="black")
    plt.text(0.65, 0.8, s=f"Median: {median:.2f} m\nStdev: {std:.2f} m", transform=plt.gca().transAxes)
    plt.xlabel("Periglacial elevation change (m)")
    plt.ylabel("Pixel frequency")
    plt.tight_layout()
    out_filepath = os.path.join(files.TEMP_DIRECTORY, "figures/periglacial_error.jpg")
    plt.savefig(out_filepath, dpi=300)

    if show:
        subprocess.run(["xdg-open", out_filepath], check=True)
    # plt.show()


def get_normalized_elevation_vs_change(redo: bool = False) -> pd.Series:
    """
    Normalise the elevation for each separate glacier and return the hypsometry + elevation change.

    :param redo: Overwrite the cache file.
    """
    # Check if a cache file exists.
    if not redo and os.path.isfile(CACHE_FILES["elevation_vs_change_data"]):
        return pd.read_csv(CACHE_FILES["elevation_vs_change_data"], header=None,
                           index_col=0, squeeze=True, dtype=np.float64)
    glaciers = gpd.read_file(files.INPUT_FILES["outlines_1935"])
    yearly_ddem = rio.open(CACHE_FILES["merged_yearly_ddem"])
    full_glacier_mask = rio.open(outlines.CACHE_FILES["glacier_mask"])

    # These arrays will iteratively be filled.
    ddem_vals = np.empty(shape=(0, 1), dtype=float)
    heights = np.empty(shape=(0, 1), dtype=float)

    for _, glacier in tqdm(glaciers.iterrows(), total=glaciers.shape[0]):
        # Reads the bounds in xmin, ymin, xmax, ymax
        bounds = glacier.geometry.bounds
        # Make slightly larger bounds than the glacier to cover it completely.
        larger_bounds = np.array([bound - bound % CONSTANTS.dem_resolution for bound in bounds])
        larger_bounds[[0, 1]] -= CONSTANTS.dem_resolution
        larger_bounds[[2, 3]] += CONSTANTS.dem_resolution
        dem_bounds = dict(zip(["west", "south", "east", "north"], larger_bounds))

        ddem = dem_tools.reproject_dem(yearly_ddem, dem_bounds, resolution=CONSTANTS.dem_resolution,
                                       resampling=rio.warp.Resampling.nearest)
        # Check that valid dDEM values exist.
        if np.all(np.isnan(ddem)):
            continue

        glacier_mask = dem_tools.reproject_dem(full_glacier_mask, dem_bounds,
                                               resolution=CONSTANTS.dem_resolution,
                                               resampling=rio.warp.Resampling.nearest) == 1
        reference_dem = dem_tools.load_reference_elevation(dem_bounds)
        # Get the mask of where valid glacier values exist in all rasters.
        valid_mask = ~np.isnan(ddem) & glacier_mask & ~np.isnan(reference_dem)
        if np.count_nonzero(valid_mask) == 0:
            continue

        # Get the old heights by multipling the dDEM with a standardised (2018 - 1930) timeframe
        # TODO: Make the timeframe dynamic.
        old_heights = reference_dem - (ddem * (2018 - 1930))
        normalized_heights = (old_heights - np.nanmin(old_heights)) / (np.nanmax(old_heights) - np.nanmin(old_heights))

        ddem_vals = np.append(ddem_vals, ddem[valid_mask])
        heights = np.append(heights, normalized_heights[valid_mask])

    # Convert it into a series and cache it.
    elevation_change = pd.Series(index=heights, data=ddem_vals)
    elevation_change.to_csv(CACHE_FILES["elevation_vs_change_data"], header=False)

    yearly_ddem.close()
    full_glacier_mask.close()

    return elevation_change


def plot_regional_mb_gradient():
    """
    Plot the Swiss-wide relationship between elevation change (m/a) and elevation.

    One plot is using glacier-wise normalized elevation vs. elevation change.
    """
    cache_filepath = os.path.join(TEMP_DIRECTORY, "heights_and_glacier_ddem.pkl")

    # Preparing the data takes a while, so it is cached to make things easier.
    if os.path.isfile(cache_filepath):
        with open(cache_filepath, "rb") as infile:
            heights, glacier_vals = pickle.load(infile)
    else:
        ddem_dataset = rio.open(CACHE_FILES["merged_yearly_ddem"])
        glacier_mask = dem_tools.read_and_crop_glacier_mask(ddem_dataset, resampling=rio.warp.Resampling.nearest)
        print("Read glacier mask")

        ddem_values = ddem_dataset.read(1)
        print("Read dDEM")

        eastings, northings = np.meshgrid(
            np.linspace(ddem_dataset.bounds.left, ddem_dataset.bounds.right, num=ddem_dataset.width),
            np.linspace(ddem_dataset.bounds.top, ddem_dataset.bounds.bottom, num=ddem_dataset.height)
        )
        glacier_vals = ddem_values[glacier_mask]
        inlier_mask = ~np.isnan(glacier_vals)
        glacier_vals = glacier_vals[inlier_mask]

        # Get the easting and northing coordinates of the valid glacier values
        filtered_eastings = eastings[glacier_mask][inlier_mask]
        filtered_northings = northings[glacier_mask][inlier_mask]

        dem = rio.open(base_dem.CACHE_FILES["base_dem"])
        print("Sampling DEM values...")
        # Sample the base DEM at the valid easting and northing coordinates.
        heights = np.fromiter(dem.sample(zip(filtered_eastings, filtered_northings)), dtype=np.float64)
        # Fix nans
        heights[heights < 0] = np.nan

        assert heights.shape == glacier_vals.shape

        with open(cache_filepath, "wb") as outfile:
            pickle.dump((heights, glacier_vals), outfile)

    # Make the old heights by multiplying the yearly dDEM values with (2018 - 1930)
    # TODO: Make the timeframe dynamic.
    old_heights = heights - glacier_vals * (2018 - 1930)
    # Retain only reasonable values
    inlier_mask = (glacier_vals < 2) & (old_heights < CONSTANTS.max_height)

    old_heights = old_heights[inlier_mask]
    glacier_vals = glacier_vals[inlier_mask]

    # Divide the values into suitable bins.
    step = 320.0
    y_bins = np.arange(
        old_heights.min() - (old_heights.min() % step),
        old_heights.max() - (old_heights.max() % step) + step,
        step=step
    )
    indices = np.digitize(old_heights, y_bins) - 1

    # Parameters for matplotlib plots.
    box_params = dict(
        vert=False,
        flierprops=dict(alpha=0.2, markersize=0.1, zorder=3),
        medianprops=dict(color="black", zorder=3),
        zorder=3
    )
    hist_params = dict(
        orientation="horizontal",
        bins=100,
        zorder=1,
        alpha=0.3,
        histtype="stepfilled",
        edgecolor="indigo"
    )

    fig = plt.figure(figsize=(8, 5))
    ax1 = plt.subplot(121)

    # Show the elevation vs. elevation change plot
    plt.boxplot(
        x=[glacier_vals[indices == index] for index in np.unique(indices)],
        positions=y_bins + step / 2,
        widths=step / 2,
        **box_params
    )

    plt.vlines(0, *plt.gca().get_ylim(), linestyles="--", color="black", zorder=0, alpha=0.5, lw=1)
    plt.text(0.01, 0.5, "Elevation (m a.s.l.)",
             transform=fig.transFigure, ha="center", va="center", rotation=90)
    plt.gca().set_yticklabels([plt.Text(text=f"{tick:.0f}")
                               for tick in plt.gca().get_yticks()])
    plt.xlim(-4, 2)
    plt.xticks([tick for tick in plt.gca().get_xticks() if tick < 2])
    plt.ylim(y_bins.min() * 0.9, y_bins.max() * 1.1)

    ax2 = plt.gca().twiny()
    # Show the area distribution (hypsometry) plot.
    ax2.hist(heights, **hist_params)
    ax1.set_zorder(ax2.get_zorder()+1)
    ax1.patch.set_visible(False)

    ax3 = plt.subplot(122)
    elevation_change = get_normalized_elevation_vs_change()
    elevation_change = elevation_change[~np.isnan(elevation_change.index.values)]

    heights = elevation_change.index.values
    elev_change = elevation_change.values

    step = 0.1
    y_bins = np.arange(
        heights.min() - (heights.min() % step),
        heights.max() - (heights.max() % step) + step,
        step=step
    )
    indices = np.digitize(heights, y_bins) - 1

    # Show the normalized elevation vs. elevation change plot
    plt.boxplot(
        x=[elev_change[indices == index] for index in np.unique(indices)],
        positions=y_bins + step / 2,
        widths=step / 2,
        **box_params
    )

    plt.vlines(0, *plt.gca().get_ylim(), linestyles="--", color="black", zorder=0, alpha=0.5, lw=1)
    plt.gca().set_yticklabels(
        [plt.Text(text=f"{val - step / 2:.1f}–{val + step / 2:.1f}") for val in plt.gca().get_yticks()])

    plt.text(0.99, 0.5, "Normalised elevation ((Z - Zₘᵢₙ) / (Zₘₐₓ - Zₘᵢₙ))",
             transform=fig.transFigure, ha="center", va="center", rotation=-90)

    plt.ylim(-0.03, 1.03)
    plt.xlim(-4, 2)
    plt.gca().yaxis.tick_right()

    ax4 = plt.gca().twiny()
    # Show the normalized area distribution (hypsometry) plot.
    ax4.hist(heights, **hist_params)
    ax3.set_zorder(ax4.get_zorder()+1)
    ax3.patch.set_visible(False)
    ax4.set_xlim(*ax4.get_xlim()[::-1])

    plt.text(0.5, 0.99, "Area distribution", transform=fig.transFigure, ha="center", va="top")
    plt.text(0.5, 0.01, "Elevation change (m · a⁻¹)", transform=fig.transFigure, ha="center")

    plt.subplots_adjust(left=0.08, right=0.90, bottom=0.1, top=0.90, wspace=0.02)

    plt.savefig(os.path.join(files.FIGURE_DIRECTORY, "elevation_vs_dhdt.jpg"), dpi=300)


class DHModel:
    """A dH/H model that is linear below the present-day glacier and a polynomial above."""

    def __init__(self, min_elevation: float):
        """
        Generate a new model.

        :param min_elevation: The present-day minimum glacier elevation.
        """
        self.min_elevation = min_elevation
        self.lower_model = sklearn.linear_model.LinearRegression()
        self.upper_model = sklearn.pipeline.make_pipeline(
            sklearn.preprocessing.PolynomialFeatures(3),
            sklearn.linear_model.LinearRegression()
        )

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        """
        Fit the model to the given training values.

        :param x_train: Elevation values.
        :param y_train: Elevation change values.
        """
        assert x_train.shape[0] == y_train.shape[0]
        lower_mask = ~np.isnan(y_train) & (x_train.flatten() < self.min_elevation)
        upper_mask = ~np.isnan(y_train) & (x_train.flatten() > self.min_elevation)

        self.lower_model.fit(x_train[lower_mask], y_train[lower_mask])
        self.upper_model.fit(x_train[upper_mask], y_train[upper_mask])

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """
        Use the model to predict elevation changes.

        :param x_test: Elevation values to get corresponding elevation change values for.
        :returns: An array of modelled elevation change values.
        """
        lower_mask = (x_test.flatten() < self.min_elevation)
        upper_mask = (x_test.flatten() > self.min_elevation)
        lower_predictions = self.lower_model.predict(x_test[lower_mask])
        upper_predictions = self.upper_model.predict(x_test[upper_mask])
        all_predictions = np.append(lower_predictions, upper_predictions)

        return all_predictions


def plot_local_hypsometric(glacier_id=10527, min_elevation=0):
    """
    Run the local hypsometric approach on a glacier (Freudinger id.) and plot it.

    plt.show() is not run and has to be done separately.

    :param glacier_id: The "EZGNR" id in the Freudinger collection.
    :param min_elevation: The minimum modern day glacier elevation.
    """
    glacier_outlines = gpd.read_file(files.INPUT_FILES["outlines_1935"])
    # Find the outline of the chosen glacier.
    outline = glacier_outlines.loc[glacier_outlines["EZGNR"] == glacier_id].iloc[0]

    dem = rio.open(base_dem.CACHE_FILES["base_dem"])
    ddem = rio.open(CACHE_FILES["merged_ddem"])

    # Crop the DEM and dDEM to the glacier outline.
    dem_cropped, _ = rio.mask.mask(dem, outline.geometry, crop=True, nodata=np.nan)
    ddem_cropped, _ = rio.mask.mask(ddem, outline.geometry, crop=True, nodata=np.nan)

    mask = ~np.isnan(ddem_cropped)
    # Generate an "old DEM" by subtracting the DEM with the dDEM
    old_heights = (dem_cropped - (ddem_cropped))[mask]
    ddem_vals = ddem_cropped[mask]

    # Generate height bins for every 50 m if the glacier has a range of more than 500 m, otherwise 10% bins.
    height_bins = np.arange(old_heights.min(), old_heights.max(), step=50) \
        if (old_heights.max() - old_heights.min()) > 500\
        else np.linspace(old_heights.min(), old_heights.max(), num=10)

    # The middle height is the height bin minus the step size
    # The first height bin is "everything underneath the np.digitize run", so exclude this value.
    height_middles = (height_bins - (height_bins[0] - height_bins[1]) / 2)[1:]

    # Bin index 0 means below height_bins.min(), bin index len(height_bins) is above height_bins.max()
    # These should not be kept
    bin_indices = np.digitize(old_heights, height_bins)
    ddem_binned, ddem_counts, ddem_errors = np.array([  # pylint: disable=unpacking-non-sequence
        [
            np.mean(ddem_vals[bin_indices == i]) if i in bin_indices else np.nan,
            len(ddem_vals[bin_indices == i]),
            np.std(ddem_vals[bin_indices == i]) if i in bin_indices else np.nan
        ] for i in np.arange(1, bin_indices.max())
    ], dtype=float).T

    # Generate and fit a dH model
    dh_model = DHModel(min_elevation=min_elevation)
    dh_model.fit(height_middles.reshape(-1, 1), ddem_binned)
    predicted_ddem = dh_model.predict(height_middles.reshape(-1, 1))

    # Predict a historic DEM using the modern elevation and the dH model.
    predicted_heights = dem_cropped.copy()
    # Offset the measured heights with predicted dH where the measured heights are not nan
    predicted_heights[~np.isnan(predicted_heights)
                      ] -= dh_model.predict(predicted_heights[~np.isnan(predicted_heights)].reshape(-1, 1))

    # Bin the historic (predicted) elevations to get the historic hypsometry.
    historic_bin_indices = np.digitize(predicted_heights.flatten(), height_bins)
    historic_hist = np.array([len(predicted_heights.flatten()[historic_bin_indices == i])
                              for i in np.arange(1, historic_bin_indices.max())])
    # Bin the modern elevations to get the modern hypsometry
    # TODO: This is not actually the hypsometry since it is not cropped by a modern outline.
    modern_bin_indices = np.digitize(dem_cropped.flatten(), height_bins)
    modern_hist = np.array([len(dem_cropped.flatten()[modern_bin_indices == i])
                            for i in np.arange(1, modern_bin_indices.max())])

    # The elevation change is the sum of the volume change divided by the sum of the area, times the glacier density.
    elevation_change = (np.sum(historic_hist * (CONSTANTS.dem_resolution ** 2) * predicted_ddem) /
                        (np.sum(np.mean([historic_hist, modern_hist], axis=0) * (CONSTANTS.dem_resolution ** 2))))\
        * CONSTANTS.ice_density

    # Get the approximate dDEM coverage by dividing the dDEM pixel counts with the estimated hypsometry.
    ddem_count_percentages = np.clip((ddem_counts / historic_hist) * 100, 0, 100)

    plt.errorbar(ddem_binned, height_middles, xerr=ddem_errors)
    for i, count in enumerate(ddem_count_percentages):
        plt.annotate(f"{count:.0f}%", (ddem_binned[i] + ddem_errors[i] + 0.01, height_middles[i]), va="center")

    plt.plot(predicted_ddem, height_middles)

    plt.title(f"Mean elevation change: {elevation_change:.2f} m w.e.")
    plt.xlabel("Elevation change (m)")
    plt.ylabel("Elevation (m a.s.l.)")


def temp_hypso():
    """Plot a quick figure showing Unteraarsgletscher and Paradisgletscher."""
    plt.figure(figsize=(12, 10))
    plt.subplot(121)
    plot_local_hypsometric(10017, min_elevation=2100)
    plt.subplot(122)
    plot_local_hypsometric(min_elevation=2300)
    plt.show()


if __name__ == "__main__":
    # try_local_hypsometric()
    temp_hypso()
    # plot_regional_mb_gradient()
    # plot_normalized_mb_gradient()

    # plot_periglacial_error(show=True)
