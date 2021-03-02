import os
import pickle
import warnings
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
import shapely
import sklearn.linear_model
import sklearn.model_selection
import sklearn.neural_network
import sklearn.pipeline
import sklearn.preprocessing
from tqdm import tqdm

import glacier_lengths
import glacier_lengths.plotting
from terra import base_dem, files
from terra.constants import CONSTANTS
from terra.preprocessing import outlines


def read_mass_balance():
    data = pd.read_csv(files.INPUT_FILES["massbalance_index"], skiprows=2,
                       delimiter=" * ", engine="python", index_col=0)

    return data


def model_mass_balance():

    mb_data = read_mass_balance()
    # mb_data.cumsum().plot()
    # plt.show()
    mb_data = mb_data[sorted(mb_data.columns, key=len, reverse=True)]

    # mb_data.columns = sorted([col[0] + col[1:].zfill(2) if len(col) > 1 else col for col in mb_data], reverse=True)

    outlines = gpd.read_file(files.INPUT_FILES["sgi_2016"]).to_crs(CONSTANTS.crs_epsg.replace("::", ":"))
    outlines.geometry = outlines.geometry.centroid
    outlines["sgi-zone"] = outlines["sgi-id"].apply(lambda text: text[:3])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        glacier_coords = {
            sgi_id: [
                np.median(outlines.loc[outlines["sgi-zone"].str.contains(sgi_id)].geometry.x),
                np.median(outlines.loc[outlines["sgi-zone"].str.contains(sgi_id)].geometry.y)
            ] for sgi_id in mb_data}

    for key in list(glacier_coords.keys()):
        if np.any(np.isnan(glacier_coords[key])):
            del glacier_coords[key]

    coords = np.array(list(glacier_coords.values()))
    keys = np.array(list(glacier_coords.keys()))

    @np.vectorize
    def get_nearest_mb(easting, northing, year):

        coord_distances = np.linalg.norm(coords - [easting, northing], axis=1)
        nearest = np.argwhere(coord_distances == coord_distances.min())[0][0]
        massbalance = mb_data.loc[year, keys[nearest]]

        return massbalance

    return get_nearest_mb


def predict_mass_balance(start_year=1927, end_year=2016, dh=-42.43, easting=658060, northing=157708):

    norm_start_year = 1880
    norm_end_year = 2020
    start_year_index = start_year - norm_start_year
    end_year_index = end_year - norm_start_year
    all_years = np.arange(norm_start_year, norm_end_year + 1)
    dh_we = dh * 0.85

    mb_model = model_mass_balance()
    mb_series = mb_model(easting, northing, np.arange(norm_start_year, norm_end_year + 1))
    mb_cumsum = np.cumsum(mb_series)

    period_mb = mb_cumsum[start_year_index: end_year_index + 1]
    conversion_factor = dh_we / (period_mb[-1] - period_mb[0])
    mb_cumsum *= conversion_factor
    dh_we += period_mb[0]

    plt.plot(all_years, mb_cumsum, zorder=0, label="Corrected mass balance")
    plt.scatter([start_year, end_year], [period_mb[0], dh_we], marker="s",
                color="k", label="Geodetic mass balance", zorder=1)
    plt.legend()
    plt.ylabel(f"Ice loss since {norm_start_year} (m w.e.)")
    plt.show()


def read_mb_bins():
    cache_filepath = "temp/values2.pkl"

    if os.path.isfile(cache_filepath):
        with open(cache_filepath, "rb") as infile:
            return pickle.load(infile)

    data = pd.read_csv("input/massbalance_fixdate_elevationbins.csv", sep=";", skiprows=list(range(0, 7)) + [8])
    outlines = gpd.read_file(files.INPUT_FILES["sgi_2016"])\
        .to_crs(CONSTANTS.crs_epsg.replace("::", ":"))\
        .set_index("sgi-id")
    outlines.geometry = outlines.geometry.centroid
    data["sgi-id"] = data["(according to Swiss Glacier Inventory)"].str.replace("/", "-", regex=False)
    data["geometry"] = outlines.loc[data["sgi-id"].values].geometry.values
    data[["slope", "aspect"]] = outlines.loc[data["sgi-id"].values][["slope_deg", "aspect_deg"]].values
    data["easting"] = data["geometry"].apply(lambda geom: geom.x)
    data["northing"] = data["geometry"].apply(lambda geom: geom.y)
    data["h_mid"] = (data["h_max"] + data["h_min"]) / 2
    data["year"] = data["date_end"].apply(lambda string: int(string[:string.index("-")]))

    data["x_dir"] = np.sin(np.deg2rad(data["aspect"]))
    data["y_dir"] = np.cos(np.deg2rad(data["aspect"]))

    data["z_dir"] = np.sin(np.deg2rad(data["slope"]))
    data["x_dir"] *= data["z_dir"]
    data["y_dir"] *= data["z_dir"]

    x_all = data[["easting", "northing", "h_mid", "year", "x_dir", "y_dir", "z_dir"]].values.astype(float)
    y_all = data["Ba"].values.astype(float)

    x_all -= np.min(x_all, axis=0)
    y_all -= np.min(y_all)
    x_all /= np.max(x_all, axis=0)
    y_all /= np.max(y_all)

    with open(cache_filepath, "wb") as outfile:
        pickle.dump((x_all, y_all), outfile)

    return x_all, y_all

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_all, y_all)

    model = sklearn.neural_network.MLPRegressor(
        hidden_layer_sizes=(512, 512),
        solver="adam",
        learning_rate_init=0.0001,
        max_iter=1000,
        tol=1e-7,
        verbose=True)

    model.fit(x_train.astype("float32"), y_train.astype("float32"))

    print(model.score(x_test, y_test))


def measure_glacier_lengths():
    centrelines = gpd.read_file(outlines.CACHE_FILES["lk50_centrelines"])
    outlines_lk50 = gpd.read_file(outlines.CACHE_FILES["lk50_outlines"])
    outlines_1973 = gpd.read_file(files.INPUT_FILES["sgi_1973"]).to_crs(outlines_lk50.crs)
    outlines_2016 = gpd.read_file(files.INPUT_FILES["sgi_2016"]).to_crs(outlines_lk50.crs)

    lengths = pd.DataFrame(columns=["SGI1973", "SGI2016", "LK50_date", "LK50_length", "LK50_length_std",
                                    "SGI1973_length", "SGI1973_std", "SGI2016_length", "SGI2016_std"])

    dem = rio.open(base_dem.CACHE_FILES["base_dem"])
    for sgi_id in tqdm(centrelines["SGI"].values, desc="Measuring lengths"):

        centreline = centrelines.loc[centrelines["SGI"] == sgi_id].iloc[-1].geometry

        elevs = list(dem.sample((centreline.coords[0], centreline.coords[-1])))

        if elevs[0][0] > elevs[1][0]:
            centreline = shapely.geometry.LineString(list(centreline.coords)[::-1])

        outline_lk50_row = outlines_lk50.loc[outlines_lk50["SGI"] == sgi_id].iloc[-1]
        outline_lk50 = outline_lk50_row.geometry

        sgi_ids_1973 = outline_lk50_row["SGI1973"].split(",")
        sgi_ids_2016 = list(map(outlines.sgi_1973_id_to_2016, sgi_ids_1973))

        outline_1973 = outlines_1973.loc[outlines_1973["SGI"].isin(sgi_ids_1973)].geometry.unary_union
        outline_2016 = outlines_2016.loc[outlines_2016["sgi-id"].isin(sgi_ids_2016)].geometry.unary_union

        try:
            centrelines_lk50 = glacier_lengths.buffer_centerline(
                centerline=centreline,
                glacier_outline=outline_lk50,
                min_radius=1,
                max_radius=50,
                buffer_count=20
            )
        except AssertionError as exception:
            warnings.warn(f"Line buffering for SGI {sgi_id} failed:\n{exception}")
            continue

        try:
            centrelines_1973 = glacier_lengths.cut_centerlines(centrelines_lk50, outline_1973)
            centrelines_2016 = glacier_lengths.cut_centerlines(centrelines_lk50, outline_2016)
        except AssertionError as exception:
            if "empty geometry" in str(exception):
                continue

        lengths_lk50 = glacier_lengths.measure_lengths(centrelines_lk50)
        lengths_1973 = glacier_lengths.measure_lengths(centrelines_1973)
        lengths_2016 = glacier_lengths.measure_lengths(centrelines_2016)

        std_variance = abs(lengths_lk50.std() - lengths_1973.std()) / lengths_lk50.std()
        if std_variance > 10:
            warnings.warn(f"{sgi_id} std variance above threshold: {std_variance:.2f}. Skipping")
            continue

        lengths.loc[sgi_id] = (",".join(sgi_ids_1973), ",".join(sgi_ids_2016), outline_lk50_row["date"],
                               lengths_lk50.mean(), lengths_lk50.std(), lengths_1973.mean(), lengths_1973.std(),
                               lengths_2016.mean(), lengths_2016.std())

    print(centrelines.shape[0] - lengths.shape[0], "failed")
    lengths["length_frac"] = lengths["SGI2016_length"] / lengths["LK50_length"]

    weighted_average = np.average(lengths["length_frac"], weights=lengths["LK50_length"])
    print(weighted_average)

    plt.hist(lengths["length_frac"], bins=50)
    print(lengths)
    plt.show()
