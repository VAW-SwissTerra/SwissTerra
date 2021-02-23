import os
import pickle
import warnings
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.model_selection
import sklearn.neural_network
import sklearn.pipeline
import sklearn.preprocessing

from terra import files
from terra.constants import CONSTANTS


def read_mass_balance():
    data = pd.read_csv(files.INPUT_FILES["massbalance_index"], skiprows=2,
                       delimiter=" * ", engine="python", index_col=0)

    return data


def model_mass_balance():

    mb_data = read_mass_balance()
    # mb_data.cumsum().plot()
    # plt.show()
    mb_data = mb_data[sorted(mb_data.columns, key=len, reverse=True)]

    #mb_data.columns = sorted([col[0] + col[1:].zfill(2) if len(col) > 1 else col for col in mb_data], reverse=True)

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
