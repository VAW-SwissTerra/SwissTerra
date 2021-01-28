import warnings
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
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

    mb_data.columns = sorted([col[0] + col[1:].zfill(2) if len(col) > 1 else col for col in mb_data], reverse=True)

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

    def get_nearest_mb(easting, northing, year):

        coord_distances = np.linalg.norm(coords - [easting, northing], axis=1)
        nearest = np.argwhere(coord_distances == coord_distances.min())[0][0]
        key = list(glacier_coords.keys())[nearest]
        massbalance = mb_data.loc[year, key]

        return massbalance

    return get_nearest_mb

    models: dict[int, Any] = {}
    for year, mass_balance_row in mb_data.iterrows():
        mass_balance = mass_balance_row[glacier_coords.keys()].values

        model = sklearn.pipeline.make_pipeline(
            sklearn.preprocessing.PolynomialFeatures(1), sklearn.linear_model.LinearRegression())
        model.fit(coords, mass_balance)
        models[year] = model

    return models


def predict_mass_balance(start_year=1940, end_year=2018, dh=-100, easting=2657131, northing=1157505):

    norm_start_year = 1920
    norm_end_year = 2020
    start_year_index = start_year - norm_start_year
    end_year_index = end_year - norm_start_year
    all_years = np.arange(norm_start_year, norm_end_year + 1)
    dh_we = dh * 0.85

    mb_series = []
    mb_model = model_mass_balance()
    for year in range(norm_start_year, norm_end_year + 1):
        mb_series.append(mb_model(easting, northing, year))
        #mb_series.append(models[year].predict(np.reshape([easting, northing], (1, -1)))[0])

    mb_cumsum = np.cumsum(mb_series)

    period_mb = mb_cumsum[start_year_index: end_year_index + 1]
    conversion_factor = dh_we / (period_mb[-1] - period_mb[0])
    mb_cumsum *= conversion_factor
    dh_we += period_mb[0]

    plt.plot(all_years, mb_cumsum, zorder=0, label="Corrected mass balance")
    plt.scatter([start_year, end_year], [period_mb[0], dh_we], marker="s",
                color="k", label="Example dH (m w.e)", zorder=1)
    plt.legend()
    plt.ylabel(f"Ice loss since {norm_start_year} (m w.e.)")
    plt.show()
