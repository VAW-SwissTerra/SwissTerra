import concurrent.futures
import os
import pickle
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import sklearn.linear_model
import sklearn.model_selection
import sklearn.neural_network
import sklearn.pipeline


def read_data():
    temp_file = "temp/values.pkl"

    if os.path.isfile(temp_file):
        with open(temp_file, "rb") as infile:
            output = pickle.load(infile)
            return output
    from terra import evaluation

    ddem = rio.open(evaluation.CACHE_FILES["merged_yearly_ddem"])
    year_change = 2018 - 1935

    bounds = dict(zip(["west", "south", "east", "north"], ddem.bounds._asdict().values()))
    eastings, northings = np.meshgrid(
        np.arange(bounds["west"], bounds["east"], step=int(ddem.res[0])),
        np.arange(bounds["south"], bounds["north"], step=int(ddem.res[0]))[::-1])

    glacier_mask = evaluation.read_and_crop_glacier_mask(ddem, resampling=rio.warp.Resampling.nearest)
    print("Read glacier mask")

    def read_raster(product):
        if product == "ddem":
            return ddem.read(1)
        raster = evaluation.load_reference_elevation(bounds, base_dem_prefix=product)
        raster[raster == -9999] = np.nan
        print(f"Read {product}")
        return raster
    # dem = evaluation.load_reference_elevation(bounds, base_dem_prefix="base_dem")
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        dem, slope, aspect, ddem_vals = list(executor.map(read_raster, ["base_dem", "slope", "aspect", "ddem"]))
    # slope = evaluation.load_reference_elevation(bounds, base_dem_prefix="slope")
    # aspect = evaluation.load_reference_elevation(bounds, base_dem_prefix="aspect")
    ddem_vals[~glacier_mask] = np.nan
    aspect[aspect == -9999] = np.nan
    slope[slope == -9999] = np.nan

    x_dir = np.sin(np.deg2rad(aspect))
    y_dir = np.cos(np.deg2rad(aspect))

    z_dir = np.sin(np.deg2rad(slope))
    x_dir *= z_dir
    y_dir *= z_dir

    old_heights = dem - ddem_vals * year_change

    mask = ~np.isnan(old_heights)

    output = np.dstack((eastings[mask], northings[mask], old_heights[mask],
                        x_dir[mask], y_dir[mask], z_dir[mask], ddem_vals[mask]))

    print(output, output.shape)
    with open("temp/values.pkl", "wb") as outfile:
        pickle.dump(output, outfile)

    return outfile


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()


def neural():
    import tensorflow as tf
    from tqdm.keras import TqdmCallback
    data = read_data()[0, :, :]

    data = data[~np.isnan(data).any(axis=1)]
    #data = data[:50000, :]
    data -= np.min(data, axis=0)
    data /= np.max(data, axis=0)

    data = data.astype("float32")

    all_xvals = data[:, :-1]
    all_yvals = data[:, -1]

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(all_xvals, all_yvals)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1)])

    model.compile(
        loss="mean_absolute_error",
        optimizer=tf.keras.optimizers.Adadelta())

    history = model.fit(x_train, y_train, epochs=1000, batch_size=64, verbose=0, callbacks=[TqdmCallback(verbose=0)])

    predictions = model.predict(x_test)

    plt.scatter(y_test, predictions)
    plt.plot([0, 1], [0, 1])
    plt.show()

    plot_loss(history)
