import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statictypes

from terra import files, metadata


@statictypes.enforce
def get_instrument_names() -> np.ndarray:
    """Return the names of each instrument in the metadata cache."""
    image_meta = metadata.image_meta.read_metadata()

    return np.unique(image_meta["Instrument"])


def plot_instrument_years() -> None:
    """Plot the temporal distribution of used instruments."""
    image_meta = metadata.image_meta.read_metadata()

    fig = plt.figure(figsize=(9, 5), dpi=80)
    fig.canvas.set_window_title('Instrument distribution')
    axis = plt.gca()

    instrument_labels = list(np.unique(image_meta["Instrument"]))
    instrument_labels.sort()
    indices = {instrument: index for index, instrument in enumerate(instrument_labels)}
    image_meta["instrument_index"] = image_meta["Instrument"].apply(lambda x: indices[x])
    image_meta["date_s"] = image_meta["date"].astype(int)
    camera_order = image_meta.groupby("instrument_index").median()["date_s"].sort_values()

    for i, instrument in enumerate(instrument_labels[index] for index in camera_order.index):
        cameras = image_meta.loc[image_meta["Instrument"] == instrument]

        axis.text(x=i, y=cameras["date"].max().year, s=cameras.shape[0], ha="center", va="bottom")
        axis.violinplot(dataset=cameras["date"].apply(
            lambda x: x.year), positions=[i], widths=0.8, showmedians=True)

    axis.set_xticks(camera_order.index)
    axis.set_xticklabels([plt.Text(x=position, text=instrument_labels[position])
                          for position in camera_order.index], rotation=45)
    plt.ylabel("Year")
    plt.xlabel("Instrument label")
    plt.grid(alpha=0.5, zorder=0)
    plt.tight_layout()

    dirname = os.path.join(files.TEMP_DIRECTORY, "figures")
    os.makedirs(dirname, exist_ok=True)
    plt.savefig(os.path.join(dirname, "instrument_years.png"), dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_instrument_years()
