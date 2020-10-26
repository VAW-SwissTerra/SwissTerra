import matplotlib.pyplot as plt
import statictypes
import numpy as np

from terra import metadata


@statictypes.enforce
def get_instrument_names() -> np.ndarray:
    """Return the names of each instrument in the metadata cache."""
    image_meta = metadata.image_meta.read_metadata()

    return np.unique(image_meta["Instrument"])



def plot_instrument_years() -> None:
    """Plot the temporal distribution of used instruments."""
    image_meta = metadata.image_meta.read_metadata()

    fig = plt.figure()
    axis = fig.add_axes([0,0,1,1])

    for instrument_name, cameras in image_meta.groupby("Instrument"):
        axis.violinplot(dataset=cameras["date"], positions=[instrument_name])


    plt.show()
        



if __name__ == "__main__":
    print(get_instrument_names())
