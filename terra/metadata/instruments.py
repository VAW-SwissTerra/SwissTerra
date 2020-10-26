import matplotlib.pyplot as plt
import numpy as np

from terra import metadata


def get_instrument_names():
    image_meta = metadata.image_meta.read_metadata()

    return np.unique(image_meta["Instrument"])


if __name__ == "__main__":
    print(get_instrument_names())
