import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from terra import files


def main():

    filenames = {
        "A_upper": "000-175-223.tif",
        "A_lower": "000-175-232.tif",
        "B_upper": "000-175-227.tif",
        "B_lower": "000-175-236.tif",
    }

    images = {}
    for key in filenames:
        img = cv2.imread(os.path.join(files.INPUT_DIRECTORIES["image_dir"], filenames[key])).astype("float32")
        img -= np.percentile(img, 10)
        img *= (255 / np.percentile(img, 95))
        img -= img * 0.1
        images[key] = np.clip(img, 0, 255).astype("uint8")

    def ul_to_extent(image, upper, left):
        return (left, left + image.shape[1], upper - image.shape[0], upper)

    plt.figure(figsize=(7, 3.5))
    plt.imshow(images["A_lower"], cmap="Greys_r", extent=ul_to_extent(images["A_lower"], -2400, 0), alpha=1)
    plt.imshow(images["A_upper"], cmap="Greys_r", extent=ul_to_extent(images["A_upper"], 0, 0), alpha=1)

    plt.imshow(images["B_lower"], cmap="Greys_r", extent=ul_to_extent(images["B_lower"], -2400, 7400), alpha=1)
    plt.imshow(images["B_upper"], cmap="Greys_r", extent=ul_to_extent(images["B_upper"], 0, 7400), alpha=1)

    plt.ylim(-2400 - images["A_lower"].shape[0], 0)
    plt.xlim(0, 7400 + images["B_lower"].shape[1])
    plt.axis("off")
    plt.subplots_adjust(0, 0, 1, 1)

    plt.savefig(os.path.join(files.FIGURE_DIRECTORY, "example_images.jpg"), dpi=300)


if __name__ == "__main__":
    main()
