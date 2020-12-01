"""Manual fiducial placement helper functions and GUIs."""
from __future__ import annotations

import csv
import os
import time
from typing import Any, NamedTuple, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PySimpleGUI as sg

from terra import files, preprocessing
from terra.constants import CONSTANTS
from terra.preprocessing import fiducials

# TODO: Get these numbers without instantiating the framematcher (it takes time..)
FIDUCIAL_LOCATIONS = CONSTANTS.zeiss_fiducial_locations
WINDOW_RADIUS = 250
POINT_RADIUS = 4
DEFAULT_FRAME_TYPE_NAME = "default"


CACHE_FILES = {
    "marked_fiducials": os.path.join(files.INPUT_ROOT_DIRECTORY, "../manual_input/marked_fiducials.csv")
}


def get_fiducials(filepath: str) -> dict[str, bytes]:
    """
    Extract the parts of an image that correspond approximately to where the fiducials are.

    param: filepath: The filepath of the image.

    return: fiducial_imgs: A dictionary of four in-memory PNG images (left, bottom, right, top).
    """
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    # Go through each corner and extract corresponding parts of the image
    fiducial_imgs: dict[str, np.ndarray] = {}
    for corner, coord in FIDUCIAL_LOCATIONS.items():
        fiducial = image[coord[0] - WINDOW_RADIUS: coord[0] + WINDOW_RADIUS,
                         coord[1] - WINDOW_RADIUS: coord[1] + WINDOW_RADIUS]

        # Increase the contrast of the fiducial to simplify the visualisation
        fiducial -= fiducial.min()
        fiducial = (fiducial * (255 / fiducial.max())).astype(fiducial.dtype)

        # Convert the image to a png in memory (for PySimpleGUI)
        imgbytes = cv2.imencode(".png", fiducial)[1].tobytes()
        fiducial_imgs[corner] = imgbytes

    return fiducial_imgs


def draw_fiducials(window: sg.Window, fiducial_imgs: dict[str, bytes]):
    """
    Draw the fiducial images onto a PySimpleGUI window.

    param: window: The PySimpleGUI window to draw on.
    param: fiducial_imgs: The fiducial images to draw.

    """

    for corner, img in fiducial_imgs.items():
        window[corner].DrawImage(data=img, location=(WINDOW_RADIUS * 2, WINDOW_RADIUS * 2))


def get_unprocessed_images() -> np.ndarray:
    """
    Get the filenames of every image without fiducial marks.
    """
    image_meta = preprocessing.image_meta.read_metadata()

    filenames = image_meta[image_meta["Instrument"].str.contains("Zeiss")]["Image file"].values

    filenames = image_meta[image_meta["Instrument"] == "Zeiss1"]["Image file"].values

    return filenames


def draw_image_by_index(index: int, filepaths: np.ndarray, window: sg.Window, marked_fiducial_points: dict[str, int], marked_fiducial_circles: dict[str, int]):
    """
    Draw an image with the corresponding index in the filepaths array.

    param: index: The index of the image to draw.
    param: filepaths: An array of filepaths to index.
    param: window: The PySimpleGUI window instance.
    param: marked_fiducial_points: The points dictionary to update accordingly.
    """
    fiducial_imgs = get_fiducials(filepaths[index])
    draw_fiducials(window, fiducial_imgs)
    for corner in fiducial_imgs:
        # Make a point slightly outside of the canvas which (when moved) represents a marked fiducial
        marked_fiducial_points[corner] = window[corner].DrawPoint((-1, -1), size=POINT_RADIUS * 2, color="red")
        marked_fiducial_circles[corner] = window[corner].DrawCircle((-1, -1), radius=POINT_RADIUS * 3, line_color="red")


class FiducialMark(NamedTuple):  # pylint: disable=inherit-non-class, too-few-public-methods
    """A manually placed fiducial mark with corresponding information."""

    filename: str
    corner: str
    x_position: Optional[int]
    y_position: Optional[int]
    frame_type: str


class MarkedFiducials:
    """A collection of fiducial marks and associated methods."""

    def __init__(self, fiducial_marks: Optional[list[FiducialMark]] = None, filepath: Optional[str] = None):
        """
        Initialise a new empty instance.

        param: fiducial_marks: Optional lsit of fiducial marks to specify.
        param: filepath: Optional cache filepath. Will be set/modified if to_csv is run.
        """
        self.fiducial_marks: list[FiducialMark] = fiducial_marks or []

        self.filepath: Optional[str] = filepath

    @ staticmethod
    def read_csv(filepath: str) -> MarkedFiducials:
        """
        Instantiate the class by reading it from a csv.

        param: filepath: The csv to read.
        """
        fiducial_marks: list[FiducialMark] = []
        with open(filepath) as infile:
            for row in csv.DictReader(infile):
                # Convert the file types appropriately (they will all be strings otherwise)
                for key in row:
                    try:
                        row[key] = int(row[key])  # type: ignore
                    # If it can't be converted to an integer, it's either a string or a string called "None"
                    except ValueError:
                        if row[key] == "None":  # Convert a "None" string to a NoneType
                            row[key] = None  # type: ignore
                        continue

                # Create a FiducialMark from the row's values
                fiducial_marks.append(FiducialMark(**row))  # type: ignore

        return MarkedFiducials(fiducial_marks=fiducial_marks, filepath=filepath)

    @ staticmethod
    def create_or_load(filepath: str) -> MarkedFiducials:
        """
        Load a fiducial mark collection if it can be found, otherwise create a new one.

        param: filepath: The csv to attempt to read.
        """
        if os.path.isfile(filepath):
            return MarkedFiducials.read_csv(filepath)

        return MarkedFiducials(filepath=filepath)

    def to_csv(self, filepath: str):
        """
        Write the fiducial mark collection to a file and set the filepath attribute accordingly.

        param: filepath: The filepath of the csv to write.
        """
        with open(filepath, "w") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=FiducialMark._fields)
            writer.writeheader()
            for fiducial_mark in self.fiducial_marks:
                writer.writerow(fiducial_mark._asdict())
        self.filepath = filepath

    def save(self):
        """Write the fiducial mark collection to the file specified with the filepath attribute."""
        if self.filepath is None:
            raise ValueError("The filepath attribute is not set.")

        self.to_csv(self.filepath)

    def add(self, fiducial_mark: FiducialMark) -> None:
        """Add a fiducial mark to the collection."""
        self.fiducial_marks.append(fiducial_mark)

    def get_fiducial_marks(self, filename: str) -> dict[str, Optional[FiducialMark]]:
        """
        Get the fiducial marks from a specific image.

        param: filename: The filename of the image.

        return: marks: The fiducial marks (if any) of the image.
        """
        fiducial_marks_for_file = [mark for mark in self.fiducial_marks if mark.filename == filename]

        # Make a dictionary of fiducial marks which are None by default
        marks: dict[str, Optional[FiducialMark]] = {corner: None for corner in FIDUCIAL_LOCATIONS}

        # Add the fiducial marks that exist (the nonexistent will still be None)
        for fiducial_mark in fiducial_marks_for_file:
            marks[fiducial_mark.corner] = fiducial_mark

        return marks

    def get_used_frame_types(self) -> list[str]:
        """Get all the frame types that have been used before."""
        # Make an empty set of strings (sets only allow unique values, allowing for a unique list to be created rapidly)
        types: set[str] = set()

        # Add the frame types to the set, possibly overwriting ones that already exist.
        for fiducial_mark in self.fiducial_marks:
            types.add(fiducial_mark.frame_type)

        return list(types)

    def get_filenames_for_frame_type(self, frame_type: str) -> list[str]:
        """Get all the filenames corresponding to the frame type."""
        if self.fiducial_marks is None:
            raise ValueError("No fiducial marks available.")

        # Get the latest frame types (newer entires overwrite the older ones for the same filename)
        frame_types: dict[str, str] = {mark.filename: mark.frame_type for mark in self.fiducial_marks}

        filenames: list[str] = [filename for filename in frame_types if frame_types[filename] == frame_type]

        return filenames


def gui():
    """Show a GUI to mark fiducials in images."""
    filenames = get_unprocessed_images()
    np.random.shuffle(filenames)

    filepaths = files.INPUT_DIRECTORIES["image_dir"] + filenames

    marked_fiducials = MarkedFiducials.create_or_load(CACHE_FILES["marked_fiducials"])
    frame_types = marked_fiducials.get_used_frame_types()

    # Create the GUI layout
    grid: list[sg.Column] = []
    # Make a column for each fiducial mark
    for i, corner in enumerate(FIDUCIAL_LOCATIONS):
        image_viewer_layout = [
            [sg.Text(f"{corner.capitalize()} fiducial")],
            [sg.Graph(  # The graph is where the fiducial and marks are rendered
                canvas_size=(WINDOW_RADIUS * 2, WINDOW_RADIUS * 2),
                graph_bottom_left=(WINDOW_RADIUS * 2, 0),
                graph_top_right=(0, WINDOW_RADIUS * 2),
                key=corner,
                enable_events=True,
                drag_submits=True
            )],
        ]

        # Add a previous/next image button to the first column
        if i == 0:
            image_viewer_layout[0].insert(1, sg.Button("Previous", key="previous"))
            image_viewer_layout[0].insert(2, sg.Button("Next", key="next"))

        grid.append(sg.Column(image_viewer_layout))

    # Add a column for selecting frame types
    grid.append(sg.Column(
        [
            # TODO: Increase the allowed column width here.
            [sg.Text(text=f"Selected type: {DEFAULT_FRAME_TYPE_NAME}", key="selected-type", size=(20, 2))],
            [sg.InputText(default_text="New type", key="new-type-text"),
             sg.Button("Submit", key="new-type-submit")],
            [sg.Listbox(values=frame_types, enable_events=True, key="types-list",
                        select_mode=sg.LISTBOX_SELECT_MODE_SINGLE, size=(20, 20))]
        ],
        key="type-layout")
    )

    # Make a window with a temporary title, using the grid that was just made
    window = sg.Window(title="Hello there", layout=[grid], return_keyboard_events=True)
    # Finalize it for whatever reason. TODO: Try removing this and see what happens
    window.Finalize()

    # Make a dictionary of points representing the marked fiducials
    marked_fiducial_points: dict[str, int] = {}
    marked_fiducial_circles: dict[str, int] = {}
    # Start a count of which image index is looked at
    image_index = 0
    # Draw the first image in the filepaths array
    draw_image_by_index(image_index, filepaths, window, marked_fiducial_points, marked_fiducial_circles)

    frame_type = DEFAULT_FRAME_TYPE_NAME

    # Create an empty dictionary of marked fiducial coordinates (in the image coordinate system)
    fiducial_coords: dict[str, Optional[list[int]]] = {}

    # Instantiate timers for when things were interacted with.
    # It's to check whether the graph or frame type was modified when the user switches picture.
    last_pressed_on_graph = 0.0
    last_updated_type = 0.0
    last_switched_image = 1.0

    # Main event loop
    while True:
        event, values = window.read()

        # Stop if the window was closed
        if event == sg.WIN_CLOSED:
            # TODO: Save the result for the active image before closing
            break

        # Switch picture
        if "next" in event or "previous" in event or event in ["Right:114", "Left:113"]:
            # Change the event name from keyboard to button style (just to make it simpler..)
            if ":" in event:
                event = "next" if event == "Right:114" else "previous"

            # Check if fiducials or frame types were modified, and save if so.
            for corner in fiducial_coords:
                # Skip if the graph or type list has not been interacted with yet on this image
                if last_pressed_on_graph < last_switched_image and last_updated_type < last_switched_image:
                    continue

                # Skip if both the last mark was set to None and this one is also None
                saved_fiducials = marked_fiducials.get_fiducial_marks(filenames[image_index])
                if fiducial_coords[corner] is None and saved_fiducials[corner] is None:
                    continue

                # Get the positions if they exist, otherwise just None
                y_position, x_position = fiducial_coords[corner] or (None, None)

                marked_fiducials.add(
                    FiducialMark(
                        filename=filenames[image_index],
                        corner=corner,
                        y_position=y_position,
                        x_position=x_position,
                        frame_type=frame_type
                    )
                )
                print(f"Saved fiducial {corner}")
            # TODO: Make this dependent on if new fiducials were actually created.
            marked_fiducials.save()

            # Update the time when the image was switched.
            last_switched_image = time.time()

            # Increment the image index up or down
            image_index += (1 if event == "next" else -1)
            # Draw the new image
            draw_image_by_index(image_index, filepaths, window, marked_fiducial_points, marked_fiducial_circles)
            # Set the title appropriately
            window.set_title(f"Examining {filenames[image_index]}")
            # Get potential previous fiducials
            saved_fiducials = marked_fiducials.get_fiducial_marks(filenames[image_index])
            # First set the pressed type to the default value, then check if a non-default value was saved before.
            #frame_type = DEFAULT_FRAME_TYPE_NAME
            # Loop through all saved fiducials (if any) and set them + the frame type accordingly
            for corner, fiducial_mark in saved_fiducials.items():
                print("Loading fiducials", fiducial_mark)
                if fiducial_mark is None:
                    # Move the graphical point outside of the canvas if there is no saved fiducial mark.
                    window[corner].RelocateFigure(figure=marked_fiducial_points[corner], x=-1, y=-1)
                    window[corner].RelocateFigure(figure=marked_fiducial_circles[corner], x=-1, y=-1)
                    continue

                # Update the frame type accordingly (the script only comes here if fiducial_mark is not None)
                frame_type = fiducial_mark.frame_type

                print(fiducial_mark)
                # Set appropriate point positions for whether or not fiducial marks exist.
                graph_x_position: int = WINDOW_RADIUS - (fiducial_mark.x_position - FIDUCIAL_LOCATIONS[corner][1])\
                    + POINT_RADIUS if fiducial_mark.x_position is not None else -1
                graph_y_position: int = WINDOW_RADIUS - (fiducial_mark.y_position - FIDUCIAL_LOCATIONS[corner][0])\
                    + POINT_RADIUS if fiducial_mark.y_position is not None else -1

                # Move the point appropriately
                window[corner].RelocateFigure(
                    figure=marked_fiducial_points[corner], x=graph_x_position, y=graph_y_position)
                window[corner].RelocateFigure(
                    figure=marked_fiducial_circles[corner], y=graph_y_position + POINT_RADIUS * 2, x=graph_x_position + POINT_RADIUS * 2)

            # Update the selected type text.
            window["selected-type"](f"Selected type: {frame_type}")

            continue  # TODO: Maybe remove this

        # Check if any of the graphs (fiducial images) were interacted with
        for corner in FIDUCIAL_LOCATIONS:
            if event != corner:
                continue

            # Update when this happened (enabling the event to be saved)
            last_pressed_on_graph = time.time()

            x_position, y_position = values[corner]

            # If the placed value was outside of the image bounds, make it None
            for value in values[corner]:
                if value < 0 or value > WINDOW_RADIUS * 2:
                    fiducial_coords[corner] = None
                    break
            # Otherwise, update the fiducial coordinates with the new value
            else:
                fiducial_coords[corner] = [WINDOW_RADIUS - [y_position, x_position][i] +
                                           FIDUCIAL_LOCATIONS[corner][i] for i in (0, 1)]

            # Move the graphical point appropriately
            window[corner].RelocateFigure(marked_fiducial_points[corner], *
                                          [value + POINT_RADIUS for value in [x_position, y_position]])
            # Move the graphical circle appropriately
            window[corner].RelocateFigure(marked_fiducial_circles[corner], *
                                          [value + POINT_RADIUS * 3 for value in [x_position, y_position]])

            # Check if a new frame type was entered in the input field
        if event == "new-type-submit":
            text = values["new-type-text"]
            # Check if the type was just "New type" (accidental click) or if it already exists
            if "New type" in text or text in frame_types:
                continue

            frame_type = text
            frame_types.append(text)
            # Update the frame types list and set the text
            window["types-list"].Update(values=frame_types)
            window["selected-type"](f"Selected type: {frame_type}")
            # Update the time so that this result will be changed if another image was pressed
            last_updated_type = time.time()
            continue

        # If a frame type was pressed in the list
        if "types-list" in event:
            frame_type = values["types-list"][0]
            window["selected-type"](f"Selected type: {frame_type}")
            last_updated_type = time.time()

    window.close()


def show_marked_fiducials(index=0):
    """
    Plot the marked fiducials on an image.

    :param index: The index of the filename to plot.
    """
    data = pd.read_csv("input/marked_fiducials.csv")
    all_filenames = np.unique(data["filename"].values)
    cam0 = data.loc[data["filename"] == all_filenames[index]]

    img0 = cv2.imread(os.path.join(
        files.INPUT_DIRECTORIES["image_dir"], all_filenames[index]), cv2.IMREAD_GRAYSCALE)

    plt.imshow(img0, cmap="Greys_r")
    plt.scatter(cam0["x_position"], cam0["y_position"])
    plt.show()


if __name__ == "__main__":
    gui()
