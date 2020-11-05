from __future__ import annotations

import csv
import os
import time
from typing import Any, NamedTuple, Optional

import cv2
import numpy as np
import PySimpleGUI as sg

from terra import files, metadata
from terra.fiducials import fiducials

FIDUCIAL_LOCATIONS = fiducials.FrameMatcher().fiducial_locations
WINDOW_RADIUS = 250
POINT_RADIUS = 4
DEFAULT_FRAME_TYPE_NAME = "default"


CACHE_FILES = {
    "marked_fiducials": os.path.join(files.INPUT_ROOT_DIRECTORY, "marked_fiducials.csv")
}


def get_fiducials(filepath: str) -> dict[str, np.ndarray]:

    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    fiducial_imgs: dict[str, np.ndarray] = {}

    for corner, coord in FIDUCIAL_LOCATIONS.items():
        fiducial = image[coord[0] - WINDOW_RADIUS: coord[0] + WINDOW_RADIUS,
                         coord[1] - WINDOW_RADIUS: coord[1] + WINDOW_RADIUS]
        imgbytes = cv2.imencode(".png", fiducial)[1].tobytes()
        fiducial_imgs[corner] = imgbytes

    return fiducial_imgs


def draw_fiducials(window: sg.Window, fiducial_imgs: dict[str, np.ndarray]) -> dict[str, sg.Graph]:

    graphs: dict[str, sg.Graph] = {}

    for corner, img in fiducial_imgs.items():
        graphs[corner] = window.Element(corner)
        graphs[corner].DrawImage(data=img, location=(WINDOW_RADIUS * 2, WINDOW_RADIUS * 2))

    return graphs


def get_unprocessed_images() -> np.ndarray:
    image_meta = metadata.image_meta.read_metadata()

    filenames = image_meta[image_meta["Instrument"].str.contains("Wild")]["Image file"].values

    return filenames


def draw_image_nr(count: int, filepaths: np.ndarray, window: sg.Window, points: dict[str, int]):
    fiducial_imgs = get_fiducials(filepaths[count])
    graphs = draw_fiducials(window, fiducial_imgs)
    for corner in fiducial_imgs:
        points[corner] = graphs[corner].DrawPoint((-1, -1), size=POINT_RADIUS * 2, color="red")

    return graphs


class FiducialMark(NamedTuple):
    filename: str
    corner: str
    x_position: Optional[int]
    y_position: Optional[int]
    frame_type: str


class MarkedFiducials:

    def __init__(self, fiducial_marks: Optional[list[FiducialMark]] = None, filepath: Optional[str] = None):
        self.fiducial_marks: list[FiducialMark] = fiducial_marks or []

        self.filepath: Optional[str] = filepath

    @staticmethod
    def read_csv(filepath: str) -> MarkedFiducials:

        fiducial_marks: list[FiducialMark] = []
        with open(filepath) as infile:
            for row in csv.DictReader(infile):
                for key in row:
                    try:
                        row[key] = int(row[key])  # type: ignore
                    except ValueError:
                        if row[key] == "None":
                            row[key] = None
                        continue
                fiducial_marks.append(FiducialMark(**row))  # type: ignore

        return MarkedFiducials(fiducial_marks=fiducial_marks, filepath=filepath)

    @staticmethod
    def create_or_load(filepath: str) -> MarkedFiducials:
        if os.path.isfile(filepath):
            return MarkedFiducials.read_csv(filepath)

        return MarkedFiducials(filepath=filepath)

    def to_csv(self, filepath: str):
        with open(filepath, "w") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=FiducialMark._fields)
            writer.writeheader()
            for fiducial_mark in self.fiducial_marks:
                writer.writerow(fiducial_mark._asdict())

    def save(self):
        if self.filepath is None:
            raise ValueError("No filepath specified")

        self.to_csv(self.filepath)

    def add(self, fiducial_mark: FiducialMark) -> None:
        self.fiducial_marks.append(fiducial_mark)

    def get_fiducial_marks(self, filename: str) -> dict[str, Optional[FiducialMark]]:
        fiducial_marks_for_file = [mark for mark in self.fiducial_marks if mark.filename == filename]

        marks: dict[str, Optional[FiducialMark]] = {corner: None for corner in FIDUCIAL_LOCATIONS}

        for fiducial_mark in fiducial_marks_for_file:
            marks[fiducial_mark.corner] = fiducial_mark

        return marks

    def get_all_previous_types(self) -> list[str]:

        types: set[str] = set()

        for fiducial_mark in self.fiducial_marks:
            types.add(fiducial_mark.frame_type)

        return list(types)


def gui():

    filenames = get_unprocessed_images()

    filepaths = files.INPUT_DIRECTORIES["image_dir"] + filenames
    np.random.shuffle(filepaths)

    count = 0

    grid: list[sg.Column] = []

    marked_fiducials = MarkedFiducials.create_or_load(CACHE_FILES["marked_fiducials"])
    types = marked_fiducials.get_all_previous_types()

    for i, corner in enumerate(FIDUCIAL_LOCATIONS):
        image_viewer_layout = [
            [sg.Text(f"{corner.capitalize()} fiducial")],
            [sg.Graph(
                canvas_size=(WINDOW_RADIUS * 2, WINDOW_RADIUS * 2),
                graph_bottom_left=(WINDOW_RADIUS * 2, 0),
                graph_top_right=(0, WINDOW_RADIUS * 2),
                key=corner,
                enable_events=True,
                drag_submits=True

            )],
        ]

        if i == 0:
            image_viewer_layout[0].insert(1, sg.Button("Previous", key="previous"))
            image_viewer_layout[0].insert(2, sg.Button("Next", key="next"))

        grid.append(sg.Column(image_viewer_layout))

    type_layout = sg.Column([
        [sg.Text(text=f"Selected type: {DEFAULT_FRAME_TYPE_NAME}", key="selected-type")],
        [sg.InputText(default_text="New type", key="new-type-text"), sg.Button("Submit", key="new-type-submit")],
        [sg.Listbox(values=types, enable_events=True, key="types-list",
                    select_mode=sg.LISTBOX_SELECT_MODE_SINGLE, size=(10, 20))]
    ], key="type-layout")

    grid.append(type_layout)

    window = sg.Window(title="Hello there", layout=[grid])
    window.Finalize()

    points: dict[str, int] = {}

    graphs = draw_image_nr(count, filepaths, window, points)

    pressed_type = DEFAULT_FRAME_TYPE_NAME

    coords: dict[str, Optional[list[int]]] = {}

    last_pressed_on_graph = 0.0
    last_updated_type = 0.0
    last_switched_image = 1.0

    while True:
        event, values = window.read()

        if event == "OK" or event == sg.WIN_CLOSED:
            break

        if "next" in event or "previous" in event:

            for corner in coords:

                # Skip if the graph or type list has not been interacted with yet on this image
                if last_pressed_on_graph < last_switched_image and last_pressed_on_graph < last_updated_type:
                    continue

                # Skip if both the last mark was set to None and this one is also None
                saved_fiducials = marked_fiducials.get_fiducial_marks(filenames[count])
                if coords[corner] is None and saved_fiducials[corner] is None:
                    continue

                # Get the positions if they exist, otherwise just None
                y_position, x_position = coords[corner] or (None, None)

                marked_fiducials.add(
                    FiducialMark(
                        filename=filenames[count],
                        corner=corner,
                        y_position=y_position,  # type: ignore
                        x_position=x_position,  # type: ignore
                        frame_type=pressed_type
                    )
                )
                print(f"Saved fiducial {corner}")
            marked_fiducials.save()

            last_switched_image = time.time()

            new_count = count + (1 if event == "next" else -1)
            count = new_count
            graphs = draw_image_nr(count, filepaths, window, points)
            saved_fiducials = marked_fiducials.get_fiducial_marks(filenames[count])
            window.set_title(f"Examining {filenames[count]}")
            pressed_type = DEFAULT_FRAME_TYPE_NAME
            for corner, fiducial_mark in saved_fiducials.items():
                print("Loading fiducials", fiducial_mark)
                if fiducial_mark is None:
                    graphs[corner].RelocateFigure(figure=points[corner], x=-1, y=-1)
                    continue
                pressed_type = fiducial_mark.frame_type

                print(fiducial_mark)
                graph_x_position = fiducial_mark.x_position + WINDOW_RADIUS - \
                    FIDUCIAL_LOCATIONS[corner][1] + POINT_RADIUS  # type: ignore
                graph_y_position = fiducial_mark.y_position + WINDOW_RADIUS - \
                    FIDUCIAL_LOCATIONS[corner][0] + POINT_RADIUS  # type: ignore

                graphs[corner].RelocateFigure(
                    figure=points[corner], x=graph_y_position, y=graph_x_position)
            window["selected-type"](f"Selected type: {pressed_type}")

            continue

        for corner in FIDUCIAL_LOCATIONS:
            if event != corner:
                continue

            last_pressed_on_graph = time.time()

            for value in values[corner]:
                if value < 0 or value > WINDOW_RADIUS * 2:
                    coords[corner] = None
                    break
            else:
                coords[corner] = [values[corner][i] - WINDOW_RADIUS + FIDUCIAL_LOCATIONS[corner][i] for i in (0, 1)]

            print(corner, coords[corner])

            graphs[corner].RelocateFigure(points[corner], *[value + POINT_RADIUS for value in values[corner]])

        if event == "new-type-submit":
            text = values["new-type-text"]
            if "New type" in text or text in types:
                continue

            pressed_type = text
            types.append(text)
            window["types-list"].Update(values=types)
            window["selected-type"](f"Selected type: {pressed_type}")
            last_updated_type = time.time()
            continue

        if "types-list" in event:
            pressed_type = values["types-list"][0]
            window["selected-type"](f"Selected type: {pressed_type}")
            last_updated_type = time.time()

    window.close()


if __name__ == "__main__":
    gui()
