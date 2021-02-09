"""Python or C-related utility functions."""
import ctypes
import datetime
import io
import os
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from typing import Any, Optional

import jinja2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
import sklearn.linear_model

libc = ctypes.CDLL(None)
c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')


@contextmanager
def no_stdout(stream=None, disable=False):
    """
    Redirect the stdout to a stream file.

    Source: https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/

    param: stream: a BytesIO object to write to.
    param: disable: whether to temporarily disable the feature.

    """
    if disable:
        yield
        return
    if stream is None:
        stream = io.BytesIO()
    # The original fd stdout points to. Usually 1 on POSIX systems.
    original_stdout_fd = sys.stdout.fileno()

    def _redirect_stdout(to_fd):
        """Redirect stdout to the given file descriptor."""
        # Flush the C-level buffer stdout
        libc.fflush(c_stdout)
        # Flush and close sys.stdout - also closes the file descriptor (fd)
        sys.stdout.close()
        # Make original_stdout_fd point to the same file as to_fd
        os.dup2(to_fd, original_stdout_fd)
        # Create a new sys.stdout that points to the redirected fd
        sys.stdout = io.TextIOWrapper(os.fdopen(original_stdout_fd, 'wb'))

    # Save a copy of the original stdout fd in saved_stdout_fd
    saved_stdout_fd = os.dup(original_stdout_fd)
    try:
        # Create a temporary file and redirect stdout to it
        tfile = tempfile.TemporaryFile(mode='w+b')
        _redirect_stdout(tfile.fileno())
        # Yield to caller, then redirect stdout back to the saved fd
        yield
        _redirect_stdout(saved_stdout_fd)
        # Copy contents of temporary file to the given stream
        tfile.flush()
        tfile.seek(0, io.SEEK_SET)
        stream.write(tfile.read())
    finally:
        tfile.close()
        os.close(saved_stdout_fd)


class ConstantType:
    """Generic readonly document constants class."""

    def __getitem__(self, key: str) -> Any:
        """
        Get an item like a dict.

        param: key: The attribute name.

        return: attribute: The value of the attribute.
        """
        attribute = self.__getattribute__(key)
        return attribute

    @staticmethod
    def raise_readonly_error(key, value):
        """Raise a readonly error if a value is trying to be set."""
        raise ValueError(f"Trying to change a constant. Key: {key}, value: {value}")

    def __setattr__(self, key, value):
        """Override the Constants.key = value action."""
        self.raise_readonly_error(key, value)

    def __setitem__(self, key, value):
        """Override the Constants['key'] = value action."""
        self.raise_readonly_error(key, value)


def notify(message: str) -> None:
    """
    Send a notification to the current user.
    """
    subprocess.run(["notify-send", message], check=True)


def is_gpu_available() -> bool:
    """Check if an NVIDIA GPU is available (if the 'nvidia-smi' command yields an error or not)."""
    with no_stdout():
        code = os.system("nvidia-smi")

    return code == 0


def plot_progress():

    ssh_commands = ["ssh", "vierzack07", "cat",
                    "Projects/ETH/SwissTerra/temp/processing/progress.log", "|", "grep", "DEMs"]
    text = subprocess.run(ssh_commands, check=True, stdout=subprocess.PIPE, encoding="utf-8").stdout

    data = pd.DataFrame(columns=["date", "dataset", "dems"])

    for i, line in enumerate(text.splitlines()):
        date_str, dataset, text = line.split(",")
        date = pd.to_datetime(date_str, format="%Z %Y/%m/%d %H:%M:%S")
        dems = int(text.replace("Made ", "").replace(" DEMs", ""))
        data.loc[i] = date, dataset, dems

    data["dems_tot"] = data["dems"].cumsum()
    data["seconds"] = data["date"].apply(lambda x: float(datetime.datetime.strftime(x, "%s")))

    christmas = datetime.datetime(year=2020, month=12, day=25, tzinfo=datetime.timezone.utc)

    after_christmas = data.loc[data["date"] > christmas]  # type: ignore

    model = sklearn.linear_model.LinearRegression()
    model.fit(after_christmas["seconds"].values.reshape(-1, 1), after_christmas["dems_tot"])

    april = datetime.datetime(year=2021, month=4, day=1, tzinfo=datetime.timezone.utc)
    dems_april = model.predict(np.reshape(float(datetime.datetime.strftime(april, "%s")), (1, -1)))[0]

    plt.figure(figsize=(6, 4))
    plt.plot([data.loc[data["date"] > christmas].iloc[0]["date"], april],
             [data.loc[data["date"] > christmas].iloc[0]["dems_tot"], dems_april],
             label="Projection",
             linestyle=":")
    plt.plot(data["date"], data["dems_tot"], linewidth=3, label="Progress")

    plt.xlim(datetime.datetime(2020, 12, 19), datetime.datetime(2021, 3, 20))
    plt.ylim(0, 2400)

    plt.hlines(2300, *plt.gca().get_xlim(), color="black", linestyles="--", label="Target")
    plt.ylabel("Number of DEMs")
    plt.legend(loc="lower right")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig("temp/figures/dem_generation_progress.jpg", dpi=300)


def plot_progress_map():
    """
    Show the west-to-east progress of the processing.

    The datasets are processed in an easterly direction (since late Jan 2021), so this function finds the latest
    finished dataset and fills all until its median (representative) easting coordinate.
    """
    ssh_commands = ["ssh", "vierzack07", "cat",
                    "Projects/ETH/SwissTerra/temp/processing/progress.log", "|", "grep", "finished"]
    text = subprocess.run(ssh_commands, check=True, stdout=subprocess.PIPE, encoding="utf-8").stdout.splitlines()[-1]

    dataset = text.split(",")[1]
    instrument, year_str = dataset.split("_")

    # Import dependencies here to avoid circular imports
    from terra.preprocessing import image_meta  # pylint: disable=import-outside-toplevel
    from terra import base_dem  # pylint: disable=import-outside-toplevel

    metadata = image_meta.read_metadata()
    metadata["year"] = metadata["date"].apply(lambda date: date.year)

    dataset_meta = metadata.loc[(metadata["Instrument"] == instrument) & (metadata["year"] == int(year_str))]

    min_easting = dataset_meta["easting"].median()

    hillshade = rio.open(base_dem.CACHE_FILES["hillshade"])
    extent = (hillshade.bounds.left, hillshade.bounds.right, hillshade.bounds.bottom, hillshade.bounds.top)
    plt.imshow(hillshade.read(1), extent=extent, cmap="Greys_r")

    plt.fill_between([hillshade.bounds.left, min_easting], hillshade.bounds.top,
                     hillshade.bounds.bottom, color="red", alpha=0.4)

    plt.ylim(hillshade.bounds.bottom, hillshade.bounds.top)
    plt.xlim(hillshade.bounds.left, hillshade.bounds.right)
    plt.title(f"Last dataset: {dataset}. Easting: {min_easting:.0f} m.")
    plt.show()


def deploy_datasets(ssh_host: str, datasets: list[str]):

    command = jinja2.Template("""bash -c 'ssh {{ ssh_host }} 'tmux new -n SwissTerra -d '
        cd ~/Projects/ETH/SwissTerra/
        eval "$(conda shell.bash hook)"
        conda activate swissterra
        while read line; do
            echo "processing $line" >> log.txt
        done <<< "$(echo "{{ datasets }}" | tr ',' '\\n')"
        '''
        """).render(
        ssh_host=ssh_host,
        datasets=",".join(datasets)
    )

    subprocess.run(command, shell=True, check=True)


def read_queue() -> list[str]:
    """
    Read the processing queue.

    :returns: A list of dataset names, or an empty list if the queue file does not exist.
    """
    queue_file = "temp/queue/queue.txt"

    if not os.path.isfile(queue_file):
        return []

    with open(queue_file) as infile:
        queue = infile.read().splitlines()

    return queue


def set_deployment_idle_status(status: bool):
    os.makedirs("temp/queue", exist_ok=True)
    with open("temp/queue/idle.txt", "w") as outfile:
        outfile.write(str(status))


def get_deployment_idle_status() -> bool:

    if not os.path.isfile("temp/queue/idle.txt"):
        return True

    with open("temp/queue/idle.txt") as infile:
        status = bool(infile.read().strip())

    return status
