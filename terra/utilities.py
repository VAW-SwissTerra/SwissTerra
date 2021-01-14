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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

    after_christmas = data.loc["2020-12-25":]  # type: ignore

    model = sklearn.linear_model.LinearRegression()
    model.fit(after_christmas["seconds"].values.reshape(-1, 1), after_christmas["dems_tot"])

    april = datetime.datetime(year=2021, month=4, day=1, tzinfo=datetime.timezone.utc)
    dems_april = model.predict(np.reshape(datetime.datetime.strftime(april, "%s"), (1, -1)))[0]

    plt.figure(figsize=(6, 4))
    plt.plot([data.loc[data["date"] > christmas].iloc[0]["date"], april],
             [data.loc[data["date"] > christmas].iloc[0]["dems_tot"], dems_april])
    plt.plot(data["date"], data["dems_tot"], linewidth=3)

    plt.xlim(datetime.datetime(2020, 12, 19), datetime.datetime(2021, 3, 20))
    plt.ylim(0, 2380)

    plt.hlines(2300, *plt.gca().get_xlim(), color="black", linestyles="--")
    plt.ylabel("Number of DEMs")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig("temp/figures/dem_generation_progress.jpg")
