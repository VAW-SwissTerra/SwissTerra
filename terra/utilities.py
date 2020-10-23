"""Python or C-related utility functions."""
import ctypes
import io
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from typing import Any

import statictypes

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

    @statictypes.enforce
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


@statictypes.enforce
def notify(message: str) -> None:
    """
    Send a notification to the current user.
    """
    subprocess.run(["notify-send", message], check=True)
