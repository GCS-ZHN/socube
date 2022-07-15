# MIT License
#
# Copyright (c) 2022 Zhang.H.N
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import warnings
import shutil
import numpy as np
import pandas as pd
import torch
import contextlib

from typing import (Callable, Optional, Dict, TypeVar, Union)
from pandas.core.generic import NDFrame

from .concurrence import parallel
from .context import ContextManagerBase
from .logging import log

__all__ = [
    "uniquePath", "mkDirs", "rm", "checkExist", "writeCsv", "writeNpy",
    "writeHdf", "loadTorchModule", "Redirect", "ReportManager"
]


def uniquePath(path: str, exists_warning: bool = False) -> str:
    """
    Check whether `path` existed in the file system. And generate a
    unique path by add index at the end of raw path.

    Parameters
    ----------------
    path : str
        The raw path to be checked.
    exists_warning : bool
        Whether to print warning when the path is existed.

    Returns
    ----------------
    str, the unique path.

    Examples
    ----------------
    >>> # pkgs/lapjv-1.3.1.tar.gz already existed
    >>> uniquePath("pkgs/lapjv-1.3.1.tar.gz")
    'pkgs/lapjv-1.3.1.tar(1).gz'

    >>> # pkgs/lapjv-1.3.1.tar.gz already existed and
    >>> # pkgs/lapjv-1.3.1.tar(1).gz already existed
    >>> uniquePath("pkgs/lapjv-1.3.1.tar.gz")
    'pkgs/lapjv-1.3.1.tar(2).gz'
    """
    name, ext = os.path.splitext(path)
    current_path = name
    idx = 1
    while os.path.exists(current_path + ext):
        current_path = "%s(%d)" % (name, idx)
        idx += 1

    if exists_warning and current_path != name:
        warnings.warn(
            f"File {name}{ext} already existed and will use {current_path}{ext} as instead!",
            RuntimeWarning,
            stacklevel=2)
    return current_path + ext


def mkDirs(path: str) -> None:
    r"""
    Create a directory if it not exists. If the parent is not existed,
    it will be created as well.

    Parameters
    ----------------
    path : str
        The path to be created.

    Examples
    ----------------
    >>> #parent is not existed
    >>> mkDirs("parent/target/")
    """
    if not os.path.exists(path):
        os.makedirs(path)


def rm(path: str) -> None:
    r"""
    Remove a directory or file.

    Parameters
    ----------------
    path : str
        The path to be removed.

    Examples
    ----------------
    >>> rm("targetFile")
    """
    if os.path.isdir(path):
        shutil.rmtree(path)
    if os.path.isfile(path):
        os.remove(path)


def checkExist(path: str,
               types: str = "file",
               raise_error: bool = True) -> bool:
    r"""
    Check whether a file or directory is existed.

    Parameters
    ---------------
    path : str
        The path to be checked.
    types : str
        The type of the path. "file" or "dir".
    raise_error : bool
        Whether to raise error when the path is not existed.

    Returns
    ---------------
    bool, whether the path is existed.

    Examples
    ---------------
    >>> checkExist("/public/file")
    >>> checkExist("/public/home/", types="dir")
    >>> checkExist("/public/home/", types="dir", raiseError=False)
    """
    if types == "file" and (not os.path.isfile(path)):
        if raise_error:
            raise FileNotFoundError(f"No such file {path}")
        return False
    elif types == "dir" and (not os.path.isdir(path)):
        if raise_error:
            raise FileNotFoundError(f"No such directory {path}")
        return False
    elif types not in ["file", "dir"]:
        raise TypeError(f"Invalid type {types}, only support file and dir")
    return True


@parallel
def writeCsv(data: Union[NDFrame, np.ndarray],
             file: str,
             callback: Optional[Callable] = None,
             **kwargs) -> None:
    """
    Write `ndarray` or any instance of `NDFrame` to a CSV file.

    Parameters
    ---------------
    data : Union[NDFrame, np.ndarray]
        The data to be written.
    file : str
        The path to be written.
    callback : Optional[Callable]
        The callback function to be called after writing.
    **kwargs
        The keyword arguments to be passed to `pandas.DataFrame.to_csv`.

    Examples
    ---------------
    >>> ndarray = np.array([1, 2, 3])
    >>> writeCsv(ndarray, "ndarray.csv", callback=lambda:print("Finish"))
    >>> series = pd.Series(ndarray)
    >>> writeCsv(series, "series.csv")
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
        kwargs["index"] = None
        kwargs["header"] = None
    elif not isinstance(data, NDFrame):
        raise TypeError(f"Invalid data type {type(data)}")
    data.to_csv(file, quoting=2, **kwargs)
    if hasattr(callback, "__call__"):
        callback()


@parallel
def writeNpy(data: np.ndarray,
             file: str,
             callback: Optional[Callable] = None,
             allow_pickle: bool = True,
             **kwargs):
    """
    Write `ndarray` to NPY binary file. NPY format is
    numpy's specific file format.

    Parameters
    --------------
    data : np.ndarray
        The data to be written.
    file : str
        The path to be written.
    callback : Optional[Callable]
        The callback function to be called after writing.
    allow_pickle : bool
        Wether to allow use pickle to serialize python object.
    **kwargs
        The keyword arguments to be passed to `numpy.save`.

    Examples
    -------------
    >>> ndarray = np.array([1, 2, 3])
    >>> writeNpy(ndarray, "ndarray.npy")
    """
    np.save(file, data, allow_pickle=allow_pickle, **kwargs)
    if hasattr(callback, "__call__"):
        callback()


@parallel
def writeHdf(data: NDFrame,
             file: str,
             key="data",
             mode="w",
             callback: Optional[Callable] = None,
             **kwargs):
    r"""
    Write `NDFrame` instance to a HDF5 binary format file.

    Parameters
    -----------------
    data : NDFrame
        The data to be written.
    file : str
        The path to be written.
    key : str
        The key to be used to store the data.
    mode : str
        The mode to be used to open the file.
    callback : Optional[Callable]
        The callback function to be called after writing.
    **kwargs
        The keyword arguments to be passed to `pandas.DataFrame.to_hdf`.

    Examples
    ----------------
    >>> data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    >>> writeHdf(data, "data.hdf", callback=lambda:print("Finish"))
    """
    data.to_hdf(file, key=key, mode=mode, **kwargs)
    if hasattr(callback, "__call__"):
        callback()


def loadTorchModule(module: torch.nn.Module,
                    file: str,
                    skipped: bool = True,
                    verbose=False):
    """
    Load a torch module from a file.

    Parameters
    ----------------
    module : torch.nn.Module
        The module waited to be updated.
    file : str
        The module file path.
    skipped : bool
        Whether to skip the loading of the module when the file is not existed.
    verbose : bool
        Whether to print the loading message.
    """
    if os.path.isfile(file):
        module.load_state_dict(torch.load(file))
        if not verbose:
            log("loadTorchModule", f"model {file} loaded")
    elif skipped:
        log("loadTorchModule", f"File {file} not found and skipped")
    else:
        raise FileNotFoundError(f"File {file} not found")
    return module


# TODO fix the bug that logging module will ignore this Redirect.
class Redirect(ContextManagerBase):
    """
    A context manager to redirect stdout and stderr.

    Parameters
    ----------------
    target : str
        The redirect target file.
    verbose : bool
        Whether to print the redirect message.

    Examples
    ----------------
    >>> with Redirect("stdout.txt", verbose=True):
    >>>     print("Hello")
    >>>     print("World")
    """
    def __init__(self, target: str, verbose: bool = True) -> None:
        parent = os.path.dirname(target)
        mkDirs(parent)
        self.__target = open(target, "wt", encoding="utf-8")
        self.__verbose = verbose
        self.__stderr = contextlib.redirect_stderr(self.__target)
        self.__stdout = contextlib.redirect_stdout(self.__target)

    def __enter__(self):
        log(Redirect.__name__,
            f"Redirect output to {self.__target.name}",
            quiet=self.__verbose)

        self.__stdout.__enter__()
        self.__stderr.__enter__()
        return self

    def __exit__(self, *args) -> bool:
        self.__stderr.__exit__(*args)
        self.__stdout.__exit__(*args)
        self.__target.close()
        return False


ReportManagerType = TypeVar('ReportManagerType', bound="ReportManager")


class ReportManager(ContextManagerBase):
    """
    A context manager to save important data.

    Parameters
    ----------------
    reports : Dict[str, NDFrame]
        a dictionary of reports, key is report filename, value
        is a `NDFrame` object waited to be saved.
    verbose : bool
        Whether to print the saving message.
        If `True`, ReportManager will work in silence.

    Examples
    ----------------
    >>> with ReportManager({"report1": df1, "report2": df2}, verbose=True) as rm:
    >>>     print("Hello")
    >>>     print("World")
    >>>     rm.addReports("report1", df3)
    >>>     rm.addReports("report2", df4)
    """

    def __init__(self,
                 reports: Dict[str, NDFrame] = dict(),
                 verbose: bool = False) -> None:
        self.__reports = dict()
        self.__check(reports)
        self.__reports = reports
        self.__verbose = verbose

    def addReports(self, reports: Dict[str, NDFrame]) -> None:
        """
        add new reports in context

        Parameters
        ----------------
        reports : Dict[str, NDFrame]
            a dictionary of reports, key is report filename, value is a
            `NDFrame` object waited to be saved.
        """
        self.__check(reports)
        self.__reports.update(reports)

    def updateReportName(self, old: str, new: str) -> None:
        """
        Update the name of a report.

        Parameters
        ----------------
        old : str
            The old report name.
        new : str
            The new report name.
        """
        if old not in self.__reports:
            raise IndexError(f"Report {old} not added yet!")
        if not isinstance(new, str):
            raise TypeError("Report name expected a str object")
        self.__reports[new] = self.__reports.pop(old)

    def __log(self, msg: str, level: str = "info") -> None:
        """
        Log the message.
        """
        log(type(self).__name__, msg, level=level, quiet=self.__verbose)

    def __check(self, reports: Dict[str, NDFrame]) -> None:
        """
        Check the reports type.
        """
        for file, df in reports.items():
            assert isinstance(file,
                              str) and file != "", f"Invalid filename {file}"
            assert isinstance(
                df, NDFrame), "Report object should be series or dataframe"

    def __enter__(self) -> ReportManagerType:
        return self

    def __exit__(self, *args) -> bool:
        for file, df in self.__reports.items():
            try:
                writeCsv(df, file)
                self.__log(f"Report {file} saved!")
            except Exception as e:
                self.__log(
                    f"An error {e} occurred while trying to save file {file}",
                    "error")
        return False
