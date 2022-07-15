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

from email.generator import Generator
import torch.cuda as cuda

from typing import Any, Iterable, List, TypeVar
from .context import ContextManagerBase
from .logging import log

"""
This module provides some tools for GPUs/CPUs Memory
"""

__all__ = ["visualBytes", "getGPUReport", "autoClearIter"]


def visualBytes(size: int) -> str:
    """
    Make memory size more friendly with proper unit.

    Parameters
    ----------
    size: a int type of bytes szie

    Returns
    ----------
    a formated string with size unit.

    Examples
    ----------
    >>> visualBytes(2<<20)
    '2.00MB'
    >>> visualBytes(18200565665)
    '16.95GB'
    """
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    idx = 0
    while size >= (1 << 10) and idx + 1 < len(units):
        size /= (1 << 10)
        idx += 1
    return "%.2f%s" % (size, units[idx])


def getGPUReport(*devices: int) -> str:
    """
    Get gpu memory usage of this program by pytorch.

    Parameters
    ----------
    devices: a list of device id

    Returns
    ----------
    A string report of gpu memory
    """
    if len(devices) == 0:
        devices = range(cuda.device_count())
    reports = map(
        lambda idx:
        f"{idx}\t cached: {visualBytes(cuda.memory_reserved(idx))}", devices)
    return "\n".join(reports)


GPUCacheManagerType = TypeVar("GPUCacheManagerType", bound="GPUContextManager")


class GPUContextManager(ContextManagerBase):
    """
    A context manager to help user automatically clean cached GPU memory.

    Parameters
    ----------
    quiet: boolean value
        Whether to remain silent

    Examples
    ----------
    >>> with GPUContextManager():
    >>>     # code using gpu by torch
    >>>     data = data.to(torch.device(0))
    >>>     pred = model(data)
    """
    def __init__(self, quiet: bool = True) -> None:
        self.quiet = quiet

    def __enter__(self) -> GPUCacheManagerType:
        cuda.empty_cache()
        return self

    def __exit__(self, *args) -> bool:
        cuda.empty_cache()
        log(GPUCacheManagerType.__name__,
            "\n" + getGPUReport(),
            quiet=self.quiet)
        return False


def autoClearIter(iter: Iterable[Any]) -> Generator:
    """
    Create a generator to automatically clean GPU memory when iterating.
    """
    for item in iter:
        with GPUContextManager():
            yield item


def parseGPUs(gpu_str: str) -> List[int]:
    """
    Parse a string of gpu ids.

    Parameters
    ----------
    gpu_str: a string of gpu ids.

    Returns
    ----------
    a list of gpu ids.

    Examples
    ----------
    >>> parseGPUs("0,1,2")
    [0, 1, 2]
    >>> parseGPUs("0")
    [0]
    >>> parseGPUs("")
    []
    """
    try:
        if gpu_str == "":
            return []
        return ["cuda:%d"%(int(idx)) for idx in gpu_str.split(",")]
    except Exception as e:
        raise ValueError(f"Invalid gpu id: {gpu_str}") from e
