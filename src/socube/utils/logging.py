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

import logging
import time
import random
"""This module provides some logging tools"""
__all__ = [
    "log", "getJobId", "DEBUG_LEVEL", "INFO_LEVEL", "WARN_LEVEL", "ERROR_LEVEL"
]

DEBUG_LEVEL = "debug"
INFO_LEVEL = "info"
WARN_LEVEL = "warn"
ERROR_LEVEL = "error"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def log(name: str,
        msg: str,
        level: str = "info",
        quiet: bool = False,
        *args,
        **kwargs) -> None:
    """
    Log a message.

    Parameters
    ----------
    name : str
        The name of the logger.
    msg : str
        The message to be logged.
    level : str
        The level of the message.
    quiet : bool
        If True, the message will not be logged.
    *args :
        The rest of the arguments will be passed to the logger.
    **kwargs :
        The rest of the keyword arguments will be passed to the logger.

    Examples
    ----------
    >>> log("mylogger", "hello world", level="debug")
    >>> log("mylogger", "hello world", level="info")
    >>> log("mylogger", "hello world", level="warn")
    >>> log("mylogger", "hello world", level="error")
    """
    if quiet:
        return
    logger = logging.getLogger(name)
    log_dict = {
        "debug": logger.debug,
        "info": logger.info,
        "warn": logger.warning,
        "error": logger.error
    }
    assert level in log_dict, f"Invalid level {level}"
    log_dict[level](msg, *args, **kwargs)


def getJobId() -> str:
    """
    Generate job id randomly. It's useful to seperate different jobs.

    Returns
    ----------
        a string id

    Examples
    ----------
    >>> getJobId()
    '20211102-141041-237'

    >>> getJobId()
    '20211102-132411-806'
    """
    s = time.strftime(r"%Y%m%d-%H%M%S", time.localtime())
    return "%s-%d" % (s, random.randint(100, 999))
