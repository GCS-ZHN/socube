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

from types import TracebackType
from typing import Any, Type
from abc import ABCMeta, abstractmethod

__all__ = ["ContextManagerBase"]


class ContextManagerBase(metaclass=ABCMeta):
    """
    Basical interface for context manager.
    """

    @abstractmethod
    def __enter__(self):
        """
        Enter the context.
        """
        return self

    @abstractmethod
    def __exit__(self, exc_type: Type[Exception], exc_obj: Exception,
                 exc_trace: TracebackType) -> bool:
        """
        __exit__ method's implemation for context manager, python interpreter
        will automatically catch exception information and transfer to this
        method as parameters. If parameters are all equal with 'None', it means
        no exception occurred and you are very lucky.

        Parameters
        ----------
        exc_type : Type[Exception]
            The type of exception.
        exc_obj : Exception
            The exception object.
        exc_trace : TracebackType
            The traceback object of the exception.

        Returns
        -------
        bool, True if the exception is handled, False otherwise.
        """
        return False

    def __call__(self, func):
        """
        As a decorator to enable the context manager.
        """

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self:
                return func(*args, **kwargs)

        return wrapper
