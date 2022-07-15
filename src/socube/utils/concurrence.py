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

import multiprocessing as mp
from concurrent.futures import (Executor, ThreadPoolExecutor,
                                ProcessPoolExecutor, Future)
from typing import Callable, List, TypeVar
from .context import ContextManagerBase
from .logging import log

import functools
"""
This module is developed for concurrence programing.
"""
__all__ = ["parallel", "ParallelManager"]

GLOBAL_POOL_STACK: List[Executor] = []
GLOBAL_TASK_STACK: List[List[Future]] = []


def parallel(func):
    """
    A decorator for asynchronous method

    Parameters
    ----------
    func : Callable
        The function to be decorated as asynchronous.

    Returns
    ----------
        a new function. If it is called under ParallelManager, it will be
        executed in a process pool.

    Examples
    ----------
    >>> @parallel
        def func(*args, **kwargs):
            ...
    >>> parallelFunc = parallel(func)
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # add task to nearest ThreadPoolExecutor
        if len(GLOBAL_TASK_STACK) > 0 and len(GLOBAL_POOL_STACK) > 0:
            for pool_index in range(len(GLOBAL_POOL_STACK)):
                if isinstance(GLOBAL_POOL_STACK[-pool_index-1], ThreadPoolExecutor):
                    task = GLOBAL_POOL_STACK[-pool_index-1].submit(func, *args, **kwargs)
                    GLOBAL_TASK_STACK[-pool_index-1].append(task)
                    return task
            else:
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return wrapper


ParallelManagerType = TypeVar("ParallelManagerType", bound="ParallelManager")


class ParallelManager(ContextManagerBase):
    """
    A context manager for multi-thread. In this context, Multiple thread is enable
    will be set as True and enable multi-thread for function decorated with `parallel`.
    Multi-level nesting is allowed, and each level has an independent asynchronous
    task pool. When exiting the context, it will wait for the tasks in the current
    asynchronous task pool to complete.

    Parameters
    ----------
    verbose : bool
        Whether to print log.
    max_workers : int
        The maximum number of workers in the asynchronous task pool.
    paral_type : str
        The type of asynchronous task pool. Only support "thread" and "process".
    raise_error : bool
        Whether to raise error when the task is not completed. Default is True.
        If you do not want interrupt the program, set it to False.

    Examples
    ----------
    >>> with ThreadManager() as tm:
            # Here is you code
            asynFunc1(*args)  # this function is decorated with `parallel`
            with tm:
                asynFunc2(*args)  # this function is decorated with `parallel`
            print("asynFunc2 finished") # this code will not be executed util asynFunc2 finished.
        print("asynFunc1 finished")  # this code will not be executed util asynFunc1finished.
    """

    def __init__(self,
                 verbose: bool = False,
                 max_workers: int = 8,
                 paral_type: str = "thread",
                 raise_error: bool = True) -> None:
        self.verbose = verbose
        self.max_workers = max_workers
        self.paral_type = paral_type
        self.pool = None
        self.raise_error = raise_error
        self.task_stack: List[Future] = []
        if paral_type not in ["process", "thread"]:
            raise TypeError(
                f"Unsupport paral_type {paral_type}, only support process/thread."
            )

    def __enter__(self) -> ParallelManagerType:
        if self.paral_type == "process":
            pool = ProcessPoolExecutor(self.max_workers, mp_context=mp.get_context("spawn"))
        elif self.paral_type == "thread":
            pool = ThreadPoolExecutor(self.max_workers)

        GLOBAL_POOL_STACK.append(pool)
        GLOBAL_TASK_STACK.append(self.task_stack)
        self.pool = pool
        log(type(self).__name__,
            f"Enter a multi-{self.paral_type} context",
            quiet=self.verbose)
        return self

    def submit(self, func: Callable, *args, **kwargs)->Future:
        """
        Add a task to the asynchronous task pool.
        For muti-thread task, you can use `parallel` decorator instead.
        But it not support multi-process task.
        """
        task = self.pool.submit(func, *args, **kwargs)
        self.task_stack.append(task)
        return task

    def __exit__(self, *args) -> bool:
        assert GLOBAL_POOL_STACK.pop() is self.pool
        assert GLOBAL_TASK_STACK.pop() is self.task_stack
        for task in self.task_stack:
            ex = task.exception()
            if ex is not None:
                if self.raise_error:
                    raise ex
                else:
                    log(type(self).__name__, f"Sub {self.paral_type} task {task} exited with error: {ex}")
        self.pool.shutdown(wait=True)
        log(type(self).__name__,
            f"Quit a multi-{self.paral_type} context",
            quiet=self.verbose)
        return False