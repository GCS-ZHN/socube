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

import torch
import numpy as np

from socube.utils.logging import log

__all__ = ["EarlyStopping"]


class EarlyStopping(object):
    """
    Early stops the training if validation loss doesn't improve after a given patience.

    Parameters
    ----------
    descend: bool, default True
        If True, the loss is minimized. If False, the loss is maximized.
    patience: int, default 7
        How long to wait after last time validation loss improved.
    verbose: int, default 0
        Prints a message which level great than `verbose` for each
         validation loss improvement.
    delta: float, default 0
        Minimum change in the monitored quantity to qualify as an improvement.
    path: str, default 'checkpoint.pt'
        Path for the checkpoint to be saved to.
    """

    def __init__(self,
                 descend: bool = True,
                 patience: int = 7,
                 threshold: float = 1e-5,
                 verbose: int = 0,
                 delta: float = 0,
                 path='checkpoint.pt'):
        self._flag = 1 if descend else -1
        self._patience = patience
        self._verbose = verbose
        self._counter = 0
        self._best = np.inf if descend else -np.inf
        self._stop = False
        self._delta = delta
        self._path = path
        self._threshold = threshold

    def __call__(self,
                 score: float,
                 model: torch.nn.Module,
                 flag: bool = True) -> bool:
        """
        Object callable function to update record

        Parameters
        --------------
        score: float
            the loss score of model
        model: torch.nn.Module
            the model to be training
        flag: bool, default True
            If True, it will skipped at this time

        Returns
        --------------
            Boolean value
        """
        if flag and (self._best + self._delta - score) * self._flag > 0:
            log(EarlyStopping.__name__,
                f"Score changes from {self._best} to {score}",
                quiet=self._verbose > 1)
            torch.save(model.state_dict(), self._path)
            self._best = score
            self._counter = 0
            return True
        else:
            self._counter += 1
            log(EarlyStopping.__name__,
                f'EarlyStopping counter: {self._counter} out of {self._patience}',
                quiet=self._verbose > 5)
            if self._counter >= self._patience:
                self._stop = True
            return False

    @property
    def earlyStop(self) -> bool:
        """
        Wether to reach early stopping point.

        Returns
        --------------
            Boolean value
        """
        return self._stop or (self._best - self._threshold) * self._flag <= 0
