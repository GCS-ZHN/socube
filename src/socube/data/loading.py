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

import pandas as pd
import numpy as np
import abc
import os
import torch
from torch.utils.data import Dataset, Sampler, Subset
from sklearn.model_selection import StratifiedKFold

__all__ = ["DatasetBase", "ConvDatasetBase"]


class DatasetBase(Dataset, metaclass=abc.ABCMeta):
    """
    Abstract base class for datasets. All SoCube extended
    datasets must inherit and implement its abstract interface.

    Parameters
    ------------
    labels: pd.DataFrame
        Dataframe containing labels for each sample.
    shuffle: bool, default False
        Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.
    seed: int, default None
        Random seed for shuffling.
    k: int, default 5
        Number of folds for k-fold cross-validation.
    task_type: str, default "classify"
        Type of task. Must be one of "classify", "regress".
    """
    def __init__(self,
                 labels: pd.DataFrame,
                 shuffle: bool = False,
                 seed: int = None,
                 k: int = 5,
                 task_type: str = "classify") -> None:

        if not shuffle:
            seed = None

        self._shuffle = shuffle
        self._seed = seed
        self._k = k
        self._labels = labels
        self._task_type = task_type

    def __len__(self) -> int:
        """Return dataset's size"""
        return self._labels.shape[0]

    @abc.abstractmethod
    def __getitem__(self, index: int) -> dict:
        """
        Abstract method to get the specified sample
        information according to the index

        Parameters
        ------------
        index: int
            Index of the sample

        Returns
        ------------
        sample: dict
            A dictionary containing the sample information,
            such as tensor data, label, etc.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sampler(self, subset: Subset) -> Sampler:
        """
        Abstract method for sampling a subset of this dataset.

        Parameters
        ------------
        subset: Subset
            A subset of this dataset.

        Returns
        ------------
        sampler: Sampler
            A sampler for the subset.
        """
        raise NotImplementedError

    @property
    def kFold(self):
        """
        Get generator for k-fold cross-validation dataset

        Returns
        ------------
        kFold: generator
            An generator for k-fold cross-validation dataset. Each iteration
            generates a tuple of two Subset objects for training and validating
        """
        skf = StratifiedKFold(n_splits=self._k,
                              random_state=self._seed,
                              shuffle=self._shuffle)

        # It will balance types if tasktype is classify
        if self._task_type == "regress":
            index = np.zeros((len(self), 1))
        elif self._task_type == "classify":
            index = self._labels
        else:
            raise ValueError("Task type must be one of 'classify', 'regress'")

        for train_index, valid_index in skf.split(index, index):
            yield (Subset(self, train_index), Subset(self, valid_index))


class ConvDatasetBase(DatasetBase):
    """
    Basical dataset designed for CNN.

    Parameters
    ----------------
    data_dir: str
        Path to the directory containing dataset.
    labels: pd.DataFrame
        Dataframe containing labels for each sample.
    transform: torch.nn.Module, default None
        Transform to apply to each sample.
    shuffle: bool, default False
        Whether to shuffle each class's samples before splitting into batches.
    seed: int, default None
        Random seed for shuffling.
    k: int, default 5
        Number of folds for k-fold cross-validation.
    task_type: str, default "classify"
        Type of task. Must be one of "classify", "regress".
    use_index: bool, default True
        Whether to use the numeric index as the sample file name, such as "0.npy",
        if `False`, then use the sample name in the labels as the sample file name,
        such as "sample_name.npy".
    """
    def __init__(self,
                 data_dir: str,
                 labels: pd.DataFrame,
                 transform: torch.nn.Module = None,
                 shuffle: bool = False,
                 seed: int = None,
                 k: int = 5,
                 task_type: str = "classify",
                 use_index: bool = True) -> None:
        super(ConvDatasetBase, self).__init__(labels, shuffle, seed,
                                              k, task_type)
        if not shuffle:
            seed = None
        self._data_dir = data_dir
        self._transform = transform
        self._use_index = use_index

    def __getitem__(self, index: int) -> dict:
        item: pd.Series = self._labels.iloc[index]
        if self._use_index:
            data_file = f"{index}.npy"
        else:
            data_file = f"{item.name}.npy"

        res = torch.from_numpy(
            np.load(os.path.join(self._data_dir, data_file)).astype("float32"))
        if self._transform:
            res = self._transform(res)
        return {'data': res, 'label': np.squeeze(item.values)}
