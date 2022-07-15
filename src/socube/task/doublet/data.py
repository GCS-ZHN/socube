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

from typing import Optional, Tuple
import torch
import os
import numpy as np
import pandas as pd

from pandas.core.generic import NDFrame
from torch.utils.data import (WeightedRandomSampler, Subset)
from socube.data.loading import ConvDatasetBase
from socube.utils.concurrence import parallel
from socube.utils.io import uniquePath, writeCsv, writeHdf
from socube.utils.logging import log

__all__ = ["ConvClassifyDataset", "generateDoublet", "checkShape", "checkData", "createTrainData"]


class ConvClassifyDataset(ConvDatasetBase):
    """
    Class of dataset for socube

    Parameters
    ------------
    data_dir: string
        the dataset's directory
    transform: torch module
        sample transform, such as `Resize`
    labels: string
        the label file csv name
    shuffle: boolean value
        if `True`, data will be shuffled while k-fold cross-valid
    seed: random seed
        random seed for k-fold cross-valid or sample
    k: integer scalar value
        k value of k-fold cross-valid
    use_index: boolean value
        If `True`, it will read sample file by index.
    """

    def __init__(self,
                 data_dir: str,
                 transform: torch.nn.Module = None,
                 labels: str = "label.csv",
                 shuffle: bool = False,
                 seed: int = None,
                 k: int = 5,
                 use_index: bool = True) -> None:

        super(ConvClassifyDataset,
              self).__init__(data_dir,
                             pd.read_csv(os.path.join(data_dir, labels),
                                         index_col=0,
                                         header=None),
                             transform=transform,
                             shuffle=shuffle,
                             seed=seed,
                             k=k,
                             use_index=use_index,
                             task_type="classify")

    def sampler(self, subset: Subset) -> WeightedRandomSampler:
        """
        Generate weighted random sampler for a subset of this dataset

        Parameters
        ------------
        subset: Subset
            the subset of this dataset

        Returns
        ------------
        Weighted random sampler
        """
        assert subset.dataset == self, "Must be a subset of this dataset"
        labels = self._labels[1][subset.indices]
        numSamples = labels.shape[0]
        labelWeights = numSamples / np.bincount(labels)
        sampleWeights = labelWeights[labels]
        generator = torch.Generator().manual_seed(
            self._seed) if self._seed is not None else None

        return WeightedRandomSampler(sampleWeights,
                                     numSamples,
                                     generator=generator)

    @property
    def typeCounts(self) -> int:
        """Numbers of different types"""
        return self._labels[1].unique().shape[0]


def generateDoublet(samples: pd.DataFrame,
                     ratio: float = 1.0,
                     adj: float = 1.0,
                     seed: Optional[int] = None,
                     size: Optional[int] = None) -> Tuple[pd.DataFrame]:
    """
    Generate training set from samples. in silico doublet
    will be simulated as positive samples.

    Parameters
    ------------
    samples: pd.DataFrame
        the samples dataframe, with row as cells (droplets, simples)
        and column as genes.
    ratio: float, default 1.0
        The ratio of the number of doublet and singlet.
    adj: float, default 1.0
        The adjustment factor for the doublet expression level. Generally,
        doublet is considered to have twice the gene expression level of
        singlet, but this is not necessarily the case in some cases. The gene
        expression level of the generated doublet is adjusted by the
        adjustment factor.
    seed: int, default None
        The random seed for the generation of the doublet.
    size: int, default None
        The size of the generated training set. If `None`, the size of the
        training set will be the same as the size of the samples.

    Returns
    ------------
    a tuple of two pd.DataFrame, the first is the positive (doublet) samples,
    the second is the negative (singlet) samples.
    """
    values = samples.values
    droplet_num = samples.shape[0]
    if size is None or size <= 0:
        size = droplet_num
    doublet_num = int(ratio * size / (ratio + 1))
    random = np.random.RandomState(seed)
    pair_index = random.choice(droplet_num, size=(doublet_num, 2))
    doublets = pd.DataFrame(values[pair_index[:, 0]] +
                            values[pair_index[:, 1]] * adj,
                            columns=samples.columns,
                            index=[f"doublet_{i}" for i in range(doublet_num)])

    singlets = samples.sample(size - doublet_num, random_state=seed)
    return singlets, doublets


def checkShape(path: str, shape: tuple = (10, None, None)) -> None:
    """
    Check dataset shape

    Parameters
    ------------
    path: string
        the dataset's directory
    shape: tuple, default (10, None, None)
        the expected shape of the dataset, None means any shape
    
    Raises
    ------------
    AssertionError: if the shape of the dataset is not the same as the expected
    """
    for file in filter(lambda x: x.endswith(".npy"), os.listdir(path)):
        test_sample: np.ndarray = np.load(os.path.join(path, file))
        assert len(test_sample.shape) == len(
            shape
        ), f"The shape of dataset {path} is not satisfied with {shape}"
        for true, expect in zip(test_sample.shape, shape):
            assert expect is None or true == expect, f"The shape of dataset {path} is not satisfied with {shape}"
        break


def checkData(data: pd.DataFrame):
    """
    Data legitimacy verification

    Parameters
    ----------
    data : pd.DataFrame
        Data to be checked, a dataframe of scRNA-seq data

    Raises
    ------
    ValueError
        If data contains NaN or inf
     IndexError
        If data contains duplicate column or row names,
        or if droplet name begins with "doublet"
    """
    def _check(x: pd.Series):
        if x.name.startswith("doublet"):
            raise IndexError(
                f"Droplet's name started with 'doublet' is not allowed, please rename it. Error in droplet: {x.name}"
            )
        elif x.isna().any():
            raise ValueError(f"A 'nan' value found at droplet {x.name}")

    if not data.index.is_unique:
        raise IndexError(
            "Duplicated droplet name detected, but name uniqueness is required"
        )

    if not data.columns.is_unique:
        raise IndexError(
            "Duplicated gene name detected, but name uniqueness is required")

    data.apply(_check, axis=1)


@parallel
def createTrainData(samples: pd.DataFrame,
                    output_path: str,
                    ratio: float = 1,
                    adj: float = 1,
                    seed: Optional[int] = None) -> Tuple[NDFrame]:
    """
    Based on the original data, doublets are generated as
    the positive data and a subset of the original data is used
    as the negative data to obtain the training dataset.

    Parameters
    ----------
    samples : pd.DataFrame
        Original data, a dataframe of scRNA-seq data. Shape is (n_droplets, n_genes)
    output_path : str
        Path to save the generated training data.
    ratio: float, default 1.0
        The ratio of the number of doublet and singlet.
    adj: float, default 1.0
        The adjustment factor for the doublet expression level. Generally,
        doublet is considered to have twice the gene expression level of
        singlet, but this is not necessarily the case in some cases. The gene
        expression level of the generated doublet is adjusted by the
        adjustment factor.
    seed: int, default None
        The random seed for the generation of the doublet.

    Returns
    -------
    A tuple of NDFrames. The first element is training data, the second element
    is the training label.
    """
    log("Generate", f"Generating doublet with ratio {ratio} and adj {adj}...")
    singlets, doublets = generateDoublet(samples,
                                          seed=seed,
                                          ratio=ratio,
                                          adj=adj,
                                          size=len(samples))
    train_data = pd.concat([samples, doublets])

    log("Generate", "Writing dataset...")
    writeHdf(
        train_data,
        uniquePath(
            os.path.join(
                output_path,
                f"02-trainData[{train_data.dtypes.iloc[0].name}][raw+samples({ratio},{adj})].h5"
            ), True))

    negative = pd.Series(np.zeros_like(singlets.index, dtype=np.int8),
                         index=singlets.index)
    positive = pd.Series(np.ones_like(doublets.index, dtype=np.int8),
                         index=doublets.index)
    train_label = pd.concat([negative, positive])
    writeCsv(train_label,
             uniquePath(
                 os.path.join(output_path, f"label[raw+samples({ratio},{adj})].csv"),
                 True),
             header=None)
    log("Generate", "Train data created")
    return train_data, train_label
