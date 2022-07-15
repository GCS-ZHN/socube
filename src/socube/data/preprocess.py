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

from typing import Tuple
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.manifold import TSNE
from umap import UMAP
from lapjv import lapjv
"""
Module for data processing API
"""
__all__ = [
    "summary", "filterData", "minmax", "std",
    "cosineDistanceMatrix", "scatterToGrid", "umap2D", "tsne2D",
    "vec2Grid", "onehot", "items"
]


def summary(data: pd.DataFrame, axis: int = 1) -> pd.DataFrame:
    """
    Data summary for each column or row.

    Parameters
    ---------------
    data : dataframe
        a dataframe with row and column
    axis : int, default 1
        0 for summary for column, 1 for summary for row

    Returns
    ---------------
    a dataframe with summary for each column or row

    Examples
    ---------------
    >>> import pandas as pd
    >>> import socube.data.preprocess as pre
    >>> data = pd.DataFrame(np.random.rand(10, 10))
    >>> pre.summary(data)
    """
    zeroCount = (data == 0).sum(axis=axis)
    summary = pd.concat([
        data.mean(axis=axis),
        data.std(axis=axis),
        data.max(axis=axis),
        data.min(axis=axis),
        data.median(axis=axis), zeroCount, zeroCount / data.shape[axis]
    ],
                        axis=1)
    summary.columns = [
        "means", "std", "max", "min", "median", "zero count", "zero percent"
    ]
    return summary.sort_values(ascending=False, by="zero percent")


def filterData(data: pd.DataFrame,
               filtered_gene_prop: float = 0.05,
               filtered_cell_prop: float = 0.05,
               mini_expr: float = 0.05,
               mini_library_size: int = 1000) -> pd.DataFrame:
    """
    Remove genes and cells which have low variation with given proportions and
    remove genes whose average expression less then `mini_expr` and remove cells
    whose cell library size less then `mini_library_size`.

    Parameters
    ---------------
    data : dataframe
        a dataframe, which row is gene and column is cell
    filtered_gene_prop : float, default 0.05
        Remove genes with low variation with this proportion
    filtered_cell_prop : float, default 0.05
        Remove cells with low variation with this proportion
    mini_expr : float, default 0.05
        Remove genes whose average expression less then `mini_expr`
    mini_library_size : int, default 1000
        Remove cells whose cell library size less then `mini_library_size`

    Returns
    ---------------
    a dataframe with filtered genes and cells
    """
    gene_std = data.std(axis=1)
    gene_rm = int(filtered_gene_prop * len(gene_std))
    cell_std = data.std()
    cell_rm = int(filtered_cell_prop * len(cell_std))

    # remove genes/cells which have low variation
    rm_gene_idx = np.argpartition(gene_std, gene_rm)[:gene_rm]
    rm_cell_idx = np.argpartition(cell_std, cell_rm)[:cell_rm]

    gene_mask = data.mean(axis=1) >= mini_expr
    cell_mask = data.sum() >= mini_library_size
    gene_mask[rm_gene_idx] = False
    cell_mask[rm_cell_idx] = False

    return data.loc[gene_mask, cell_mask].copy()


def minmax(data: pd.DataFrame,
           range: Tuple[int] = (0, 1),
           flag: int = 0,
           dtype: str = "float32") -> pd.DataFrame:
    """
    Perform maximum-minimum normalization

    Parameters
    ---------------
    data : dataframe
        a dataframe, which row is sample and column is feature
    range : tuple, default (0, 1)
        The maximum and minimum values of the normalized
        data, normalized to 0~1 by default
    flag : int, default 0
        Equal to 0 for minmax by columns, greater than 0 for minmax by
        rows, less than 0 for minmax by global.
    dtype : str, default "float32"
        The data type of the normalized data

    Returns
    ---------------
    a dataframe with normalized data

    Examples
    ---------------
    >>> import pandas as pd
    >>> import socube.data.preprocess as pre
    >>> data = pd.DataFrame(np.random.rand(10, 10))
    >>> pre.minmax(data)
    """
    if flag < 0:
        data_max = data.values.max()
        data_min = data.values.min()
        return (data - data_min) / (data_max - data_min)

    minmax = MinMaxScaler(feature_range=range)
    index = data.index
    columns = data.columns
    if flag > 0:
        data = data.T
    norm_data = minmax.fit_transform(data)
    if flag > 0:
        norm_data = norm_data.T
    return pd.DataFrame(norm_data, index=index, columns=columns).astype(dtype)


def std(data: pd.DataFrame,
        horizontal: bool = False,
        dtype: str = "float32",
        global_minmax: bool = False) -> pd.DataFrame:
    """
    Standardization of data

    Parameters
    ---------------
    data : dataframe
        a dataframe, which row is sample and column is feature
    horizontal : bool, default False
        If True, perform standardization horizontally
    dtype : str, default "float32"
        The data type of the standardized data
    global_minmax : bool, default False
        If True, perform global standardization,
        otherwise standardization by row or column

    Returns
    ---------------
    a dataframe with standardized data

    Examples
    ---------------
    >>> import pandas as pd
    >>> import socube.data.preprocess as pre
    >>> data = pd.DataFrame(np.random.rand(10, 10))
    >>> pre.std(data)
    """
    if global_minmax:
        data_max = data.values.max()
        data_min = data.values.min()
        return ((data - data_min) / (data_max - data_min)).astype(dtype)
    else:
        scaler = StandardScaler()
        index = data.index
        columns = data.columns
        if horizontal:
            data = data.T
        std_data = scaler.fit_transform(data)
        if horizontal:
            std_data = std_data.T
        return pd.DataFrame(std_data, index=index,
                            columns=columns).astype(dtype)


def cosineDistanceMatrix(x1: torch.Tensor,
                         x2: torch.Tensor = None,
                         device_name: str = "cpu") -> torch.Tensor:
    """
    Calculate the cosine distance matrix between the two sets of samples.

    Parameters
    ---------------
    x1 : torch.Tensor
        a tensor of samples, with shape (n1, d)
    x2 : torch.Tensor, default None
        a tensor of samples, with shape (n2, d), if None, x2 = x1
    device_name : str, default "cpu"
        the device used for accelerating the calculation
        such as "cpu", "cuda:0", "cuda:1", etc.

    Returns
    ---------------
    a tensor of cosine distance matrix, with shape (n1, n2)

    Examples
    ---------------
    >>> import torch
    >>> import socube.data.preprocess as pre
    >>> x1 = torch.rand(10, 10)
    >>> x2 = torch.rand(10, 10)
    >>> pre.cosineDistanceMatrix(x1, x2)
    """
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    e = None
    if not isinstance(x2, torch.Tensor):
        x2 = x1
        e = 1 - torch.eye(x1.shape[0]).to(device)
    assert x1.shape[1] == x2.shape[
        1], "Feature vector length should be the same!"
    delta = 3e-6
    x1 = x1.to(device)
    x2 = x2.to(device)
    x1_mod = x1.square().sum(dim=1).sqrt().unsqueeze(1)
    x2_mod = x2.square().sum(dim=1).sqrt().unsqueeze(0)
    dist_matrix = (1 - x1.matmul(x2.T) / (x1_mod.mul(x2_mod) + delta))
    dist_matrix[dist_matrix <= delta] = 0
    if isinstance(e, torch.Tensor):
        dist_matrix = dist_matrix.mul(e)
    return dist_matrix.to(torch.device("cpu"))


def scatterToGrid(scatters2d: torch.Tensor,
                  transform: Tuple[int] = (1, -1),
                  device_name: str = "cpu") -> torch.Tensor:
    """
    Scattered coordinates-grid coordinate mapping
    based on J-V linear assignment algorithm

    Parameters
    ---------------
    scatters2d : torch.Tensor
        a tensor of scattered coordinates, with shape (n, 2)
    transform : tuple, default (1, -1)
        the transformation of the coordinates, such as (1, -1).
        Greater than 0 means that the corresponding coordinates
        do not change direction, less than 0 means that the
        corresponding coordinates are reversed. For scatter and grid plots,
        the y-axis directions are often opposite, and for visual
        consistency, the y-axis needs to be transformed
    device_name : str, default "cpu"
        the device used for accelerating the calculation
        such as "cpu", "cuda:0", "cuda:1", etc.

    Returns
    ---------------
    a tensor of grid coordinates, with shape (n, 2)

    Examples
    ---------------
    >>> import torch
    >>> import socube.data.preprocess as pre
    >>> scatters2d = torch.rand(10, 2)
    >>> pre.scatterToGrid(scatters2d)
    """
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    scatters2d = scatters2d * torch.Tensor([transform])
    scatters2d = scatters2d.to(device)
    scatters3d = scatters2d.unsqueeze(1)
    width = int(np.sqrt(scatters2d.shape[0]))
    if width * width < scatters2d.shape[0]:
        width += 1
    grid3d = torch.arange(
        scatters2d.shape[0]).unsqueeze(0).unsqueeze(2).to(device)
    # Horizontal (x) and vertical (y) coordinates of the grid
    gridLoc3d = torch.cat(
        (grid3d % width, grid3d.div(width, rounding_mode='floor')), dim=2)
    distance2d = (scatters3d - gridLoc3d).square().sum(dim=2)
    loc = lapjv(distance2d.cpu().numpy())[0]
    gridLoc2d = gridLoc3d.squeeze(0)[loc]
    return gridLoc2d.cpu()


def umap2D(data: pd.DataFrame,
           metric: str = 'correlation',
           neighbors: int = 5,
           seed: int = None) -> pd.DataFrame:
    """
    Reducing high-dimensional data to 2D using UMAP

    Parameters
    ---------------
    data : pd.DataFrame
        a dataframe of high-dimensional data, with shape (n, d).
        n is the number of samples, d is the dimension of the data
    metric : str, default 'correlation'
        the metric used for calculating the distance between samples.
        such as 'correlation', 'euclidean', 'manhattan', etc.
    neighbors : int, default 5
        the number of neighbors used for UMAP.
    seed : int, default None
        the random seed used for UMAP.
    Returns
    ---------------
    a dataframe of two-dimensional data, with shape (n, 2)

    Examples
    ---------------
    >>> import pandas as pd
    >>> import socube.data.preprocess as pre
    >>> data = pd.DataFrame(np.random.rand(10, 10))
    >>> pre.umap2D(data)
    """
    umapObject = UMAP(n_neighbors=neighbors,
                      n_components=2,
                      metric=metric,
                      min_dist=0.1,
                      random_state=seed)

    embedd = umapObject.fit(data.values)
    return pd.DataFrame(embedd.embedding_,
                        index=data.index,
                        columns=['x', 'y'])


def tsne2D(data: pd.DataFrame,
           metric: str = 'correlation',
           seed: int = None) -> pd.DataFrame:
    """
     Reducing high-dimensional data to 2D using t-SNE

    Parameters
    ---------------
    data : pd.DataFrame
        a dataframe of high-dimensional data, with shape (n, d).
        n is the number of samples, d is the dimension of the data
    metric : str, default 'correlation'
        the metric used for calculating the distance between samples.
        such as 'correlation', 'euclidean', 'manhattan', etc.
    seed : int, default None
        the random seed used for t-SNE.

    Returns
    ---------------
    a dataframe of two-dimensional data, with shape (n, 2)

    Examples
    ---------------
    >>> import pandas as pd
    >>> import socube.data.preprocess as pre
    >>> data = pd.DataFrame(np.random.rand(10, 10))
    >>> pre.tsne2D(data)
    """
    tsneObject = TSNE(n_components=2,
                      random_state=seed,
                      metric=metric,
                      verbose=2)
    embedd = tsneObject.fit(data.values)
    return pd.DataFrame(embedd.embedding_,
                        index=data.index,
                        columns=['x', 'y'])


def vec2Grid(vector: np.ndarray,
             shuffle: bool = False,
             seed: int = None) -> pd.DataFrame:
    """
    Converts a one-dimensional vector to a two-dimensional grid

    Parameters
    ---------------
    vector : np.ndarray
        a one-dimensional vector, with shape (n,).
        n is the number of samples
    shuffle : bool, default False
        whether to shuffle the vector
    seed : int, default None
        the random seed used for shuffling the vector

    Returns
    ---------------
    a dataframe of two-dimensional data, with shape (n, 2),
    each row represents a grid point with horizontal (x) and vertical (y) coordinates

    Examples
    ---------------
    >>> import numpy as np
    >>> import socube.data.preprocess as pre
    >>> vector = np.random.rand(10)
    >>> pre.vec2Grid(vector)
    """
    if not (vector.ndim == 1):
        raise ValueError('vector must be one-dimensional')
    if not (vector.shape[0] > 1):
        raise ValueError('vector must have more than one element')
    if not (isinstance(vector, np.ndarray)):
        raise ValueError('vector must be a numpy array')

    width = int(np.sqrt(vector.shape[0]))
    if width * width < vector.shape[0]:
        width += 1
    index = np.arange(vector.shape[0])
    x, y = index % width, index // width
    gridData = pd.DataFrame()
    gridData["x"] = x
    gridData["y"] = y
    gridData.index = vector
    if shuffle:
        gridData = gridData.sample(frac=1, random_state=seed)
        gridData.index = vector
    return gridData


def onehot(label: np.ndarray, class_nums: int = None) -> np.ndarray:
    """
    Convert 1D multi-label vector (each element is a sample's label)
    to onehot matrix. The label should be a integer

    Parameters
    ---------------
    label : np.ndarray
        a one-dimensional integer vector, with shape (n,).
        n is the number of samples
    class_nums : int, default None
        the number of classes. If None, the number of classes is
        automatically determined.

    Returns
    ---------------
    a ndarray of onehot matrix with shape (n, class_nums)

    Examples
    ---------------
    >>> onehot(np.array([1,2,4]))
    array([[0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1]], dtype=int32)

    >>> onehot(np.array([1,2,4]), 6)
    array([[0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0]], dtype=int32)
    """
    if not (label.ndim == 1):
        raise ValueError('label must be one-dimensional')
    if not (isinstance(label, np.ndarray)):
        raise ValueError('label must be a numpy array')
    if class_nums is None:
        class_nums = label.max() + 1
    return np.eye(class_nums, dtype=np.int32)[label]


def items(data: pd.DataFrame) -> pd.DataFrame:
    r"""
    Convert a dataFrame to a dataframe with row, col, val three columns

    Parameters
    ---------------
    data : pd.DataFrame

    Returns
    ---------------
    a dataframe with row, col, val three columns

    Examples
    ---------------
    >>> import pandas as pd
    >>> import socube.data.preprocess as pre
    >>> data = pd.DataFrame(np.random.rand(10, 10))
    >>> pre.items(data)
    """
    rowIndex = (np.zeros(data.shape) +
                np.expand_dims(np.arange(data.shape[0]), 1)).flatten()
    colIndex = (np.zeros(data.shape) +
                np.expand_dims(np.arange(data.shape[1]), 0)).flatten()
    value = data.values.flatten()
    return pd.DataFrame({
        "row": rowIndex.astype(np.int16),
        "col": colIndex.astype(np.int16),
        "val": value
    })
