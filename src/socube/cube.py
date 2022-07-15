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

import json
from typing import (Optional, Tuple, TypeVar, Union)
import pandas as pd
import numpy as np
import torch
import abc

from os.path import join

from sklearn.decomposition import PCA
from .data import (std, minmax, scatterToGrid, tsne2D, umap2D)
from .utils.logging import log, getJobId
from .utils.io import (writeHdf, mkDirs, writeCsv, writeNpy, checkExist)
from .utils.concurrence import ParallelManager
from tqdm.auto import tqdm

__all__ = ["SoCube"]
SoCubeType = TypeVar("SoCubeType", bound="SoCube")


class SoCube(metaclass=abc.ABCMeta):
    """
    Abstract class of SoCube.

    Parameters
    ----------
    data_path : str
        Path of original scRNA-seq data file
    home_path : str, optional
        Path of output home directory. If not specified, it will be set to
        the same directory as data_path.
    metric: str, optional
        Metric to use for distance calculation. supported:
        'correlation', 'cosine', 'euclidean', 'manhattan', etc.
        default is 'correlation'.
    dtype: str, optional
        Data type of output data. isupported: 'float32', 'float64', etc.
        default is 'float32'.
    verbose: int, optional
        Verbose level. default is 1.
    """

    def __init__(self,
                 data_path: str,
                 home_path: Optional[str] = None,
                 metric: str = 'correlation',
                 dtype: str = "float32",
                 verbose: int = 1,
                 **kwargs) -> None:
        checkExist(data_path, "dir")
        if home_path is None:
            home_path = data_path

        embedding_path = join(home_path, "embedding")
        mkDirs(embedding_path)
        self._config = {
            "data_path": data_path,
            "embedding_path": embedding_path,
            "metric": metric,
            "dtype": dtype,
            "home_path": home_path,
            "verbose": verbose,
            "socube": self.__class__.__name__
        }
        self._config.update(kwargs)

        self._manifold_methods = {"umap": umap2D, "tsne": tsne2D}
        self._grid = None
        self._scatter = None
        self._shape = None

    @abc.abstractmethod
    def fit(self, **kwargs):
        """all subclass must implement it"""
        for k, v in kwargs.items():
            self._config[k] = v

    @staticmethod
    def instance(map_type: str,
                 data_path: str,
                 home_path: Optional[str] = None,
                 metric: str = 'correlation',
                 dtype: str = "float32",
                 verbose: int = 1,
                 **kwargs) -> SoCubeType:
        """
        SoCube is designed using the factory pattern, and this method
        is used to create SoCube instance objects

        Parameters
        ----------
        map_type : str
            Type of map to use. supported: 'pca'
        data_path : str
            Path of original scRNA-seq data file
        home_path : str, optional
            Path of output home directory. If not specified, it will be set to
            the same directory as data_path.
        metric: str, optional
            Metric to use for distance calculation. supported:
            'correlation', 'cosine', 'euclidean', 'manhattan', etc.
            default is 'correlation'.
        dtype: str, optional
            Data type of output data. isupported: 'float32', 'float64', etc.
            default is 'float32'.
        verbose: int, optional
            Verbose level. default is 1.

        Returns
        -------
        SoCube instance object
        """
        avail = SoCube.available()
        if map_type in avail:
            return avail[map_type](data_path=data_path,
                                   home_path=home_path,
                                   metric=metric,
                                   dtype=dtype,
                                   verbose=verbose,
                                   **kwargs)
        else:
            raise TypeError(f"Only support: {avail.keys()}")

    @staticmethod
    def available() -> dict:
        """
        Return available map types
        """
        return {"pca": SoPCACube}

    def _log(self, message: str, verbose_level: int = 1, **kwargs):
        """
        The logging methods used inside the SoCube class are
        wrappers for the socube.utils.logging module

        Parameters
        ----------
        message : str
            Message to be logged
        verbose_level : int, optional
            Verbose level. default is 1. The higher the verbose value
            the higher the priority of the log
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the logging module
        """
        log(self.__class__.__name__,
            message,
            quiet=self._config["verbose"] > verbose_level,
            **kwargs)

    def _std(self, data: pd.DataFrame, file: str = None) -> pd.DataFrame:
        """
        Standardize the data. A wrapper for socube.data.std

        Parameters
        ----------
        data : pd.DataFrame
            Data to be standardized
        file : str, optional
            Path to save standardized data. If not specified, the standardized
            data will not be saved.

        Returns
        -------
        pd.DataFrame
            Standardized data
        """
        self._log("Standard data by z-score")
        data = std(data, dtype=self._config["dtype"])
        if file is not None:
            writeHdf(data,
                     file,
                     callback=lambda: self._log("Standard data saved"))
        return data

    def _normalize(self, data: pd.DataFrame, file: str = None) -> pd.DataFrame:
        """
        Normalize the data. A wrapper for socube.data.minmax

        Parameters
        ----------
        data : pd.DataFrame
            Data to be normalized
        file : str, optional
            Path to save normalized data. If not specified, the normalized
            data will not be saved.

        Returns
        -------
        pd.DataFrame
            Normalized data
        """
        self._log("Normalize data by min-max scale")
        data = minmax(data, dtype=self._config["dtype"])
        if file is not None:
            writeHdf(data,
                     file,
                     callback=lambda: self._log("Normalized data saved"))
        return data

    def _alignment(self, data: pd.DataFrame, file: str = None) -> pd.DataFrame:
        """
        Align the data. A wrapper for socube.data.scatterToGrid

        Parameters
        ----------
        data : pd.DataFrame
            Data to be aligned, it must be a dataframe with columns 'x' and 'y'
            which are the x and y coordinates of the data in the scatter plot
        file : str, optional
            Path to save aligned data. If not specified, the aligned
            data will not be saved.

        Returns
        -------
        pd.DataFrame
            Aligned data, which is a dataframe with two columns 'x' and 'y',
            representing the x and y coordinates of the data in the grid
        """
        assert data.shape[1] == 2, f"Expect 2 col but got {data.shape[1]}"
        self._log("JV alignment to grid")
        grid_tensor = scatterToGrid(torch.from_numpy(data.values),
                                    device_name=self._config["device"])
        grid_data = pd.DataFrame(grid_tensor.numpy(),
                                 index=data.index,
                                 columns=["x", "y"])

        # grid shape row * col
        self.shape = (int(grid_data.y.max()) + 1, int(grid_data.x.max()) + 1)
        self._log(f"Grid shape is {self.shape}")
        if file is not None:
            writeCsv(grid_data,
                     file,
                     callback=lambda: self._log("Grid data saved"))
        return grid_data

    @property
    def grid(self) -> pd.DataFrame:
        """Return the grid data"""
        assert isinstance(self._grid, pd.DataFrame), "Uninitialized grid data"
        return self._grid.copy()

    @grid.setter
    def grid(self, data: pd.DataFrame):
        """Set the grid data"""
        assert isinstance(data, pd.DataFrame) and len(
            data.shape) == 2, "Invalid grid data"
        self._grid = data.copy()
        self._grid.columns = ["x", "y"]
        self.shape = (int(data.y.max()) + 1, int(data.x.max()) + 1)

    @property
    def scatter(self) -> pd.DataFrame:
        """Return the scatter data"""
        assert isinstance(self._scatter,
                          pd.DataFrame), "Uninitialized scatter data"
        return self._scatter.copy()

    @scatter.setter
    def scatter(self, data: pd.DataFrame):
        """Set the scatter data"""
        assert isinstance(data, pd.DataFrame) and len(
            data.shape) == 2, "Invalid scatter data"
        self._scatter = data.copy()

    @property
    def shape(self) -> tuple:
        """
        Return the socube 2d grid's shape
        """
        assert self._shape, "Uninitialized shape property"
        return self._shape

    @shape.setter
    def shape(self, new_shape: Tuple[int]):
        """
        Set the socube 2d grid's shape.
        """
        new_shape = tuple(filter(lambda obj: isinstance(obj, int), new_shape))
        assert len(new_shape) == 2, "Invalid grid shape change"
        self._shape = (new_shape[0], new_shape[1])

    def _manifold(self,
                  data: pd.DataFrame,
                  methods: str = 'umap',
                  file: str = None) -> pd.DataFrame:
        """
        Dimensionality reduction to two-dimensional space using
        manifold learning algorithm

        Parameters
        ----------
        data : pd.DataFrame
            Data to be reducted, rows represent genes and cols
            represent cells
        file : str, optional
            Path to save reducted scatter data. If not specified,
            the data will not be saved.

        Returns
        -------
        pd.DataFrame
            Aligned data, which is a dataframe with two columns 'x' and 'y',
            representing the x and y coordinates of the data in the scatter
        """
        assert methods in self._manifold_methods, "Unsupport manifold methods"
        self._log(f"Manifold learning by {methods}")
        data = self._manifold_methods[methods](data, self._config["metric"])

        if file is not None:
            writeCsv(data,
                     file,
                     callback=lambda: self._log("Manifold 2d result saved"))
        return data

    @abc.abstractmethod
    def transform(self, sample: pd.Series, **kwargs) -> np.ndarray:
        """
        Transform a droplet sample vector to
        its embedding form. for batch transform,
        see `batchTransform` api.

        Parameters
        ----------
        sample: pd.Series
            a droplet sample vector

        Returns
        ----------
        embedding form
        """
        raise NotImplementedError

    def batchTransform(self,
                       samples: pd.DataFrame,
                       data_dir: Optional[str] = None,
                       use_index: bool = False,
                       **kwargs) -> np.ndarray:
        r"""
        The sample data batch is subjected to feature transformation
        by socube to obtain the feature 3d matrix in the form of NCHW

        Parameters
        ----------------
        samples: pd.DataFrame
            The sample data to be transformed, row as samples, columns as features
        data_dir: str
            Path to save samples.  If given, it will save samples
        use_index: bool
            If True, it will save samples with using numerical index as filename. such as `0.npy`
            otherwise, sample name used instead.

        Returns
        -----------
            The generated embedding feature matrix and gene matching
        """
        gene_match = [0, 0]
        if data_dir is not None:
            mkDirs(data_dir)

        with ParallelManager(verbose=True):

            def _transform(sample: pd.Series):
                res, match = self.transform(sample, **kwargs)
                gene_match[0] = match[0]
                gene_match[1] = match[1]
                if data_dir is not None:
                    if use_index:
                        name = join(
                            data_dir,
                            f"{samples.index.get_loc(sample.name)}.npy")
                    else:
                        name = join(data_dir, f"{sample.name}.npy")
                    writeNpy(res, name)
                return res

            tqdm.pandas(desc="Transform")
            self._log(f"Batch transform {samples.shape[0]} samples")
            res = np.stack(samples.progress_apply(_transform, axis=1))
        self._log("Gene matched:{:d}, not matched:{:d}".format(*gene_match))
        return res, gene_match

    @abc.abstractmethod
    def load(self, cube_path: str) -> bool:
        """
        Load pretrained Socube
        """
        raise NotImplementedError


class SoPCACube(SoCube):
    """
    SoCube implemented with PCA latent gene feature
    """

    def __init__(self,
                 data_path: str,
                 home_path: Optional[str] = None,
                 metric: str = 'correlation',
                 dtype: str = "float32",
                 verbose: int = 1) -> None:

        super(SoPCACube, self).__init__(data_path=data_path,
                                        metric=metric,
                                        dtype=dtype,
                                        home_path=home_path,
                                        verbose=verbose)
        self._latent = None

    def fit(self,
            data: Union[str, pd.DataFrame],
            latent_dim: int,
            with_log: bool = True,
            with_std: bool = True,
            with_norm: bool = True,
            device: str = "cpu",
            manifold: str = 'umap',
            job_id: Optional[str] = None,
            seed: Optional[int] = None) -> str:
        """
        Fitting the most suitable cube representation to
        the currently specified data

        Parameters
        -------------
        data: str or pd.DataFrame
            If a str passed, it will be regarded as the input data's filename,
            otherwise a dataframe passed as the input data.
        latent_dim: int
            gene latent feature's target dimension. it is the third
            dimension of socube.
        with_log: bool
            Whether process data with logarithmic transform.
        with_std: bool
            Whether standardized data
        with_norm: bool
            Whether normalized data
        device: str
            Device name. cpu or cuda
        manifold: str
            Manifold algorithm, only 'tsne' and 'umap' supported
        job_id: str
            Job id, if is not specified, will be automatically generated.
        seed: int
            Random seed

        Returns
        -------
        Job id returned
        """
        if job_id is None:
            job_id = getJobId()
        super(SoPCACube, self).fit(jobid=job_id,
                                   device=device,
                                   log=with_log,
                                   std=with_std,
                                   norm=with_norm,
                                   latent_dim=latent_dim,
                                   manifold=manifold,
                                   seed=seed)
        if isinstance(data, str):
            self._config["data"] = data

        self._config["embedding_path"] = join(self._config["embedding_path"],
                                              job_id)
        config_path = join(self._config["embedding_path"], "config.json")
        json.dump(self._config, open(config_path, "w"))
        self._log("Current socube job config will be saved to {}".format(
            config_path))

        mkDirs(self._config["embedding_path"])

        # read data
        if isinstance(data, str):
            infile = join(self._config["data_path"], data)
            checkExist(infile)
            self._log(f"Load data from {infile}")
            data = pd.read_hdf(infile)
            self._log("Data loaded")
        elif not isinstance(data, pd.DataFrame):
            raise TypeError(f"Invalid data type '{type(data)}'")

        if with_log:
            data = np.log(data + 1)

        if with_std:
            data = self._std(data,
                             file=join(self._config["embedding_path"],
                                       "std.h5"))

        if with_norm:
            data = self._normalize(data,
                                   file=join(self._config["embedding_path"],
                                             "norm.h5"))

        data = data.sample(frac=1, random_state=seed)
        self.latent = self._pca(data,
                                latent_dim,
                                seed=seed,
                                file=join(self._config["embedding_path"],
                                          "latent.h5"))

        self.scatter = self._manifold(data,
                                      manifold,
                                      file=join(self._config["embedding_path"],
                                                "scatter.csv"))

        self.grid = self._alignment(self.scatter,
                                    file=join(self._config["embedding_path"],
                                              "grid.csv"))
        return job_id

    def _pca(self,
             data: pd.DataFrame,
             latent_dim: int,
             seed: Optional[int] = None,
             file: Optional[str] = None) -> pd.DataFrame:
        """
        PCA reduction used for getting gene latent feature

        Parameters
        ----------
        data: pd.DataFrame
            Input data
        latent_dim: int
            Target latent feature dimension
        seed: int
            Random seed
        file: str
            Npy file name to save the latent feature

        Returns
        -------
        pd.DataFrame
            Latent feature
        """
        pca = PCA(n_components=latent_dim, random_state=seed)
        transform = pca.fit_transform(data.values)
        transform = pd.DataFrame(transform, index=data.index)
        if file is not None:
            self._log("Write pca result to file")
            writeHdf(transform,
                     file,
                     callback=lambda: self._log("Gene pca data saved"))
        return transform

    @property
    def latent(self) -> pd.DataFrame:
        """Return latent feature"""
        assert isinstance(self._latent,
                          pd.DataFrame), "Uninitialize gene latent"
        return self._latent.copy()

    @latent.setter
    def latent(self, data: pd.DataFrame):
        """Set latent feature"""
        assert isinstance(data, pd.DataFrame) and len(
            data.shape) == 2, "Invalid latent"
        self._latent = data.copy()

    def transform(self, sample: pd.Series) -> np.ndarray:
        """Implementation of load method"""
        df = self.grid.join(sample)
        counts: dict = df.iloc[:, 2].isna().value_counts().to_dict()
        geneMatch = (counts.setdefault(False, 0), counts.setdefault(True, 0))
        df = df.fillna(0)
        df.columns = ['col', 'row', 'val']
        latent2d: np.ndarray = self.latent.loc[df.index].values
        # negative means this dim infered by others
        # weighted calculation
        latent2d *= df.val.values.reshape((-1, 1))
        # imputate and convert to 3D
        shape = self.shape
        latent3d = np.concatenate([
            latent2d,
            np.zeros(
                (shape[0] * shape[1] - latent2d.shape[0], latent2d.shape[1]))
        ]).reshape((*shape, -1))
        # sorted gene by idx, imputation pixel always at the end of matrix and no influence to sort
        latent3d[df.row.values, df.col.values] = latent2d
        # Convert HWC to CHW
        latent3d = np.transpose(latent3d, [2, 0, 1])
        return latent3d, geneMatch

    def load(self, cube_path: str) -> bool:
        """Implementation of load method"""
        try:
            self.grid = pd.read_csv(join(cube_path, "grid.csv"), index_col=0)
            self.scatter = pd.read_csv(join(cube_path, "scatter.csv"),
                                       index_col=0)
            self.latent = pd.read_hdf(join(cube_path, "latent.h5"))
            self._config = json.load(open(join(cube_path, "config.json")))
            self._log(f"Load socube job from {cube_path}")
            return True
        except Exception as e:
            self._log(f"Load socube failed: {e}")
            return False
