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

import abc
import torch

from torch.nn import Module
from torch.nn.parameter import Parameter
from typing import Any, Iterator, Tuple, Type

__all__ = ["ModuleMetaBase", "ModuleBase", "NetBase"]


class ModuleMetaBase(abc.ABCMeta):
    """
    Metaclass for all neural network model implemented in this package.

    Parameters
    ----------
    cls : type
        The class being created, which is a instance of ModuleMetaBase and
        a subclass of torch.nn.Module.
    in_channels : int
        The number of input channels. Any type created by this metaclass
        must contain this constructor parameter.
    out_channels : int
        The number of output channels. Any type created by this metaclass
        must contain this constructor parameter.
    *args : Any
        The arguments for the class constructor
    **kwargs : Any
        The keyword arguments for the class constructor

    Returns
    -------
    A new instance of the class being created.
    """
    def __call__(cls: Type[Module], in_channels: int, out_channels: int,
                 *args: Any, **kwargs: Any) -> Module:
        if not issubclass(cls, Module):
            raise TypeError("cls must be a subclass of torch.nn.Module")

        obj = cls.__new__(cls)
        obj.__init__(in_channels, out_channels, *args, **kwargs)
        return obj


class ModuleBase(Module, metaclass=ModuleMetaBase):
    """
    Basical abstract class of all neural network modules.
    Any subclass of this class must implement the following
    abstract methods:

    Parameters
    ----------
    in_channels : int
        The number of input channels.
    out_channels : int
        The number of output channels.
    **kwargs : Any
    """
    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super(ModuleBase, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        # store any other parameters required by the model
        self._kwargs = kwargs

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Data forward for a neural network waited to be implemented.

        Parameters
        ----------
        x : torch.Tensor
            The input data.

        Returns
        -------
        The output data.
        """
        raise NotImplementedError

    def parameters(self,
                   recurse: bool = True,
                   skip_no_grad: bool = False) -> Iterator[Parameter]:
        """
        Return model paramters with or without no grad parameters.
        If some layers are freezed, these parameters should not be
        updated by optimizer. Though no grad parameters won't be
        updated even if they are sent to optimizer. But it is a waste
        of calculation resource.

        Parameters
        ----------
        recurse : bool
            If True, return all parameters of all sublayers.
        skip_no_grad : bool, default False
            If True, return all parameters with grad.
             it advised set as True when using optimizer

        Returns
        -------
        An iterator of all parameters.
        """
        params = super().parameters(recurse=recurse)
        if skip_no_grad:
            return filter(lambda param: param.requires_grad, params)
        return params

    def freeze(self, layer: Module) -> None:
        """
        Freeze parameters of a layer and its sublayers

        Parameters
        ----------
        layer : Module
            The layer to be frozen.
        """
        for param in layer.parameters(recurse=True):
            param.requires_grad = False

    def unfreeze(self, layer: Module) -> None:
        """
        Unfreeze parameters of a layer and its sublayers

        Parameters
        ----------
        layer : Module
            The layer to be unfrozen.
        """
        for param in layer.parameters(recurse=True):
            param.requires_grad = True


class NetBase(ModuleBase):
    """
    Basic abstract class of all neural network models.
    Any subclass of this class must implement the following
    abstract methods:

    Parameters
    ----------
    in_channels : int
        The number of input channels.
    out_channels : int
        The number of output channels.
    shape : Tuple[int]
        The shape of a sample.
    **kwargs : Any
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 shape: Tuple[int] = None,
                 **kwargs) -> None:
        r"""
        Parameters:
        ----------
        - inChannels
        - outChannels
        - shape  the shape of a sample.
        """
        super().__init__(in_channels, out_channels, **kwargs)
        self._binary = False
        self._shape = shape

    @abc.abstractmethod
    def criterion(self, y_predict: torch.Tensor,
                  y_true: torch.Tensor) -> torch.Tensor:
        """
        Abstract methods to calculate the loss of the model.

        Parameters
        ----------
        y_predict : torch.Tensor
            The predicted data.
        y_true : torch.Tensor
            The true data.

        Returns
        -------
        The loss of the model.
        """
        raise NotImplementedError

    @property
    def shape(self) -> Tuple[int]:
        """
        Return the shape of a sample.

        Returns
        -------
        The shape of a sample.
        """
        if self._shape is None:
            raise TypeError("Uninitialized input tensor shape")
        return self._shape

    @shape.setter
    def shape(self, s: Tuple[int]):
        """
        Set the shape of a sample.

        Parameters
        ----------
        s : Tuple[int]
            The shape of a sample.

        Raises
        ------
        TypeError
            If the shape is not a tuple of int.
        """
        if len(s) <= 0:
            raise TypeError("Shape's length should greater than zero")

        for dim in s:
            if dim is not None and not isinstance(dim, int):
                raise TypeError("Invalid dimension")

        else:
            self._shape = s
