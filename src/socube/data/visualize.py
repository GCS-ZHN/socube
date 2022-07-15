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
import matplotlib.pyplot as plt

from highcharts import Highchart
from typing import Dict, Optional, Sequence, Tuple
from ..utils.logging import log
"""
A module for data visualization
"""

__all__ = [
    "getHeatColor", "convertHexToRGB", "convertRGBToHex", "plotScatter", "plotGrid", "plotAUC"
]


def getHeatColor(intensity: float,
                 topColor: str,
                 bottomColor: str = "#ffffff") -> str:
    """
    Calculation of heat map color values based on intensity values

    Parameters
    ----------
    intensity: float value
        color intensity values between 0 and 1
    topColor: hexadecimal color string
       Color value when intensity value is 1
    bottomColor: hexadecimal color string
     Color value when intensity value is 0, default is white

    Returns
    ----------
        The hexadecimal color string corresponding to the intensity

    Examples
    ----------
    >>> getHeatColor(0.5, "#ff0000")
    '#ff7f7f'
    """
    assert intensity >= 0 and intensity <= 1 + 1e-8, f"Intensity should range from 0 to 1, But got {intensity}"
    bRGBColor = convertHexToRGB(bottomColor)
    tRGBColor = convertHexToRGB(topColor)
    color = ((1 - intensity) * bRGBColor[i] + intensity * tRGBColor[i]
             for i in range(3))
    return convertRGBToHex(tuple(color))


def convertHexToRGB(hex_color: str) -> Tuple[int]:
    """
    Convert hexadecimal color strings to RGB tri-color integer tuples

    Parameters
    ----------
    hex_color: hexadecimal color string, such as '#ff0000'

    Returns
    ----------
    RGB tri-color integer tuples

    Examples
    ----------
    >>> hexToRGB('#ff0000')
    (255, 0, 0)
    """
    if not isinstance(hex_color, str):
        raise ValueError("hex_color should be a string")
    if hex_color[0] != "#":
        raise ValueError("hex_color should start with '#'")
    if len(hex_color) != 7:
        raise ValueError("hex_color should be 7 characters long")

    try:
        d = int(hex_color.replace("#", ""), base=16)
        b, d = d % 256, d // 256
        g, d = d % 256, d // 256
        r, d = d % 256, d // 256
        return r, g, b
    except Exception:
        raise ValueError("Ilegal hex color format")


def convertRGBToHex(color: Tuple[int]) -> str:
    """
    Convert RGB tricolor integer tuple to hexadecimal color string

    Parameters
    ----------
    color: RGB tricolor integer tuple
        such as (255, 0, 0)

    Returns
    ----------
    hexadecimal color string, such as '#ff0000'

    Examples
    ----------
    >>> rgbToHex((255, 0, 0))
    '#ff0000'
    """
    assert color is not None and len(color) == 3, "Ilegal rgb color format"
    try:
        r = int(color[0] % 256)
        g = int(color[1] % 256)
        b = int(color[2] % 256)
        return "#%02x%02x%02x" % (r, g, b)
    except Exception:
        raise ValueError("Ilegal rgb color format")


# see https://api.highcharts.com/highcharts
def plotScatter(data2d: pd.DataFrame,
                colormap: Dict[str, str],
                title: str,
                subtitle: str,
                filename: str = None,
                scatter_symbol: str = "circle",
                width: int = 1000,
                height: int = 850,
                radius: int = 3,
                x_min: Optional[int] = None,
                y_min: Optional[int] = None,
                x_max: Optional[int] = None,
                y_max: Optional[int] = None,
                x_title: Optional[str] = None,
                y_title: Optional[str] = None):
    """
    Draw the scatter image of socube

    Parameters
    ----------
    data2d: pandas.DataFrame
        The data to be plotted, with columns of x, y, label and subtype,
        if subtype is float, it regarded as intensity value, and the color will be
        calculated based on the intensity value.
    colormap: Dict[str, str]
        The color map for the subtype, with key as subtype name and value
        as hexadecimal color string. If the subtype is float, colormap's key
        should be '0' and '1', '0' for low intensity color and '1' for high intensity color.
    title: str
        The title of the plot
    subtitle: str
        The subtitle of the plot
    filename: str
        The filename of the plot, if None, the plot will not be saved. format is html and
        the filename extension will automatically be added and you should not add it.
    width: int
        The width of the plot, unit is pixel
    height: int
        The height of the plot, unit is pixel
    radius: int
        The radius of the scatter point, unit is pixel

    Returns
    ----------
    The plot object

    Examples
    ----------
    >>> data2d = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6], 'label': ['a', 'b', 'c'], 'subtype': [0.5, 0.7, 0.9]})
    >>> colormap = {'0': '#ff0000', '1': '#00ff00'}
    >>> plotScatter(data2d, colormap, 'title', 'subtitle', 'test.html')
    """
    data2d = data2d.copy()
    # Create drawing objects and configure global parameters
    H = Highchart(width=width, height=height)

    # Set jquery CDN to load
    H.JSsource[0] = "http://code.jquery.com/jquery-1.9.1.min.js"
    H.set_options('chart', {'type': 'scatter', 'zoomType': 'xy'})
    H.set_options('title', {'text': title})
    H.set_options('subtitle', {'text': subtitle})
    H.set_options(
        'xAxis',
        {
            'title': {
                'enabled': True,
                'text': x_title,
                'style': {
                    'fontSize': 20
                }
            },
            'labels': {
                'style': {
                    'fontSize': 20
                }
            },
            'showLastLabel': True,
            'min': x_min,
            'max': x_max
        })
    H.set_options(
        'yAxis',
        {
            'title': {
                'text': y_title,
                'style': {
                    'fontSize': 20
                }
            },
            'labels': {
                'style': {
                    'fontSize': 20
                }
            },
            'min': y_min,
            'max': y_max
        })
    H.set_options(
        'legend', {
            'align': 'right',
            'layout': 'vertical',
            'margin': 1,
            'verticalAlign': 'top',
            'y': 40,
            'symbolHeight': 12,
            'floating': False,
        })
    H.set_options(
        'plotOptions', {
            'scatter': {
                'marker': {
                    'radius': radius,
                    'states': {
                        'hover': {
                            'enabled': True,
                            'lineColor': 'rgb(100,100,100)'
                        }
                    },
                    "symbol": scatter_symbol
                },
                'states': {
                    'hover': {
                        'marker': {
                            'enabled': False
                        }
                    }
                },
                'tooltip': {
                    'headerFormat': '<b>{series.name}</b><br>',
                    'pointFormat': 'Label: {point.label}'
                }
            },
            'series': {
                'turboThreshold': 50000
            }
        })

    if data2d.subtype.dtype.name == "object":
        # Add series by subtype
        for subtype, color in colormap.items():
            dfi = data2d[data2d.subtype.apply(
                lambda subtypes: subtype in subtypes.split("||"))]
            if len(dfi) == 0:
                continue
            data = dfi.to_dict('records')
            H.add_data_set(data, 'scatter', subtype, color=color)
    elif "float" in data2d.subtype.dtype.name:
        data2d["color"] = data2d.subtype.apply(lambda intensity: getHeatColor(
            intensity, colormap["1"], colormap["0"]))
        H.add_data_set(data2d.to_dict('records'), 'scatter')
    else:
        raise NotImplementedError("Unsupport 'subtype' column's dtype")

    if filename:
        H.save_file(filename)
        log(__name__, f"Save to {filename}.html")

    return H


def plotGrid(data2d: pd.DataFrame,
             colormap: Dict[str, str],
             shape: Tuple[int],
             title: str,
             subtitle: str,
             filename: str,
             width: int = 1000,
             height: int = 850) -> Highchart:
    """
    Draw socube's Grid image

    Parameters
    ----------
    data2d: pandas.DataFrame
        The data to be plotted, with columns of x, y, label and subtype,
        if subtype is float, it regarded as intensity value, and the color will be
        calculated based on the intensity value.
    colormap: Dict[str, str]
        The color map for the subtype, with key as subtype name and value
        as hexadecimal color string. If the subtype is float, colormap's key
        should be '0' and '1', '0' for low intensity color and '1' for high intensity color.
    shape: Tuple[int]
        The shape of the grid, (row, col)
    title: str
        The title of the plot
    subtitle: str
        The subtitle of the plot
    filename: str
        The filename of the plot, if None, the plot will not be saved. format is html and
        the filename extension will automatically be added and you should not add it.
    width: int
        The width of the plot, unit is pixel
    height: int
        The height of the plot, unit is pixel

    Returns
    ----------
    The plot object

    Examples
    ----------
    >>> data2d = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6], 'label': ['a', 'b', 'c'], 'subtype': [0.5, 0.7, 0.9]})
    >>> colormap = {'0': '#ff0000', '1': '#00ff00'}
    >>> plotGrid(data2d, colormap, (6, 3), 'title', 'subtitle', 'test.html')
    """
    data2d = data2d.copy()
    H = Highchart(width=width, height=height)
    H.JSsource[0] = "http://code.jquery.com/jquery-1.9.1.min.js"
    H.set_options('chart', {'type': 'heatmap', 'zoomType': 'xy'})
    H.set_options('title', {'text': title})
    H.set_options('subtitle', {'text': subtitle})
    H.set_options(
        'xAxis', {
            'title': None,
            'min': 0,
            'max': shape[1],
            'startOnTick': False,
            'endOnTick': False,
            'allowDecimals': False,
            'labels': {
                'style': {
                    'fontSize': 20
                }
            }
        })
    H.set_options(
        'yAxis', {
            'title': {
                'text': ' ',
                'style': {
                    'fontSize': 20
                }
            },
            'startOnTick': False,
            'endOnTick': False,
            'gridLineWidth': 0,
            'reversed': True,
            'min': 0,
            'max': shape[0],
            'allowDecimals': False,
            'labels': {
                'style': {
                    'fontSize': 20
                }
            }
        })
    H.set_options(
        'legend', {
            'align': 'right',
            'layout': 'vertical',
            'margin': 1,
            'verticalAlign': 'top',
            'y': 60,
            'symbolHeight': 12,
            'floating': False,
        })
    H.set_options(
        'tooltip', {
            'headerFormat': '<b>{series.name}</b><br>',
            'pointFormat': '{point.label} {point.x} {point.y}'
        })

    # plotOptions.series.turboThreshold is the threshold of serie point.
    # For detail , See https://api.highcharts.com/highcharts/plotOptions.series.turboThreshold
    H.set_options('plotOptions', {'series': {'turboThreshold': 50000}})
    if data2d.subtype.dtype.name == "object":
        for subtype, color in colormap.items():
            dfi = data2d[data2d.subtype.apply(
                lambda subtypes: subtype in subtypes.split("||"))]
            if len(dfi) == 0:
                continue
            data = dfi.to_dict('records')
            H.add_data_set(data, 'heatmap', subtype, color=color)
    elif "float" in data2d.subtype.dtype.name:
        data2d["color"] = data2d.subtype.apply(lambda intensity: getHeatColor(
            intensity, colormap["1"], colormap["0"]))
        H.add_data_set(data2d.to_dict('records'), 'heatmap')
    else:
        raise NotImplementedError("Unsupport 'subtype' column's dtype")

    if filename:
        H.save_file(filename)
        log(__name__, f"Save to {filename}.html")
    return H


def plotAUC(data: Dict[str, Tuple[Sequence]],
            title: str,
            xlabel: str,
            ylabel: str,
            file: Optional[str] = None,
            slash: int = 0):
    """
    Plot a AUC curve

    Parameters
    ------------------
    data: Dict[str, Tuple[Sequence]]
        The data to be plotted, with key as the name of the curve and value
        as the (x, y) data.
    title: str
        The title of the plot
    xlabel: str
        The xlabel of the plot
    ylabel: str
        The ylabel of the plot
    file: str
        The filename of the plot, if None, the plot will not be saved
    slash: int
        if `slash` is positve, will plot a forward slash,
        if `slash` is negative, will plot a back slash.

    Examples
    ----------
    >>> data = {'AUC': ([0.1, 0.2, 0.3], [0.1, 0.2, 0.3]), 'AUC2': ([0.1, 0.2, 0.3], [0.1, 0.2, 0.3])}
    >>> plotAUC(data, 'AUC', 'xlabel', 'ylabel', 'auc.png')
    """
    # reuse figure object to avoid open too many figures and cosume too much memory
    plt.figure(1024)
    for legend, vals in data.items():
        plt.plot(vals[0], vals[1], lw=2, label=legend)

    if slash > 0:
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    elif slash < 0:
        plt.plot([0, 1], [1, 0], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="lower right")
    if file is None:
        return
    plt.savefig(file)

    # clean current activate figure for reusing.
    plt.clf()
    plt.close(1024)
