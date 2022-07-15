import pandas as pd
from socube.data.preprocess import items, std, minmax, global_minmax, filterData, umap2D, vec2Grid


def test_std():
    data = pd.DataFrame([[1, 2, 5], [2, 6, 6], [3, 5, 7], [4, 6, 0]],
                        columns=["col_1", "col_2", "col_3"])

    std_h_data = std(data)
    assert (
        std_h_data.columns == data.columns).all(), "Column name should be same"


def test_global_minmax():
    data = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    expect = pd.DataFrame([[0.0, 0.125, 0.25], [0.375, 0.5, 0.625], [0.75, 0.875, 1.0]])
    assert (global_minmax(data) == expect).values.all(), "global_minmax error"


def test_filterData():
    filterData
    raise NotImplementedError


def test_minmax():
    minmax
    raise NotImplementedError


def test_umap2d():
    umap2D
    raise NotImplementedError


def test_vector2grid():
    vec2Grid
    raise NotImplementedError


def test_items():
    items
    raise NotImplementedError
