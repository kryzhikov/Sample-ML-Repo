import numpy as np
import pytest
import hypothesis.strategies as st
from hypothesis import assume, given
from hypothesis.strategies import integers as ints, floats

from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr

from utils import euclidean_distance, euclidean_similarity, pearson_similarity, apk, mapk

import warnings
warnings.filterwarnings(action='ignore')


def same_len_lists(min_value=1, max_value=100):
    return ints(min_value=min_value, max_value=max_value).flatmap(lambda n: st.lists(
        st.lists(
            floats(min_value=-1e6, max_value=1e6, allow_nan=False),
            min_size=n,
            max_size=n
        ),
        min_size=2,
        max_size=2
    ))


@given(same_len_lists())
def test_euclidean_distance(lists):
    x, y = lists
    x, y = np.array(x, dtype=np.float64), np.array(y, dtype=np.float64)
    np.testing.assert_almost_equal(euclidean_distance(x, y), euclidean(x, y), decimal=5)


@pytest.mark.parametrize(
    'x, y, expected_score',
    (
        ([0, 0, 0], [0, 0, 0], 1.0),
        ([0, 0, 0], [1, 1, 1], 0.36602540378443865),
        ([-1, -1, -1], [1, 1, 1], 0.2240092377397959),
    )
)
def test_euclidean_similarity(x, y, expected_score):
    x, y = np.array(x, dtype=np.float64), np.array(y, dtype=np.float64)
    np.testing.assert_almost_equal(euclidean_similarity(x, y), expected_score, decimal=5)


@pytest.mark.filterwarnings("ignore:pearsonr")
@given(same_len_lists(2, 100))
def test_pearson_similarity(lists):
    x, y = lists
    x, y = np.array(x, dtype=np.float64), np.array(y, dtype=np.float64)
    np.testing.assert_almost_equal(pearson_similarity(x, y), pearsonr(x, y)[0], decimal=5)


@pytest.mark.parametrize(
    'predicted, expected_score',
    (
        ([1, 4, 5], 0.3333333333333333),
        ([4, 1, 5], 0.16666666666666666),
        ([5, 4, 1], 0.1111111111111111),
        ([4, 1, 2], 0.38888888888888884),
        ([3, 2, 1], 1.0),
    )
)
def test_apk(predicted, expected_score):
    actual = np.array([1, 2, 3])
    np.testing.assert_almost_equal(apk(actual, predicted, k=3), expected_score)


def test_mapk():
    predicted = [
        [1, 4, 5],
        [4, 1, 5],
        [5, 4, 1],
        [4, 1, 2],
        [3, 2, 1],
    ]
    actual = [[1,2,3]] * len(predicted)
    expected_score = 0.4
    np.testing.assert_almost_equal(mapk(actual, predicted, k=3), expected_score)
