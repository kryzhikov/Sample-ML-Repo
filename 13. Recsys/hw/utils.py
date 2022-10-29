import numpy as np
from typing import Optional, Union, Tuple


def euclidean_distance(x: np.array, y: np.array) -> float:
    """
    Calculate euclidean distance between points x and y
    Args:
        x, y: two points in Euclidean n-space
    Returns:
        Length of the line segment connecting given points
    """
    pass


def euclidean_similarity(x: np.array, y: np.array) -> float:
    """
    Calculate euclidean similarity between points x and y
    Args:
        x, y: two points in Euclidean n-space
    Returns:
        Similarity between points x and y
    """
    pass


def pearson_similarity(x: np.array, y: np.array) -> float:
    """
    Calculate a Pearson correlation coefficient given 1-D data arrays x and y
    Args:
        x, y: two points in n-space
    Returns:
        Pearson correlation between x and y
    """
    pass


def apk(actual: np.array, predicted: np.array, k: int = 10) -> float:
    """
    Compute the average precision at k
    Args:
        actual: a list of elements that are to be predicted (order doesn't matter)
        predicted: a list of predicted elements (order does matter)
        k: the maximum number of predicted elements
    Returns:
        The average precision at k over the input lists
    """
    pass


def mapk(actual: np.array, predicted: np.array, k: int = 10) -> float:
    """
    Compute the mean average precision at k
    Args:
        actual: a list of lists of elements that are to be predicted
        predicted: a list of lists of predicted elements
        k: the maximum number of predicted elements
    Returns:
        The mean average precision at k over the input lists
    """
    pass
