"""
Equation and examples of the Mean Absolute Error (MAE) performance measure.  This performance measure is linear, giving
the same weight to slight errors as it does large errors.
Author: Andrew Jarombek
Date: 10/5/2020
"""

from typing import Callable
import math

import pandas as pd


def mae(feature_values: pd.DataFrame, m: int, h: Callable[[pd.Series], int], label: str) -> float:
    """
    Implementation of the Mean Absolute Error (MAE) performance measure.
    :param feature_values: Feature values (including the label) for the problem.
    :param m: The number of instances in the dataset.
    :param h: Prediction function (hypothesis) for the problem.
    :param label: Label of the problem (the field containing the expected output).
    :return: The result of the performance measure which shows the amount of error found in the solution.
    """
    error_sum = 0
    for _, feature in feature_values.iterrows():
        error_sum += math.fabs(h(feature.drop(label)) - feature[label])

    return (1 / m) * error_sum
