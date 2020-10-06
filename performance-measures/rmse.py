"""
Equation and examples of the Root Mean Square Error (RMSE) performance measure.
Author: Andrew Jarombek
Date: 10/5/2020
"""

from typing import Callable
import math

import pandas as pd


def rmse(feature_values: pd.DataFrame, m: int, h: Callable[[pd.Series], int], label: str) -> float:
    """
    Implementation of the Root Mean Square Error (RMSE) performance measure.
    :param feature_values: Feature values (including the label) for the problem.
    :param m: The number of instances in the dataset.
    :param h: Prediction function (hypothesis) for the problem.
    :param label: Label of the problem (the field containing the expected output).
    :return: The result of the performance measure which shows the amount of error found in the solution.
    """
    error_sum = 0
    for _, feature in feature_values.iterrows():
        error_sum += (h(feature.drop(label)) - feature[label]) ** 2

    return math.sqrt((1 / m) * error_sum)


if __name__ == '__main__':
    m = 8
    X = pd.DataFrame({
        'miles': [2.99, 15.5, 11.34, 6.57, 4.5, 4.47, 4.5, 2.95],
        'pace': [432, 409, 384, 402, 410, 415, 420, 427]
    })


    def h(run: pd.Series) -> int:
        """
        A naive hypothesis function (which doesn't use machine learning) to try and predict the pace of a run.  It
        predicts for each run, the pace gets two seconds slower for each additional mile,
        starting at a baseline of 6:40.
        :param run: Information about a run (without the label).
        :return: The predicted pace in seconds.
        """
        return 400 + (run['miles'] * 2)

    result = rmse(X, m, h, 'pace')
    print(result)
