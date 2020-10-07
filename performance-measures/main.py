"""
Main method for running the Machine Learning performance measures.
Author: Andrew Jarombek
Date: 10/6/2020
"""

import pandas as pd

from mae import mae
from rmse import rmse

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

    result = mae(X, m, h, 'pace')
    print(result)
