"""
Tests for the Machine Learning performance measures.
Author: Andrew Jarombek
Date: 10/6/2020
"""

import unittest

import pandas as pd

from rmse import rmse
from mae import mae


class PerformanceMeasureTests(unittest.TestCase):

    def test_rmse(self):
        m = 8
        X = pd.DataFrame({
            'miles': [2.99, 15.5, 11.34, 6.57, 4.5, 4.47, 4.5, 2.95],
            'pace': [432, 409, 384, 402, 410, 415, 420, 427]
        })
        h = lambda run: 400 + (run['miles'] * 2)
        result = rmse(X, m, h, 'pace')
        self.assertAlmostEqual(result, 20.57, places=2)

    def test_rmse_single_item(self):
        m = 1
        X = pd.DataFrame({'miles': [4.5], 'pace': [421]})
        h = lambda run: 425
        result = rmse(X, m, h, 'pace')
        self.assertAlmostEqual(result, 4)

    def test_mae(self):
        m = 8
        X = pd.DataFrame({
            'miles': [2.99, 15.5, 11.34, 6.57, 4.5, 4.47, 4.5, 2.95],
            'pace': [432, 409, 384, 402, 410, 415, 420, 427]
        })
        h = lambda run: 400 + (run['miles'] * 2)
        result = mae(X, m, h, 'pace')
        self.assertAlmostEqual(result, 17.125, places=2)

    def test_mae_single_item(self):
        m = 1
        X = pd.DataFrame({'miles': [4.5], 'pace': [421]})
        h = lambda run: 425
        result = mae(X, m, h, 'pace')
        self.assertAlmostEqual(result, 4)


if __name__ == '__main__':
    unittest.main()
