"""
A fun first machine learning problem.  Inspired by me seeing rat friends almost every day on my runs
through central park :).
I hope you have a wonderful holiday :).
Author: Andrew Jarombek
Date: 10/6/2020
"""

import urllib.request
import os
import zipfile
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

# https://www.kaggle.com/new-york-city/nyc-rat-sightings/download
DATASET_URL = "https://bit.ly/2GAhRwT"


def fetch_data() -> None:
    try:
        os.mkdir('data')
    except FileExistsError:
        print('data directory already exists')

    with urllib.request.urlopen(DATASET_URL) as file:
        with open('data/source.zip', 'wb') as saved_file:
            saved_file.write(file.read())

    with zipfile.ZipFile('data/source.zip', 'r') as zip_ref:
        zip_ref.extractall('data')


def load_data() -> pd.DataFrame:
    return pd.read_csv('data/Rat_Sightings.csv')


def split_train_test(all_data: pd.DataFrame, test_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the full data set into two sets - one for training the machine learning algorithm (training set) and one for
    testing the performance of the machine learning algorithm (test set).  scikit-learn has a built-in utility function
    for performing this training/test set split.
    :param all_data: The full data set.
    :param test_ratio: The ratio of data that is placed in the training set.
    :return: A tuple containing the training set and test set.
    """
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(all_data) * test_ratio)
    train_indices = shuffled_indices[test_set_size:]
    test_indices = shuffled_indices[:test_set_size]
    return all_data.iloc[train_indices], all_data.iloc[test_indices]


if __name__ == '__main__':
    fetch_data()
    data = load_data()
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

    StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
