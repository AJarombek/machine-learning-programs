"""
A fun first machine learning problem.  Inspired by me seeing rat friends almost every day on my runs
through central park :).
Author: Andrew Jarombek
Date: 10/6/2020
"""

import urllib.request
import os
import zipfile

import pandas as pd

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


if __name__ == '__main__':
    fetch_data()
    data = load_data()
