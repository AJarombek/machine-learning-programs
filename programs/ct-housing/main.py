"""
Home for Christmas, decided to make a machine learning program predicting home prices in Connecticut.
Author: Andrew Jarombek
Date: 12/25/2020
"""

import urllib.request
import os

# https://data.ct.gov/Housing-and-Development/Real-Estate-Sales-2001-2018/5mzw-sjtu
HOUSING_DATASET_URL = "https://data.ct.gov/api/views/5mzw-sjtu/rows.csv?accessType=DOWNLOAD"


def fetch_ct_housing_data() -> None:
    """
    Fetch the dataset containing Connecticut housing data.
    """
    try:
        os.mkdir('data')
    except FileExistsError:
        print('data directory already exists')

    if not os.path.exists('data/CT_Housing.csv'):
        with urllib.request.urlopen(HOUSING_DATASET_URL) as file:
            with open('data/CT_Housing.csv', 'wb') as saved_file:
                saved_file.write(file.read())


if __name__ == '__main__':
    fetch_ct_housing_data()
