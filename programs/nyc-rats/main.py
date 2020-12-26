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
RAT_DATASET_URL = "https://bit.ly/2GAhRwT"

# https://data.beta.nyc/dataset/0ff93d2d-90ba-457c-9f7e-39e47bf2ac5f/resource/7caac650-d082-4aea-9f9b-3681d568e8a5/download/nyc_zip_borough_neighborhoods_pop.csv
NYC_POPULATION_DATASET_URL = "https://bit.ly/38ysHxo"


def fetch_rat_sighting_data() -> None:
    """
    Fetch the dataset containing the NYC rat sighting data.
    """
    try:
        os.mkdir('data')
    except FileExistsError:
        print('data directory already exists')

    if not os.path.exists('data/Rat_Sightings.csv'):
        with urllib.request.urlopen(RAT_DATASET_URL) as file:
            with open('data/source.zip', 'wb') as saved_file:
                saved_file.write(file.read())

        with zipfile.ZipFile('data/source.zip', 'r') as zip_ref:
            zip_ref.extractall('data')


def fetch_nyc_population_data() -> None:
    """
    Fetch the dataset containing NYC population data (bucketed by zip code).
    """
    if not os.path.exists('data/NYC_Population.csv'):
        with urllib.request.urlopen(NYC_POPULATION_DATASET_URL) as file:
            with open('data/NYC_Population.csv', 'wb') as saved_file:
                saved_file.write(file.read())


def load_rat_sighting_data() -> pd.DataFrame:
    """
    Load the CSV file containing rat sighting data into a pandas DataFrame object.
    """
    return pd.read_csv('data/Rat_Sightings.csv')


def load_population_data() -> pd.DataFrame:
    """
    Load the CSV file containing nyc population data into a pandas DataFrame object.
    """
    return pd.read_csv('data/NYC_Population.csv')


def combine_datasets(rat_sighting_data: pd.DataFrame, population_data: pd.DataFrame) -> pd.DataFrame:
    """
    Combine the rat sighting and population by zip code data sets.
    :param rat_sighting_data: The rat sighting data.
    :param population_data: The population by zip code data.
    :return: The merged data set as a pandas data frame.
    """
    pop_data = population_data.rename(columns={
        'zip': 'Zip',
        'borough': 'Borough',
        'post_office': 'Post Office',
        'neighborhood': 'Neighborhood',
        'population': 'Population',
        'density': 'Density'
    })
    pop_data = pop_data.drop(columns=['Borough'])
    full_data_set = rat_sighting_data.join(pop_data.set_index('Zip'), on='Incident Zip')

    full_data_set = full_data_set.loc[:, [
        'Incident Zip',
        'Location Type',
        'Incident Address',
        'Status',
        'Borough',
        'Latitude',
        'Longitude',
        'Post Office',
        'Neighborhood',
        'Population',
        'Density'
    ]]

    full_data_set = full_data_set.dropna(subset=['Incident Zip'])
    full_data_set['Incident Zip'] = full_data_set['Incident Zip'].apply(lambda x: str(int(x)))
    return full_data_set


def split_train_test_sample(all_data: pd.DataFrame, test_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the full data set into two sets - one for training the machine learning algorithm (training set) and one for
    testing the performance of the machine learning algorithm (test set).  scikit-learn has a built-in utility function
    for performing this training/test set split.
    :param all_data: The full data set.
    :param test_ratio: The ratio of data that is placed in the training set.
    :return: A tuple containing the training set and test set.
    """
    shuffled_indices = np.random.permutation(len(all_data))
    test_set_size = int(len(all_data) * test_ratio)
    train_indices = shuffled_indices[test_set_size:]
    test_indices = shuffled_indices[:test_set_size]
    return all_data.iloc[train_indices], all_data.iloc[test_indices]


def train_test_split_random_sampling(all_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into a training set and a test set.  This is done with random sampling (picking data for the
    two sets randomly).
    :param all_data: The data before its split into separate sets.
    :return: A tuple containing the training set and test set.
    """
    return train_test_split(all_data, test_size=0.2, random_state=42)


def train_test_split_stratified_sampling(all_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into a training set and a test set.  This is done with stratified sampling (picking a specific number
    of data rows from subgroups [strata] so that the data accurately reflects the overall population).
    :param all_data: The data before its split into separate sets.
    :return: A tuple containing the training set and test set.
    """
    zip_code_counts = all_data['Incident Zip'].astype('category').value_counts()
    zip_codes_with_one_row = zip_code_counts[zip_code_counts == 1].index

    cleaned_data = all_data[~all_data['Incident Zip'].isin(zip_codes_with_one_row)]

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    strat_train_set = None
    strat_test_set = None

    for train_index, test_index in split.split(cleaned_data, cleaned_data['Incident Zip']):
        strat_train_set = cleaned_data.reindex(train_index)
        strat_test_set = cleaned_data.reindex(test_index)

    return strat_train_set, strat_test_set


if __name__ == '__main__':
    fetch_rat_sighting_data()
    fetch_nyc_population_data()

    rat_sighting_data = load_rat_sighting_data()
    population_data = load_population_data()

    data = combine_datasets(rat_sighting_data, population_data)

    train_set, test_set = train_test_split_random_sampling(data)
    train_test_split_stratified_sampling(data)
