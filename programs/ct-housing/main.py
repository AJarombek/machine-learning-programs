"""
Home for Christmas, decided to make a machine learning program predicting home prices in Connecticut.
Author: Andrew Jarombek
Date: 12/25/2020
"""

import urllib.request
import os
import time
from datetime import datetime
from typing import Tuple

import pandas as pd
import scipy.sparse
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

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


def load_ct_housing_data() -> pd.DataFrame:
    """
    Load the CSV file containing Connecticut housing data.
    """
    return pd.read_csv('data/CT_Housing.csv')


def clean_ct_housing_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Connecticut housing data by removing null values and mapping dates to their corresponding unix values.
    :param data: The housing data set.
    :return: The cleaned data.
    """
    clean_data = data.drop(['Remarks', 'Address', 'SerialNumber', 'NonUseCode'], axis=1)
    clean_data = clean_data[~clean_data['DateRecorded'].isnull()]
    clean_data = clean_data[~clean_data['AssessedValue'].isnull()]
    clean_data = clean_data[~clean_data['SaleAmount'].isnull()]
    clean_data = clean_data[~clean_data['SalesRatio'].isnull()]

    def convert_to_unix(x: str) -> int:
        ts = datetime.strptime(x, "%m/%d/%Y")
        return int(time.mktime(ts.timetuple()))

    clean_data['DateRecorded'] = clean_data['DateRecorded'].apply(convert_to_unix)
    return clean_data


def split_labels_and_data(data: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Split the housing data into two sets.  The first set is the label (the correct answer to the ML problem).  In this
    scenario, the label is the price that the house was sold for.  The second set is the data that is fed to the machine
    learning algorithm.  It is the entire initial dataset minus the label.
    :param data: The housing data set.
    :return: Label data (tuple index 0) and the training and testing data (tuple index 1).
    """
    return data['SaleAmount'].copy(), data.drop('SaleAmount', axis=1)


def transform_and_encode_ct_housing_data(data: pd.DataFrame) -> scipy.sparse:
    """
    Transform the data so that it performs better as a machine learning training model.
    :param data: The housing data (without the label)
    :return: The data which is passed to the ML algorithm.
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', OneHotEncoder())
    ])

    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, ['ListYear', 'DateRecorded', 'AssessedValue', 'SalesRatio', 'NumberOfBuildings']),
        ('cat', cat_pipeline, ['Town', 'PropertyType', 'ResidentialType'])
    ])

    return full_pipeline.fit_transform(data)


def train_ct_housing_linear_regression_model(prepared_data: scipy.sparse, labels: pd.Series) -> LinearRegression:
    """
    Train the data using a linear regression model.
    :param prepared_data: Data that is prepared to be used by the model.
    :param labels: Labels (the answer to the ML problem) for the data.
    :return: The linear regression model object.
    """
    lin_reg = LinearRegression()
    lin_reg.fit(prepared_data, labels)
    return lin_reg


def train_ct_housing_decision_tree_regressor_model(prepared_data: scipy.sparse, labels: pd.Series) \
        -> DecisionTreeRegressor:
    """
    Train the data using a decision tree regressor model.
    :param prepared_data: Data that is prepared to be used by the model.
    :param labels: Labels (the answer to the ML problem) for the data.
    :return: The decision tree regressor model object.
    """
    decision_tree_reg = DecisionTreeRegressor()
    decision_tree_reg.fit(prepared_data, labels)
    return decision_tree_reg


def train_ct_housing_random_forst_regressor_model(prepared_data: scipy.sparse, labels: pd.Series) \
        -> RandomForestRegressor:
    """
    Train the data using a random forest regressor model.
    :param prepared_data: Data that is prepared to be used by the model.
    :param labels: Labels (the answer to the ML problem) for the data.
    :return: The random forest regressor model object.
    """
    random_forest_reg = RandomForestRegressor()
    random_forest_reg.fit(prepared_data, labels)
    return random_forest_reg


if __name__ == '__main__':
    ''' 
    As I'm sure you know, you are always welcome to interrupt my day or have someone else interrupt my day 
    if you think I can help you :) 
    '''
    fetch_ct_housing_data()
    initial_housing_data = load_ct_housing_data()
    train_set, test_set = train_test_split(initial_housing_data, test_size=0.2, random_state=42)
    cleaned_housing_data = clean_ct_housing_data(train_set)
    housing_data_labels, housing_data = split_labels_and_data(cleaned_housing_data)
