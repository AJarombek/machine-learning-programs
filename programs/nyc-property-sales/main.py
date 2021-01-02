"""
A machine learning problem that predicts property sale prices in New York City.
https://www.kaggle.com/new-york-city/nyc-property-sales/download
Author: Andrew Jarombek
Date: 12/30/2020
"""

import time
from typing import Tuple
from datetime import datetime

import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.sparse
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

num_attributes = [
    'BLOCK',
    'LOT',
    'ZIP CODE',
    'RESIDENTIAL UNITS',
    'COMMERCIAL UNITS',
    'TOTAL UNITS',
    'LAND SQUARE FEET',
    'GROSS SQUARE FEET',
    'YEAR BUILT',
    'TAX CLASS AT TIME OF SALE',
    'SALE DATE'
]

cat_attributes = [
    'BOROUGH',
    'NEIGHBORHOOD',
    'BUILDING CLASS CATEGORY',
    'TAX CLASS AT PRESENT',
    'BUILDING CLASS AT PRESENT',
    'BUILDING CLASS AT TIME OF SALE'
]


def load_data() -> pd.DataFrame:
    """
    Load the CSV file containing NYC property sales data.
    """
    return pd.read_csv('data/NYC_Property_Sales.csv')


def clean_data_set(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the NYC property sales data.
    :param data: The property sales data set.
    :return: The cleaned data.
    """
    clean_data = data.drop(['Unnamed: 0', 'EASE-MENT', 'ADDRESS', 'APARTMENT NUMBER'], axis=1)

    clean_data['SALE PRICE'] = clean_data['SALE PRICE'].replace(r'^\s+-\s+$', '0', regex=True)
    clean_data['SALE PRICE'] = clean_data['SALE PRICE'].astype(int)
    clean_data = clean_data[~clean_data['SALE PRICE'].isnull()]
    clean_data = clean_data[clean_data['SALE PRICE'] > 100_000]

    clean_data['LAND SQUARE FEET'] = \
        clean_data['LAND SQUARE FEET'] \
            .replace(r'^\s+-\s+$', -1, regex=True) \
            .astype(int) \
            .replace(-1, np.nan)

    clean_data['GROSS SQUARE FEET'] = \
        clean_data['GROSS SQUARE FEET'] \
            .replace(r'^\s+-\s+$', -1, regex=True) \
            .astype(int) \
            .replace(-1, np.nan)

    def convert_to_unix(x: str) -> int:
        ts = datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
        return int(time.mktime(ts.timetuple()))

    clean_data['SALE DATE'] = clean_data['SALE DATE'].apply(convert_to_unix)

    clean_data['BOROUGH'] = clean_data['BOROUGH'].map({
        1: 'MANHATTAN',
        2: 'BRONX',
        3: 'BROOKLYN',
        4: 'QUEENS',
        5: 'STATEN ISLAND'
    })

    return clean_data


def split_labels_and_data(data: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Split the data into two sets.  The first set is the label (the correct answer to the ML problem).  In this scenario,
    the label is the price that the property was sold for.  The second set is the data that is fed to the machine
    learning algorithm.  It is the entire initial dataset minus the label.
    :param data: The property sale data set.
    :return: Label data (tuple index 0) and the training and testing data (tuple index 1).
    """
    return data['SaleAmount'].copy(), data.drop('SaleAmount', axis=1)


def transform_and_encode_data(data: pd.DataFrame) -> scipy.sparse:
    """
    Transform the data so that it performs better as a machine learning training model (for the scikit-learn library).
    :param data: The property sale data (without the label)
    :return: The data which is passed to the ML algorithm.
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('encoder', OneHotEncoder())
    ])

    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attributes),
        ('cat', cat_pipeline, cat_attributes)
    ])

    return full_pipeline.fit_transform(data)


def train_linear_regression_model(prepared_data: scipy.sparse, labels: pd.Series) -> LinearRegression:
    """
    Train the data using a linear regression model.
    :param prepared_data: Data that is prepared to be used by the model.
    :param labels: Labels (the answer to the ML problem) for the data.
    :return: The linear regression model object.
    """
    lin_reg = LinearRegression()
    lin_reg.fit(prepared_data, labels)
    return lin_reg


def train_decision_tree_regressor_model(prepared_data: scipy.sparse, labels: pd.Series) \
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


def train_random_forst_regressor_model(prepared_data: scipy.sparse, labels: pd.Series) \
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


def transform_and_encode_data_statsmodels(data: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the data so that it performs better as a machine learning training model (for the statsmodels library).
    :param data: The property sale data (without the label)
    :return: The data which is passed to the ML algorithm.
    """
    nyc_property_sales_data_no_nulls = data.fillna(0)

    num_columns = nyc_property_sales_data_no_nulls.drop(cat_attributes, axis=1)
    num_columns_transformed = PowerTransformer().fit_transform(num_columns)

    num_columns_transformed = pd.DataFrame(num_columns_transformed)
    num_columns_transformed.columns = num_columns.columns
    num_columns_transformed.index = num_columns.index

    cat_columns = nyc_property_sales_data_no_nulls.drop(num_attributes, axis=1)
    cat_columns = pd.get_dummies(cat_columns, columns=cat_attributes, drop_first=True)

    return pd.concat([num_columns_transformed, cat_columns], axis=1)


def train_ordinary_least_squares_model(prepared_data: pd.DataFrame, labels: pd.Series) \
        -> sm.regression.linear_model.RegressionResults:
    """
    Train the data using an ordinary least squares model.
    :param prepared_data: Data that is prepared to be used by the model.
    :param labels: Labels (the answer to the ML problem) for the data.
    :return: The ordinary least squares model result object.
    """
    constant_data = sm.add_constant(prepared_data)
    ols = sm.OLS(labels, constant_data)
    ols_model = ols.fit()
    return ols_model


if __name__ == '__main__':
    initial_data = load_data()
    train_set, test_set = train_test_split(initial_data, test_size=0.2, random_state=42)
    cleaned_data = clean_data_set(train_set)
    labels, data_set = split_labels_and_data(cleaned_data)

    # Train models using scikit-learn
    final_data = transform_and_encode_data(data_set)
    linear_regression_model = train_linear_regression_model(final_data, labels)
    decision_tree_regressor_model = train_decision_tree_regressor_model(final_data, labels)
    random_forst_regressor_model = train_random_forst_regressor_model(final_data, labels)

    # Train models using statsmodels
    final_data_statsmodels = transform_and_encode_data_statsmodels(data_set)
    ols_model_results = train_ordinary_least_squares_model(final_data_statsmodels, labels)
