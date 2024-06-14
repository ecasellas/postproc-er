"""Module to calculate multiple linear regressions.
"""
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import linear_model
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression


def get_station_predictors(station_data: DataFrame, model_data: DataFrame,
                           predictand: str, predictors: list) -> dict:
    """Calculates multiple linear regression parameters for a specific
    location using NWP model data.

    Args:
        station_data (pd.DataFrame): Historical observational data for a
                                     specific location.
        model_data (pd.DataFrame): Historical NWP model data.
        predictand (str): Predictand variable of the regression.
        predictors (list): Predictor variables of the regression.

    Returns:
        dict: Multiple linear regression parameters (score, coefficients,
              intercept and predictors used).
    """
    min_predictand_improvement = 0.02

    station_data = station_data.assign(obs=station_data[
        station_data['variable'] == predictand]['value'])

    model_data = model_data.drop_duplicates()
    model_data = model_data.pivot(index=['datetime'], columns=['variable'],
                                  values='value').reset_index()

    if predictors is None:
        predictors = model_data.columns[1:]

    data = model_data.merge(station_data[['datetime', 'obs']], on=['datetime'])

    if len(data) < 850:
        return None

    y_values = np.array(data['obs'])
    x_values = np.array(data[predictors])

    sfs_forward = SequentialFeatureSelector(LinearRegression(),
                                            n_features_to_select="auto",
                                            tol=min_predictand_improvement,
                                            scoring='r2',
                                            direction="forward").fit(x_values,
                                                                     y_values)

    predictors_used = np.array(predictors)[sfs_forward.get_support()]

    x_transformed = sfs_forward.transform(x_values)
    clf = linear_model.LinearRegression()

    clf.fit(x_transformed, y_values)

    return {'score': clf.score(x_transformed, y_values),
            'coefs': clf.coef_.tolist(),
            'intercept': clf.intercept_,
            'predictors': predictors_used}


def train_regressions(stat, model_data, station_data, lead_time, var):
    model_f = model_data[model_data['station_id'] == stat]
    dt_column = model_f['run_datetime'] + \
        pd.to_timedelta(model_f['lead_time'], unit='hours')
    model_f = model_f.assign(datetime=dt_column)

    station_f = station_data[station_data['station_id'] == stat]

    if len(station_f) > 365:
        params = get_station_predictors(station_f, model_f, predictand=var)
        if params is None:
            return None
        params['lead_time'] = lead_time
        params['station_id'] = stat
        params['predictand'] = var

        return params

    return None