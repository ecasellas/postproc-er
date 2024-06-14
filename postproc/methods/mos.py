"""Module to calculate multiple linear regressions.
"""

from os.path import exists

import duckdb
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import linear_model
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression


def get_station_predictors(
    station_data: DataFrame, model_data: DataFrame, predictand: str, predictors: list
) -> dict:
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

    station_data = station_data.assign(
        obs=station_data[station_data["variable"] == predictand]["value"]
    )

    model_data = model_data.drop_duplicates()
    model_data = model_data.pivot(
        index=["datetime"], columns=["variable"], values="value"
    ).reset_index()

    if predictors is None:
        predictors = model_data.columns[1:]

    data = model_data.merge(station_data[["datetime", "obs"]], on=["datetime"])

    if len(data) < 850:
        return None

    y_values = np.array(data["obs"])
    x_values = np.array(data[predictors])

    sfs_forward = SequentialFeatureSelector(
        LinearRegression(),
        n_features_to_select="auto",
        tol=min_predictand_improvement,
        scoring="r2",
        direction="forward",
    ).fit(x_values, y_values)

    predictors_used = np.array(predictors)[sfs_forward.get_support()]

    x_transformed = sfs_forward.transform(x_values)
    clf = linear_model.LinearRegression()

    clf.fit(x_transformed, y_values)

    return {
        "score": clf.score(x_transformed, y_values),
        "coefs": clf.coef_.tolist(),
        "intercept": clf.intercept_,
        "predictors": predictors_used,
    }


def train_regressions(stat, model_data, station_data, lead_time, var):
    model_f = model_data[model_data["station_id"] == stat]
    dt_column = model_f["run_datetime"] + pd.to_timedelta(
        model_f["lead_time"], unit="hours"
    )
    model_f = model_f.assign(datetime=dt_column)

    station_f = station_data[station_data["station_id"] == stat]

    if len(station_f) > 365:
        params = get_station_predictors(station_f, model_f, predictand=var)
        if params is None:
            return None
        params["lead_time"] = lead_time
        params["station_id"] = stat
        params["predictand"] = var

        return params

    return None


class Forecaster:
    """Class to obtain MOS forecasts from multiple linear regressions."""

    def __init__(self, regression_parquet: str):
        """Inits Forecaster class with a regression parquet file.

        Args:
            regression_parquet (str): Path to a regression parquet file.

        Raises:
            FileNotFoundError: If 'regression_parquet' is not found.
            KeyError: If 'regression_parquet' does not contain at least
                      required columns.
        """
        if not exists(regression_parquet):
            raise FileNotFoundError(regression_parquet + " not found.")

        self.df_regression = duckdb.query(
            "SELECT * FROM '" + regression_parquet + "'"
        ).df()

        if not (
            set(self.df_regression.columns)
            >= {
                "score",
                "coefs",
                "intercept",
                "predictors",
                "lead_time",
                "station_id",
                "predictand",
            }
        ):
            raise KeyError(
                regression_parquet + " must contain at least the "
                "following columns: {'score', 'coefs', "
                "'intercept', 'predictors', 'lead_time', "
                "'station_id', 'predictand'}"
            )

    def forecast_point(
        self, station_id: str, model_data: DataFrame, predictand: str
    ) -> float:
        """Calculates the forecast for a specific point and lead time using
        selected multiple linear regression and NWP model data.

        Args:
            station_id (str): Point or station identification code.
            model_data (pd.DataFrame): DataFrame of NWP model data for a
                                       specific lead time.
            predictand (str): Variable to use as predictand.

        Raises:
            ValueError: If 'station_id' not in regressions parquet file.
            ValueError: If 'predictand' not in regressions parquet file.
            ValueError: If 'station_id' not in 'model_data' DataFrame.
            ValueError: If 'model_data' includes more than one lead time.

        Returns:
            float: Forecast for station_id.
        """
        if station_id not in list(self.df_regression["station_id"]):
            raise ValueError(
                "Point " + station_id + " not found in " "regressions parquet file."
            )
        if station_id not in list(model_data["station_id"]):
            raise ValueError(
                "Point " + station_id + " not found in model_data" " DataFrame."
            )
        if len(set(model_data["lead_time"])) != 1:
            raise ValueError(
                "Too many lead times in model_data. model_data "
                "must contain only data corresponding to one lead"
                " time."
            )
        if predictand not in list(set(self.df_regression["predictand"])):
            raise ValueError(
                predictand + " not found in regression file. "
                "Predictands available: "
                + str(list(set(self.df_regression["predictand"])))
            )

        lead_time = int(model_data["lead_time"].iloc[0])
        point_data = model_data.loc[model_data["station_id"] == station_id]

        point_regression = self.df_regression[
            (self.df_regression["station_id"] == station_id)
            & (self.df_regression["lead_time"] == lead_time)
            & (self.df_regression["predictand"] == predictand)
        ].drop_duplicates(["lead_time", "station_id", "predictand"])

        point_regression = point_regression.reset_index()

        predictor_values = []
        coefficients = []
        for i, var in enumerate(point_regression["predictors"][0]):
            predictor_values.append(
                float(point_data.loc[point_data["variable"] == var, "value"].iloc[0])
            )
            coefficients.append(point_regression["coefs"][0][i])

        forecast = (
            sum(np.array(predictor_values) * np.array(coefficients))
            + point_regression["intercept"]
        )

        return float(forecast.iloc[0])

    def forecast_points(
        self, stations_id: list, model_data: DataFrame, predictand: str
    ) -> list:
        """Calculates the forecast for multiple points and a specific lead time
        using selected multiple linear regression and NWP model data. If a
        station_id not found in regressions parquet file, np.nan is assigend to
        forecast for that point.

        Args:
            stations_id (list): List of point or station identification codes.
            model_data (pd.DataFrame): DataFrame of NWP model data for a
                                       specific lead time.
            predictand (str): Variable to use as predictand.

        Raises:
            ValueError: If 'predictand' not in regressions parquet file.
            ValueError: If 'station_id' not in 'model_data' DataFrame.
            ValueError: If 'model_data' includes more than one lead time.

        Returns:
            list: Forecast for stations_id.
        """
        lead_time = int(model_data["lead_time"].iloc[0])
        forecast = []
        for station_id in stations_id:
            try:
                fct_point = self.forecast_point(station_id, model_data, predictand)
                forecast.append(
                    {
                        "run_datetime": model_data["run_datetime"].iloc[0],
                        "station_id": station_id,
                        "lead_time": lead_time,
                        "forecast": fct_point,
                    }
                )
            except ValueError as err:
                if "Point " + station_id + " not found in regressions" in str(err):
                    forecast.append(
                        {
                            "run_datetime": model_data["run_datetime"].iloc[0],
                            "station_id": station_id,
                            "lead_time": lead_time,
                            "forecast": np.nan,
                        }
                    )
                else:
                    raise

        return forecast
