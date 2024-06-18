from pandas import DataFrame
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle
import pandas as pd


def get_station_rf_model(
    station_data: DataFrame, model_data: DataFrame, predictand: str, predictors: list
) -> dict:
    """Calculates Random Forest model for a specific location using NWP model data.

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

    rf_regr = RandomForestRegressor().fit(x_values, y_values)

    return rf_regr


def forecast_hourly(
    model_data: pd.DataFrame, stations_id: list, predictand: str, config: dict
) -> pd.DataFrame:
    """Obtains hourly forecasts of a specified predictand for each station in
    stations_id.

    Args:
        model_data (DataFrame): Data from a NWP model for specific points.
        stations_id (list): Station id points to obtain a forecast.
        predictand (str): Variable to forecast.
        config (dict): Configuration dictionary.

    Returns:
        DataFrame: Hourly forecast for a specific variable and for each
                   station.
    """
    results = []
    for lead_time in range(config["lead_times"]):
        regression_file = config["random_forest"]["rf_model"].format(
            lead_time=lead_time
        )
        with open(regression_file, "rb") as f:
            regressions = pickle.load(f)
        model_lt = model_data.loc[model_data["lead_time"] == lead_time]
        for station_id in stations_id:
            if len(regressions[station_id]) == 0:
                continue
            point_data = model_lt.loc[model_lt["station_id"] == station_id]
            point_data = point_data.pivot(
                index=["lead_time", "run_datetime", "station_id"],
                columns="variable",
                values="value",
            ).iloc[0]
            point_data = np.array(point_data).reshape(1, -1)

            fct_point = regressions[station_id][0].predict(point_data)

            forecast = {
                "run_datetime": model_data["run_datetime"].iloc[0],
                "station_id": station_id,
                "lead_time": lead_time,
                "forecast": fct_point[0],
            }

            results.append(forecast)

    return results


def train_rf_model(
    station_id: str, model_data: pd.DataFrame, station_data: pd.DataFrame, var: str
) -> RandomForestRegressor:
    """Trains a Random Forest model using the provided data for a specific station.

    Args:
        station_id (str): Station identification code.
        model_data (pd.DataFrame): The DataFrame containing the model input features for training.
        station_data (pd.DataFrame): The DataFrame containing the station input target for training.
        var (str): Name of the variable to predict.

    Returns:
        RandomForestRegressor: The trained Random Forest model.
    """    
    model_f = model_data[model_data["station_id"] == station_id]
    dt_column = model_f["run_datetime"] + pd.to_timedelta(
        model_f["lead_time"], unit="hours"
    )
    model_f = model_f.assign(datetime=dt_column)

    station_f = station_data[station_data["station_id"] == station_id]

    if len(station_f) > 365:
        rf_model = get_station_rf_model(
            station_f, model_f, predictand=var, predictors=None
        )
        if rf_model is None:
            return None

        return rf_model

    return None
