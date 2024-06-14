#!/usr/bin/env python
"""Script principal per a l'execució del MOS.
"""
import traceback
from datetime import datetime


import pandas as pd


from postproc.methods.mos import Forecaster

from postproc.io.parquet import get_model_run

from postproc.utils.config import load_config


def forecast_hourly(
    model_data: pd.DataFrame,
    stations_id: list,
    predictand: str,
    regression_file: str,
    config: dict,
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
    regression = Forecaster(regression_file)

    results = []
    for lead_time in range(config["lead_times"]):
        model_lt = model_data.loc[model_data["lead_time"] == lead_time]
        fct_hourly = regression.forecast_points(stations_id, model_lt, predictand)
        results = results + fct_hourly

    return results


def main():
    """Main function of the script."""
    config = load_config("config_pymos_tfm.json")

    stations_md = pd.read_parquet(config["station_metadata_pq"])
    stations_id = list(stations_md["station_id"])

    start_date = datetime(2023, 3, 1)
    end_date = datetime(2024, 3, 31)

    dates = pd.date_range(start_date, end_date, freq="1D")

    resultat_2t = []

    for date in dates:
        print(
            "Inici del càlcul del mos pel %s", date.strftime("%Y-%m-%d %H") + "."
        )
        time_0 = datetime.utcnow()

        try:
            regression_file = (config["regressions_pq"])
        except Exception as err:
            print("Error recuperant el fitxer de les regressions.")
            print(err)
            print(traceback.format_exc())
            raise

        try:
            model_data = get_model_run(config["model_dir_pq"], date)
            if len(model_data) == 0:
                print("No hi ha dades de model per a aquest dia.")
                continue
        except FileNotFoundError as err:
            print(
                "El fitxer de model no està disponible o encara no existeix."
            )
            raise
        except Exception as err:
            print("Error durant la importació del model.")
            print(err)
            print(traceback.format_exc())
            raise
        print("Importació del model - OK")

        try:
            result_2t = forecast_hourly(
                model_data, stations_id, "2t", regression_file, config
            )

            resultat_2t = resultat_2t + result_2t
        except Exception as err:
            print("Error durant el pronòstic horari.")
            print(err)
            print(traceback.format_exc())
            raise
        print("Pronòstic horari 2t - OK")

        elapsed_time = (datetime.utcnow() - time_0).total_seconds() / 60
        print("Temps d'execució: %2.2f minuts.", round(elapsed_time, 1))

    pd.DataFrame(resultat_2t).to_parquet("forecast_mos_2t_2d.parquet")
    print("Pronòstic MOS - OK")


if __name__ == "__main__":

    main()
