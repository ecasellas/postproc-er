"""Script per a l'entrenament dels punts d'estacions i l'obtenció d'un
fitxer .parquet amb les regressions i els corresponents paràmetres.
"""
import sys
import traceback
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from postproc.methods.mos import train_regressions
from postproc.utils.config import load_config
from postproc.io.parquet import get_model_lt_data, get_station_var_data


if __name__ == "__main__":

    start_date = datetime(2020, 2, 1)
    end_date = datetime(2023, 2, 1)

    try:
        config = load_config("/home/ecm/projects/postproc-er/config_grib.json")
    except Exception as err:
        print("Error while loading the configuration file.")
        print(err)
        raise

    lead_times = config["lead_times"]
    vars_to_train = ["2t"]
    model_parquet = config["model_dir_pq"] + "*.parquet"
    station_parquet = config["station_dir_pq"] + "*.parquet"

    metadata = pd.read_parquet(config["station_metadata_pq"])
    station_list = list(metadata["station_id"])

    if len(station_list) < 20:
        print("Nombre insuficient d'estacions.")
        sys.exit()

    run_datetime_0 = start_date.strftime("%Y-%m-%d %H:%M:%S")
    run_datetime_1 = end_date.strftime("%Y-%m-%d %H:%M:%S")

    print("[1/2] Entrenament - Inici")
    regressions = []
    try:
        for lead_time in lead_times:
            pbar = tqdm(
                total=len(station_list) * len(vars_to_train),
                desc="Entrenament - lt " + str(lead_time),
            )
            for var in vars_to_train:
                model_data = get_model_lt_data(
                    model_parquet, lead_time, run_datetime_0, run_datetime_1
                )

                station_data = get_station_var_data(
                    station_parquet, var, run_datetime_0, run_datetime_1
                )

                for station in station_list:
                    regressions.append(
                        train_regressions(
                            station, model_data, station_data, lead_time, var
                        )
                    )
                    pbar.update(1)
            print("      Horitzó pronòstic " + str(lead_time).zfill(2) + " - OK")
            pbar.close()
    except Exception as err:
        print("Error no controlat durant l'entrenament.")
        print(err)
        print(traceback.format_exc())
        raise

    # Eliminem els None per aquells casos en què no hi ha dades i no es pot
    # entrenar
    regressions = [regr for regr in regressions if regr is not None]

    # Guardem les regressions en un fitxer .parquet
    try:
        pd.DataFrame(regressions).to_parquet(config["regressions_pq"])
    except Exception as err:
        print("Error durant la conversió a .parquet del DataFrame.")
        print(err)
        print(traceback.format_exc())
        raise

    print("[2/2] Entrenament - OK")
