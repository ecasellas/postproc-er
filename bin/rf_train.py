"""Script per a l'entrenament dels punts d'estacions i l'obtenció d'un
fitxer .parquet amb les regressions i els corresponents paràmetres.
"""
import pickle
import sys
import traceback
from collections import defaultdict
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from postproc.io.parquet import get_model_lt_data, get_station_var_data
from postproc.methods.random_forest import train_rf_model
from postproc.utils.config import load_config

if __name__ == "__main__":

    start_date = datetime(2020, 3, 1)
    end_date = datetime(2023, 3, 1)

    try:
        config = load_config(
            "/home/ecm/projects/uoc/tfm/code/pymos/config_pymos_tfm.json"
        )
    except Exception as err:
        print("Error while loading the configuration file.")
        print(err)
        raise

    lead_times = range(49)
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
    try:
        for lead_time in lead_times:
            regressions = defaultdict(list)
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
                station_data = station_data.dropna()

                for station in station_list:
                    regr = train_rf_model(station, model_data, station_data, var)
                    if regr is not None:
                        regressions[station].append(regr)
                    pbar.update(1)
            print("      Horitzó pronòstic " + str(lead_time).zfill(2) + " - OK")
            pbar.close()

            # Eliminem els None per aquells casos en què no hi ha dades i no es pot
            # entrenar
            # regressions = [regr for regr in regressions if regr is not None]

            # Guardem les regressions en un fitxer .parquet
            try:
                with open(
                    "random_forest_regressions_lt_" + str(lead_time) + ".pickle", "wb"
                ) as f:
                    pickle.dump(regressions, f)
                regressions = None
                del regressions
            except Exception as err:
                print("Error durant la conversió a .parquet del DataFrame.")
                print(err)
                print(traceback.format_exc())
                raise

    except Exception as err:
        print("Error no controlat durant l'entrenament.")
        print(err)
        print(traceback.format_exc())
        raise

    print("[2/2] Entrenament - OK")
