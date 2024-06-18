
from datetime import datetime
from multiprocessing.pool import Pool

import pandas as pd
from tqdm import tqdm
from unimodel.io.readers_nwp import read_moloch_grib

from postproc.io.importers import import_nwp_grib
from postproc.utils.config import load_config
from postproc.utils.dates import end_of_month
from postproc.utils.geotools import get_model_points


def __get_model_data(grib_file, var, stepType):

    grib_data = read_moloch_grib(grib_file, var, "cosmo-2I_er", {"stepType": stepType})

    lead_times = grib_data.step.data.astype("timedelta64[h]").astype(int)

    lsm = read_moloch_grib(grib_file, "fr_land", "cosmo-2I_er", {"stepType": "instant"})

    dict_row_col = get_model_points(lsm, stations_md)

    model_run = pd.Timestamp(int(grib_data.time.data)).to_pydatetime()

    _model_data = []

    for station_id, coords in dict_row_col.items():

        for i, lead_time in enumerate(lead_times):
            _model_data.append(
                {
                    "station_id": station_id,
                    "run_datetime": model_run,
                    "lead_time": lead_time,
                    "value": grib_data[i].values[coords[0], coords[1]],
                    "variable": var,
                }
            )

    return _model_data


if __name__ == "__main__":

    config = load_config("/home/ecm/projects/postproc-er/config_grib.json")

    stations_md = pd.read_parquet(config["stations_metadata_pq"])

    variables = [
        "2t",
        "2d",
        "tp",
        "10u",
        "10v",
        "vmax_10m",
        "clct",
        "clch",
        "clcm",
        "clcl",
        "qv_s",
        "hzerocl",
        "alb_rad",
        "sp",
    ]


    start_date = datetime(2024, 2, 1)
    end_date = datetime(2024, 3, 31)

    dates = pd.date_range(start_date, end_date, freq="1D")

    model_data = []

    pbar = tqdm(total=len(dates), desc="Creating model parquet")

    for date in dates:
        try:
            grib_file = import_nwp_grib(date, "cosmo-2I_er", config)
            pbar.update(1)
        except Exception as err:
            if end_of_month(date) or date == dates[-1]:
                pd.DataFrame(model_data).to_parquet(
                    config["model_dir_pq"]
                    + "cosmo_"
                    + date.strftime("%Y%m")
                    + ".parquet"
                )
                model_data = []
                print("File saved.")
            print(err)
            continue

        arguments = []

        for var in variables:

            if var in ["tp", "vmax_10m"]:
                stepType = "accum"
            else:
                stepType = "instant"

            arguments.append((grib_file, var, stepType))
        try:
            with Pool(processes=6) as pool:
                pooled_model_data = pool.starmap(__get_model_data, arguments)
        except Exception as err:
            print(err)
            continue

        model_data = model_data + pooled_model_data

        if end_of_month(date) or date == dates[-1]:
            pd.DataFrame(model_data).to_parquet(
                config["model_dir_pq"]
                + "cosmo_"
                + date.strftime("%Y%m")
                + ".parquet"
            )
            model_data = []
            print("File saved.")

        pbar.update(1)
    pbar.close()
