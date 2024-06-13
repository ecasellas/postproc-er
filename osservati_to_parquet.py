import gzip
import json
from datetime import datetime
from glob import glob
from os.path import basename

import pandas as pd
from postproc.utils.config import load_config


if __name__ == "__main__":

    config = load_config("/home/ecm/projects/postproc-er/config_grib.json")

    osservati_files = glob("/home/ecm/projects/uoc/tfm/data/osservati/*.json.gz")

    for osservati_file in osservati_files:

        osservati_data = []

        with gzip.open(osservati_file, "r") as f:
            for line in f:
                line_dict = json.loads(line)

                line_data = {}
                for element in line_dict["data"]:
                    if element["vars"].keys() == {"B12101"}:
                        if (
                            element["timerange"] == [0, 0, 3600]
                            and element["level"][0:2] == [103, 2000]
                            or element["timerange"] == [254, 0, 0]
                        ):
                            line_data["variable"] = "2t"
                            line_data["value"] = element["vars"]["B12101"]["v"]

                            line_data["id"] = line_dict["data"][0]["vars"]["B01019"][
                                "v"
                            ]
                            line_data["datetime"] = datetime.strptime(
                                line_dict["date"], "%Y-%m-%dT%H:%M:%SZ"
                            )
                            line_data["lon"] = round(line_dict["lon"] * 1e-5, 5)
                            line_data["lat"] = round(line_dict["lat"] * 1e-5, 5)

                            osservati_data.append(line_data)

                            break

        osservati_data = pd.DataFrame(osservati_data)

        osservati_data.to_parquet(
            "/home/ecm/projects/uoc/tfm/data/osservati/"
            + basename(osservati_file)[:7]
            + ".parquet"
        )

        print(basename(osservati_file)[:7] + ".parquet")
