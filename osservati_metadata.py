import duckdb

from postproc.utils.config import load_config


if __name__ == "__main__":

    config = load_config("/home/ecm/projects/postproc-er/config_grib.json")

    station_data = duckdb.query(
        "SELECT DISTINCT id, lon, lat FROM '/home/ecm/projects/uoc/tfm/data/osservati/*.parquet'"
    ).df()

    station_data.columns = ["station_id", "lon", "lat"]

    station_data.to_parquet("/home/ecm/projects/uoc/tfm/data/osservati_metadata.parquet")
