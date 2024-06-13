import duckdb


def get_model_lt_data(parquet_file, lead_time, start_date, end_date):
    model_data = duckdb.query(
                        "SELECT * FROM '"
                        + parquet_file
                        + "' WHERE LEAD_TIME = "
                        + str(lead_time)
                        + " AND RUN_DATETIME >= '"
                        + start_date
                        + "' AND RUN_DATETIME <= '"
                        + end_date
                        + "'"
                    ).df()
    
    model_data.dropna(inplace=True)

    if len(model_data) == 0:
        raise ValueError("")

    return model_data


def get_station_var_data(parquet_file, variable, start_date, end_date):

    station_data = duckdb.query(
                    "SELECT * FROM '"
                    + parquet_file
                    + "' WHERE VARIABLE = '"
                    + variable
                    + "' AND DATETIME >= '"
                    + start_date
                    + "' AND DATETIME <= '"
                    + end_date
                    + "'"
                ).df()
    
    station_data.dropna(inplace=True)

    if len(station_data) == 0:
        raise ValueError("")

    return station_data