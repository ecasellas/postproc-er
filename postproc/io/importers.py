"""Module to import NWP grib files.
"""
import os
import zipfile
from datetime import datetime
from glob import glob
from os import makedirs, remove
from posixpath import basename
from shutil import copyfile

from genericpath import exists


def __get_datetime_formatted__(date: datetime) -> dict:
    year = date.strftime("%Y")
    month = date.strftime("%m")
    day = date.strftime("%d")
    hour = date.strftime("%H")
    minute = date.strftime("%M")
    second = date.strftime("%S")

    return {
        "year": year,
        "month": month,
        "day": day,
        "hour": hour,
        "minute": minute,
        "second": second,
    }


def import_nwp_grib(
    date_run: datetime, model: str, config: dict, lead_time: int = None
) -> str:
    """Imports a NWP model grib file.

    Args:
        date_run (datetime): Date and time of the NWP model run.
        model (str): Alias of the NWP model selected.
        config (dict): Configuration dictionary with NWP model parameters
                       including {'nwp_dir': Intermediate folder to deal with
                       grib files, 'lead_times': Lead times to get for each NWP
                       run, 'alias_model': {'src': Source path including file
                       name, 'projection': regular_ll or rotated_ll,
                       'compacted': True (all lead times in one file),
                       'compressed': src is a compressed file}}.
        lead_time (int, optional): Lead time of the NWP grib to import if
                                   'compacted' is False. Defaults to None.

    Raises:
        KeyError: If 'model' not in the configuration dictionary.
        KeyError: If 'compressed' set to True and 'src_tar' not in the
                  configuration dictionary.
        FileNotFoundError: If 'compressed' set to True and .tar file not found.
        ValueError: If 'compacted' set to False and lead_time not provided.
        FileNotFoundError: If NWP grib file not found.

    Returns:
        str: Path of the imported NWP grib file.
    """
    if model not in config.keys():
        raise KeyError(model + " not in configuration dictionary.")

    model_dir = config["nwp_dir"] + model + "/"
    if not exists(model_dir):
        makedirs(model_dir)

    prev_files_tar = glob(model_dir + "*.zip")
    prev_files = glob(model_dir + "*[!.zip]")

    date_run_f = __get_datetime_formatted__(date_run)

    if config[model]["compressed"]:
        if "src_tar" not in config[model].keys():
            raise KeyError("src_tar must be included if compressed is set to " "True.")
        tar_file = config[model]["src_tar"].format(
            year=date_run_f["year"],
            month=date_run_f["month"],
            day=date_run_f["day"],
            run=date_run_f["hour"],
        )

        if not exists(model_dir + basename(tar_file)):
            for prev_file in prev_files_tar:
                remove(prev_file)
            for prev_file in prev_files:
                remove(prev_file)
            if exists(tar_file):
                copyfile(tar_file, model_dir + basename(tar_file))
            else:
                raise FileNotFoundError(tar_file + " not found.")

    if config[model]["compacted"]:
        nwp_file = config[model]["src"].format(
            year=date_run_f["year"],
            month=date_run_f["month"],
            day=date_run_f["day"],
            hour=date_run_f["hour"],
        )
    else:
        if lead_time is None:
            raise ValueError("If compacted is False, lt must be supplied.")
        nwp_file = config[model]["src"].format(
            year=date_run_f["year"],
            month=date_run_f["month"],
            day=date_run_f["day"],
            run=date_run_f["hour"],
            lt=str(lead_time).zfill(2),
        )

    # If NWP grib file already exists in stage directory, this part is skipped
    if not exists(model_dir + basename(nwp_file)):
        # If NWP grib file is from a compressed source, it is extracted
        if config[model]["compressed"]:
            with zipfile.ZipFile(model_dir + basename(tar_file), "r") as _zip:
                for member in _zip.infolist():
                    with _zip.open(member) as source, open(
                        os.path.join(model_dir, member.filename), "wb"
                    ) as target:
                        target.write(source.read())

        else:
            if not exists(nwp_file):
                raise FileNotFoundError(nwp_file + " not found.")
            for prev_file in prev_files:
                remove(prev_file)
            copyfile(nwp_file, model_dir + basename(nwp_file))

        # Otherwise, if exists, it is directly copied to stage directory
        if not exists(model_dir + basename(nwp_file)):
            raise FileNotFoundError(nwp_file + " not found.")

    return model_dir + basename(nwp_file)
