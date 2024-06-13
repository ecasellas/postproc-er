import numpy as np
import pyproj
import xarray
from sklearn.neighbors import NearestNeighbor


def get_model_points(lsm: xarray.DataArray, stations_md) -> dict:
    """Determine model point position (row, col) for each station.
    Only model grid point where landsea mask equals 1 are considered.

    Args:
        lsm (xarray): Land-sea mask xarray.
        stations_md (pd.Dataframe): Stations metadatada.
    Returns:
        dict: Model point position for each station follwoing
        {'station_id': (row, col)}.
    """
    dictionary_row_col = {}

    for stat in stations_md.itertuples():
        inproj = pyproj.CRS("EPSG:4326")
        outproj = pyproj.CRS(lsm.rio.crs)

        proj = pyproj.Transformer.from_crs(inproj, outproj)

        x1 = lsm.x.values
        y1 = lsm.y.values

        x2, y2 = proj.transform(stat.lat, stat.lon)

        d_x = np.abs(x1 - np.array([x2]))
        col = np.argmin(d_x)

        d_y = np.abs(y1 - np.array([y2]))
        row = np.argmin(d_y)

        if lsm[row, col] < 1:
            neigh_candidates = np.where(lsm == 1)
            neigh_candidates = np.vstack((neigh_candidates[0], neigh_candidates[1])).T
            neigh_needed = np.vstack((row, col)).T

            nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(
                neigh_candidates
            )

            _, indices = nbrs.kneighbors(neigh_needed)

            row, col = (
                neigh_candidates[indices].flatten()[0],
                neigh_candidates[indices].flatten()[1],
            )

        dictionary_row_col[stat.station_id] = (row, col)

    return dictionary_row_col
