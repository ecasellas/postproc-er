from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from postproc.methods.neural_networks import create_dnn_architecture
from postproc.utils.config import load_config

config = load_config("/home/ecm/projects/postproc-er/config_grib.json")


# Carreguem els noms dels fitxers amb dades de model
model_files = glob(config["model_dir_pq"] + "*.parquet")

# Llegim les dades de model i en definim la variable datetime
model_data = pd.read_parquet(model_files)
model_data["datetime"] = model_data["run_datetime"] + pd.to_timedelta(
    model_data["lead_time"], unit="hours"
)

# Adaptem les dades tal que les variables són columnes (features)
model_data = model_data.pivot(
    index=["lead_time", "run_datetime", "station_id", "datetime"],
    columns="variable",
    values="value",
)
model_data = model_data.reset_index()
model_data = model_data.dropna()

# Carreguem els noms dels fitxers amb dades d'estacions
obs_files = glob(config["station_dir_pq"] + "*.parquet")

# Llegim les dades d'estacions (target)
obs_data = pd.read_parquet(obs_files)
obs_data = obs_data.dropna()

# Seleccionem les estacions amb un mínim de dades
valid_id = obs_data.groupby("station_id").count().reset_index()
valid_id = valid_id[valid_id["variable"] > 850]["station_id"]

obs_data = obs_data[obs_data["station_id"].isin(valid_id)]

# Seleccionem les columnes i redefinim value -> obs
obs_data = obs_data[["station_id", "datetime", "value"]]
obs_data.columns = ["station_id", "datetime", "obs"]

# Unim els dos conjunts de dades
data = pd.merge(model_data, obs_data, on=["datetime", "station_id"])

# Creem un token per a cada identificador d'estació
station_ids = data["station_id"].unique()
station_ids = pd.DataFrame(
    {"station_id": station_ids, "station_token_id": np.arange(0, len(station_ids), 1)}
)

# Ho unim amb la resta de dades
data = pd.merge(data, station_ids, on="station_id")

# Definim el conjunt d'entrenament i el de validació
train_data = data[data["datetime"] < datetime(2023, 3, 1)]
val_data = data[data["datetime"] >= datetime(2023, 3, 1)]

# Convertim a array les dades d'entrenament i validació (features i target)
features_train = np.array(
    train_data[
        [
            "lead_time",
            "10u",
            "10v",
            "2d",
            "2t",
            "alb_rad",
            "clch",
            "clcl",
            "clcm",
            "clct",
            "hzerocl",
            "qv_s",
            "sp",
            "tp",
            "vmax_10m",
        ]
    ]
)
embedding_train = np.array(train_data["station_token_id"])
target_train = np.array(train_data["obs"])

features_val = np.array(
    val_data[
        [
            "lead_time",
            "10u",
            "10v",
            "2d",
            "2t",
            "alb_rad",
            "clch",
            "clcl",
            "clcm",
            "clct",
            "hzerocl",
            "qv_s",
            "sp",
            "tp",
            "vmax_10m",
        ]
    ]
)
embedding_val = np.array(val_data["station_token_id"])
target_val = np.array(val_data["obs"])

# Creem el model DNN
dnn_model = create_dnn_architecture(n_features=15, embedding_dim=6, max_id=124)

# Definim l'optimitzador
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0007)

# Compliem el model
dnn_model.compile(optimizer=optimizer, loss="mae", metrics=["mae"])


time_1 = datetime.utcnow()
dnn_model.fit(
    [np.array(features_train), np.array(embedding_train)],
    y=np.array(target_train),
    validation_data=(
        [np.array(features_val), np.array(embedding_val)],
        np.array(target_val),
    ),
    epochs=500,
    batch_size=512,
    verbose=1,
    callbacks=[
        EarlyStopping(patience=20, restore_best_weights=True),
        ReduceLROnPlateau(patience=3, factor=0.8, min_lr=1e-7),
    ],
)
time_2 = datetime.utcnow()

print((time_2 - time_1).total_seconds() / 60, "minuts.")


results = dnn_model.predict([np.array(features_val), np.array(embedding_val)])
