import tensorflow as tf


def create_dnn_architecture(n_features: int, embedding_dim: int, max_id: int):
    """Creates a Dense Neural Network Architecture.

    Args:
        n_features (int): Number of input features.
        embedding_dim (int): Number of embedding dimensions.
        max_id (int): Maximum number of embedding vocabulary.

    Returns:
        tf.keras.Model: DNN model.
    """    
    features_in = tf.keras.layers.Input(shape=(n_features,))
    norm_layer = tf.keras.layers.Normalization(axis=None)
    features_in = norm_layer(features_in)
    id_in = tf.keras.layers.Input(shape=(1,))
    emb = tf.keras.layers.Embedding(max_id + 1, embedding_dim)(id_in)
    emb = tf.keras.layers.Flatten()(emb)
    x = tf.keras.layers.Concatenate()([features_in, emb])
    x = tf.keras.layers.Dense(256, activation="leaky_relu")(x)
    x = tf.keras.layers.Dense(256, activation="leaky_relu")(x)
    x = tf.keras.layers.Dense(1, activation="linear")(x)

    dnn = tf.keras.Model(inputs=[features_in, id_in], outputs=x)

    return dnn
