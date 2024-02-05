import keras
import tensorflow as tf
from keras import layers


def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = layers.Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature


def encode_categorical_feature(feature, name, dataset, is_string):
    lookup_class = layers.StringLookup if is_string else layers.IntegerLookup
    # Create a lookup layer which will turn strings into integer indices
    lookup = lookup_class(output_mode="binary")

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    lookup.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = lookup(feature)
    return encoded_feature


def make_binary_model(
    col_list,
    continuous_cols,
    train_ds,
    loss_func,
    output_activation,
    learning_rate,
    dropout_rate,
):
    # Categorical
    cat_inputs = [keras.Input(shape=(1,), name=col, dtype="int64") for col in col_list]
    cat_encoded = [
        encode_categorical_feature(inp, name, train_ds, is_string=False)
        for inp, name in zip(cat_inputs, col_list)
    ]

    # Numerical
    num_inputs = [keras.Input(shape=(1,), name=col) for col in continuous_cols]
    num_encoded = [
        encode_numerical_feature(inp, name, train_ds)
        for inp, name in zip(num_inputs, continuous_cols)
    ]
    all_inputs = num_inputs + cat_inputs
    x = layers.Concatenate()(num_encoded + cat_encoded)
    # x = layers.Flatten()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    output = layers.Dense(1, activation=output_activation)(x)
    model = keras.Model(all_inputs, output)
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss=loss_func, metrics=["accuracy"])
    return model
