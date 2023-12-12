import os
import tensorflow as tf
from PIL import Image
import pandas as pd


def model_builder(input_shape, num_classes, kernel_size=(3, 3), pool_size=(2, 2)):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Conv2D(32, kernel_size, padding="same", activation="relu", input_shape=input_shape),
            tf.keras.layers.MaxPooling2D(pool_size, strides=2),
            tf.keras.layers.Conv2D(64, kernel_size, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size, strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def dataset_iter(dataset_dir):
    """Iterate over the dataset directory and yield the image paths and labels."""
    for label in os.listdir(dataset_dir):
        for image_name in os.listdir(os.path.join(dataset_dir, label)):
            yield Image.open(os.path.join(dataset_dir, label, image_name)), label


def mk_dataset_df(dataset_dir):
    """Create a pandas dataframe from the dataset directory."""
    df = pd.DataFrame(dataset_iter(dataset_dir), columns=["image", "label"])
    replace_dict = {label: i for i, label in enumerate(df["label"].unique())}
    df["label"] = df["label"].replace(replace_dict)
    return df