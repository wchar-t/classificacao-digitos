"""Parte 1: treina CNN no MNIST.

Arquitetura:
  Conv(32,3) -> BN -> ReLU -> Conv(32,3) -> BN -> ReLU -> MaxPool(2) -> Dropout(0.25)
  Conv(64,3) -> BN -> ReLU -> Conv(64,3) -> BN -> ReLU -> MaxPool(2) -> Dropout(0.25)
  Flatten -> Dense(128) -> BN -> ReLU -> Dropout(0.5) -> Dense(10, softmax)

Justificativa no relatorio.md.
"""
import json
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

(x_tr, y_tr), (x_te, y_te) = keras.datasets.mnist.load_data()
x_tr = (x_tr / 255.0).astype("float32")[..., None]
x_te = (x_te / 255.0).astype("float32")[..., None]

modelo = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, 3, padding="same"),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Conv2D(32, 3, padding="same"),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(2),
    layers.Dropout(0.25),

    layers.Conv2D(64, 3, padding="same"),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Conv2D(64, 3, padding="same"),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(2),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(128),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),
])

modelo.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

modelo.summary()

cb = keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=5, restore_best_weights=True
)

history = modelo.fit(
    x_tr, y_tr,
    validation_split=0.1,
    epochs=30,
    batch_size=128,
    callbacks=[cb],
    verbose=2,
)

modelo.save("modelo.keras")

with open("history.json", "w") as f:
    json.dump(history.history, f)

print("\nÉpocas rodadas:", len(history.history["loss"]))
print("Melhor val_accuracy:", max(history.history["val_accuracy"]))
