"""Parte 3: exportar para TensorFlow.js."""
import tensorflowjs as tfjs
from tensorflow import keras

modelo = keras.models.load_model("modelo.keras")
tfjs.converters.save_keras_model(modelo, "modelo_web")
print("OK")
