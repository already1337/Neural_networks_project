# L1 регуляризация — это метод регуляризации, который добавляет штраф
    # на абсолютное значение весов в функцию потерь, чтобы уменьшить влияние
    # ненужных или малозначимых признаков на модель.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_test.astype("float32") / 255.0
x_train.astype("float32") / 255.0

model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2)
])
early = EarlyStopping(monitor="val_loss", patience=2)
model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

history = model.fit(x_train, y_train, epochs=10, validation_split=0.1, callbacks=[early])

plt.plot(history.history["loss"], label="Потери")
plt.plot(history.history["val_loss"], label="Потери val")
plt.title("Потери в обучении")
plt.xlabel("Эпоха")
plt.ylabel("Потеря")
plt.legend()
plt.show()

plt.plot(history.history["accuracy"], label="Точность")
plt.plot(history.history["val_accuracy"], label="Точность val")
plt.title("Точность тренировки")
plt.xlabel("Эпоха")
plt.ylabel("Точность")
plt.legend()
plt.show()


