# L2  - регуляризация, также известная как гребневая регрессия (RidgeRegression),
# является методом регуляризации, который добавляет к функции потерь штраф
# на квадраты весов, чтобы уменьшить влияние больших значений весов на модель.

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.python.keras import regularizers
from tensorflow.keras.datasets import cifar10

# Загружаем модель 70 000 изображений 60 000 тренировок и 10 000 тестов
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Нормализация данных (упрощение и сведение к одному диапазону от 0 до 1 и от -1 до 1 сводим к диапазону от 0 до 1)
x_test.astype("float32") / 255.0
x_train.astype("float32") / 255.0

# Построение структуры модели c L2-регулизацией
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.001),
                  input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.2),
    layers.Conv2D(64, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.2),
    layers.Conv2D(128, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(10, activation="softmax"),
])
# эрлистопинг - это когда один из нейронов не приносит пользы, это тот нейрон, который пытается переобучиться:
# Когда потери будут составлять 2 эпохи, отключим нейрон
early = EarlyStopping(monitor="val_loss", patience=2)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.1, callbacks=[early])


plt.plot(history.history["loss"], label="Потери")
plt.plot(history.history["val_loss"], label="Потери Val")
plt.title("Потери в обучении")
# Подпишем оси
plt.xlabel("Эпоха")
plt.ylabel("Потеря")
plt.legend()
plt.show()

plt.plot(history.history["accuracy"], label="Точность")
plt.plot(history.history["val_accuracy"], label="Точность Val")
plt.title("Точность тренировки")
# Подпишем оси
plt.xlabel("Эпоха")
plt.ylabel("Точность")
plt.legend()
plt.show()


# Tensorflow на процессоре, скомпилировав его из исходного кода с соответствующими флагами,
# чтобы учесть наборы инструкций AVX AVX2 вашего процессора. При этом вы сможете обучать простые модели
# только по соображениям времени вычислений.


