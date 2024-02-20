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
# Это сделано для того, чтобы помочь с масштабом математики, участвующей в создании прогноза для каждого изображения.
x_test.astype("float32") / 255.0
x_train.astype("float32") / 255.0


# Построение структуры модели c L2-регулизацией
# L2 регуляризация, также известная как гребневая регрессия (RidgeRegression), является методом регуляризации,
    # который добавляет к функции потерь штраф
    # на квадраты весов, чтобы уменьшить влияние больших значений весов на модель.
# Первым компонентом этого является tf.keras.models.Sequential() вызов.
# Цель этого кода - указать, какие слои будут присутствовать в нашей нейронной сети.
# Все, что делает эта функция,- это начинает создавать линейное (или «последовательное») расположение слоев.
model = keras.Sequential([
    # Используем сверточный слой нейройной сети с процентом штрафа на каждом уровне с размерами 32 * 32
    # Чтобы размерность карты после свертки оставалась той-же, добавляют padding: матрицу
    layers.Conv2D(32, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.001),
                  input_shape=(32, 32, 3)),
    # Добавим батч нормализацию к выходам предыдущего слоя
    layers.BatchNormalization(),
    # Используем сверточный слой нейройной сети с одним процента штрафа на каждом уровне без учёта размера
    layers.Conv2D(32, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.001)),
    # Добавим батч нормализацию к выходам предыдущего слоя
    layers.BatchNormalization(),
    # Уменьшаем масштаб в 2 раза
    layers.MaxPooling2D(pool_size=(2, 2)),
    # С вероятностью 20% будем отключать по 1 нейрону, чтобы избежать переобучения нейронной сети
    layers.Dropout(0.2),
    # Снова Используем сверточный слой нейройной сети с одним процента штрафа на каждом уровне без учёта размера
    layers.Conv2D(64, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.001)),
    # Еще раз Добавим батч нормализацию к выходам предыдущего слоя
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.001)),
    # Еще раз Добавим батч нормализацию к выходам предыдущего слоя
    layers.BatchNormalization(),
    # Еще раз Уменьшаем масштаб в 2 раза
    layers.MaxPooling2D(pool_size=(2, 2)),
    # С вероятностью 30% будем отключать по 1 нейрону, чтобы избежать переобучения нейронной сети
    layers.Dropout(0.2),
    # Все тоже самое со 128 нейронами
    layers.Conv2D(128, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.001)),
    # Еще раз Добавим батч нормализацию к выходам предыдущего слоя
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.001)),
    # Еще раз Добавим батч нормализацию к выходам предыдущего слоя
    layers.BatchNormalization(),
    # Еще раз Уменьшаем масштаб в 2 раза
    layers.MaxPooling2D(pool_size=(2, 2)),
    # С вероятностью 40% будем отключать по 1 нейрону, чтобы избежать переобучения нейронной сети
    layers.Dropout(0.2),

    #Переводим из многомерного массива в одномерный
    layers.Flatten(),
    # Другой вид слоя, который мы видим в модели, создан с использованием tf.keras.layers.Dense()который создает то,
    # что называется полностью связанным или плотно связанным слоем. Это можно сравнить с разреженным слоем,
    # и различие связано с тем, как информация передается между узлами в соседних слоях.
    # Здесь одна распространенная функция активации, и та, которая используется во случае Dense(),
    # называется «softmax».
    layers.Dense(30, activation="softmax"),
    # Softmax берет логиты, вычисленные по взвешенной сумме активаций из предыдущего слоя,
    # и преобразует их в вероятности, которые составляют 1,0. Это делает его чрезвычайно полезной функцией
    # активации для использования в нашем выходном слое, поскольку она обеспечивает легко интерпретируемые результаты
])
# эрлистопинг - это когда один из нейронов не приносит пользы, это тот нейрон, который пытается переобучиться:
# Когда потери будут составлять 2 эпохи, отключим нейрон
early = EarlyStopping(monitor="val_loss", patience=2)

# Теперь, когда мы определили, как выглядит наша нейронная сеть, следующий шаг - рассказать Tensorflow, как ее тренировать.
# Компиляция модели (оптимизация функции потерь (насколько нейронка была менее точной предыдущей модели)
# Укажем конфигурацию обучения (оптимизатор, функция потерь, метрики)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# Определим метрику точности "accuracy"
# optimizer="Adam" - среднеквадратичное распределение
# Это важные особенности того, как нейронная сеть дает свои окончательные прогнозы.

# Наконец-то наступает время обучения модели, и с TensorFlow 2.0 это очень легко сделать.
history = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.1, callbacks=[early])

# Эта строка кода довольно интуитивно понятна, она передает обучающие данные и правильные метки этих данных.
# Параметр эпохи в model.fit()функция - это количество раз, которое модель видит все данных обучения.
# Причина, по которой мы хотим, чтобы модель видела все обучающие данные несколько раз, состоит в том,
# что одного проходного действия может быть недостаточно для того, чтобы модель достаточно обновляла свои веса
# при вычислении взвешенных сумм для заметного улучшения предсказательной силы.


# Мы создали объект взаимодействия истории
# Получение модели (через цикл обучения модели - эпохи чем больше, тем лучше)
# validation_split=0.1 - тонкая настройка обучения, с функцией поиска переобученного нейрона
# Мы передаем валидационные данные для x_train, y_train

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


