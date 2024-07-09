relative_path = '../datasets/bike-sharing-dataset_nan_without_deviations.csv'
import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import json
import pickle


def get_metrics(y_test, y_pred):
    '''
    Вычисление и вывод метрик: MAE, RMSE, R2.
    На основе сравнения проверочных и вычисленных.
    :param y_test: - проверочные значения целевой переменный
    :param y_pred: - вычисленные значения целевой переменный
    '''
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y_test, y_pred)

    print("MAE : {:>9,.3f} (средняя абсолютная ошибка)".format(mae))
    print("MSE : {:>9,.6f} (среднеквадратичная ошибка)".format(mse))
    print("RMSE: {:>9,.6f} (кв. корень из среднеквадратичной ошибки)".format(rmse))
    print("R2  : {:>9,.3f} (коэфф. детерминации)".format(r2))
    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}


# --------------------------------------------------------------------------


df = pd.read_csv(relative_path,  # путь к файлу, (используй автодотолнение)
                 sep=',',  # разделитель данных в файле
                 header=0,  # номер строки с заголовками, нумерация с нуля
                 # header='None', # если заголовки отсутствуют
                 )

features = ['hum', 'temp', 'hr', 'weekday', 'windspeed']
target = ['cnt']

dfX = df[features]
dfY = df[target]

scalerNormForX = MinMaxScaler()
scalerNormForY = MinMaxScaler()

scalerNormForX.fit(dfX)
scalerNormForY.fit(dfY)

dfXNorm = pd.DataFrame(
    data=scalerNormForX.transform(dfX),  # значения ячеек    <<--
    columns=dfX.columns,  # названия столбцов
    index=dfX.index  # идентификаторы строк
)
dfYNorm = pd.DataFrame(
    data=scalerNormForY.transform(dfY),  # значения ячеек    <<--
    columns=dfY.columns,  # названия столбцов
    index=dfY.index  # идентификаторы строк
)

# формирование выборок
valid_size = 0.3  # доля тестовой части в выборке
rand_seed = 8  # начальное состояние генератора случ. чисел

xTrain, xTest, yTrain, yTest = train_test_split(
    df[features],  # исходные данные X
    df[target],  # исходные данные y

    test_size=valid_size,  # доля тестовой части в выборке
    random_state=rand_seed,  # начальное состояние генератора случ. чисел
    shuffle=True  # перемешивание
)

xNormTrain, xNormTest, yNormTrain, yNormTest = train_test_split(
    dfXNorm[features],  # исходные данные X
    dfYNorm[target],  # исходные данные y

    test_size=valid_size,  # доля тестовой части в выборке
    random_state=rand_seed,  # начальное состояние генератора случ. чисел
    shuffle=True  # перемешивание
)

# ----------m1----------
# cols_m1=['windspeed','hum','temp']
cols_m1 = features
model_lin_empty = linear_model.LinearRegression()
# Вычислить коэфф. Ki в функции y(x)=Ki*xi + .... + B
# на тренировочном наборе, т.е. обучить модель
m1 = model_lin_empty.fit(
    xTrain[cols_m1],  # или тут от всех признаков нужно?
    yTrain[['cnt']]
)

# Получить вычисленные(predicted) зн. на проверочном наборе
yPred = m1.predict(xTest[cols_m1])

m1_metrics = get_metrics(yTest, yPred)

# --------------m2------------------------
epochForTrain = 100
# cols_m2=['hum', 'temp',  'hr',]
cols_m2 = features
with tf.device('/CPU:0'):
    totalHistoryLossTrain = []  # Вспомогательный список для хранение полной истории обучения
    totalHistoryLossTest = []  # Вспомогательный список для хранение полной истории обучения

    ###########################################
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    # 1) Параметры структуры
    input_size = len(cols_m2)  # кол-во входных  узлов
    output_size = 1  # кол-во выходных узлов
    hiddenLayer_size = 10  # кол-во узлов на скрытом слое

    # 2.1) Построение модели нейронной сети
    # Многослойная
    m2 = tf.keras.models.Sequential()

    m2.add(tf.keras.layers.Input(shape=(input_size,)))  # Входной слой
    # параметр "units" - кол-во узлов/нейронов на данном слое
    # параметр "activation" - вид функции активации
    # ...

    m2.add(tf.keras.layers.Dense(units=hiddenLayer_size,
                                 activation=tf.keras.activations.sigmoid))  # Выходной слой, с линейной функцией активации
    m2.add(tf.keras.layers.Dense(units=output_size,
                                 activation=tf.keras.activations.linear))  # Выходной слой, с линейной функцией активации

    # 2.2) Ф. потерь и оптимизации
    fLoss = tf.keras.losses.mean_squared_error
    # fOptimizer=tf.keras.optimizers.SGD(learning_rate=0.01)
    fOptimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    fMetric = [tf.keras.losses.mean_squared_error]

    m2.compile(
        loss=fLoss,
        optimizer=fOptimizer,
        metrics=[fMetric]
    )

    print(m2.summary())
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    ###########################################

with tf.device('/CPU:0'):
    # 3) Цикл обучения/продолжения обучения сети
    # xNorm_train, xNorm_test, yNorm_train, yNorm_test
    history = m2.fit(

        xNormTrain[cols_m2],
        # обучающие X  # требуется передать таблицу с кол-ом столбцов равным кол-ву входных нейронов
        yNormTrain,  # обучающие Y

        validation_data=(  # опционально: проверочные X и Y
            xNormTest[cols_m2],  # требуется передать таблицу с кол-ом столбцов равным кол-ву входных нейронов
            yNormTest),

        epochs=epochForTrain,  # кол-во эпох обучения
        batch_size=100,  # кол-во образцов в каждой эпохе
        verbose=1,
    )

    totalHistoryLossTrain.extend(history.history['loss'])
    if 'val_loss' in history.history.keys():
        totalHistoryLossTest.extend(history.history['val_loss'])
    totalHistoryLossTrain.extend(history.history['loss'])
    if 'val_loss' in history.history.keys():
        totalHistoryLossTest.extend(history.history['val_loss'])

# Опрос модели
with tf.device('/CPU:0'):
    # Calculate predictions
    yNormPred = m2.predict(
        xNormTest[cols_m2])  # требуется передать таблицу с кол-ом столбцов равным кол-ву входных нейронов

m2_metrics = get_metrics(yNormTest, yNormPred)

# ------Save8-------
# a
m1_dict = dict()
m1_dict['id'] = 1
m1_dict['model_type'] = str(type(m1))
m1_dict['features'] = cols_m1
m1_dict['r2'] = m1_metrics['r2']
m1_dict['rmse'] = m1_metrics['rmse']

m2_dict = dict()
m2_dict['id'] = 2
m2_dict['model_type'] = str(type(m2))
m2_dict['features'] = cols_m2
m2_dict['r2'] = m2_metrics['r2']
m2_dict['rmse'] = m2_metrics['rmse']

# b
with open('models/characteristic_m1.json', 'w') as f:
    json.dump(m1_dict, f)

with open('models/characteristic_m2.json', 'w') as f:
    json.dump(m2_dict, f)
# c
with open('models/scalerX.pickle', 'wb') as f:
    pickle.dump(scalerNormForX, f)

with open('models/scalerY.pickle', 'wb') as f:
    pickle.dump(scalerNormForY, f)

# d
with open('models/m1.pickle', 'wb') as f:
    pickle.dump(m1, f)

m2.save('models/m2.h5',
        overwrite=True,
        save_format='h5'
        )
