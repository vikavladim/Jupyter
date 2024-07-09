relative_path = '../datasets/bike-sharing-dataset_nan_without_deviations.csv'
import streamlit as st
import tensorflow as tf
import json
import pickle

with open('models/characteristic_m1.json', 'r') as f:
    m1_metrics = json.load(f)

with open('models/characteristic_m2.json', 'r') as f:
    m2_metrics = json.load(f)

with open('models/scalerX.pickle', 'rb') as f:
    scalerNormForX = pickle.load(f)

with open('models/scalerY.pickle', 'rb') as f:
    scalerNormForY = pickle.load(f)

with open('models/m1.pickle', 'rb') as f:
    m1 = pickle.load(f)

m2 = tf.keras.models.load_model('models/m2.h5')

# -----Рисование-------
# st.sidebar.header("sidebar Заголовок")
hum = st.sidebar.number_input('Влажность:', min_value=0., max_value=1., step=0.01, value=0.5)
temp = st.sidebar.number_input('Температура:', min_value=0., max_value=1., step=0.01, value=0.5)
windspeed = st.sidebar.number_input('Скорость ветра:', min_value=0., max_value=1., step=0.01, value=0.5)

hr = st.sidebar.slider('Время',
                       min_value=0,
                       max_value=23,
                       value=12, step=1)
days = ['Понедельник', 'Вторник', 'Среда',
        'Четверг', 'Пятница', 'Суббота', 'Воскресенье']
weekday = st.sidebar.selectbox(
    'День недели',
    days)

st.write('Основная часть страницы')

new_features = [hum, temp, hr, days.index(weekday), windspeed]

col1, col2 = st.columns(2)
with col1:
    st.write(f'LinearRegression\n',
             f'- r2={m1_metrics["r2"]}\n',
             f'- rmse={m1_metrics["rmse"]}\n',
             f'- Количество велосипедов в исходной шкале{m1.predict([new_features])}\n',
             f'- Количество велосипедов в нормализованной шкале{scalerNormForY.transform(m1.predict([new_features]))}'
             )
with col2:
    st.write(f'Нейронная сеть\n',
             f'- r2={m2_metrics["r2"]}\n',
             f'- rmse={m2_metrics["rmse"]}\n',
             f'- Количество велосипедов в исходной шкале{scalerNormForY.inverse_transform(m2.predict(scalerNormForX.transform([new_features])))}\n',
             f'- Количество велосипедов в нормализованной шкале{m2.predict([scalerNormForX.transform([new_features])])}')
