import pandas as pd
import numpy as np
import glob
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import sklearn.preprocessing as sk
import time

from keras.layers import LSTM, Dense
from keras.models import Sequential

# Метрики
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr


# Для удобства
from pprint import pprint


# Function for waiting input from user for easy reading info
def waitor():
    _ = input("Нажмите для продолжения")
    return 1


directory = r"C:\Users\Николай\Desktop\Дане, которому постоянно что то нужно\Проект погода\archive"
files = listdir(directory)
print(files)
# Выше виден список всех файлов с данными, эти данные необьходимо объединить в один файл, с которым потом будем работать

path = directory
all_files = glob.glob(path + "/*.csv")
lst = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    lst.append(df)
data = pd.concat(lst, axis=0, ignore_index=False)
data = data.reset_index(drop=True)
pprint(data.head(10))
waitor()


# ОБРАБОТКА ДАННЫХ
pprint(data.info)
waitor()

del data['Unnamed: 0']

pprint(data.head(20))
waitor()

# Date
# Latitude
# Longitude
# cld: Cloud cover (%)
# dtr: Temperature range (°C)
# frs: Number of frost days
# pet: Potential evapotranspiration (mm)
# pre: Precipitation (mm)
# tmn: Minimum temperature (°C)
# tmp: Mean temperature (°C)
# tmx: Maximum temperature (°C)
# vap: Relative humidity (%)
# wet: Number of wet days

pprint(data.count(axis=0, numeric_only=False))
waitor()

signs_arr = ['Date', 'Latitude', 'Longitude', 'dtr', 'frs', 'pet', 'pre', 'tmn', 'tmp', 'tmx', 'vap', 'wet']
result = 'cld'

for i in signs_arr:
    print(len(data[data[i] == '']) / len(data[data[i] != '']))
waitor()

data.drop(data[data['Longitude'] < 73].index, inplace=True)
data.info()
waitor()

data['Date'] = pd.to_datetime(data['Date'])
data = data.reset_index(drop=True)
for column in data:
    data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())
# data.rename(columns={"Date": "0",
#                      "Latitude": "1",
#                      "Longitude": "2",
#                      "cld": "3",
#                      "dtr": "4",
#                      "frs": "5",
#                      "pet": "6",
#                      "pre": "7",
#                      "tmn": "8",
#                      "tmp": "9",
#                      "tmx": "10",
#                      "vap": "11",
#                      "wet": "12"},
#             inplace=True)


pprint(data.head())
waitor()

# СОЗДАНИЕ ВЫБОРОК, МОДЕЛЬ И ЕЕ ОБУЧЕНИЕ
# Преобразуем данные в последовательности (в этом нам поможет оконное разбиение данных)
window_size = 100000  # SITUATIONAL
count_input_signs = len(signs_arr)

pprint(data.loc[:, signs_arr].head())
# Создание последовательностей

# Разбиение на окна
X, y = [], []
for i in range(0, len(data) - window_size, 10000):
    print(i, i+window_size, sep='---')  # !CLEAN after debugging
    window = data[i:i + window_size]  # Создание окна
    X.append(window.loc[:, signs_arr])
    y.append(window.loc[:, result])

# Преобразоываем в массивы np
X = np.array(X)
y = np.array(y)

# Делим на обучающую и тестовую выборки
train_size = int(0.99 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Создаем модель
model = Sequential()
model.add(LSTM(50, input_shape=(window_size, count_input_signs)))
model.add(Dense(1))

# Обучаем модель
print("Обучение...")
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=1, batch_size=32)

y_test = y_test.reshape(-1, 1)

# Избавимся от окон для предсказаний
tmp = []
for i in X_test:
    tmp.extend(i)
X_test = tmp
print(len(X_test))
# Получим метрики для данной модели без учета вероятностей
print("Без вероятностей")
y_pred = model.predict(X_test)
print(len(y_pred))
print(len(y_test))
pprint(f'Среднеквадратичная ошибка -- {mean_squared_error(y_test, y_pred)} \n'
       f'Средняя абсолютная ошибка -- {mean_absolute_error(y_test, y_pred)}\n'
       f'Коэффициент детерминации -- {r2_score(y_test, y_pred)}\n'
       f'Коэффициент корреляции Пирсона -- {pearsonr(y_test, y_pred)}\n')

# Получим ROC_AUC метрики для данной модели с учетом вероятностей
print("С вероятностями")
y_pred = model.predict_proba(X_test)[:, 1]
pprint(f'Среднеквадратичная ошибка -- {mean_squared_error(y_test, y_pred)} \n'
       f'Средняя абсолютная ошибка -- {mean_absolute_error(y_test, y_pred)}\n'
       f'Коэффициент детерминации -- {r2_score(y_test, y_pred)}\n'
       f'Коэффициент корреляции Пирсона -- {pearsonr(y_test, y_pred)}\n')