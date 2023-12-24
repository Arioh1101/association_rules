import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from sklearn.svm import SVC
import xgboost as xgb
from xgboost import XGBClassifier

from pprint import pprint
# Создаем объект DataFrame с данными из файла tinkoff_credits
data = pd.read_csv('tinkoff_credits.csv', encoding='cp1251', delimiter=';')

# Выводим первые 5 строк
pprint(data.head(5))

# Выводим общую информацию
pprint(data.info())

# Удаление ненужных столбцов - в нашем случае это столбец с ID клиента - для нас он не важен
data.drop('client_id', axis=1, inplace=True)

# Преобразуем строковые значения в числовые в столбцах score_shk, credit_sum
for col in ['score_shk', 'credit_sum']:
    data[col] = data[col].str.replace(',', '.').astype(float)

# Дозаполним пустые ячейки в столбцах
# score_shk, credit_sum, age, credit_count и overdue_credit_count, monthly_income
# медианными значениями
for col in ['score_shk', 'credit_sum', 'age', 'credit_count', 'overdue_credit_count', 'monthly_income']:
    data[col] = data[col].fillna(data[col].median())

# Столбец пола кодируем методом факторизации
data['gender'] = pd.factorize(data['gender'])[0]

# Смотрим, что осталось заменить
pprint(data.info())

# Обработаем столбцы
# job_position, education, marital_status
# методом дамми-кодирования
data = pd.concat([data,
                 pd.get_dummies(data['job_position'], prefix='job_position'),
                 pd.get_dummies(data['education'], prefix='education'),
                 pd.get_dummies(data['marital_status'], prefix='marital_status')],
                 axis=1
                 )

# Удалим столбцы с категориальными данными
data.drop(['job_position', 'education', 'marital_status'], axis=1, inplace=True)
# Удалим столбец с регионом - он не влияет на целевую переменную
data.drop('living_region', axis=1, inplace=True)

pprint(data.info())


# Теперь все столбцы являются числовыми

# Считаем сколько раз зависимая переменная принимает свои значения
pprint(data['open_account_flg'].value_counts(dropna=False))

# Посчитаем точность модели при условии, что она предсказывала бы всем подряд
print(f'Точность модели d={data["open_account_flg"].value_counts(dropna=False)[0] / len(data)}')

# Запишем в переменные входные и выходные данные
y = data['open_account_flg']
x = data.drop('open_account_flg', axis=1)

# Разделим данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=21)

# Обучим модель и подберем гиперпараметры
params = [
    # Для XGBoost
    {
    'n_estimators': [i*10 for i in range(1, 10)],
    'max_depth': [i for i in range(3, 10)]
    },
    # Для SVM
    {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
    }
]
# cl = XGBClassifier(n_estimators=10, max_depth=2, random_state=21)

# cl = GridSearchCV(cl, params[0], cv=3)
# cl.fit(X_train, y_train)
# Посмотрим на точность
# print(grid_search.score(X_test, y_test))


# WARNING временное решение
# Посмотрим на лучшие гиперпараметры (Чтобы не обучать модель каждый раз, просто подставим их)
# pprint(f'Лучшие гиперпараметры для XGBoost: {cl.best_params_}')

cl = XGBClassifier(n_estimators=20, max_depth=5, random_state=21)
cl.fit(X_train, y_train)
print(cl.score(X_test, y_test))

# END OF WARNING

# Получим прогноз
y_pred = cl.predict(X_test)
pprint(confusion_matrix(y_test, y_pred))
# Получим ROC_AUC метрику для данной модели
y_pred = cl.predict_proba(X_test)[:, 1]
print(roc_auc_score(y_test, y_pred))

# Построим график
xgb.plot_importance(cl)
plt.show()

# попробуем алгоритм SVM
# Создадим и обучим модель, подберем к ней гиперпараметры
svm = SVC(C=0.1, kernel='rbf', gamma='scale', random_state=21)
svm = GridSearchCV(svm, params[1], cv=3)
svm.fit(X_train, y_train)
pprint(f"Лучшие гиперпараметры: {svm.best_params_}")
# Посмотрим на точность
print(svm.score(X_test, y_test))

# Получим прогноз
y_pred = svm.predict(X_test)
pprint(confusion_matrix(y_test, y_pred))
# Получим ROC_AUC метрику для данной модели
y_pred = svm.predict(X_test)
print(roc_auc_score(y_test, y_pred))

# попробуем объединить 2 алгоритма
# Используем SVM как фильтр для обработки данных и очистки их от шумов (фильтрации)
X_train_filtered = svm.predict(X_train)
X_test_filtered = svm.predict(X_test)

# Обучим модель на фильтрованных данных
cl.fit(X_train_filtered, y_train)

# Получим прогноз
y_pred = cl.predict(X_test_filtered)
pprint(confusion_matrix(y_test, y_pred))
# Получим ROC_AUC метрику для данной модели
y_pred = cl.predict_proba(X_test_filtered)[:, 1]
print(roc_auc_score(y_test, y_pred))
