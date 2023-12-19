import pandas as pd
from apyori import apriori
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

from pprint import pprint


def clusterize(rules, rules_list, n):
    # Create matrix from rules for hierarchies
    matrix = []
    for rule in rules:
        line = [round(rule[1], 3), round(rule[2][0][2], 3), round(rule[2][0][3], 3)]
        matrix.append(line)

    # Инициализация алгоритма иерархической кластериациии
    model = AgglomerativeClustering(n_clusters=n)
    # Применяем алгоритм к данным
    model.fit(matrix)
    # Получаем метки кластеров
    labels = model.labels_

    # Сортируем правила по кластерам, записывая их в словарь
    clustered_rules = {}
    for i in range(n):
        clustered_rules[i] = []
    for i, labels in enumerate(labels):
        clustered_rules[labels].append(rules_list[i])

    return clustered_rules


def get_association_rules(data,
                          form_list_by,
                          group_by='Номер заказа',
                          min_support=0.005,
                          min_confidence=0.5,
                          min_lift=1.5,
                          clust=True,
                          count_clusters=3):
    data = data.groupby(group_by, observed=False)[form_list_by].apply(list).values.tolist()
    #pprint(data)

    # Apply apriori algorithm
    rules = apriori(data, min_support=min_support, min_confidence=min_confidence, min_lift=min_lift, min_length=2)
    rules = list(rules)
    #pprint(rules)

    # Doing pretty
    rules_list = []
    for rule in rules:
        pair = rule[2][0]
        base = [str(item) for item in pair.items_base]
        add = [str(item) for item in pair.items_add]
        output_str = f"--{', '.join(base)} ==> {', '.join(add)}: sup = {str(round(rule[1], 3))}, conf = {str(round(rule[2][0][2], 3))}, lift = {str(round(rule[2][0][3], 3))}"
        if not clust:
            print(output_str)
        # Для дальнейшего использования "красивых" правил - запием их в список
        rules_list.append(output_str)

    if clust:
        if len(rules_list) >= 2:
            pprint(clusterize(rules, rules_list, count_clusters))
        else:
            print("Недостаточно данных для кластеризации")

# open file
transactions = pd.read_excel("transactions.xlsx", skiprows=2)

# View first 10 lines
print(transactions.head(10))

# Видим, что одна транзакция представлена несколькими строками - значит группировать будем по номеру транзакции
get_association_rules(transactions, 'Товар', clust=True)

# Попробуем модель FP_Growth
data = transactions.groupby('Номер заказа')["Товар"].apply(list).values.tolist()


te = TransactionEncoder()
te_ary = te.fit_transform(data)

df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = fpgrowth(df, min_support=0.05, use_colnames=True)
# Посмотрим на все товары, которые покупают часто (в 5% покупок)
print(frequent_itemsets)

# Выведем только комбинации продуктов, которые часто поупают
for item in frequent_itemsets.itertuples():
    if len(item[2]) > 1:
        print(f"Набор: {', '.join(item.itemsets)} - покупают с частотой {round(item.support, 3)}")

# попробуем найти зависимость по другой модели

# trying with category - "Категория"   ###
get_association_rules(transactions, 'Категория', clust=True)
# Всего одно правило - можем просто упомянуть его - а можем и убрать нахер

## trying with category - "Цена"   ###
#get_association_rules(transactions, 'Цена', clust=True)

# Зависимостей много - но как их анализировать...

# Добавим столбец в датафрейм с ценовой категорией
transactions["Ценовая категория"] = pd.cut(transactions["Цена"],
                                           bins=[0, 100, 500, 1000, 100000],
                                           labels=["малая цена", "средняя цена", "высокая цена", "очень высокая цена"])
get_association_rules(transactions, 'Ценовая категория', clust=True)

# Как и в предыдущем случае - можем и убрать нахер

