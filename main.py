import pandas as pd
from apyori import apriori
from sklearn.cluster import AgglomerativeClustering
import numpy as np

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
        clustered_rules[str(i)] = []
    for i, labels in enumerate(labels):
        clustered_rules[str(labels)].append(rules_list[i])

    return clustered_rules


def get_association_rules(data,
                          form_list_by,
                          group_by='Номер заказа',
                          min_support=0.005,
                          min_confidence=0.5,
                          min_lift=1.5,
                          clust=True,
                          count_clusters=3):
    data = data.groupby(group_by)[form_list_by].apply(list).values.tolist()
    # pprint(data)

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

# trying with category - "Категория"   ###
get_association_rules(transactions, 'Категория', clust=True)
# Всего одно правило - можем просто упомянуть его - а можем и убрать нахер

### trying with category - "Цена"   ###
get_association_rules(transactions, 'Цена', clust=True)

# Зависимостей много - но как их анализировать...