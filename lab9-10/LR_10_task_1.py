"""
Лабораторна робота №10. Завдання 10.1
Знаходження оптимальних навчальних параметрів за допомогою сіткового пошуку
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from utilities import visualize_classifier


if __name__ == '__main__':
    # Завантаження вхідних даних
    input_file = '!materials/data_random_forests.txt'
    data = np.loadtxt(input_file, delimiter=',')
    X, y = data[:, :-1], data[:, -1]

    # Розбиття даних на три класи на підставі міток
    class_0 = np.array(X[y == 0])
    class_1 = np.array(X[y == 1])
    class_2 = np.array(X[y == 2])

    # Розбивка даних на навчальний та тестовий набори
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=5
    )

    # Задаємо сітку значень параметрів, де будемо тестувати класифікатор.
    # Зазвичай ми підтримуємо постійним значення одного параметра та варіюємо інші.
    # Потім ця процедура повторюється кожного з параметрів.
    # На разі ми хочемо знайти найкращі значення параметрів n_estimators і max_depth.
    # Визначимо сітку значень параметрів.

    parameter_grid = [
        {'n_estimators': [100],
         'max_depth': [2, 4, 7, 12, 16]},
        {'max_depth': [4], 'n_estimators': [25, 50, 100, 250]}
    ]

    # Визначимо метричні характеристики, які має використовувати класифікатор для знаходження найкращої комбінації параметрів.
    metrics = ['precision_weighted', 'recall_weighted']

    # Для кожної метрики необхідно виконати сітковий пошук, під час якого навчатимемо класифікатор конкретній комбінації параметрів.
    for metric in metrics:
        print("\n##### Searching optimal parameters for", metric)
        classifier = GridSearchCV(
            ExtraTreesClassifier(random_state=0),
            parameter_grid,
            cv=5,
            scoring=metric
        )
        classifier.fit(X_train, y_train)

        # Виведемо оцінку для кожної комбінації параметрів.
        print("\nGrid scores for the parameter grid:")
        # Для нових версій sklearn змінюємо доступ до результатів
        if hasattr(classifier, 'cv_results_'):
            # Для sklearn >= 0.20
            for i in range(len(classifier.cv_results_['params'])):
                print(classifier.cv_results_['params'][i], '-->',
                      round(classifier.cv_results_['mean_test_score'][i], 3))
        else:
            # Для старіших версій sklearn
            for params, avg_score, _ in classifier.grid_scores_:
                print(params, '-->', round(avg_score, 3))

        print("\nBest parameters:", classifier.best_params_)

        # Виведемо звіт із результатами роботи класифікатора
        y_pred = classifier.predict(X_test)
        print("\nPerformance report:\n")
        print(classification_report(y_test, y_pred))

