"""
Лабораторна робота №10. Завдання 10.2
Обчислення відносної важливості ознак
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


if __name__ == '__main__':
    # Завантаження даних із цінами на нерухомість
    housing_data = datasets.load_boston()

    # Перемішування даних
    X, y = shuffle(housing_data.data, housing_data.target, random_state=7)

    # Розбивка даних на навчальний та тестовий набори
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7
    )

    # Модель на основі AdaBoost
    regressor = AdaBoostRegressor(
        DecisionTreeRegressor(max_depth=4),
        n_estimators=400, random_state=7)
    regressor.fit(X_train, y_train)

    # Обчислення показників ефективності регресора AdaBoost
    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    print("\nADABOOST REGRESSOR")
    print("Mean squared error =", round(mse, 2))
    print("Explained variance score =", round(evs, 2))

    # Вилучення важливості ознак
    feature_importances = regressor.feature_importances_
    feature_names = housing_data.feature_names

    # Нормалізація значень важливості ознак
    feature_importances = 100.0 * (feature_importances / max(feature_importances))

    # Сортування та перестановка значень
    index_sorted = np.flipud(np.argsort(feature_importances))

    # Розміщення міток уздовж осі X
    pos = np.arange(index_sorted.shape[0]) + 0.5

    # Побудова стовпчастої діаграми
    plt.figure()
    plt.bar(pos, feature_importances[index_sorted], align='center')
    plt.xticks(pos, feature_names[index_sorted])
    plt.ylabel('Відносна важливість')
    plt.xlabel('Ознаки')
    plt.title('Важливість ознак з використанням регресора AdaBoost')
    plt.show()

    print("\n===== Аналіз важливості ознак =====")
    print("\nОзнаки за спаданням важливості:")
    for i, idx in enumerate(index_sorted):
        print(f"{i+1}. {feature_names[idx]}: {feature_importances[idx]:.2f}%")

    print("\n===== Висновки =====")
    print("Найбільш важливі ознаки (>50%):")
    for idx in index_sorted:
        if feature_importances[idx] > 50:
            print(f"  - {feature_names[idx]}: {feature_importances[idx]:.2f}%")

    print("\nМенш важливі ознаки (<20%), якими можна знехтувати:")
    for idx in index_sorted:
        if feature_importances[idx] < 20:
            print(f"  - {feature_names[idx]}: {feature_importances[idx]:.2f}%")

