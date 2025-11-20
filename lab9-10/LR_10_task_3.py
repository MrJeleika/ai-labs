"""
Лабораторна робота №10. Завдання 10.3
Прогнозування інтенсивності дорожнього руху за допомогою класифікатора 
на основі гранично випадкових лісів
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesRegressor


if __name__ == '__main__':
    # Завантаження вхідних даних
    input_file = '!materials/traffic_data.txt'
    data = []
    with open(input_file, 'r') as f:
        for line in f.readlines():
            items = line[:-1].split(',')
            data.append(items)

    data = np.array(data)

    # Перетворення рядкових даних на числові
    label_encoder = []
    X_encoded = np.empty(data.shape)
    for i, item in enumerate(data[0]):
        if item.isdigit():
            X_encoded[:, i] = data[:, i]
        else:
            label_encoder.append(preprocessing.LabelEncoder())
            X_encoded[:, i] = label_encoder[-1].fit_transform(data[:, i])

    X = X_encoded[:, :-1].astype(int)
    y = X_encoded[:, -1].astype(int)

    # Розбиття даних на навчальний та тестовий набори
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=5)

    # Регресор на основі гранично випадкових лісів
    params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
    regressor = ExtraTreesRegressor(**params)
    regressor.fit(X_train, y_train)

    # Обчислення характеристик ефективності регресора на тестових даних
    y_pred = regressor.predict(X_test)
    print("Mean absolute error:", round(mean_absolute_error(y_test, y_pred), 2))

    # Тестування кодування на одиночному прикладі
    test_datapoint = ['Saturday', '10:20', 'Atlanta', 'no']
    test_datapoint_encoded = [-1] * len(test_datapoint)
    count = 0
    for i, item in enumerate(test_datapoint):
        if item.isdigit():
            test_datapoint_encoded[i] = int(item)
        else:
            test_datapoint_encoded[i] = int(label_encoder[count].transform([item])[0])
            count += 1

    test_datapoint_encoded = np.array(test_datapoint_encoded)

    # Прогнозування результату для тестової точки даних
    print("Predicted traffic:",
          int(regressor.predict([test_datapoint_encoded])[0]))

    # Додаткове тестування на кількох точках даних
    print("\n===== Додаткові тестові прогнози =====")
    test_cases = [
        ['Monday', '08:00', 'Atlanta', 'yes'],
        ['Friday', '18:30', 'Atlanta', 'yes'],
        ['Sunday', '14:00', 'Atlanta', 'no'],
        ['Wednesday', '12:00', 'Atlanta', 'no']
    ]

    for test_case in test_cases:
        test_encoded = [-1] * len(test_case)
        count = 0
        for i, item in enumerate(test_case):
            if item.isdigit():
                test_encoded[i] = int(item)
            else:
                test_encoded[i] = int(label_encoder[count].transform([item])[0])
                count += 1
        test_encoded = np.array(test_encoded)
        predicted_traffic = int(regressor.predict([test_encoded])[0])
        print(f"День: {test_case[0]}, Час: {test_case[1]}, "
              f"Команда: {test_case[2]}, Матч: {test_case[3]} -> "
              f"Прогноз трафіку: {predicted_traffic}")

