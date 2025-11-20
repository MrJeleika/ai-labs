"""
Лабораторна робота 7. Завдання 7.2
Передбачення за допомогою регресії однієї змінної
Варіант 1: файл data_regr_1.txt
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

# Завантажуємо вхідні дані (Варіант 1)
input_file = '!materials/data_regr_1.txt'
data = np.loadtxt(input_file, delimiter=',')

# Розділяємо дані на X та y
X = data[:, :-1]
y = data[:, -1]

# Розбиваємо дані на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5
)

# Створюємо об'єкт лінійного регресора
regressor = linear_model.LinearRegression()

# Навчаємо модель на навчальних даних
regressor.fit(X_train, y_train)

# Прогнозуємо результат для тестового набору даних
y_test_pred = regressor.predict(X_test)

# Візуалізація результатів
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Навчальні дані')
plt.scatter(X_test, y_test, color='green', label='Тестові дані')
plt.plot(X_test, y_test_pred, color='red', linewidth=2, label='Передбачення')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Регресія однієї змінної - Варіант 1')
plt.legend()
plt.grid(True)
plt.savefig('task_7_2_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Обчислюємо метричні параметри регресора
print("Метрики якості регресора (Варіант 1):")
print("Середня абсолютна помилка (MAE):", round(mean_absolute_error(y_test, y_test_pred), 2))
print("Середньоквадратична помилка (MSE):", round(mean_squared_error(y_test, y_test_pred), 2))
print("Коефіцієнт детермінації (R2):", round(r2_score(y_test, y_test_pred), 2))

# Зберігаємо модель у файл
output_model_file = 'saved_model_variant1.pkl'
with open(output_model_file, 'wb') as f:
    pickle.dump(regressor, f)

print(f"\nМодель збережено у файл: {output_model_file}")

# Завантажуємо модель з файлу та робимо прогноз
with open(output_model_file, 'rb') as f:
    regressor_model = pickle.load(f)

# Робимо прогноз з завантаженою моделлю
y_test_pred_new = regressor_model.predict(X_test)
print("\nПрогноз з завантаженою моделлю:")
print("Середня абсолютна помилка (MAE):", round(mean_absolute_error(y_test, y_test_pred_new), 2))

