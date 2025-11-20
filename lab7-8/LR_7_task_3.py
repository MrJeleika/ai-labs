"""
Лабораторна робота 7. Завдання 7.3
Створення багатовимірного регресора
"""

import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error, explained_variance_score

# Завантажуємо вхідні дані
input_file = '!materials/data_multivar_regr.txt'
data = np.loadtxt(input_file, delimiter=',')

# Розділяємо дані на X та y
X = data[:, :-1]
y = data[:, -1]

# Розбиваємо дані на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5
)

# ==================== ЛІНІЙНА РЕГРЕСІЯ ====================
# Створюємо об'єкт лінійного регресора
linear_regressor = linear_model.LinearRegression()

# Навчаємо модель на навчальних даних
linear_regressor.fit(X_train, y_train)

# Прогнозуємо результат для тестового набору даних
y_test_pred_linear = linear_regressor.predict(X_test)

# Виводимо метрики якості лінійної регресії
print("=" * 60)
print("ЛІНІЙНА РЕГРЕСІЯ - Метрики якості:")
print("=" * 60)
print("Mean absolute error (MAE):", round(mean_absolute_error(y_test, y_test_pred_linear), 2))
print("Mean squared error (MSE):", round(mean_squared_error(y_test, y_test_pred_linear), 2))
print("Median absolute error:", round(median_absolute_error(y_test, y_test_pred_linear), 2))
print("Explained variance score:", round(explained_variance_score(y_test, y_test_pred_linear), 2))
print("R2 score:", round(r2_score(y_test, y_test_pred_linear), 2))

# ==================== ПОЛІНОМІАЛЬНА РЕГРЕСІЯ ====================
# Створюємо поліноміальний регресор ступеня 10
polynomial = PolynomialFeatures(degree=10)
X_train_poly = polynomial.fit_transform(X_train)
X_test_poly = polynomial.transform(X_test)

# Навчаємо лінійний регресор на поліноміальних ознаках
poly_regressor = linear_model.LinearRegression()
poly_regressor.fit(X_train_poly, y_train)

# Прогнозуємо результат для тестового набору даних
y_test_pred_poly = poly_regressor.predict(X_test_poly)

# Виводимо метрики якості поліноміальної регресії
print("\n" + "=" * 60)
print("ПОЛІНОМІАЛЬНА РЕГРЕСІЯ (степінь 10) - Метрики якості:")
print("=" * 60)
print("Mean absolute error (MAE):", round(mean_absolute_error(y_test, y_test_pred_poly), 2))
print("Mean squared error (MSE):", round(mean_squared_error(y_test, y_test_pred_poly), 2))
print("Median absolute error:", round(median_absolute_error(y_test, y_test_pred_poly), 2))
print("Explained variance score:", round(explained_variance_score(y_test, y_test_pred_poly), 2))
print("R2 score:", round(r2_score(y_test, y_test_pred_poly), 2))

# ==================== ПРОГНОЗ ДЛЯ ОКРЕМОЇ ТОЧКИ ====================
# Беремо тестову точку, близьку до [7.66, 6.29, 5.66] (рядок 11 в файлі, очікуване значення ~41.35)
datapoint = np.array([[7.75, 6.35, 5.56]])

print("\n" + "=" * 60)
print("ПРОГНОЗ ДЛЯ ТЕСТОВОЇ ТОЧКИ:", datapoint[0])
print("=" * 60)

# Перетворюємо точку в поліном
poly_datapoint = polynomial.transform(datapoint)

# Прогноз за допомогою лінійного регресора
poly_linear_pred = linear_regressor.predict(datapoint)
print("Лінійний регресор - прогноз:", round(poly_linear_pred[0], 2))

# Прогноз за допомогою поліноміального регресора
poly_pred = poly_regressor.predict(poly_datapoint)
print("Поліноміальний регресор - прогноз:", round(poly_pred[0], 2))
print("Очікуване значення: ~41.35")

print("\n" + "=" * 60)
print("ВИСНОВОК:")
print("=" * 60)
print("Поліноміальний регресор дає результат ближчий до очікуваного")
print("значення, що свідчить про кращу якість моделі для цих даних.")
print("=" * 60)

