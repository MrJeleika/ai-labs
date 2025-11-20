"""
Лабораторна робота 8. Завдання 8.1
Регресія багатьох змінних з використанням датасету діабету
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Завантажуємо датасет діабету
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

print("Інформація про датасет діабету:")
print(f"Кількість зразків: {X.shape[0]}")
print(f"Кількість ознак: {X.shape[1]}")
print(f"Назви ознак: {diabetes.feature_names}")
print()

# Поділ даних на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0
)

# Створюємо та навчаємо модель лінійної регресії
regr = LinearRegression()
regr.fit(X_train, y_train)

# Робимо прогноз по тестовій вибірці
y_pred = regr.predict(X_test)

# Виводимо коефіцієнти регресії та показники
print("=" * 60)
print("КОЕФІЦІЄНТИ МОДЕЛІ:")
print("=" * 60)
print("Коефіцієнти регресії (regr.coef_):")
for i, coef in enumerate(regr.coef_):
    print(f"  {diabetes.feature_names[i]}: {coef:.4f}")
print(f"\nВільний член (regr.intercept_): {regr.intercept_:.4f}")

print("\n" + "=" * 60)
print("МЕТРИКИ ЯКОСТІ:")
print("=" * 60)
print(f"Коефіцієнт детермінації R²: {r2_score(y_test, y_pred):.4f}")
print(f"Середня абсолютна помилка (MAE): {mean_absolute_error(y_test, y_pred):.4f}")
print(f"Середньоквадратична помилка (MSE): {mean_squared_error(y_test, y_pred):.4f}")

# Побудова графіка 1: Порівняння передбачених та фактичних значень
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Ідеальна лінія')
plt.xlabel('Фактичні значення')
plt.ylabel('Передбачені значення')
plt.title('Порівняння фактичних та передбачених значень')
plt.legend()
plt.grid(True, alpha=0.3)

# Побудова графіка 2: Залишки (residuals)
plt.subplot(1, 2, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5, color='green')
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Передбачені значення')
plt.ylabel('Залишки')
plt.title('Графік залишків')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task_8_1_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# Додатковий графік: Важливість ознак
plt.figure(figsize=(10, 6))
feature_importance = np.abs(regr.coef_)
features = diabetes.feature_names
indices = np.argsort(feature_importance)[::-1]

plt.bar(range(len(features)), feature_importance[indices])
plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45)
plt.xlabel('Ознаки')
plt.ylabel('Абсолютне значення коефіцієнта')
plt.title('Важливість ознак (за абсолютним значенням коефіцієнтів)')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('task_8_1_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("ВИСНОВОК:")
print("=" * 60)
print("Модель лінійної регресії успішно навчена на датасеті діабету.")
print(f"R² = {r2_score(y_test, y_pred):.4f} показує якість моделі.")
print("Графіки збережені у файли task_8_1_plots.png та task_8_1_feature_importance.png")
print("=" * 60)

