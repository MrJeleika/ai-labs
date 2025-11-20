"""
Лабораторна робота 8. Завдання 8.2 та 8.3
Самостійна побудова регресії та побудова кривих навчання
Варіант 1: m = 100, X = 6 * np.random.rand(m, 1) - 5, y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_squared_error, r2_score

# Встановлюємо seed для відтворюваності результатів
np.random.seed(42)

# ==================== ГЕНЕРАЦІЯ ДАНИХ (Варіант 1) ====================
print("=" * 60)
print("ВАРІАНТ 1: Генерація даних")
print("=" * 60)
print("Модельне рівняння: y = 0.5*X² + X + 2 + шум")
print()

m = 100
X = 6 * np.random.rand(m, 1) - 5
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

# Виведення даних на графік
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X, y, alpha=0.6, color='blue')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Вихідні дані (Варіант 1)')
plt.grid(True, alpha=0.3)

# ==================== ЛІНІЙНА РЕГРЕСІЯ ====================
print("=" * 60)
print("ЛІНІЙНА РЕГРЕСІЯ")
print("=" * 60)

lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

# Метрики для лінійної регресії
mse_lin = mean_squared_error(y, y_pred_lin)
r2_lin = r2_score(y, y_pred_lin)

print(f"Коефіцієнти: {lin_reg.coef_[0]}")
print(f"Вільний член: {lin_reg.intercept_[0]:.4f}")
print(f"MSE: {mse_lin:.4f}")
print(f"R²: {r2_lin:.4f}")
print(f"Рівняння: y = {lin_reg.coef_[0][0]:.4f}*X + {lin_reg.intercept_[0]:.4f}")

# Графік лінійної регресії
plt.subplot(1, 3, 2)
plt.scatter(X, y, alpha=0.6, color='blue', label='Дані')
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_plot_lin = lin_reg.predict(X_plot)
plt.plot(X_plot, y_plot_lin, 'r-', linewidth=2, label='Лінійна регресія')
plt.xlabel('X')
plt.ylabel('y')
plt.title(f'Лінійна регресія (R² = {r2_lin:.3f})')
plt.legend()
plt.grid(True, alpha=0.3)

# ==================== ПОЛІНОМІАЛЬНА РЕГРЕСІЯ (степінь 2) ====================
print("\n" + "=" * 60)
print("ПОЛІНОМІАЛЬНА РЕГРЕСІЯ (степінь 2)")
print("=" * 60)

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)

# Метрики для поліноміальної регресії
mse_poly = mean_squared_error(y, y_pred_poly)
r2_poly = r2_score(y, y_pred_poly)

print(f"Коефіцієнти: {poly_reg.coef_[0]}")
print(f"Вільний член: {poly_reg.intercept_[0]:.4f}")
print(f"MSE: {mse_poly:.4f}")
print(f"R²: {r2_poly:.4f}")
print(f"Рівняння: y = {poly_reg.coef_[0][1]:.4f}*X² + {poly_reg.coef_[0][0]:.4f}*X + {poly_reg.intercept_[0]:.4f}")

# Графік поліноміальної регресії
plt.subplot(1, 3, 3)
plt.scatter(X, y, alpha=0.6, color='blue', label='Дані')
X_plot_poly = poly_features.transform(X_plot)
y_plot_poly = poly_reg.predict(X_plot_poly)
plt.plot(X_plot, y_plot_poly, 'g-', linewidth=2, label='Поліноміальна регресія')
plt.xlabel('X')
plt.ylabel('y')
plt.title(f'Поліноміальна регресія (R² = {r2_poly:.3f})')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task_8_2_regression.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("ПОРІВНЯННЯ МОДЕЛЕЙ:")
print("=" * 60)
print("Модельне рівняння:     y = 0.5*X² + 1.0*X + 2.0 + шум")
print(f"Передбачене рівняння:  y = {poly_reg.coef_[0][1]:.2f}*X² + {poly_reg.coef_[0][0]:.2f}*X + {poly_reg.intercept_[0]:.2f}")
print(f"\nПоліноміальна модель (R² = {r2_poly:.3f}) значно краща за лінійну (R² = {r2_lin:.3f})")


# ==================== ЗАВДАННЯ 8.3: КРИВІ НАВЧАННЯ ====================
print("\n" + "=" * 80)
print("ЗАВДАННЯ 8.3: КРИВІ НАВЧАННЯ")
print("=" * 80)

def plot_learning_curves(model, X, y, title, degree=None):
    """
    Функція для побудови кривих навчання
    """
    if degree is not None:
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        X = poly_features.fit_transform(X)
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    train_errors = -train_scores.mean(axis=1)
    val_errors = -val_scores.mean(axis=1)
    
    plt.plot(train_sizes, train_errors, 'o-', color='blue', linewidth=2, label='Навчальна вибірка')
    plt.plot(train_sizes, val_errors, 'o-', color='red', linewidth=2, label='Валідаційна вибірка')
    plt.xlabel('Розмір навчального набору')
    plt.ylabel('MSE')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

# Створюємо графіки кривих навчання
plt.figure(figsize=(15, 5))

# 1. Криві навчання для лінійної моделі
plt.subplot(1, 3, 1)
plot_learning_curves(LinearRegression(), X, y, 'Криві навчання: Лінійна модель')

# 2. Криві навчання для поліноміальної моделі 10-го ступеня
plt.subplot(1, 3, 2)
plot_learning_curves(LinearRegression(), X, y, 
                     'Криві навчання: Поліноміальна модель (степінь 10)', 
                     degree=10)

# 3. Криві навчання для поліноміальної моделі 2-го ступеня
plt.subplot(1, 3, 3)
plot_learning_curves(LinearRegression(), X, y, 
                     'Криві навчання: Поліноміальна модель (степінь 2)', 
                     degree=2)

plt.tight_layout()
plt.savefig('task_8_3_learning_curves.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 80)
print("АНАЛІЗ КРИВИХ НАВЧАННЯ:")
print("=" * 80)
print("1. ЛІНІЙНА МОДЕЛЬ (недонавчання):")
print("   - Обидві криві стабілізуються на високому рівні помилки")
print("   - Криві розташовані близько одна до одної")
print("   - Модель занадто проста для даних")
print()
print("2. ПОЛІНОМІАЛЬНА МОДЕЛЬ 10-ГО СТУПЕНЯ (перенавчання):")
print("   - Помилка на навчальних даних дуже низька")
print("   - Великий проміжок між кривими")
print("   - Модель занадто складна, перенавчилась")
print()
print("3. ПОЛІНОМІАЛЬНА МОДЕЛЬ 2-ГО СТУПЕНЯ (оптимальна):")
print("   - Криві близькі одна до одної")
print("   - Помилка валідації низька")
print("   - Найкращий баланс між зміщенням та дисперсією")
print("=" * 80)

# ==================== ДОДАТКОВИЙ АНАЛІЗ: ПОРІВНЯННЯ РІЗНИХ СТУПЕНІВ ====================
print("\n" + "=" * 80)
print("ПОРІВНЯННЯ ПОЛІНОМІАЛЬНИХ МОДЕЛЕЙ РІЗНИХ СТУПЕНІВ")
print("=" * 80)

degrees = [1, 2, 3, 5, 10, 15, 20]
train_errors = []
test_errors = []

# Розділяємо дані на навчальну та тестову вибірки
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for degree in degrees:
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    train_pred = model.predict(X_train_poly)
    test_pred = model.predict(X_test_poly)
    
    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    
    train_errors.append(train_mse)
    test_errors.append(test_mse)
    
    print(f"Ступінь {degree:2d}: Train MSE = {train_mse:8.4f}, Test MSE = {test_mse:8.4f}")

# Графік залежності помилки від ступеня полінома
plt.figure(figsize=(10, 6))
plt.plot(degrees, train_errors, 'o-', linewidth=2, label='Навчальна помилка')
plt.plot(degrees, test_errors, 'o-', linewidth=2, label='Тестова помилка')
plt.xlabel('Ступінь полінома')
plt.ylabel('MSE')
plt.title('Залежність помилки від ступеня полінома')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(degrees)
plt.savefig('task_8_2_degree_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 80)
print("ВИСНОВКИ:")
print("=" * 80)
print("1. Дані згенеровані за квадратичною залежністю (y = 0.5*X² + X + 2)")
print("2. Лінійна модель недонавчилась (високий зсув)")
print("3. Поліноміальна модель 2-го ступеня є оптимальною")
print("4. Моделі вищих ступенів (10+) схильні до перенавчання")
print("5. Криві навчання допомагають діагностувати недо/перенавчання")
print("=" * 80)

