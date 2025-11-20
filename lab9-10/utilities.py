"""
Допоміжні функції для візуалізації класифікаторів
"""

import numpy as np
import matplotlib.pyplot as plt


def visualize_classifier(classifier, X, y, title=''):
    """
    Візуалізація меж рішень класифікатора
    
    Параметри:
    classifier - навчений класифікатор
    X - вхідні дані (numpy array з 2 ознаками)
    y - цільові мітки
    title - назва графіку
    """
    # Визначення мінімальних та максимальних значень для побудови сітки
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    
    # Створення сітки точок з кроком 0.01
    mesh_step_size = 0.01
    x_vals, y_vals = np.meshgrid(
        np.arange(min_x, max_x, mesh_step_size),
        np.arange(min_y, max_y, mesh_step_size)
    )
    
    # Прогнозування результатів для всіх точок сітки
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
    output = output.reshape(x_vals.shape)
    
    # Створення графіку
    plt.figure()
    
    # Вибір кольорової схеми
    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray, shading='auto')
    
    # Нанесення тренувальних точок на графік
    plt.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='black', 
                linewidth=1, cmap=plt.cm.Paired)
    
    # Встановлення меж графіку
    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())
    
    # Налаштування осей
    plt.xticks((np.arange(int(min_x), int(max_x), 1.0)))
    plt.yticks((np.arange(int(min_y), int(max_y), 1.0)))
    
    plt.title(title)
    plt.show()

