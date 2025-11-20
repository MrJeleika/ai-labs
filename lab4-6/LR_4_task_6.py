import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from utilities import visualize_classifier

# Вхідний файл, який містить дані
input_file = 'data_multivar_nb.txt'

# Завантаження даних із вхідного файлу
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Розбивка даних на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

print("="*60)
print("КЛАСИФІКАЦІЯ ЗА ДОПОМОГОЮ SVM")
print("="*60)

# Створення та навчання SVM класифікатора
svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_classifier.fit(X_train, y_train)

# Прогнозування на тестових даних
y_test_pred_svm = svm_classifier.predict(X_test)

# Обчислення метрик для SVM
svm_accuracy = accuracy_score(y_test, y_test_pred_svm)
svm_precision = precision_score(y_test, y_test_pred_svm, average='weighted')
svm_recall = recall_score(y_test, y_test_pred_svm, average='weighted')
svm_f1 = f1_score(y_test, y_test_pred_svm, average='weighted')

print("\nМетрики якості для SVM:")
print("-" * 60)
print(f"Accuracy (Точність):  {svm_accuracy:.3f} ({svm_accuracy*100:.2f}%)")
print(f"Precision (Точність): {svm_precision:.3f}")
print(f"Recall (Повнота):     {svm_recall:.3f}")
print(f"F1-score:              {svm_f1:.3f}")

# Confusion matrix для SVM
cm_svm = confusion_matrix(y_test, y_test_pred_svm)
print("\nConfusion Matrix для SVM:")
print(cm_svm)

# Перехресна перевірка для SVM
num_folds = 3
svm_cv_accuracy = cross_val_score(svm_classifier, X, y, scoring='accuracy', cv=num_folds)
svm_cv_precision = cross_val_score(svm_classifier, X, y, scoring='precision_weighted', cv=num_folds)
svm_cv_recall = cross_val_score(svm_classifier, X, y, scoring='recall_weighted', cv=num_folds)
svm_cv_f1 = cross_val_score(svm_classifier, X, y, scoring='f1_weighted', cv=num_folds)

print("\nПерехресна перевірка для SVM (3-fold):")
print(f"Accuracy:  {svm_cv_accuracy.mean():.3f} (+/- {svm_cv_accuracy.std()*2:.3f})")
print(f"Precision: {svm_cv_precision.mean():.3f} (+/- {svm_cv_precision.std()*2:.3f})")
print(f"Recall:    {svm_cv_recall.mean():.3f} (+/- {svm_cv_recall.std()*2:.3f})")
print(f"F1-score:  {svm_cv_f1.mean():.3f} (+/- {svm_cv_f1.std()*2:.3f})")

# Візуалізація результатів SVM
print("\nВізуалізація результатів SVM класифікатора...")
# Створюємо власну візуалізацію для збереження графіка
min_x, max_x = X_test[:, 0].min() - 1.0, X_test[:, 0].max() + 1.0
min_y, max_y = X_test[:, 1].min() - 1.0, X_test[:, 1].max() + 1.0
mesh_step_size = 0.01
x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size), 
                              np.arange(min_y, max_y, mesh_step_size))
output = svm_classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
output = output.reshape(x_vals.shape)

plt.figure(figsize=(10, 8))
plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=75, edgecolors='black', 
            linewidth=1, cmap=plt.cm.Paired)
plt.xlim(x_vals.min(), x_vals.max())
plt.ylim(y_vals.min(), y_vals.max())
plt.xticks((np.arange(int(X_test[:, 0].min() - 1), int(X_test[:, 0].max() + 1), 1.0)))
plt.yticks((np.arange(int(X_test[:, 1].min() - 1), int(X_test[:, 1].max() + 1), 1.0)))
plt.title('SVM Classifier Results')
plt.savefig('svm_classifier_results.png', dpi=300, bbox_inches='tight')
plt.close()  # Закриваємо фігуру після збереження

# Показуємо графік через visualize_classifier
visualize_classifier(svm_classifier, X_test, y_test)

print("\n" + "="*60)
print("КЛАСИФІКАЦІЯ ЗА ДОПОМОГОЮ NAIVE BAYES")
print("="*60)

# Створення та навчання Naive Bayes класифікатора
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Прогнозування на тестових даних
y_test_pred_nb = nb_classifier.predict(X_test)

# Обчислення метрик для Naive Bayes
nb_accuracy = accuracy_score(y_test, y_test_pred_nb)
nb_precision = precision_score(y_test, y_test_pred_nb, average='weighted')
nb_recall = recall_score(y_test, y_test_pred_nb, average='weighted')
nb_f1 = f1_score(y_test, y_test_pred_nb, average='weighted')

print("\nМетрики якості для Naive Bayes:")
print("-" * 60)
print(f"Accuracy (Точність):  {nb_accuracy:.3f} ({nb_accuracy*100:.2f}%)")
print(f"Precision (Точність): {nb_precision:.3f}")
print(f"Recall (Повнота):     {nb_recall:.3f}")
print(f"F1-score:              {nb_f1:.3f}")

# Confusion matrix для Naive Bayes
cm_nb = confusion_matrix(y_test, y_test_pred_nb)
print("\nConfusion Matrix для Naive Bayes:")
print(cm_nb)

# Перехресна перевірка для Naive Bayes
nb_cv_accuracy = cross_val_score(nb_classifier, X, y, scoring='accuracy', cv=num_folds)
nb_cv_precision = cross_val_score(nb_classifier, X, y, scoring='precision_weighted', cv=num_folds)
nb_cv_recall = cross_val_score(nb_classifier, X, y, scoring='recall_weighted', cv=num_folds)
nb_cv_f1 = cross_val_score(nb_classifier, X, y, scoring='f1_weighted', cv=num_folds)

print("\nПерехресна перевірка для Naive Bayes (3-fold):")
print(f"Accuracy:  {nb_cv_accuracy.mean():.3f} (+/- {nb_cv_accuracy.std()*2:.3f})")
print(f"Precision: {nb_cv_precision.mean():.3f} (+/- {nb_cv_precision.std()*2:.3f})")
print(f"Recall:    {nb_cv_recall.mean():.3f} (+/- {nb_cv_recall.std()*2:.3f})")
print(f"F1-score:  {nb_cv_f1.mean():.3f} (+/- {nb_cv_f1.std()*2:.3f})")

# Візуалізація результатів Naive Bayes
print("\nВізуалізація результатів Naive Bayes класифікатора...")
# Створюємо власну візуалізацію для збереження графіка
min_x, max_x = X_test[:, 0].min() - 1.0, X_test[:, 0].max() + 1.0
min_y, max_y = X_test[:, 1].min() - 1.0, X_test[:, 1].max() + 1.0
mesh_step_size = 0.01
x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size), 
                              np.arange(min_y, max_y, mesh_step_size))
output = nb_classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
output = output.reshape(x_vals.shape)

plt.figure(figsize=(10, 8))
plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=75, edgecolors='black', 
            linewidth=1, cmap=plt.cm.Paired)
plt.xlim(x_vals.min(), x_vals.max())
plt.ylim(y_vals.min(), y_vals.max())
plt.xticks((np.arange(int(X_test[:, 0].min() - 1), int(X_test[:, 0].max() + 1), 1.0)))
plt.yticks((np.arange(int(X_test[:, 1].min() - 1), int(X_test[:, 1].max() + 1), 1.0)))
plt.title('Naive Bayes Classifier Results')
plt.savefig('nb_classifier_results.png', dpi=300, bbox_inches='tight')
plt.close()  # Закриваємо фігуру після збереження

# Показуємо графік через visualize_classifier
visualize_classifier(nb_classifier, X_test, y_test)

# Порівняння моделей
print("\n" + "="*60)
print("ПОРІВНЯННЯ МОДЕЛЕЙ")
print("="*60)
print("\nМетрика\t\t\tSVM\t\tNaive Bayes\tКраща модель")
print("-" * 60)
print(f"Accuracy\t\t{svm_accuracy:.3f}\t\t{nb_accuracy:.3f}\t\t{'SVM' if svm_accuracy > nb_accuracy else 'Naive Bayes'}")
print(f"Precision\t\t{svm_precision:.3f}\t\t{nb_precision:.3f}\t\t{'SVM' if svm_precision > nb_precision else 'Naive Bayes'}")
print(f"Recall\t\t\t{svm_recall:.3f}\t\t{nb_recall:.3f}\t\t{'SVM' if svm_recall > nb_recall else 'Naive Bayes'}")
print(f"F1-score\t\t{svm_f1:.3f}\t\t{nb_f1:.3f}\t\t{'SVM' if svm_f1 > nb_f1 else 'Naive Bayes'}")

print("\n" + "="*60)
print("ВИСНОВКИ")
print("="*60)
if svm_accuracy > nb_accuracy:
    print("1. Модель SVM показала кращі результати за всіма метриками.")
    print("2. SVM класифікатор краще підходить для даного завдання класифікації.")
    print("3. Переваги SVM:")
    print("   - Краща точність класифікації")
    print("   - Краще узагальнення на нових даних")
    print("   - Ефективна робота з нелінійно розділеними класами")
else:
    print("1. Модель Naive Bayes показала кращі результати.")
    print("2. Naive Bayes класифікатор краще підходить для даного завдання.")
    print("3. Переваги Naive Bayes:")
    print("   - Швидке навчання та прогнозування")
    print("   - Простота реалізації")
    print("   - Ефективна робота з малими наборами даних")

print("\n4. Рекомендації:")
print("   - Для продакшн системи рекомендовано використовувати модель з кращими метриками")
print("   - Важливо враховувати час навчання та прогнозування")
print("   - Для великих наборів даних SVM може бути повільнішим")
print("   - Naive Bayes швидший, але може бути менш точним на складних даних")

