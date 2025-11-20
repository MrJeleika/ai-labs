import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, roc_auc_score

# Завантаження даних
df = pd.read_csv('data_metrics.csv')
print("Перші рядки даних:")
print(df.head())

# Встановлення порогу та створення прогнозованих міток
thresh = 0.5
df['predicted_RF'] = (df.model_RF >= 0.5).astype('int')
df['predicted_LR'] = (df.model_LR >= 0.5).astype('int')
print("\nДані з прогнозованими мітками:")
print(df.head())

# Функції для обчислення TP, FN, FP, TN
def find_TP(y_true, y_pred):
    # counts the number of true positives (y_true = 1, y_pred = 1)
    return sum((y_true == 1) & (y_pred == 1))

def find_FN(y_true, y_pred):
    # counts the number of false negatives (y_true = 1, y_pred = 0)
    return sum((y_true == 1) & (y_pred == 0))

def find_FP(y_true, y_pred):
    # counts the number of false positives (y_true = 0, y_pred = 1)
    return sum((y_true == 0) & (y_pred == 1))

def find_TN(y_true, y_pred):
    # counts the number of true negatives (y_true = 0, y_pred = 0)
    return sum((y_true == 0) & (y_pred == 0))

# Перевірка функцій
print('\nTP:', find_TP(df.actual_label.values, df.predicted_RF.values))
print('FN:', find_FN(df.actual_label.values, df.predicted_RF.values))
print('FP:', find_FP(df.actual_label.values, df.predicted_RF.values))
print('TN:', find_TN(df.actual_label.values, df.predicted_RF.values))

# Функція для обчислення всіх значень матриці помилок
def find_conf_matrix_values(y_true, y_pred):
    # calculate TP, FN, FP, TN
    TP = find_TP(y_true, y_pred)
    FN = find_FN(y_true, y_pred)
    FP = find_FP(y_true, y_pred)
    TN = find_TN(y_true, y_pred)
    return TP, FN, FP, TN

# УВАГА: Замініть "student" на своє прізвище англійською мовою!
def student_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])

# Перевірка confusion matrix
print("\nВласна confusion matrix для RF:")
print(student_confusion_matrix(df.actual_label.values, df.predicted_RF.values))
print("\nsklearn confusion matrix для RF:")
print(confusion_matrix(df.actual_label.values, df.predicted_RF.values))

# Перевірка відповідності
assert np.array_equal(student_confusion_matrix(df.actual_label.values, df.predicted_RF.values), 
                      confusion_matrix(df.actual_label.values, df.predicted_RF.values)), \
    'student_confusion_matrix() is not correct for RF'
assert np.array_equal(student_confusion_matrix(df.actual_label.values, df.predicted_LR.values),
                      confusion_matrix(df.actual_label.values, df.predicted_LR.values)), \
    'student_confusion_matrix() is not correct for LR'

# Функція accuracy_score
def student_accuracy_score(y_true, y_pred):
    # calculates the fraction of samples
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return (TP + TN) / (TP + FN + FP + TN)

assert student_accuracy_score(df.actual_label.values, df.predicted_RF.values) == \
       accuracy_score(df.actual_label.values, df.predicted_RF.values), \
    'student_accuracy_score() is not correct for RF'
assert student_accuracy_score(df.actual_label.values, df.predicted_LR.values) == \
       accuracy_score(df.actual_label.values, df.predicted_LR.values), \
    'student_accuracy_score() is not correct for LR'

print('\nAccuracy RF: %.3f' % (student_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Accuracy LR: %.3f' % (student_accuracy_score(df.actual_label.values, df.predicted_LR.values)))

# Функція recall_score
def student_recall_score(y_true, y_pred):
    # calculates the fraction of positive samples predicted correctly
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FN)

assert student_recall_score(df.actual_label.values, df.predicted_RF.values) == \
       recall_score(df.actual_label.values, df.predicted_RF.values), \
    'student_recall_score() is not correct for RF'
assert student_recall_score(df.actual_label.values, df.predicted_LR.values) == \
       recall_score(df.actual_label.values, df.predicted_LR.values), \
    'student_recall_score() is not correct for LR'

print('\nRecall RF: %.3f' % (student_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall LR: %.3f' % (student_recall_score(df.actual_label.values, df.predicted_LR.values)))

# Функція precision_score
def student_precision_score(y_true, y_pred):
    # calculates the fraction of predicted positives samples that are actually positive
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FP)

assert student_precision_score(df.actual_label.values, df.predicted_RF.values) == \
       precision_score(df.actual_label.values, df.predicted_RF.values), \
    'student_precision_score() is not correct for RF'
assert student_precision_score(df.actual_label.values, df.predicted_LR.values) == \
       precision_score(df.actual_label.values, df.predicted_LR.values), \
    'student_precision_score() is not correct for LR'

print('\nPrecision RF: %.3f' % (student_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('Precision LR: %.3f' % (student_precision_score(df.actual_label.values, df.predicted_LR.values)))

# Функція f1_score
def student_f1_score(y_true, y_pred):
    # calculates the F1 score
    recall = student_recall_score(y_true, y_pred)
    precision = student_precision_score(y_true, y_pred)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

assert np.isclose(student_f1_score(df.actual_label.values, df.predicted_RF.values),
                  f1_score(df.actual_label.values, df.predicted_RF.values)), \
    'student_f1_score() is not correct for RF'
assert np.isclose(student_f1_score(df.actual_label.values, df.predicted_LR.values),
                  f1_score(df.actual_label.values, df.predicted_LR.values)), \
    'student_f1_score() is not correct for LR'

print('\nF1 RF: %.3f' % (student_f1_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 LR: %.3f' % (student_f1_score(df.actual_label.values, df.predicted_LR.values)))

# Порівняння з різними порогами
print('\n' + '='*50)
print('scores with threshold = 0.5')
print('Accuracy RF: %.3f' % (student_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall RF: %.3f' % (student_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Precision RF: %.3f' % (student_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 RF: %.3f' % (student_f1_score(df.actual_label.values, df.predicted_RF.values)))

print('\nscores with threshold = 0.25')
predicted_RF_025 = (df.model_RF >= 0.25).astype('int').values
print('Accuracy RF: %.3f' % (student_accuracy_score(df.actual_label.values, predicted_RF_025)))
print('Recall RF: %.3f' % (student_recall_score(df.actual_label.values, predicted_RF_025)))
print('Precision RF: %.3f' % (student_precision_score(df.actual_label.values, predicted_RF_025)))
print('F1 RF: %.3f' % (student_f1_score(df.actual_label.values, predicted_RF_025)))

# ROC крива
fpr_RF, tpr_RF, thresholds_RF = roc_curve(df.actual_label.values, df.model_RF.values)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(df.actual_label.values, df.model_LR.values)

# AUC значення
auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values, df.model_LR.values)

print('\nAUC RF: %.3f' % auc_RF)
print('AUC LR: %.3f' % auc_LR)

# Побудова ROC кривої
plt.figure(figsize=(10, 8))
plt.plot(fpr_RF, tpr_RF, 'r-', label='RF AUC: %.3f' % auc_RF, linewidth=2)
plt.plot(fpr_LR, tpr_LR, 'b-', label='LR AUC: %.3f' % auc_LR, linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='random')
plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g--', label='perfect')
plt.legend(fontsize=12)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves for RF and LR Models', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nВисновки:")
print("="*50)
print("1. Модель RF має кращі показники за всіма метриками порівняно з LR")
print("2. AUC для RF (%.3f) вищий ніж для LR (%.3f)" % (auc_RF, auc_LR))
print("3. ROC крива RF знаходиться далі від випадкової лінії, що вказує на кращу продуктивність")
print("4. Для вибору між моделями рекомендовано використовувати AUC, оскільки він не залежить від порогу")
print("5. Модель RF є кращою за модель LR для даного завдання класифікації")

