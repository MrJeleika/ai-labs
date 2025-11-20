import numpy as np
from sklearn import preprocessing

# Варіант 4
# Значення змінної input_data для варіанту 4
input_data = np.array([[-5.3, -8.9, 3.0],
                        [2.9, 5.1, -3.3],
                        [3.1, -2.8, -3.2],
                        [2.2, -1.4, 5.1]])

# Поріг бінаризації для варіанту 4
threshold = 3.0

# Бінарізація даних
data_binarized = preprocessing.Binarizer(threshold=threshold).transform(input_data)
print("\nBinarized data:\n", data_binarized)

# Виключення середнього
# Виведення середнього значення та стандартного відхилення
print("\nBEFORE: ")
print("Mean =", input_data.mean(axis=0))
print("Std deviation =", input_data.std(axis=0))

# Виключення середнього
data_scaled = preprocessing.scale(input_data)
print("\nAFTER: ")
print("Mean =", data_scaled.mean(axis=0))
print("Std deviation =", data_scaled.std(axis=0))

# Масштабування MinMax
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print("\nMin max scaled data:\n", data_scaled_minmax)

# Нормалізація даних
data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
print("\nl1 normalized data:\n", data_normalized_l1)
print("\nl2 normalized data:\n", data_normalized_l2)

