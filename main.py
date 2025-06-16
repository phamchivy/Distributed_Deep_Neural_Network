import numpy as np
import csv
from tensorflow.keras.datasets import mnist

# Tải dữ liệu từ Keras
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Flatten ảnh từ 28x28 thành 784
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

def save_to_csv(images, labels, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for img, label in zip(images, labels):
            row = [label] + img.tolist()
            writer.writerow(row)

# Ghi ra file CSV
save_to_csv(x_train, y_train, 'mnist_train.csv')
save_to_csv(x_test, y_test, 'mnist_test.csv')
