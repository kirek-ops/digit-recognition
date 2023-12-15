import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout

import csv
import json
import numpy as np

(x_train, y_train), (x_test, y_test) = data = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

additional_data_path = 'feedback_data.csv'
additional_x_train = []
additional_y_train = []

with open(additional_data_path, 'r') as csvfile:
    csv_reader = csv.reader(csvfile)

    header = next(csv_reader)
    print(f'Now read: {header}')

    for row in csv_reader:
        if row:
            features = json.loads(row[0])
            label = int(row[1])
            additional_x_train.append(features)
            additional_y_train.append(label)
        else:
            print("Something wrong with file")

additional_x_train = np.array(additional_x_train)
additional_y_train = np.array(additional_y_train)

additional_x_train = additional_x_train.reshape(additional_x_train.shape[0], 28, 28)

x_train = np.concatenate((x_train, additional_x_train))
y_train = np.concatenate((y_train, additional_y_train))

model = Sequential([
    Flatten(input_shape = (28, 28)),
    Dense(128, activation = 'relu'),
    Dense(64, activation = 'relu'),
    Dense(10, activation = 'softmax'),
])

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 3)  

model.save('digits_recognition.model')
