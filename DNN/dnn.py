import ast
import matplotlib.pyplot as plt
import keras
import pandas as pd
import tensorflow as tf

# Helper libraries
import numpy as np

from keras import backend as K
from matplotlib import pyplot

from sklearn.model_selection import train_test_split
import pickle

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


file_dir = "D:\\SDN Project\\Data\\"#"D:\\SDN Project\\"


names = ['protocol','src_ip' , 'src_port', 'dst_ip', 'dst_port', 'ndpi_proto_num', 'src2dst_packets',
        'src2dst_bytes', 'dst2src_packets', 'dst2src_bytes', 'ndpi_proto', 'class']
df = pd.read_csv(file_dir + 'total_class.csv', names=names)
array = df.values

X = np.asarray(df[['protocol', 'src_port', 'dst_port', 'src2dst_packets', 'src2dst_bytes', 'dst2src_packets','dst2src_bytes']][1:])
y = []

my_tags = []
classes = open("dnn.txt", "w+")
for i in df['class'][1:]:
    if i not in my_tags:
        my_tags.append(i)
for i in df['class'][1:]:
    classes.write(i+" "+str(my_tags.index(i))+"\n")
    y.append(my_tags.index(i))
y = np.asarray(y)
print(X.shape, y.shape)


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)
features = len(x_train[0])

model = keras.Sequential([
    keras.layers.Dense(features, kernel_regularizer=tf.keras.regularizers.l1(0.1)),
    keras.layers.Dense(512, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.1)),
    keras.layers.Dense(256, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l1(0.1)),
    keras.layers.Dense(128, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.1)),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['acc', f1_m, precision_m, recall_m])

history = model.fit(x_train, y_train, batch_size=1000, epochs=400)
loss, accuracy, f1_score, precision, recall = model.evaluate(x_test, np.array(y_test), verbose=0)
model.save("dnn_model1.sav")
y_predict = model.predict(x_test)

print("Saving the model")
filename = 'dnn_model.sav'
pickle.dump(model, open(filename, 'wb'))

for i in y_predict:
    print(np.argmax(i))
print(f'loss: {loss}, acc: {accuracy}, f1_score: {f1_score}, precision: {precision}, recall: {recall}')
print(model.summary())
print(x_test[0])
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['acc'], label='train')
pyplot.legend()
pyplot.show()
