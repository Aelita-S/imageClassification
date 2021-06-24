import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras

from charts import line
from classify import Classifier, get_data, get_topk, timer, run

# import other library

# load train data
train_images, train_labels = get_data(kind='train')

# load test data
test_images, test_labels = get_data(kind='t10k')
train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

if __name__ == '__main__':
    clf = Classifier(train_labels, train_images, algorithm=model, is_object=True)
    train_history = clf.train(epochs=20)
    clf.train_predict(probability=False)
    clf.predict(test_labels, test_images, probability=False)

    # 绘制训练 & 验证的准确率值
    line(train_history.history['accuracy'], None, title='Model accuracy', ylabel='Accuracy', xlabel='Epoch', legend='Train')
