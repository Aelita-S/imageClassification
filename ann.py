import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras

from classify import Classifier, get_data, get_topk, timer, run

# import other library

# load train data
train_images, train_labels = get_data(kind='train')

# load test data
test_images, test_labels = get_data(kind='t10k')
train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images.reshape(-1, 28, 28)
test_images = test_images.reshape(-1, 28, 28)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

if __name__ == '__main__':
    clf = Classifier(train_labels, train_images, algorithm=model, is_object=True)
    train_history = clf.train(epochs=10)
    clf.train_predict(probability=False)
    clf.predict(test_labels, test_images, probability=False)

    # 绘制训练 & 验证的准确率值
    plt.plot(train_history.history['accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend('Train', loc='upper left')
    plt.show()
