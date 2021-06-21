import tensorflow as tf
from tensorflow import keras

from classify import get_data, get_topk, timer

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


@timer
def run_fit(epochs=5):
    model.fit(train_images, train_labels, epochs=epochs)

    predicted_data_proba = model.predict(test_images)
    top1 = get_topk(target=test_labels, data_set=predicted_data_proba, k=1)
    top2 = get_topk(target=test_labels, data_set=predicted_data_proba, k=2)

    print(top1, top2)


if __name__ == '__main__':
    run_fit(100)
