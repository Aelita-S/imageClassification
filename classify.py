import gzip
import numpy as np
from pathlib import Path
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

from random import randint
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

data_dir = Path(__file__).resolve(strict=True).parent / "fashion-mnist" / "data" / "fashion"


class Classifier:
    def __init__(self, train_data, train_target_data, algorithm, **kwargs):
        """训练

        :param train_data: 训练集数据
        :param train_target_data: 训练集分类
        :param algorithm: 训练算法
        """
        self.clf = algorithm(**kwargs)
        self.clf.fit(train_data, train_target_data)
        print("训练集准确率: ", accuracy_score(train_target_data, self.clf.predict(train_data)))

    def classify(self, test_data, test_target_data):
        """测试

        :param test_data: 测试集数据
        :param test_target_data: 测试集分类
        :return: 训练集准确率、召回率、F1 Score、混淆矩阵
        """
        predicted_data = self.clf.predict(test_data)
        accuracy = accuracy_score(test_target_data, predicted_data)
        recall = recall_score(test_target_data, predicted_data, average='macro')
        f1 = f1_score(test_target_data, predicted_data, average='macro')
        confusion_mat = confusion_matrix(test_target_data, predicted_data)
        return accuracy, recall, f1, confusion_mat


def get_data(kind):
    labels_path, images_path = data_dir / f'{kind}-labels-idx1-ubyte.gz', data_dir / f'{kind}-images-idx3-ubyte.gz'
    with gzip.open(labels_path, 'rb') as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as f:
        images = np.frombuffer(f.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)
    return images, labels


train_images, train_labels = get_data('train')

test_images, test_labels = get_data('t10k')

if __name__ == '__main__':
    clf = Classifier(train_images, train_labels, SVC)
    print("SVC", clf.classify(test_images, test_labels))
