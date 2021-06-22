import gzip
import time
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from pathlib import Path

from sklearn.metrics import recall_score, f1_score, confusion_matrix
from keras.metrics import sparse_top_k_categorical_accuracy

import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

data_dir = Path(__file__).resolve(strict=True).parent / "fashion-mnist" / "data" / "fashion"


def timer(func):
    def deco(*args, **kwargs):
        print(f'\n函数 {func.__name__} 开始运行：')
        start_time = time.time()

        res = func(*args, **kwargs)
        end_time = time.time()
        print(f'函数 {func.__name__} 运行了 {round(end_time - start_time, 3)}秒')
        return res

    return deco


class Classifier:
    def __init__(self, train_target_data, train_data, algorithm, is_object=False, **kwargs):
        """训练

        :param train_target_data: 训练集分类
        :param train_data: 训练集数据
        :param algorithm: 训练算法
        :param is_object: algorithm是否为对象（True则为类）
        """
        self.train_target_data = train_target_data
        self.train_data = train_data

        if is_object:
            self.clf = algorithm
        else:
            self.clf = algorithm(**kwargs)
        # self.train_predict()

    @timer
    def train(self, **kwargs):
        return self.clf.fit(self.train_data, self.train_target_data, **kwargs)

    @timer
    def train_predict(self, probability=True):
        """测试

        :param probability: 是否返回预测概率向量（机器学习算法）
        :return: 训练集Top1准确率、Top2准确率
        """
        if probability:
            train_predicted_data_proba = self.clf.predict_proba(self.train_data)
        else:
            train_predicted_data_proba = self.clf.predict(self.train_data)
        print("训练集top1准确率: ", get_topk(target=self.train_target_data, data_set=train_predicted_data_proba, k=1))
        print("训练集top2准确率: ", get_topk(target=self.train_target_data, data_set=train_predicted_data_proba, k=2))

    @timer
    def predict(self, test_target_data, test_data, probability=True):
        """测试

        :param probability: 是否返回预测概率向量（机器学习算法）
        :param test_target_data: 测试集分类
        :param test_data: 测试集数据
        :return: 训练集Top1准确率、Top2准确率、召回率、F1 Score、混淆矩阵
        """
        if probability:
            predicted_data_proba = self.clf.predict_proba(test_data)
        else:
            predicted_data_proba = self.clf.predict(test_data)

        top1 = get_topk(target=test_target_data, data_set=predicted_data_proba, k=1)
        top2 = get_topk(target=test_target_data, data_set=predicted_data_proba, k=2)
        print("测试集top1准确率: ", top1)
        print("测试集top2准确率: ", top2)

        return top1, top2


def get_topk(target, data_set, k=2):
    return np.count_nonzero(sparse_top_k_categorical_accuracy(target, data_set, k=k)) / len(target)


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

MODE = ''


def run(classification):
    print("算法：", classification['algorithm'].__name__)
    run_train_images = train_images
    run_train_labels = train_labels
    if MODE == 'DEV':
        run_train_images = run_train_images[0: 500]
        run_train_labels = run_train_labels[0: 500]

    clf = Classifier(train_target_data=run_train_labels, train_data=run_train_images, **classification)
    clf.train()
    print(clf.predict(test_labels, test_images))


if __name__ == '__main__':
    algorithms = {
        "SVC": dict(algorithm=SVC, C=10, probability=True, verbose=True),
        "KNN": dict(algorithm=KNeighborsClassifier, n_neighbors=10, n_jobs=8),
        "GNB": dict(algorithm=GaussianNB)
    }

    run(algorithms['KNN'])
