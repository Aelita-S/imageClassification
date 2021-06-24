import gzip
import time
import os

from sklearn.decomposition import PCA

from charts import line

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
from skimage import feature

data_dir = Path(__file__).resolve(strict=True).parent / "fashion-mnist" / "data" / "fashion"


def timer(func):
    def deco(*args, **kwargs):
        print(f'\n函数 {func.__name__} 开始运行：')
        start_time = time.time()

        res = func(*args, **kwargs)
        cost = round(time.time() - start_time, 3)
        print(f'函数 {func.__name__} 运行了 {cost} 秒')
        return res, cost

    return deco


def lbp_texture(data: np.ndarray):
    """LBP特征提取"""
    radius = 1
    n_point = radius * 8
    images = data.reshape(-1, 28, 28)

    data_hist = np.zeros((len(images), 256))
    for idx, img in enumerate(images):
        # 使用skimage LBP方法提取图像的纹理特征
        lbp = feature.local_binary_pattern(img, n_point, radius)
        # hist size:256
        max_bins = int(lbp.max()) + 1
        data_hist[idx], _ = np.histogram(lbp, bins=max_bins, range=(0, max_bins))

    return data_hist


def get_pca(data: np.ndarray, n=2) -> PCA:
    pca = PCA(n_components=n)
    pca.fit(data)  # 训练PCA模型
    return pca


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

        self.name = kwargs.pop('name')

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

        cm = confusion_matrix(test_target_data, self.clf.predict(test_data))

        plot_confusion_matrix(cm, title=self.name)

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


def plot_confusion_matrix(cm, title='Confusion Matrix'):
    classes = [i for i in range(len(cm))]
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=5, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.show()


def run(classification, train_set, test_set):
    print("算法：", classification['algorithm'].__name__)
    run_train_images = train_set
    run_train_labels = train_labels
    if MODE == 'DEV':
        run_train_images = run_train_images[0: 500]
        run_train_labels = run_train_labels[0: 500]

    clf = Classifier(train_target_data=run_train_labels, train_data=run_train_images, **classification)
    clf.train()
    res = clf.predict(test_labels, test_set)
    print(res)
    print("\n\n")
    return res


if __name__ == '__main__':
    algorithms = {
        # "SVC": dict(algorithm=SVC, C=10, probability=True, name='SVC'),
        "KNN": dict(algorithm=KNeighborsClassifier, n_neighbors=10, n_jobs=8, name='KNN'),
        "GNB": dict(algorithm=GaussianNB, name='GNB')
    }
    # print("PCA维数：", 15)
    # pca = get_pca(train_images, 15)
    # X_train, X_test = pca.transform(train_images), pca.transform(test_images)
    X_train, X_test = lbp_texture(train_images), lbp_texture(test_images)
    for a in algorithms.values():
        run(a, X_train, X_test)
        print()
        run(a, train_images, test_images)

    # top1_scores = []
    # cost = []
    # for i in range(1, 16):
    #     print("PCA维数：", i)
    #     pca = get_pca(train_images, i)
    #
    #     res = run(algorithms['KNN'], pca.transform(train_images), pca.transform(test_images))
    #     top1_scores.append(res[0][0])
    #     cost.append(res[1])
    #     print()
    # res = run(algorithms['KNN'], train_images, test_images)
    # top1_scores.append(res[0][0])
    # cost.append(res[1])
    # ax = list(range(1, 16))
    # ax.append("784")
    # line(top1_scores, ax, title="Accuracy", ylabel="accuracy", xlabel="n_components")
    # line(cost, ax, title="Time cost", ylabel="s", xlabel="n_components")
