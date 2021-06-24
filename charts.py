from matplotlib import pyplot as plt


def line(yAxis, xAxis=None, title="", ylabel="", xlabel="", legend=None):
    if xAxis:
        plt.plot(xAxis, yAxis)
    else:
        plt.plot(yAxis)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if legend:
        plt.legend(legend, loc='upper left')
    plt.show()


def bar(yxAis, xAxis, title="", ylabel="", xlabel=""):
    for a, b in zip(xAxis, yxAis):
        plt.text(a, b + 0.02, '%.3f' % b, ha='center', va='bottom', fontsize=7)
    plt.bar(xAxis, height=yxAis)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()


if __name__ == '__main__':
    classification = ["SVC", "KNN(8 jobs)", "GNB", "ANN(20 Epochs)"]
    bar([0.9011, 0.8629, 0.5856, 0.8863], classification, title="Top1 accuracy", ylabel="accuracy", xlabel="class")  # Top1
    bar([0.9725, 0.9643, 0.9456, 0.9678], classification, title="Top2 accuracy", ylabel="accuracy", xlabel="class")  # Top2
    bar([1208, 0.003, 0.514, 75.125], classification, title="Time cost(Training)", ylabel="s", xlabel="class")  # 训练时间
    bar([101, 19.587, 0.705, 0.406], classification, title="Time cost(Test)", ylabel="s", xlabel="class")  # 测试集分类时间

