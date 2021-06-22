from matplotlib import pyplot as plt


def line(yAxis, xAxis, title="", ylabel="", xlabel="", legend=""):
    plt.plot(yAxis)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(legend, loc='upper left')
    plt.show()


def bar(yxAis, xAxis, title="", ylabel="", xlabel="", legend=""):
    plt.bar(yxAis, height=xAxis)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(legend, loc='upper left')
    plt.show()


if __name__ == '__main__':
    bar([1, 2, 3], ["1", "2", "3"])
