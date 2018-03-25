import math
import matplotlib.pyplot as plt
import numpy as np


class Simple2dDataSet:

    def __init__(self, n, m=1.0, c=0.0, f1=0, f2=0):
        self._n = n
        self._m = m
        self._c = c
        self._x, self._y = np.random.rand(2, n)
        self._class = []

        self._modify_clusters(f1, f2)
        self._assign_class()

    def _assign_class(self):
        for x, y in zip(self._x, self._y):
            self._class.append(0 if (y < self._fline(x)) else 1)
        self._class = np.array(self._class)

    def _modify_clusters(self, f1, f2):
        if f1:
            self._x = self._x - f1 * (self._x - 0.5)
            self._y = self._y - f1 * (self._y - 0.5)
        if f2:
            dy = self._y - self._fline(self._x)
            shifpoints = np.abs(dy) < f2
            self._y[shifpoints] += np.sign(dy[shifpoints]) * f2
        self._y[self._y > 1.0] = 1.0
        self._y[self._y < 0.0] = 0.0

    def _fline(self, x):
        return self._m*x + self._c

    def plot(self, fig_num=1, subplot=111, title="Simple 2D", show=True):

        def plot_class(ax, class_value, line_style):
            ax.plot(self._x[self._class == class_value], self._y[self._class == class_value], line_style)

        def plot_dividing_line(ax, line_style):
            ax.plot([0, 1], [self._fline(0), self._fline(1)], line_style)

        fig = plt.figure(fig_num)
        ax = fig.add_subplot(subplot)
        plot_class(ax, 0, 'ro')
        plot_class(ax, 1, 'bo')
        plot_dividing_line(ax, 'g--')
        ax.set_title(title)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        if show:
            plt.show()


class DataSet:

    def __init__(self, vectors, classes):
        self.vectors = vectors
        self.classes = classes

    @property
    def size(self):
        assert (len(self.vectors) == len(self.classes))
        return len(self.vectors)

    def __add__(self, other):
        vectors = np.vstack((self.vectors, other.vectors))
        classes = np.append(self.classes, other.classes)
        return DataSet(vectors, classes)


class Circular2dDataSet(DataSet):

    def __init__(self, n, centre, radius, cls_int):
        self._centre = centre
        self._radius = radius
        self.vectors = self._calculate_vectors(n)
        self.classes = cls_int * np.ones(n)

    @staticmethod
    def _polar_to_cartesian(r, theta, c=(0, 0)):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return (x + c[0], y + c[1])

    def _calculate_vectors(self, n):
        r = self._radius * np.random.rand(n)
        theta = 2.0 * math.pi * np.random.rand(n)
        x, y = self._polar_to_cartesian(r, theta, self._centre)
        return np.vstack((x, y)).T

    def line(self, steps=100):
        theta = np.linspace(0, 2.0 * math.pi, steps)
        return self._polar_to_cartesian(self._radius, theta, self._centre)



class KNearestNeighbours:

    def __init__(self, k=None):
        self._k = k
        self._training_data = None

    def fit(self, vectors, classes):
        self._training_data = DataSet(vectors, classes)

    def predict(self, vectors):
        k = self._k if self._k else self._training_data.size
        predicted = []
        for w in vectors:
            distance = [np.linalg.norm(w - w_) for w_ in self._training_data.vectors]
            sorted_indices = np.argsort(distance)
            k_nearest_classes = self._training_data.classes[sorted_indices][:k]
            predicted.append(np.round(np.mean(k_nearest_classes)))
        return predicted


class GaussianNaiveBayes:

    def __init__(self):
        self._gassian_means = []
        self._gassian_stds = []
        self._training_data = None

    def fit(self, vectors, classes):
        self._training_data = DataSet(vectors, classes)
        for kk in set(classes):  # Assumes classes are 0, 1, 2, 3
            for feature_values in vectors.T:
                



    def _mean(self):



# def plot_2d_data(vectors1, vectors2, line1=None, line2=None, fig_num=1, subplot_num=111, title = "", show=True):
#
#         fig = plt.figure(fig_num)
#         ax = fig.add_subplot(subplot_num)
#
#         x, y = vectors1.T
#         ax.plot(x, y, 'ro')
#         if line1 is not None:
#             x, y = line1
#             ax.plot(x, y, 'r--')
#         x, y = vectors2.T
#         ax.plot(x, y, 'bo')
#         if line2 is not None:
#             x, y = line2
#             ax.plot(x, y, 'b--')
#
#         ax.set_title(title)
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#
#         if show:
#             plt.show()


def plot_2d_dataset(ax, dataset, dots=['rs', 'bo']):
    x, y = dataset.vectors.T
    for cls, dot in zip(set(dataset.classes), dots):
        ax.plot(x[dataset.classes == cls], y[dataset.classes == cls], dot)


if __name__ == "__main__":

    #c1 = (0.50, 0.50)
    c1 = (0.40, 0.40)
    r1 = 0.15
    c2 = (0.65, 0.65)
    r2 = 0.18

    # Generate training dataset (100 points)
    dataset1 = Circular2dDataSet(n=50, centre=c1, radius=r1, cls_int=0)
    dataset2 = Circular2dDataSet(n=50, centre=c2, radius=r2, cls_int=1)
    training_dataset = dataset1 + dataset2

    # Generate testing dataset (20 points)
    dataset1 = Circular2dDataSet(n=50, centre=c1, radius=r1, cls_int=0)
    dataset2 = Circular2dDataSet(n=50, centre=c2, radius=r2, cls_int=1)
    testing_dataset = dataset1 + dataset2

    # Predict using K nearest neighbours
    clf = KNearestNeighbours(k=5)
    clf.fit(training_dataset.vectors, training_dataset.classes)
    predicted = clf.predict(testing_dataset.vectors)
    actual = testing_dataset.classes

    predicted_dataset = DataSet(testing_dataset.vectors, predicted)

    # TODO: asses

    # Plot training dataset
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    # Plot bounding circles
    x, y = dataset1.line()
    ax.plot(x, y, 'r--')
    x, y = dataset2.line()
    ax.plot(x, y, 'b--')
    # Plot training dataset
    plot_2d_dataset(ax, training_dataset)
    # Setup and show
    ax.set_title("Training")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    #plt.show()

    # Plot training dataset
    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    # Plot bounding circles
    x, y = dataset1.line()
    ax.plot(x, y, 'r--')
    x, y = dataset2.line()
    ax.plot(x, y, 'b--')
    # Plot training dataset
    for ii, (act, pred) in enumerate(zip(actual, predicted)):
        if act != pred:
            predicted_dataset.classes[ii] += 2
    plot_2d_dataset(ax, predicted_dataset, dots=['rs', 'bo', 'bx', 'rx'])
    # Setup and show
    ax.set_title("Precited")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.show()




