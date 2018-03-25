import csv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
import os
from sklearn import metrics
from sklearn import neural_network
from sklearn import svm
import warnings


SRCDIR = os.path.dirname(__file__)
DATADIR = os.path.join(SRCDIR, "data")


class GFEData:

    def __init__(self, user_label, class_label):
        self._time = []
        self._frame_data = []
        self._target_values = []
        self._processed_data1 = []
        self._processed_data2 = []
        self._user_label = user_label
        self._class_label = class_label
        self._key_length_indices = [( 2, 17), (10, 27), ( 2, 89), (10, 89),
                                    (48, 54), (39, 89), (44, 89), (57, 51),
                                    (17, 27), (39, 57), (44, 57)]
        a = 87 # 4
        b = 12 # 10
        self._key_angle_indices = [(48, 89, 54), (51, 54, 57), (51, 48, 57),
                                   (57, 54, 89), (57, 48, 89), ( b,  a, 27),
                                   (17,   b, a)]
        self._key_z_indices = [2, 4, 10, 12, 16, 17, 20, 26, 27, 30, 39, 44, 48, 51, 54, 57, 89]
        self._read_data()
        self._process_data()

    @property
    def nFrames(self):
        return len(self._target_values)

    @property
    def frame_data(self):
        return self._frame_data

    def target(self, window_size=1):
        return self._target_values[:self.nFrames - window_size + 1]

    def _window_data(self, data, window_size):
        windowed_data = []
        for ii in range(self.nFrames - window_size + 1):
            windowed_data.append(np.concatenate(self._processed_data1[ii:ii+window_size]))
        return windowed_data

    def data1(self, window_size=1):
        return self._window_data(self._processed_data1, window_size)

    def data2(self, window_size=1):
        return self._window_data(self._processed_data1, window_size)

    def data(self, representation_type, window_size=1):
        return [self.data1(window_size), self.data2(window_size)][representation_type - 1]

    def _read_data(self):
        # Read data points
        datapoints_file_path = os.path.join(DATADIR, "{}_{}_datapoints.txt".format(self._user_label, self._class_label))
        with open(datapoints_file_path, "r") as f:
            reader = csv.reader(f, delimiter=' ')
            next(reader)  # Skip header
            for row in reader:
                values = [float(s) for s in row]
                self._time.append(values.pop(0))
                frame_data = np.transpose(np.array(values).reshape(-1, 3))
                self._frame_data.append(frame_data)
        # Read target values
        targets_file_path = os.path.join(DATADIR, "{}_{}_targets.txt".format(self._user_label, self._class_label))
        with open(targets_file_path, "r") as f:
            for line in f:
                self._target_values.append(int(line))

    def _process_data(self):
        for frame_data in self._frame_data:
            x, y, z = frame_data
            # Key distance values
            key_lengths = []
            for i1, i2 in self._key_length_indices:
                a = np.array([x[i1], y[i1]])
                b = np.array([x[i2], y[i2]])
                length = np.linalg.norm(b - a)
                key_lengths.append(length)
            # Key angle values
            key_angles = []
            for i1, i2, i3 in self._key_angle_indices:
                # https://stackoverflow.com/questions/35176451/python-code-to-calculate-angle-between-three-point-using-their-3d-coordinates
                a = np.array([x[i1], y[i1]])
                b = np.array([x[i2], y[i2]])
                c = np.array([x[i3], y[i3]])
                ba = a - b
                bc = c - b
                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                angle = np.arccos(cosine_angle)
                key_angles.append(angle)
            # Key z values
            key_z_values = list(z[self._key_z_indices])
            # combine
            self._processed_data1.append(np.array(key_lengths + key_angles))
            self._processed_data2.append(np.array(key_lengths + key_angles + key_z_values))


class GFEDataPlotter:

    def __init__(self, gfe_data):
        self._gfe_data = gfe_data
        self._polygon_data = [
            (  8, True),
            ( 16, True),
            ( 26, True),
            ( 36, True),
            ( 48, False),
            ( 60, True),
            ( 68, True),
            ( 87, False),
            ( 90, True),
            ( 95, False),
            (100, False)]

    def _set_2d_ax_properties(self, ax):
        ax.set_xlim([255, 355])
        ax.set_ylim([-270, -160])
        ax.set_aspect("equal")
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")

    def _set_3d_ax_properties(self, ax):
        ax.set_xlabel("x (pixels)")
        ax.set_zlabel("y (pixels)")
        ax.set_ylabel("z (mm)")
        ax.view_init(elev=-166, azim=73)

    def _polylines(self, frame_number):
        x, y, z = self._gfe_data.frame_data[frame_number]
        y = -y  # invert in y
        llim = 0
        line_data = []
        for ii, (ulim, closed) in enumerate(self._polygon_data):
            x_, y_, z_ = x[llim:ulim], y[llim:ulim], z[llim:ulim]
            if closed:
                x_ = np.append(x_, x_[0])
                y_ = np.append(y_, y_[0])
                z_ = np.append(z_, z_[0])
            line_colour = "r" if self._gfe_data.target[frame_number] else "k"
            line_style = "--" if ii == 8 else "-"
            line_data.append((x_, y_, z_, line_style + line_colour))
            llim = ulim
        return line_data

    def plot2d(self, frame_number, subplot=111, fig_number=None, annotate=False, draw_polygons=False, color_lines=True, show_plot=False):
        x, y, z = self._gfe_data.frame_data[frame_number]
        y = -y  # invert in y
        fig = plt.figure(fig_number)
        ax = fig.add_subplot(subplot)
        ax.scatter(x, y)
        if draw_polygons:
            for x_, y_, z_, ls in self._polylines(frame_number):
                ax.plot(x_, y_, ls)
        if annotate:
            for ii, xy in enumerate(zip(x, y)):
                ax.annotate(str(ii), xy)
        self._set_2d_ax_properties(ax)
        if show_plot:
            plt.show()

    def plot3d(self, frame_number, subplot=111, fig_number=None, draw_polygons=False, color_lines=True, show_plot=False):
        x, y, z = self._gfe_data.frame_data[frame_number]
        y = y  # Invert in y
        z[z == 0.0] = None  # Remove z errors
        fig = plt.figure(fig_number)
        ax = fig.add_subplot(subplot, projection = '3d')
        ax.scatter(x, z, y)
        if draw_polygons:
            for x_, y_, z_, ls in self._polylines(frame_number):
                ax.plot(x_, z_, -y_, ls)
        self._set_3d_ax_properties(ax)
        if show_plot:
            plt.show()

    def animate2d(self, subplot=111, fig_number=None, show_plot=False):

        def init():
            for x_, y_, z_, ls in self._polylines(0):
                y_[:] = None
                ax.plot(x_, y_, ls)
            for ii, line in enumerate(ax.lines):
                lines.append(line)
            self._set_2d_ax_properties(ax)
            text.set_text("Frame: 0")
            return lines + [text,]

        def update(frame_number):
            global frame_counter
            for ii, (x_, y_, z_, ls) in enumerate(self._polylines(frame_number)):
                lines[ii].set_data(x_, y_)
                lines[ii].set_color(ls[-1])
            text.set_text("Frame: %i" % frame_number)
            return lines + [text,]

        fig = plt.figure(fig_number)
        ax = fig.add_subplot(subplot)
        empty_data = [[] for _ in self._polygon_data]
        lines = plt.plot(empty_data, empty_data, animated=True)
        text = ax.text(0.05, 0.95, '', horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
        ani = FuncAnimation(fig, update, frames=np.arange(self._gfe_data.nFrames), init_func=init, blit=True, interval=1000/27, repeat=False)
        if show_plot:
            plt.show()


def train_and_predict(clf, training_data, testing_data, write_report=True):
    frp_strings = []
    error_counter = 0
    for representation_type in [1, 2]:
        for window_size in [1, 5]:
            clf.fit(training_data.data(representation_type, window_size), training_data.target(window_size))
            predicted = clf.predict(testing_data.data(representation_type, window_size))
            actual = testing_data.target(window_size)
            while True:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    classification_report = metrics.classification_report(actual, predicted, target_names=["Affirmative", "Negative"])
                if write_report:
                    print(classification_report)
                    #print(metrics.confusion_matrix(actual, predicted, labels=[0, 1]))
                ###### affirm_line = classification_report.split('\n')[2]
                affirm_line = classification_report.split('\n')[2]
                p, r, f1 = affirm_line.split()[1:-1]
                if "0.00" in (p, r, f1):
                    if error_counter < 3:
                        error_counter += 1
                        continue
                    else:
                        p = r = f1 = "----"
                error_counter = 0
                break
            frp_strings.append("%s-%s-%s" % (f1, r, p))
    return frp_strings


if __name__=="__main__":

    gfe_labels = ["affirmative", "conditional", "doubt_question", "emphasis", "negative",
                  "relative", "topics", "wh_question", "yn_question"]


    # Plot
    # plotter = GFEDataPlotter(training_data)
    # plotter.plot2d(66, annotate=True, draw_polygons=True, show_plot=True)
    # plotter.plot3d(66, draw_polygons=True, show_plot=True)
    # plotter.animate2d(show_plot=True)

    # Sci-Kit Learn Support Vector Machine Classifier with user set gamma and C
    # clf = svm.SVC()

    # Sci-Kit Learn Multi-Layer Perceptron
    gfe_label = "emphasis"
    training_data = GFEData("a", gfe_label)
    testing_data = GFEData("b", gfe_label)
    print("\nAlpha variation in multi-layer perceptron (%s)" % gfe_label)
    print("\n| alpha             |       w = 1              w > 1       |       w = 1              w > 1       |")
    print(  "---------------------------------------------------------------------------------------------------")
    for alpha in np.logspace(-5, -1, 5):
        clf = neural_network.MLPClassifier(solver='lbfgs', alpha=alpha, hidden_layer_sizes=(10,))
        frp1, frp2, frp3, frp4 = train_and_predict(clf, training_data, testing_data, False)
        print("| {:<17} | {:>15}  {:>15} (5) | {:>15}  {:>15} (5) |".format(alpha, frp1, frp2, frp3, frp4))
    print()



    clf = neural_network.MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(10,))
    print("\nMulti-layer perceptron with lbfgs solver, 10 hidden layers and alpha = 1e-4")
    print("\n| GFE               |       w = 1              w > 1       |       w = 1              w > 1       |")
    print(  "---------------------------------------------------------------------------------------------------")
    for gfe_label in gfe_labels:
        training_data = GFEData("a", gfe_label)
        testing_data = GFEData("b", gfe_label)
        frp1, frp2, frp3, frp4 = train_and_predict(clf, training_data, testing_data, False)
        print("| {:<17} | {:>15}  {:>15} (5) | {:>15}  {:>15} (5) |".format(gfe_label, frp1, frp2, frp3, frp4))
    print()


    # Sci-Kit Learn Multi-Layer Perceptron
    gfe_label = "emphasis"
    training_data = GFEData("a", gfe_label)
    testing_data = GFEData("b", gfe_label)
    print("\nC variation in support vector machine (%s)" % gfe_label)
    print("\n| C                 |       w = 1              w > 1       |       w = 1              w > 1       |")
    print(  "---------------------------------------------------------------------------------------------------")
    for c_value in [1., 10., 100.]:
        clf = svm.SVC(C=c_value)
        frp1, frp2, frp3, frp4 = train_and_predict(clf, training_data, testing_data, False)
        print("| {:<17} | {:>15}  {:>15} (5) | {:>15}  {:>15} (5) |".format(c_value, frp1, frp2, frp3, frp4))
    print()


    # Sci-Kit Learn Support Vector Machine Classifier with user set gamma and C
    clf = svm.SVC(C=100.)
    print("\nMulti-layer perceptron with lbfgs solver, 10 hidden layers and alpha = 1e-4")
    print("\n| GFE               |       w = 1              w > 1       |       w = 1              w > 1       |")
    print(  "---------------------------------------------------------------------------------------------------")
    for gfe_label in gfe_labels:
        training_data = GFEData("a", gfe_label)
        testing_data = GFEData("b", gfe_label)
        frp1, frp2, frp3, frp4 = train_and_predict(clf, training_data, testing_data, False)
        print("| {:<17} | {:>15}  {:>15} (5) | {:>15}  {:>15} (5) |".format(gfe_label, frp1, frp2, frp3, frp4))
    print()