import csv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
import os
from pprint import pprint


class UserData:

    def __init__(self, user_labels, class_labels, data_dir, z_correction=None):
        self._user_labels = user_labels
        self._class_labels = class_labels
        self._data_dir = data_dir
        self._time = []
        self._frames = []
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
            (100, False),
            ]
        self._read_files(user_labels[0], class_labels[3], z_correction)

    def _read_files(self, user_label, class_label, z_correction):
        datapoints_file_path = os.path.join(self._data_dir, "{}_{}_datapoints.txt".format(user_label, class_label))
        with open(datapoints_file_path, "r") as f:
            reader = csv.reader(f, delimiter=' ')
            next(reader)  # Skip header
            for row in reader:
                values = [float(s) for s in row]
                self._time.append(float(values.pop(0)))
                frame_data = np.transpose(np.array(values).reshape(-1, 3))
                if z_correction:
                    z = frame_data[2, :]
                    mean = np.mean(z)
                    z[z > (1 + z_correction) * mean] = None
                    z[z < (1 - z_correction) * mean] = None
                    frame_data[2, :] = z
                self._frames.append(frame_data)

    def _polylines(self, frame_number):
        x, y, z = self._frames[frame_number]
        y = -y  # invert in y
        llim = 0
        line_data = []
        for ii, (ulim, closed) in enumerate(self._polygon_data):
            x_, y_, z_ = x[llim:ulim], y[llim:ulim], z[llim:ulim]
            if closed:
                x_ = np.append(x_, x_[0])
                y_ = np.append(y_, y_[0])
                z_ = np.append(z_, z_[0])
            line_style = "--k" if ii == 8 else "k"
            line_data.append((x_, y_, z_, line_style))
            llim = ulim
        return line_data

    def _set_ax_properties(self, ax):
        ax.set_xlim([255, 345])
        ax.set_ylim([-260, -160])
        ax.set_aspect("equal")
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")

    def print_frame_data(self, frame_number):
        x, y, z = self._frames[frame_number]
        print("{:5} {:20} {:20} {:20}".format(" ", "x", "y", "y"))
        for ii, (x, y, z) in enumerate(zip(x, y, z)):
            print("{5} {:20} {:20} {:20}".format(x, y, z))

    def plot2d(self, frame_number, subplot=111, fig_number=None, annotate=False, draw_polygons=False, color_lines=True, show_plot=False):
        x, y, z = self._frames[frame_number]
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
        self._set_ax_properties(ax)
        if show_plot:
            plt.show()

    def plot3d(self, frame_number, subplot=111, fig_number=None, draw_polygons=False, color_lines=True, show_plot=False):
        x, y, z = self._frames[frame_number]
        y = y  # invert in y
        fig = plt.figure(fig_number)
        ax = fig.add_subplot(subplot, projection = '3d')
        ax.scatter(x, z, y)
        if draw_polygons:
            for x_, y_, z_, ls in self._polylines(frame_number):
                ax.plot(x_, z_, -y_, ls)
        ax.set_xlabel("x (pixels)")
        ax.set_zlabel("y (pixels)")
        ax.set_ylabel("z (mm)")
        ax.view_init(elev=-166, azim=73)
        if show_plot:
            plt.show()

    def animate2d(self, subplot=111, fig_number=None, show_plot=False):

        def init():
            for x_, y_, z_, ls in self._polylines(0):
                y_[:] = None
                ax.plot(x_, y_, ls)
            for ii, line in enumerate(ax.lines):
                print("ii: ", ii)
                lines.append(line)
            self._set_ax_properties(ax)
            return lines

        def update(frame_number):
            for ii, (x_, y_, z_, ls) in enumerate(self._polylines(frame_number)):
                lines[ii].set_data(x_, y_)
            return lines

        fig = plt.figure(fig_number)
        ax = fig.add_subplot(subplot)
        empty_data = [[] for _ in self._polygon_data]
        lines = plt.plot(empty_data, empty_data, animated=True)
        ani = FuncAnimation(fig, update, frames=np.arange(100), init_func=init, blit=True, interval=75)
        if show_plot:
            plt.show()


if __name__=="__main__":

    src_dir = os.path.dirname(__file__)
    data_dir = os.path.join(src_dir, "data")
    user_labels = ("a", "b")
    class_labels = ("affirmative", "conditional", "doubth", "emphasis", "negative", "relative", "topics", "wh", "yn")

    user_data = UserData(user_labels, class_labels, data_dir, z_correction=0.1)

    user_data.plot2d(66, annotate=True, draw_polygons=True)
    user_data.plot3d(66, draw_polygons=True)
    user_data.animate2d(show_plot=True)
