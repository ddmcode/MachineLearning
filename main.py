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
        self._t = []
        self._frames = []
        self._lines = []
        self._polygon_data = [
            (8, True, ("-", "r")),
            (16, True, ("-", "r")),
            (26, True, ("-", "g")),
            (36, True, ("-", "g")),
            (48, False, ("-", "k")),
            (60, True, ("-", "m")),
            (68, True, ("-", "b")),
            (87, False, ("-", "k")),
            (90, True, ("--", "k")),
            (95, False, ("-", "r")),
            (100, False, ("-", "r"))
            ]
        self._read_files(user_labels[0], class_labels[0], z_correction)

    def _read_files(self, user_label, class_label, z_correction):
        datapoints_file_path = os.path.join(self._data_dir, "{}_{}_datapoints.txt".format(user_label, class_label))
        with open(datapoints_file_path, "r") as f:
            reader = csv.reader(f, delimiter=' ')
            next(reader)  # Skip header
            for row in reader:
                values = [float(s) for s in row]
                self._t.append(float(values.pop(0)))
                frame_data = np.transpose(np.array(values).reshape(-1, 3))
                if z_correction:
                    z = frame_data[2, :]
                    mean = np.mean(z)
                    z[z > (1 + z_correction) * mean] = None
                    z[z < (1 - z_correction) * mean] = None
                    frame_data[2, :] = z
                self._frames.append(frame_data)

    def print_frame_data(self, frame_number):
        x, y, z = self._frames[frame_number]
        print("{:5} {:20} {:20} {:20}".format(" ", "x", "y", "y"))
        for ii, (x, y, z) in enumerate(zip(x, y, z)):
            print("{5} {:20} {:20} {:20}".format(x, y, z))

    def plot2d(self, frame_number, subplot=None, fig_number=None, annotate=False, draw_polygons=False, color_lines=True, show_plot=False):
        x, y, z = self._frames[frame_number]
        y = -y  # invert in y
        fig = plt.figure(fig_number)
        ax = fig.add_subplot(subplot)
        ax.scatter(x, y)
        if draw_polygons:
            llim = 0
            for ulim, closed, ls in self._polygon_data:
                x_, y_ = x[llim:ulim], y[llim:ulim]
                if closed:
                    x_ = np.append(x_, x_[0])
                    y_ = np.append(y_, y_[0])
                llim = ulim
                line_style = ''.join(ls) if color_lines else ls[0]
                ax.plot(x_, y_, line_style)
        if annotate:
            for ii, xy in enumerate(zip(x, y)):
                ax.annotate(str(ii), xy)
        ax.set_xlim([260, 340])
        ax.set_ylim([-260, -180])
        ax.set_aspect("equal")
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
        if show_plot:
            plt.show()

    def plot3d(self, frame_number, subplot=None, fig_number=None, draw_polygons=False, color_lines=True, show_plot=False):
        x, y, z = self._frames[frame_number]
        y = -y  # invert in y
        y, z = z, y # swap y, z
        fig = plt.figure(fig_number)
        ax = fig.add_subplot(subplot, projection = '3d')
        ax.scatter(x, y, z)
        if draw_polygons:
            llim = 0
            for ulim, closed, ls in self._polygon_data:
                x_, y_, z_ = x[llim:ulim], y[llim:ulim], z[llim:ulim]
                if closed:
                    x_ = np.append(x_, x_[0])
                    y_ = np.append(y_, y_[0])
                    z_ = np.append(z_, z_[0])
                llim = ulim
                line_style = ''.join(ls) if color_lines else ls[0]
                ax.plot(x_, y_, z_, line_style)
        ax.set_xlim([265, 335])
        ax.set_zlim([-180, -260])
        ax.set_xlabel("x (pixels)")
        ax.set_zlabel("y (pixels)")
        ax.set_ylabel("z (mm)")
        ax.view_init(elev=-166, azim=73)
        if show_plot:
            plt.show()

    def animate2d(self, fig_number=None, show_plot=False):

        fig = plt.figure(fig_number)
        ax = fig.add_subplot(111)
        empty_data = [[] for _ in self._polygon_data]
        lines = plt.plot(empty_data, empty_data, animated=True)

        def init():
            x, y, z = self._frames[0]
            y = -y  # invert in y
            llim = 0
            for ulim, closed, ls in self._polygon_data:
                x_, y_ = x[llim:ulim], y[llim:ulim]
                if closed:
                    x_ = np.append(x_, x_[0])
                    y_ = np.append(y_, y_[0])
                llim = ulim
                y_[:] = None
                ax.plot(x_, y_, "k")
            ax.set_xlim([260, 340])
            ax.set_ylim([-260, -180])
            ax.set_aspect("equal")
            ax.set_xlabel("x (pixels)")
            ax.set_ylabel("y (pixels)")
            #print("len(lines): ", len(lines))
            for ii, line in enumerate(ax.lines):
                print("ii: ", ii)
                lines.append(line)
            return lines

        def update(iframe):
            x, y, z = self._frames[iframe]
            y = -y  # invert in y
            llim = 0
            for ii, (ulim, closed, _) in enumerate(self._polygon_data):
                x_, y_ = x[llim:ulim], y[llim:ulim]
                if closed:
                    x_ = np.append(x_, x_[0])
                    y_ = np.append(y_, y_[0])
                llim = ulim
                lines[ii].set_data(x_, y_)
            return lines
                #ax.plot(x_, y_, line_style[0] + "k")

        ani = FuncAnimation(fig, update, frames=np.arange(98), init_func=init, blit=True)
        plt.show()

    #def animate(self, annotate=False, show_z=False, show_plot=False):


if __name__=="__main__":

    src_dir = os.path.dirname(__file__)
    data_dir = os.path.join(src_dir, "data")
    user_labels = ("a", "b")
    class_labels = ("affirmative", "conditional", "doubth", "emphasis", "negative", "relative", "topics", "wh", "yn")

    user_data = UserData(user_labels, class_labels, data_dir, z_correction=0.1)
    user_data.plot2d(66, subplot=111, fig_number=3, annotate=True, draw_polygons=True, show_plot=False)
    user_data.plot3d(66, subplot=111, fig_number=4, draw_polygons=True, show_plot=False)
    user_data.animate2d(fig_number=5, show_plot=True)
