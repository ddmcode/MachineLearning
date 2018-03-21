import csv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
import os


class UserData:

    def __init__(self, user_label, class_label, data_dir, z_correction=None):
        self._user_label = user_label
        self._class_label = class_label

        self._time = []
        self._frames = []
        self._targets = []
        self._data1 = []
        self._data2 = []

        self._key_length_indices = [( 2, 17), (10, 27), ( 2, 89), (10, 89),
                                    (48, 54), (39, 89), (44, 89), (57, 51),
                                    (17, 27), (39, 57), (44, 57)]
        a = 87 # 4
        b = 12 # 10
        self._key_angle_indices = [(48, 89, 54), (51, 54, 57), (51, 48, 57),
                                   (57, 54, 89), (57, 48, 89), ( b,  a, 27),
                                   (17,   b, a)]
        self._key_z_indices = [2, 4, 10, 12, 16, 17, 20, 26, 27, 30, 39, 44, 48, 51, 54, 57, 89]

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

        self._read_files(data_dir, z_correction)
        self._extract_data()

    def _read_files(self, data_dir, z_correction):
        def remove_z_outliers(data):
            z = data[2, :]
            mean = np.mean(z)
            z[z > (1 + z_correction) * mean] = None
            z[z < (1 - z_correction) * mean] = None
            data[2, :] = z
        # Datapoints
        datapoints_file_path = os.path.join(data_dir, "{}_{}_datapoints.txt".format(self._user_label, self._class_label))
        with open(datapoints_file_path, "r") as f:
            reader = csv.reader(f, delimiter=' ')
            next(reader)  # Skip header
            for row in reader:
                values = [float(s) for s in row]
                self._time.append(values.pop(0))
                frame_data = np.transpose(np.array(values).reshape(-1, 3))
                if z_correction:
                    remove_z_outliers(frame_data)
                self._frames.append(frame_data)
        # Targets
        targets_file_path = os.path.join(data_dir, "{}_{}_targets.txt".format(self._user_label, self._class_label))
        with open(targets_file_path, "r") as f:
            for line in f:
                self._targets.append(int(line))

    def _extract_data(self):
        for frame_data in self._frames:
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
            self._data1.append(np.array(key_lengths + key_angles))
            self._data2.append(np.array(key_lengths + key_angles + key_z_values))

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
            line_colour = "r" if self._targets[frame_number] else "k"
            line_style = "--" if ii == 8 else "-"
            line_data.append((x_, y_, z_, line_style + line_colour))
            llim = ulim
        return line_data

    def _set_ax_properties(self, ax):
        ax.set_xlim([255, 355])
        ax.set_ylim([-270, -160])
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
                lines.append(line)
            self._set_ax_properties(ax)
            return lines

        def update(frame_number):
            for ii, (x_, y_, z_, ls) in enumerate(self._polylines(frame_number)):
                lines[ii].set_data(x_, y_)
                lines[ii].set_color(ls[-1])
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

    # user labels: "a", "b"
    # class labels: "affirmative", "conditional", "doubt_question", "emphasis", "negative", "relative", "topics", "wh", "yn"
    user_data = UserData("a", "doubt_question", data_dir, z_correction=0.1)

    # plot
    user_data.plot2d(66, annotate=True, draw_polygons=True, show_plot=True)
    user_data.plot3d(66, draw_polygons=True, show_plot=True)
    user_data.animate2d(show_plot=True)
