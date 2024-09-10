import numpy as np
import pandas as pd
from colorama import Fore


def to_2d(loc, dims):
    # loc is a one-dimensional index
    # NOTE: this indexes from the lower left corner
    i, j = loc // dims[1], loc % dims[1]
    return [i, j]


def to_1d(array, dims):
    # array is a two-dimensional index
    # NOTE: this indexes from the lower left corner
    loc = array[0] * dims[1] + array[1]
    return loc


def neighbors_of_four(dims, target):
    size = dims[0] * dims[1]
    my_points = np.arange(0, size, 1)
    coords = to_2d(my_points, dims)
    target = to_2d(target, dims)

    x_distance = 2 - (coords[1] - target[1]) / 12
    y_distance = 2 - (coords[0] - target[0]) / 12
    dist = (x_distance**2 + y_distance ** 2) ** 0.5
    coords = np.concatenate(
        (np.reshape(coords[0], (size, 1)), np.reshape(coords[1], (size, 1))), axis=1
    )
    # print(coords)
    # coords = []
    # for point in my_points:
    #     coords.append(to_2d(point, dims))  # todo: can this be a numpy meshgrid?
    #
    # print(coords)
    # note: i, j indexing -> i is the row and j is the column
    movements = [[0, -1], [0, 1], [-1, 0], [1, 0]]  # left, right, down, up

    # find all four neighbors
    left_neighbors = [
        [coords[i][0] + movements[0][0], coords[i][1] + movements[0][1]]
        for i in range(len(coords))
    ]
    right_neighbors = [
        [coords[i][0] + movements[1][0], coords[i][1] + movements[1][1]]
        for i in range(len(coords))
    ]
    down_neighbors = [
        [coords[i][0] + movements[2][0], coords[i][1] + movements[2][1]]
        for i in range(len(coords))
    ]
    up_neighbors = [
        [coords[i][0] + movements[3][0], coords[i][1] + movements[3][1]]
        for i in range(len(coords))
    ]

    # find if anything is out of bounds
    for i in range(len(left_neighbors)):
        if (0 <= left_neighbors[i][0] < dims[0]) & (
            0 <= left_neighbors[i][1] < dims[1]
        ):
            left_neighbors[i] = to_1d(left_neighbors[i], dims)
        else:
            left_neighbors[i] = max(my_points) + 1

        if (0 <= right_neighbors[i][0] < dims[0]) & (
            0 <= right_neighbors[i][1] < dims[1]
        ):
            right_neighbors[i] = to_1d(right_neighbors[i], dims)
        else:
            right_neighbors[i] = max(my_points) + 1

        if (0 <= down_neighbors[i][0] < dims[0]) & (
            0 <= down_neighbors[i][1] < dims[1]
        ):
            down_neighbors[i] = to_1d(down_neighbors[i], dims)
        else:
            down_neighbors[i] = max(my_points) + 1

        if (0 <= up_neighbors[i][0] < dims[0]) & (0 <= up_neighbors[i][1] < dims[1]):
            up_neighbors[i] = to_1d(up_neighbors[i], dims)
        else:
            up_neighbors[i] = max(my_points) + 1

    my_neighbors = pd.DataFrame(
        {
            "points": my_points,
            "left": left_neighbors,
            "right": right_neighbors,
            "up": up_neighbors,
            "down": down_neighbors,
            "dist": dist,
        }
    )
    return my_neighbors


def neighbors(vertex, dim):
    x_coord, y_coord = vertex

    neighbor_verts = []
    action = []
    # left neighbor
    if x_coord - 1 >= 0:
        neighbor_verts.append([x_coord - 1, y_coord])
        action.append(3)
    # right neighbor:
    if x_coord + 1 < dim[0]:
        neighbor_verts.append([x_coord + 1, y_coord])
        action.append(1)
    # lower neighbor
    if y_coord - 1 >= 0:
        neighbor_verts.append([x_coord, y_coord - 1])
        action.append(2)
    # upper neighbor:
    if y_coord + 1 < dim[1]:
        neighbor_verts.append([x_coord, y_coord + 1])
        action.append(0)

    return neighbor_verts, action


def neighbors_features(vertex, dim):
    x_coord, y_coord = vertex

    neighbor_verts = []
    # left neighbor
    if x_coord - 1 >= 0:
        neighbor_verts.append([x_coord - 1, y_coord])
    else:
        neighbor_verts.append([np.inf, np.inf])
    # right neighbor:
    if x_coord + 1 < dim[0]:
        neighbor_verts.append([x_coord + 1, y_coord])
    else:
        neighbor_verts.append([np.inf, np.inf])
    # lower neighbor
    if y_coord - 1 >= 0:
        neighbor_verts.append([x_coord, y_coord - 1])
    else:
        neighbor_verts.append([np.inf, np.inf])
    # upper neighbor:
    if y_coord + 1 < dim[1]:
        neighbor_verts.append([x_coord, y_coord + 1])
    else:
        neighbor_verts.append([np.inf, np.inf])

    return neighbor_verts


def coord_to_label(coords, dims=(25, 25)):
    labels = np.arange(1, dims[0] * dims[1] + 0.5, 1)
    labels = np.reshape(labels, dims)
    my_list = []
    for coord in coords:
        label = labels[coord[0], coord[1]]
        my_list.append(int(label))
    return my_list


def fixed_length_path(path, length):
    if len(path) > length:
        path = path[:length]
    # elif len(path) < 25:
    #     repetitions = 25 - len(path)
    #     for _ in range(repetitions):
    #         path.append(destination)
    return np.array(path)


def new_position(point, action, log):
    x_coord, y_coord = point
    success = True
    if action == 0:
        move_to = [x_coord, y_coord + 1]
    elif action == 1:
        move_to = [x_coord + 1, y_coord]
    elif action == 2:
        move_to = [x_coord, y_coord - 1]
    elif action == 3:
        move_to = [x_coord - 1, y_coord]
    else:
        raise Warning("Invalid action specified")

    for i, val in enumerate(move_to):
        if val > 24:
            # log.debug("Invalid Coordinate Specified: " + str(val))
            move_to[i] = 24
            success = False
        elif val < 0:
            # log.debug("Invalid Coordinate Specified: " + str(val))
            move_to[i] = 0
            success = False

    return move_to, success


class MyLogger:
    def __init__(self, logging=False, debug_msgs=False, show_msgs=False):
        self.logging_msgs = logging
        self.debug_msgs = debug_msgs
        self.show_msgs = show_msgs

    def _displayMessage(self, message, level=None, color=None):
        if level is not False:
            if color == "red":
                print(Fore.RED, level, message)
            elif color == "blue":
                print(Fore.BLUE, level, message)
            else:
                print(Fore.RESET, level, message)
        elif self.show_msgs is not False:
            print(message)

    def debug(self, message, color=None):
        if self.debug_msgs is not False:
            self._displayMessage(message, level="[DEBUG]: ", color=color)

    def info(self, message):
        if self.logging_msgs is not False:
            self._displayMessage(message, level="[INFO]: ")


if __name__ == "__main__":
    x = np.flip(np.reshape(np.arange(0, 25, 1), (5, 5)), axis=0)
    print(x)
    df = neighbors_of_four((5, 5), 24)
    print(6)
    print(to_2d(6, (5, 5)))
    print(to_1d([1, 1], (5, 5)))
    print(df)
    y = np.array([1, 3, 5])
    [a, b] = to_2d(np.array([1, 2]), (5, 5))
    print(type(a))
    print(b)
    # print(df.loc[y, "left"])
