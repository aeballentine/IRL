import numpy as np
import pandas as pd


def optimal_path(
    value_function,
    threat,
    start_index,
    end_index,
    eightby=False,
    offset=0,
    dim=(25, 25),
    disp=True,
    max_points=500,
):
    # value function should be a numpy array with dimensions as specified
    # start and end index should be an array. The indices reference the value function (2D array) -> [row, column]
    ith = start_index[0]
    jth = start_index[1]

    # does the path converge?
    success = True

    # value function of the starting point (use this to check the total cost of the path)
    value = value_function[ith, jth]

    # this will be updated throughout the function. There is no incurred cost from the starting location
    total_cost = 0

    # track the path, but keep track of [x, y] coordinates. For consistency, [0, 0] is the upper left corner of the plot
    # and [0, 25] is the lower left if using a Seaborn heatmap. The column is the x coordinate and the row is the
    # y coordinate. This is reversed from indexing into the value function and threat field
    # Offset is an optional input if adding a buffer around the edge of the graph,
    # or shifting to center the plot (Seaborn)
    path = pd.DataFrame(
        {
            "x_index": [jth + offset],
            "y_index": [ith + offset],
            "x_val": [-1 + jth / 12],
            "y_val": [1 - ith / 12],
        }
    )

    # track the path so far...make sure to not double back
    path_tracker = [str(ith) + " " + str(jth)]

    while ith != end_index[0] or jth != end_index[1]:
        adjacent = neighbors([ith, jth], dim, eightby)
        mini = float("Inf")
        cost = float("Inf")
        for coords in adjacent:
            coord_string = str(coords[0]) + " " + str(coords[1])
            if not any(coord_string == x for x in path_tracker):
                if value_function[coords[0], coords[1]] < mini:
                    mini = value_function[coords[0], coords[1]]
                    cost = threat[coords[0], coords[1]]
                    ith = coords[0]
                    jth = coords[1]
                if value_function[coords[0], coords[1]] == mini:
                    if threat[coords[0], coords[1]] < cost:
                        cost = threat[coords[0], coords[1]]
                        ith = coords[0]
                        jth = coords[1]
            # else:
            # print("repeat value")

        # path_tracker.append(str(ith) + " " + str(jth))

        new_val = pd.DataFrame(
            {
                "x_index": [jth + offset],
                "y_index": [ith + offset],
                "x_val": [-1 + jth / 12],
                "y_val": [1 - ith / 12],
            }
        )
        path = pd.concat([path, new_val], ignore_index=True)
        total_cost += threat[ith, jth]
        if len(path["x_index"]) > max_points:
            success = False
            if disp:
                print("Error - path failed")
            break
    if disp:
        print("The final value function: ", np.round(value_function[ith, jth], 3))
        print("The final coordinate:", ith, jth)
        print("The value function from the initial location:", np.round(value, 3))
        print(
            "The value function adding the cost along the way:", np.round(total_cost, 3)
        )
    return path, total_cost, success


def neighbors(vertex, dim, eightby=False):
    x_coord, y_coord = vertex

    if eightby:
        # find all possible x and y coordinates
        x_poss = np.array([x_coord - 1, x_coord, x_coord + 1])
        x_poss = x_poss[(x_poss >= 0) & (x_poss < dim[0])]
        y_poss = np.array([y_coord - 1, y_coord, y_coord + 1])
        y_poss = y_poss[(y_poss >= 0) & (y_poss < dim[0])]
        # combine these coordinates in all ways
        neighbor_verts = [[x, y] for x in x_poss for y in y_poss]
        neighbor_verts = [x for x in neighbor_verts if x != [x_coord, y_coord]]
    else:
        neighbor_verts = []
        # left neighbor
        if x_coord - 1 >= 0:
            neighbor_verts.append([x_coord - 1, y_coord])
        # right neighbor:
        if x_coord + 1 < dim[0]:
            neighbor_verts.append([x_coord + 1, y_coord])
        # lower neighbor
        if y_coord - 1 >= 0:
            neighbor_verts.append([x_coord, y_coord - 1])
        # upper neighbor:
        if y_coord + 1 < dim[1]:
            neighbor_verts.append([x_coord, y_coord + 1])

    return neighbor_verts
