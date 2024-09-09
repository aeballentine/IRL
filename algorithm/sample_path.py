import sys
import torch
import pandas as pd
import numpy as np

from create_expert_demonstrations import find_optimal_path, create_feature_map
from IRL_utilities import neighbors_of_four, to_2d

sys.path.append("..")


def find_next_state(loc, my_action, features, my_neighbors):
    next_loc = my_neighbors.iloc[loc, my_action + 1]
    if next_loc == 624:
        terminated = False
        finished = True

    elif next_loc == 625:
        terminated = True
        finished = False

    else:
        terminated = False
        finished = False

    # formatting
    return terminated, finished, next_loc


# def create_dataframe(values, dims):
#     # values should be a one-dimensional vector
#     # dims is a tuple: number of x points by number of y points
#
#     # reshape the input vector. The input is listed from the lower left corner, left to right, and up
#     # reshape the input to the specified dimensions, and then flip the vector
#     values = np.flip(np.reshape(values, dims), axis=0)
#
#     # create the names of the columns and the rows. The columns will become the x-axis coordinates and the rows will
#     # become the y-axis
#     columns = np.round(np.linspace(-1, 1, dims[0]), 2)
#     indices = np.round(np.linspace(1, -1, dims[1]), 2)
#     ind = []
#     for x in indices:
#         if np.where(indices == x)[0][0] % 2 == 0:
#             ind.append(str(x))
#         else:
#             ind.append(" ")
#
#     cols = []
#     for x in columns:
#         if np.where(columns == x)[0][0] % 2 == 0:
#             cols.append(str(x))
#         else:
#             cols.append(" ")
#     # indices = [str(x) for x in indices]
#
#     # create a data frame with the x coordinates as the columns, the y coordinates as the indices, and the values as
#     # the input vector
#     value_map = pd.DataFrame(values, columns=cols, index=ind)
#
#     return value_map, values
#
#
# def feature_map(state, threat, target, dims=(25, 25)):
#     # want to return a torch tensor
#     # features: current threat and threat for each of the four neighboring cells, x- and y-distance to goal
#
#     my_threat = threat[state[0], state[1]]
#     features = [my_threat]
#     for neighbor in neighbors_features(state, dims):
#         if neighbor != [np.inf, np.inf]:
#             features.append(threat[neighbor[0], neighbor[1]])
#         else:
#             features.append(
#                 np.inf
#             )  # todo: might want to play around with this value a little
#     features.append(target[0] - state[0])  # x distance
#     features.append(target[0] - state[1])  # y distance
#
#     features = np.array(features, dtype=np.float32)
#
#     return features
#
#
# def neighbors_features(vertex, dim):
#     x_coord, y_coord = vertex
#
#     neighbor_verts = []
#     # left neighbor
#     if x_coord - 1 >= 0:
#         neighbor_verts.append([x_coord - 1, y_coord])
#     else:
#         neighbor_verts.append([np.inf, np.inf])
#     # right neighbor:
#     if x_coord + 1 < dim[0]:
#         neighbor_verts.append([x_coord + 1, y_coord])
#     else:
#         neighbor_verts.append([np.inf, np.inf])
#     # lower neighbor
#     if y_coord - 1 >= 0:
#         neighbor_verts.append([x_coord, y_coord - 1])
#     else:
#         neighbor_verts.append([np.inf, np.inf])
#     # upper neighbor:
#     if y_coord + 1 < dim[1]:
#         neighbor_verts.append([x_coord, y_coord + 1])
#     else:
#         neighbor_verts.append([np.inf, np.inf])
#
#     return neighbor_verts
#
#
# def new_position(point, action, log):
#     x_coord, y_coord = point
#     success = True
#     if action == 0:
#         move_to = [x_coord, y_coord + 1]
#     elif action == 1:
#         move_to = [x_coord + 1, y_coord]
#     elif action == 2:
#         move_to = [x_coord, y_coord - 1]
#     elif action == 3:
#         move_to = [x_coord - 1, y_coord]
#     else:
#         raise Warning("Invalid action specified")
#
#     for i, val in enumerate(move_to):
#         if val > 24:
#             # log.debug("Invalid Coordinate Specified: " + str(val))
#             move_to[i] = 24
#             success = False
#         elif val < 0:
#             # log.debug("Invalid Coordinate Specified: " + str(val))
#             move_to[i] = 0
#             success = False
#
#     return move_to, success


policy = torch.load("results/policy_model_SINGLE.pth")
data = pd.read_csv("cost_function/001.csv")
threat_field = data["Threat Intensity"].to_numpy()
threat_field = np.append(threat_field, np.inf)
value_function = data['Value'].to_numpy()
value_function = np.append(value_function, np.inf)
starting_coordinate = 598
final_coordinate = 624

neighbors = neighbors_of_four((25, 25), 624)
optimal_path, success = find_optimal_path(value_function, threat_field, starting_coordinate, final_coordinate, neighbors)
optimal_path = to_2d(np.array(optimal_path), (25, 25))
optimal_path = np.stack((optimal_path[0], optimal_path[1]), axis=1)
gen_path = []
my_loc = starting_coordinate

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

for _ in range(50):
    feature_map = create_feature_map(threat_field, neighbors)
    my_features = feature_map[my_loc]
    my_feature = torch.from_numpy(my_features).float().to(device)
    action = policy(my_feature).max(0).indices.cpu().numpy()
    # loc, my_action, features, my_neighbors
    term, done, new_loc = find_next_state(my_loc, action, feature_map, neighbors)
    gen_path.append(int(my_loc))
    my_loc = new_loc
    if term or done:
        break

gen_path = to_2d(np.array(gen_path),  (25, 25))
gen_path = np.stack((gen_path[0], gen_path[1]), axis=1)
print(gen_path)

print(optimal_path)
