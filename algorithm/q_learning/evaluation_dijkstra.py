import torch
from torch import nn
import numpy as np
import pandas as pd
# from torchviz import make_dot
import torch.nn.functional as func
import matplotlib.pyplot as plt
import matplotlib
# import seaborn as sns

from dijkstra import dijkstra
from IRL_utilities import neighbors_of_four


def to_2d(loc, dims):
    # loc is a one-dimensional index
    # NOTE: this indexes from the lower left corner
    i, j = loc // dims[1], loc % dims[1]
    return [i, j]


def create_dataframe(values, dims):
    # values should be a one-dimensional vector
    # dims is a tuple: number of x points by number of y points

    # reshape the input vector. The input is listed from the lower left corner, left to right, and up
    # reshape the input to the specified dimensions, and then flip the vector
    values = np.flip(np.reshape(values, dims), axis=0)

    # create the names of the columns and the rows. The columns will become the x-axis coordinates and the rows will
    # become the y-axis
    columns = np.round(np.linspace(-1, 1, dims[0]), 2)
    indices = np.round(np.linspace(1, -1, dims[1]), 2)
    ind = []
    for x in indices:
        if np.where(indices == x)[0][0] % 2 == 0:
            ind.append(str(x))
        else:
            ind.append(" ")

    cols = []
    for x in columns:
        if np.where(columns == x)[0][0] % 2 == 0:
            cols.append(str(x))
        else:
            cols.append(" ")
    # indices = [str(x) for x in indices]

    # create a data frame with the x coordinates as the columns, the y coordinates as the indices, and the values as
    # the input vector
    value_map = pd.DataFrame(values, columns=cols, index=ind)

    return value_map, values


# for the only remaining valid networks
class DQN(nn.Module):
    """
    Deep Q-Learning network
    For this application, using a single linear layer
    """

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 625)
        self.layer3 = nn.Linear(625, 128)
        self.layer4 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


def find_nn_path(feature_function, starting_coord):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    feature_function = feature_function.view(-1, 20)
    new_coord = starting_coord
    new_feat = feature_function[new_coord].to(device)
    path = [new_coord]

    success = True
    left = False
    for step in range(100):
        with torch.no_grad():
            action = (
                policy_net(new_feat).max(0).indices.clone().detach().cpu().numpy()
            )  # this should be max(1) for multi-threat
        # print("Neural network-chosen action: \t", action)
        # neural network rewards
        new_coord = neighbors.iloc[new_coord, int(action) + 1]
        path.append(int(new_coord))

        if new_coord == 625:
            success = False
            left = True
            # print('terminated outside the graph')
            break

        elif new_coord == 624:
            # print("Success!")
            break

        # print("Neural network next coordinate: \t", new_coord)
        new_feat = feature_function[new_coord].to(device)

        if step == 99:
            success = False
            # print('failed: path - \t', new_coord, feature_function[new_coord][0])
            # print(' ')

    return path, success, left, new_coord, feature_function[new_coord][0] + 2 * feature_function[new_coord][1]


if __name__ == "__main__":
    # neural network
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    policy_net = torch.load('policy_model_final_sample_diff.pth', weights_only=False, map_location=device)
    neighbors = neighbors_of_four(dims=(25, 25), target=624)

    data = pd.read_pickle('single_threat_Bayesian_2.pkl')
    feature_function_ = data.feature_map
    feature_function_ = np.reshape(feature_function_[0], (626, 20))

    # for Dijkstra's algorithm
    vertices = np.arange(0, 625, 1)
    threat_field_ = data.threat_field.to_numpy()[0]

    # chose starting coordinates
    starting_coords = np.arange(0, 624, 1)

    # vectors to hold results
    average_discrepancy = []
    dijkstra_average = []
    nn_average = []

    # path length
    dijkstra_length = []
    nn_length = []

    # number of failures of the neural network
    n_failures = 0
    n_departures = 0

    failed_loc = []
    failed_threat = []

    for coord in starting_coords:
        # call dijsktras and the nn
        nn_path, info, outside, fail_loc, fail_val = find_nn_path(
            feature_function=torch.from_numpy(feature_function_).float(),
            starting_coord=int(coord))
        dijkstra_info, counter = dijkstra(feature_function=feature_function_, vertices=vertices, source=int(coord),
                                          node_f=624, neighbors=neighbors)

        # if the network fails, terminate the loop
        if info is False:
            if outside is True:
                n_departures += 1
            else:
                n_failures += 1
                failed_loc.append(fail_loc)
                failed_threat.append(fail_val)
            continue

        # recover the Dijkstra algorithm path
        node = 624
        dijkstra_path = [node]
        while node != coord:
            previous_node = dijkstra_info[node].parent
            dijkstra_path.append(previous_node)
            node = previous_node
        dijkstra_path = dijkstra_path[::-1]

        # determine the cost of the neural network path and of Dijkstra's algorithm
        dijkstra_cost = 0
        for node in dijkstra_path[1:]:
            # dijkstra_cost += feature_function_[node][0] + 2 * feature_function_[node][1]
            dijkstra_cost += feature_function_[node][0]
            # if node == 624:
            #     dijkstra_cost += 10

        nn_cost = 0
        for node in nn_path[1:]:
            # nn_cost += feature_function_[node][0] + 2 * feature_function_[node][1]
            nn_cost += feature_function_[node][0]
            # if node == 624:
            #     nn_cost += 10

        # add our values to keep track of them
        nn_length.append(len(nn_path))
        dijkstra_length.append(len(dijkstra_path))
        average_discrepancy.append(float(dijkstra_cost - nn_cost))
        dijkstra_average.append(float(dijkstra_cost))
        nn_average.append(float(nn_cost))

        # compare the NN path and the Dijkstra's path
        # costs = feature_function_[:-1, 0] + 2 * feature_function_[:-1, 1]
        # threat_field, _ = create_dataframe(costs, (25, 25))
        #
        # nn_2d = to_2d(np.array(nn_path), (25, 25))
        # dijkstra_2d = to_2d(np.array(dijkstra_path), (25, 25))
        # sns.heatmap(threat_field, cbar=False, annot=True, fmt='g')
        # plt.plot(np.array(nn_2d[1]) + 0.5, 24 - np.array(nn_2d[0]) + 0.5)
        # plt.plot(np.array(dijkstra_2d[1]) + 0.5, 24 - np.array(dijkstra_2d[0]) + 0.5)
        # plt.show()

    print('Total number of failures: \t', n_failures)
    print('Final location if the algorithm failed: \t', np.unique(np.array(failed_loc)))
    print('Value at the final point if the algorithm failed: \t', np.unique(np.array(failed_threat)))
    print('Number of times the algorithm left the graph: \t', n_departures)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams['font.size'] = 18

    # mean and standard deviation
    error = -np.array(dijkstra_average) + np.array(nn_average)
    percent_error = error / np.array(dijkstra_average)
    mean = np.round(np.mean(100 * percent_error), 3)
    std = np.round(np.std(100 * percent_error), 3)
    # plot the difference between the neural network and Dijkstra's algorithm (positive means Dijkstra's won)
    plt.scatter(np.array(dijkstra_length), percent_error * 100, c='tab:blue', s=10)
    plt.xlabel('Optimal Path Length', fontdict={'size': 20})
    plt.ylabel(r'Percent Error - $J_{NN}$ and $J^*$', fontdict={'size': 20})
    plt.ylim([-0.1, 100])
    plt.xlim([0, 60])

    plt.annotate(r'$\bar{x}$: ' + str(mean), xy=(2, 95), horizontalalignment='left', verticalalignment='top',
                 fontsize=15)
    plt.annotate(r'$\sigma$: ' + str(std), xy=(2, 90), horizontalalignment='left', verticalalignment='top',
                 fontsize=15)
    plt.annotate(r'Non-Convergent Paths: ' + str(n_failures), xy=(2, 85), horizontalalignment='left',
                 verticalalignment='top',
                 fontsize=15)

    plt.show()

    # plt.scatter(dijkstra_length, np.array(dijkstra_length) - np.array(nn_length))
    # plt.show()

    # find mean and standard deviation
    print('mean = ', np.mean(error))
    print('sd = ', np.std(error))
