import numpy as np
import copy
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from IRL_utilities import neighbors_of_four


class DQN(nn.Module):
    """
    Deep Q-Learning network
    For this application, using a single linear layer
    """

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)


policy_net = torch.load('policy_net_gamma_06.pth', weights_only=False)
neighbors = neighbors_of_four(dims=(25, 25), target=624)


def find_feature_expectation(feature_function):
    starting_coords = np.arange(0, 624, 1)
    n_observations = 5
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    path_length = 30

    n_threats = 1
    feature_function = feature_function.view(-1, n_observations)

    coords = np.tile(starting_coords, n_threats)
    coords_conv = np.repeat(626 * np.arange(0, n_threats, 1), len(starting_coords))
    # 626 because we've added a 626th row to the feature function for outside the boundary

    my_features = (
        feature_function[coords + coords_conv]
    )  # features at all the starting coordinates
    new_features = copy.deepcopy(my_features).to(device)  # feature values to use to decide each action
    # my_features = my_features[:, :4].view(-1, 4).to(device)

    errors = 0
    rewards_vec = []
    min_threat_rewards = []

    for coord in coords:
        # for coord in [0, 1]:
        features = feature_function[coord].to(device)
        nn_reward = 0
        my_reward = 0

        new_coord = copy.deepcopy(coord)
        new_feat = feature_function[coord].to(device)
        # log.debug("Starting coordinate:  \t" + str(coord))
        # print("Starting features: \t", new_feat)

        i = 0

        for step in range(path_length - 1):

            i += 1

            with torch.no_grad():
                action = (
                    policy_net(new_feat).max(0).indices.clone().detach().cpu().numpy()
                )  # this should be max(1) for multi-threat

            # print("Neural network-chosen action: \t", action)
            # neural network rewards
            new_coord = neighbors.iloc[new_coord, int(action)]

            if new_coord == 625:
                break
            # print("Neural network next coordinate: \t", new_coord)
            new_feat = feature_function[new_coord].to(device)

            # print("Neural network next feature: \t", new_feat)
            reward = 10 - new_feat[0]
            nn_reward += reward.cpu().numpy()

            # my rewards: moving only toward the minimum threat
            my_action = feature_function[coord].min(0).indices
            # print("Action according to the minimum threat: \t", action)
            coord = neighbors.iloc[coord, int(my_action)]
            # print("My new coordinate: \t", coord)
            # print("New feature function: \t", feature_function[coord])
            reward = 10 - feature_function[coord, 0]
            my_reward += reward.cpu().numpy()
            #
            # if action == 0:
            #     # print(coord)
            #     # print(nn_reward)
            #     break
        rewards_vec.append(nn_reward / i)
        min_threat_rewards.append(my_reward / i)

    # print(rewards_vec)
    # print(min_threat_rewards)

    plt.scatter(coords, rewards_vec, label='Neural Network', marker='s')
    plt.scatter(coords, min_threat_rewards, label='Greedy Algorithm', marker='+')
    plt.legend()
    plt.title(r'$\gamma$=0.6')
    plt.xlabel('Starting Coordinate')
    plt.ylabel(r'$R_{avg}$')
    plt.show()

    rewards_vec = np.array(rewards_vec)
    min_threat_rewards = np.array(min_threat_rewards)
    plt.scatter(coords, rewards_vec - min_threat_rewards)
    plt.title(r'$\gamma$=0.6')
    plt.xlabel('Starting Coordinate')
    plt.ylabel(r'$R_{avg, NN} - R_{avg, greedy}$')
    plt.show()


if __name__ == "__main__":
    data = pd.read_pickle('multi_threat(6).pkl')
    feature_function_ = data.feature_map
    feature_function_ = np.reshape(feature_function_[0][:, [0, 2, 4, 6, 8]], (1, 626, 5))
    find_feature_expectation(feature_function=torch.from_numpy(feature_function_).float().abs())
