"""
This file provides the helper functions for path_to_value.py (inverse reinforcement learning)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
from IRL_utilities import to_2d
from IRL_utilities import neighbors_features


# todo: check this against a real threat field
def feature_avg(state, threat, target, neighbor_coords, dims=(25, 25), gamma=0.99):
    # features: current threat and threat for each of the four neighboring cells, x- and y-distance to goal
    # NOTE: state and threat should be numpy arrays; target should be an integer; neighbor_coords is a dataframe
    # todo: want a second one of these that outputs a torch tensor...it'll make it easier for Q learning

    discount_factor = list(map(lambda x: pow(gamma, x), range(len(state))))

    my_threat = np.sum(discount_factor * threat[state])

    # avg of left neighbors
    left_neighbors = neighbor_coords.loc[state, "left"].to_numpy()
    left_threat = np.sum(discount_factor * threat[left_neighbors])

    # avg of right neighbors
    right_neighbors = neighbor_coords.loc[state, "right"].to_numpy()
    right_threat = np.sum(discount_factor * threat[right_neighbors])

    # avg of up neighbors
    up_neighbors = neighbor_coords.loc[state, "up"].to_numpy()
    up_threat = np.sum(discount_factor * threat[up_neighbors])

    # avg of down neighbors
    down_neighbors = neighbor_coords.loc[state, "down"].to_numpy()
    down_threat = np.sum(discount_factor * threat[down_neighbors])

    # x distance
    x_distance = np.sum(
        discount_factor * neighbor_coords.loc[state, "x_dist"].to_numpy()
    )
    # y distance
    y_distance = np.sum(
        discount_factor * neighbor_coords.loc[state, "y_dist"].to_numpy()
    )

    features = np.array(
        [
            my_threat,
            left_threat,
            right_threat,
            up_threat,
            down_threat,
            x_distance,
            y_distance,
        ],
        dtype=np.float32,
    )

    return features


class RewardFunction(nn.Module):
    """
    Assuming the reward function is a linear combination of the features
    R(s) = w^T * phi(s)
    """

    def __init__(self, feature_dim):
        super(RewardFunction, self).__init__()
        # initialize the weights as zeros
        # self.weights = nn.Parameter(torch.zeros(feature_dim))
        self.layer1 = nn.Linear(feature_dim, 1)

    def forward(self, features):
        # return the feature tensor
        # return the anticipated reward function
        # using matmul: does the dot product if one arg is a matrix: matrix must be first
        # the second dimension of the matrix (# of columns) should be equal to the length of the vector
        # return torch.dot(self.weights, features)
        f1 = F.elu(features)
        return self.layer1(f1)


# def feature_expectation(trajectories, reward_function, gamma=0.99):
#     total_reward = torch.zeros_like(reward_function.weights)
#     for trajectory in trajectories:
#         for t, (state, action) in enumerate(trajectory):
#             # find the discounted feature expectation
#             discount = gamma**t
#             features = feature_avg(state)
#             total_reward += discount * features
#
#     return total_reward


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()

        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

        # Called with either one element to determine next action, or a batch
        # during optimization. Returns tensor([[left0exp,right0exp]...]).

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class CustomRewardDataset(Dataset):
    def __init__(self, feature_map, expert_expectation):
        self.feature_map = feature_map
        self.expert_expectation = expert_expectation

    def __len__(self):
        return len(self.expert_expectation)

    def __getitem__(self, item):
        feature_map = self.feature_map[item]
        expert_expectation = self.expert_expectation[item]

        feature_map = torch.from_numpy(feature_map).float()
        expert_expectation = torch.from_numpy(expert_expectation).float()

        return feature_map, expert_expectation


class CustomPolicyDataset(Dataset):
    def __init__(self, my_features, actions, rewards, next_features):
        self.features = my_features
        self.actions = torch.unsqueeze(actions, 1)
        self.rewards = rewards
        self.next_features = next_features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        features = self.features[item]
        action = self.actions[item]
        reward = self.rewards[item]
        next_features = self.next_features[item]
        return features, action, reward, next_features
