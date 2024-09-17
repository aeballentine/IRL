"""
This file provides the helper functions for IRL_algorithm (inverse reinforcement learning)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class RewardFunction(nn.Module):
    """
    Assuming the reward function is a linear combination of the features
    R(s, a, s') = w^T * phi(s)
    """

    def __init__(self, feature_dim):
        super(RewardFunction, self).__init__()
        # initialize the weights as ones
        self.weights = nn.Parameter(torch.zeros(feature_dim))

    def forward(self, features):
        # return the anticipated reward function
        f1 = torch.matmul(features, self.weights)   # using matmul to allow for 2d inputs
        return f1


class CustomRewardDataset(Dataset):
    """
    Dataset for the reward function:
    Two returns: feature map and the associated expert expectation
    """
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
