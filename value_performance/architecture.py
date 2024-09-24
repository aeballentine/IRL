import torch.nn as nn
import torch
import numpy as np
from collections import OrderedDict
from torch.utils.data import Dataset


def fun_make_sparse(threat, increment, dim):
    # input the threat field
    # increment should be the number of rows and columns **kept** (same in both dimensions)
    # intentionally not flipping the threat field, so this is upside down
    threat = np.flip(np.reshape(threat, dim), axis=0)
    threat = threat[0::increment]
    threat = [threat[i][0::increment] for i in range(0, len(threat))]
    threat = np.flip(threat, axis=0)
    threat = np.reshape(threat, (1, len(threat) ** 2))
    return threat[0]


class NeuralNetwork(nn.Module):
    """
    Create the activation function between each layer of the neural network
    Input is the size of each layer
    __init__ createss the correct transitions between each layer:
    - layer 1: linear activation 50->75; layer 2: linear activation 75->50, etc...
    """

    def __init__(self, dimensions, function=None):
        super().__init__()
        self.flatten = nn.Flatten()
        stacking = OrderedDict()
        i = 0
        while len(dimensions) > 2:
            stacking["lin" + str(i)] = nn.Linear(dimensions[0], dimensions[1])
            if function:
                stacking["act" + str(i)] = function
            dimensions.pop(0)
            i += 1
        stacking["lin" + str(i)] = nn.Linear(dimensions[0], dimensions[1])
        self.linear_stack = nn.Sequential(stacking)

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits


class CustomDataset(Dataset):
    def __init__(self, threat_vals, value, sparse=False, increment=None, dim=None):
        self.threat_map = threat_vals
        self.val = value
        self.sparse = sparse
        self.increment = increment
        self.dim = dim

    def __len__(self):
        return len(self.threat_map)

    @staticmethod
    def make_sparse(threat, increment, dim):
        # input the threat field
        # increment should be the number of rows and columns **kept** (same in both dimensions)
        # intentionally not flipping the threat field, so this is upside down
        threat = np.flip(np.reshape(threat, dim), axis=0)
        threat = threat[0::increment]
        threat = [threat[i][0::increment] for i in range(0, len(threat))]
        threat = np.flip(threat, axis=0)
        threat = np.reshape(threat, (1, len(threat) ** 2))
        return threat[0]

    def __getitem__(self, item):
        threat_field = self.threat_map[item]
        value = self.val[item]
        full_threat = torch.from_numpy(threat_field)
        value = torch.from_numpy(value)
        if self.sparse:
            sparse_threat = self.make_sparse(
                threat_field, increment=self.increment, dim=self.dim
            )
            sparse_threat = torch.from_numpy(sparse_threat)
            return sparse_threat.float(), value.float(), full_threat.float()
        else:
            return full_threat.float(), value.float()
