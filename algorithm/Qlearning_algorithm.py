"""
This is the module used for q-learning in the main training loop
Goal: train an agent to act according to the reward function
Possible movements: left, right, up, and down
"""

import copy
import numpy as np
import math
import random
from collections import namedtuple, deque
import torch
from torch import nn
from torch import optim

from IRL_utilities import MyLogger

log = MyLogger(logging=False, debug_msgs=True)
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    """
   Class to hold state, next_state, action, reward information
   Add to this throughout q-learning: this is the training data
   """
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    """
    Deep Q-Learning network
    Trying a single linear layer
    """
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, n_actions)
        # self.layer2 = nn.Linear(128, 128)
        # self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        # x = func.relu(self.layer1(x))
        # x = func.relu(self.layer2(x))
        return self.layer1(x)


class DeepQ:
    def __init__(
        self, n_observations, n_actions, device, LR, neighbors, gamma, target_loc, min_accuracy, memory_length, tau,
            num_epochs, batch_size, criterion, path_length
    ):
        # basic parameters
        self.n_observations = n_observations    # number of characteristics of the state
        self.n_actions = n_actions  # number of possible actions
        self.LR = LR    # learning rate
        self.min_accuracy = min_accuracy  # value to terminate Q-learning
        self.batch_size = batch_size    # number of datapoints per epoch
        self.num_epochs = num_epochs    # number of epochs to run
        self.tau = tau  # parameter to update the target network
        self.target_loc = target_loc    # target location
        self.path_length = path_length

        self.loss = 0

        # variables that will store the dataset and networks for Q-learning
        self.memory = None  # memory class
        self.memory_length = memory_length  # how many past movements to store in memory

        # policy and target networks
        self.policy_net = DQN(
            n_observations=self.n_observations, n_actions=self.n_actions
        ).to(device)
        self.target_net = DQN(
            n_observations=self.n_observations, n_actions=self.n_actions
        ).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.LR, amsgrad=True)

        self.criterion = criterion
        self.device = device    # should always be mps
        self.reward = None  # reward neural network (updated from main code)

        # epsilon parameters
        self.steps_done = 0     # to track for decay
        self.EPS_START = 0.9    # starting value
        self.EPS_END = 0.051    # lowest possible value
        self.EPS_DECAY = 500  # this was originally 1000

        # for movement tracking
        self.neighbors = neighbors  # dataframe of neighbors

        # for reward calculations
        self.gamma = gamma  # discount factor

        # coords to calculate the feature expectation
        self.starting_coords = [341, 126, 26, 620, 299, 208, 148, 150, 27, 302, 134, 460, 513, 200, 1, 598, 69, 309,
                                111, 504, 393, 588, 83, 27, 250]

    def select_action(self, loc, features):
        # input: loc (1d coordinate) and features (feature vector, 626x20)
        state = features[loc]
        sample = random.random()    # random number generator
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(
            -1 * self.steps_done / self.EPS_DECAY
        )   # threshold, based on iterations of the network, so this decreases as the network learns
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                action = (
                    self.policy_net(state.to(self.device))
                    .max(0)
                    .indices.clone()
                    .detach()
                    .cpu()
                    .numpy()
                )    # choose an action according to the policy network
                return action
        else:
            return np.random.randint(4)   # return a random action otherwise

            # option: otherwise move to the cell with the smallest threat
            # threat_old = -np.inf
            # action = np.inf
            # neighbors = self.neighbors.loc[loc][1:5]
            # for i, neighbor in enumerate(neighbors.to_numpy(dtype=np.uint32)):
            #     threat = features[neighbor, 0]
            #     if threat > threat_old:
            #         action = i
            #         threat_old = threat
            # return action

    def find_next_state(self, loc, action, features):
        next_loc = self.neighbors.iloc[loc, action]    # given a known action, find the corresponding location
        next_state = features[next_loc].to(self.device)
        reward = self.reward(next_state).unsqueeze(0)

        # formatting
        state = features[loc].to(self.device).unsqueeze(0)
        action = torch.from_numpy(np.array([action])).to(self.device).unsqueeze(0)

        if next_loc == 624:
            terminated = False
            finished = True
            next_state = next_state.unsqueeze(0)

        elif next_loc == 625:
            terminated = True
            finished = False
            next_state = None

        else:
            terminated = False
            finished = False
            next_state = next_state.unsqueeze(0)

        return terminated, finished, next_state, reward, state, action, next_loc

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)   # generate a random sample for training
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)  # value according to the policy

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )   # value at the next state

        # want loss between q_{my state} and R + gamma * q_{next state}
        expected_state_action_values = (next_state_values.unsqueeze(1) * self.gamma) + reward_batch.unsqueeze(1)
        loss = self.criterion(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return loss

    def run_q_learning(self, features):
        # input features (a nx626x20 vector: first dimension indicates the number of threat fields used for training)
        if self.loss > 1:     # if the prior q-learning loop didn't perform well, re-generate random weights
            self.policy_net = DQN(
                n_observations=self.n_observations, n_actions=self.n_actions
            ).to(self.device)
            self.target_net = DQN(
                n_observations=self.n_observations, n_actions=self.n_actions
            ).to(self.device)
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.LR, amsgrad=True)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.memory = ReplayMemory(capacity=self.memory_length) # restart the memory

        self.steps_done = 0
        loss_memory = []

        for episode in range(self.num_epochs):
            # pick a random place to start
            loc = np.random.randint(624)

            # pick one of the threat fields and just rotate through as we continue training
            feature = features[episode % len(features)]

            # choose an action based on the starting location
            action = self.select_action(loc, features=feature)
            terminated, finished, next_state, reward, state, action, loc = (
                self.find_next_state(loc=loc, action=action, features=feature)
            )

            # add the action to memory
            self.memory.push(state, action, next_state, reward)

            # run the optimizer
            loss = self.optimize_model()

            # update the target network with a soft update: θ′ ← τ θ + (1 - τ )θ′
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * self.tau + target_net_state_dict[key] * (1 - self.tau)
            self.target_net.load_state_dict(target_net_state_dict)

            if not loss:
                loss = 10
            else:
                loss = loss.item()

            if episode % 200 == 0:
                log.debug(
                    "Epoch: \t"
                    + str(episode)
                    + " \t Final Loss Calculated: \t"
                    + str(np.round(loss, 6))
                )   # print the loss during training

            if loss < self.min_accuracy:    # if we reach the specified minimum accuracy, break the loop
                break
            loss_memory.append(loss)

        log.debug(color='red', message='Final loss: \t' + str(np.round(loss, 4)))
        self.loss = loss
        sums = self.find_feature_expectation(feature_function=features)
        return sums

    def find_feature_expectation(self, feature_function):
        n_threats = len(feature_function)   # number of threat fields used for training
        # map the feature function to one long vector (nx20 instead of nx626x20)
        feature_function = feature_function.view(-1, self.n_observations)

        # tile the starting coordinates
        coords = np.tile(self.starting_coords, n_threats)
        # make a "conversion" vector: this allows us to access the 2nd and 3rd (and so on) feature functions
        coords_conv = np.repeat(626 * np.arange(0, n_threats, 1), len(self.starting_coords))
        # 626 because we've added a 626th row to the feature function for outside the boundary

        # starting features
        my_features = (
            feature_function[coords + coords_conv]
        )
        # features used to calculate the desired action
        new_features = copy.deepcopy(my_features).to(self.device)
        # my_features = my_features[:, :4].view(-1, 4).to(self.device)
        my_features = my_features.to(self.device)

        # mask for any finished paths (or terminated paths)
        mask = np.ones(coords.shape, dtype=bool)
        finished_mask = np.ones(coords.shape, dtype=bool)

        for step in range(self.path_length - 1):    # path length - 1 because the first coordinate counts too

            with torch.no_grad():
                action = (
                    self.policy_net(new_features).max(1).indices.cpu().numpy()
                )  # determine the action according to the policy network

                coords[mask] = list(
                    map(
                        lambda index: self.neighbors.iloc[
                            index[1], action[index[0]] + 1
                        ],
                        enumerate(coords[mask]),
                    )
                )   # find the next coordinate according to the initial location and action

                ind = np.where(coords == self.target_loc)
                if ind:
                    mask[ind] = False
                    finished_mask[ind] = False
                failures = np.where(coords == self.target_loc + 1)
                if failures:
                    mask[failures] = False

                new_features = (
                    feature_function[coords[finished_mask] + coords_conv[finished_mask]].view(-1, 20).to(self.device)
                )   # find the features at the new location
                # now add the features: should be gamma^t * new_features for t in [0, T]
                # step starts at 0, we start at 1 because this is the 2nd point in the path
                my_features[finished_mask] += self.gamma ** (step + 1) * new_features

        # total number of paths, then finishes and failures
        total_paths = len(coords)
        not_finishes_failures = sum(mask)
        not_finishes = sum(finished_mask)
        finishes = total_paths - not_finishes
        failures = total_paths - not_finishes_failures - finishes
        log.debug(color='red', message='Number of failures \t' + str(failures))
        log.debug(color='red', message='Number of successes \t' + str(finishes))

        # formatting to return the calculated feature function
        n_returns = len(self.starting_coords)
        reshaped_features = my_features.view(-1, n_returns, my_features.size(1))
        feature_sums = reshaped_features.sum(dim=1) / len(self.starting_coords)
        return feature_sums
