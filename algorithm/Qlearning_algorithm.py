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
from itertools import count
import torch
from torch import nn
from torch import optim
# import wandb
from torch.nn import functional as func

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
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 625)
        self.layer3 = nn.Linear(625, 128)
        self.layer4 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


class DeepQ:
    def __init__(
        self, n_observations, n_actions, device, LR, neighbors, gamma, target_loc, min_accuracy, memory_length, tau,
            num_epochs, batch_size, criterion, path_length, expert_paths, starting_coords
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
        self.expert_paths = expert_paths[0]

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

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)

        self.criterion = criterion
        self.device = device    # should always be mps
        self.reward = None  # reward neural network (updated from main code)

        # epsilon parameters
        self.steps_done = 0     # to track for decay
        self.EPS_START = 0.85    # starting value
        self.EPS_END = 0.051    # lowest possible value
        self.EPS_DECAY = 500  # this was originally 1000

        # for movement tracking
        self.neighbors = neighbors  # dataframe of neighbors

        # for reward calculations
        self.gamma = gamma  # discount factor

        # coords to calculate the feature expectation
        # self.starting_coords = [341, 126, 26, 620, 299, 208, 148, 150, 27, 302, 134, 460, 513, 200, 1, 598, 69, 309,
        #                         111, 504, 393, 588, 83, 27, 250]
        self.starting_coords = starting_coords

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
                return int(action)
        else:
            return np.random.randint(4)   # return a random action otherwise

    def find_next_state(self, loc, action, features):
        next_loc = self.neighbors.iloc[loc, action + 1]    # given a known action, find the corresponding location
        next_state = features[next_loc].to(self.device)
        with torch.no_grad():
            reward = self.reward(next_state[:4]).unsqueeze(0)

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
            next_state = next_state.unsqueeze(0)

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

        # wandb.log({'q_loss': loss})
        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return loss.item()

    def run_q_learning(self, features):
        # input features (a nx626x20 vector: first dimension indicates the number of threat fields used for training)
        if self.loss > 5:     # if the prior q-learning loop didn't perform well, re-generate random weights
            self.policy_net = DQN(
                n_observations=self.n_observations, n_actions=self.n_actions
            ).to(self.device)
            self.target_net = DQN(
                n_observations=self.n_observations, n_actions=self.n_actions
            ).to(self.device)
            self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.memory = ReplayMemory(capacity=self.memory_length) # restart the memory

        self.steps_done = 0
        loss_memory = []
        possible_actions = np.array([-1, 1, 25, -25])

        for episode in range(self.num_epochs):
            path_indexer = 0
            path_num = np.random.randint(len(self.expert_paths))

            # pick a random place to start
            loc = np.random.randint(624)
            for t in count():

                # pick one of the threat fields and just rotate through as we continue training
                # feature = features[episode % len(features)]
                feature = features[0]

                if (episode % 10 == 0) and episode < 100:
                    loc = self.expert_paths[path_num][path_indexer]
                    action = self.expert_paths[path_num][path_indexer + 1] - loc
                    action = np.where(possible_actions == action)[0][0]
                    path_indexer += 1
                else:
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
                    loss = loss

                if loc == 624:
                    break
                elif loc == 625:
                    break
                elif t > 75:
                    break

            if loss < self.min_accuracy:    # if we reach the specified minimum accuracy, break the loop
                break
            loss_memory.append(loss)

        self.loss = loss
        sums = self.find_feature_expectation(feature_function=features)
        return sums, loss

    def find_feature_expectation(self, feature_function):
        n_threats = len(feature_function)   # number of threat fields used for training
        # map the feature function to one long vector (nx20 instead of nx626x20)
        feature_function = feature_function.view(-1, self.n_observations)

        # tile the starting coordinates
        coords = np.tile(self.starting_coords, n_threats)
        # make a "conversion" vector: this allows us to access the 2nd and 3rd (and so on) feature functions

        finishes = 0
        failures = 0

        my_features = torch.tensor([]).to(self.device)
        for coord in coords:    # todo: match this to the expert demonstrations, check each point
            new_features = feature_function[[coord]].to(self.device)
            # print(new_features)
            my_features = torch.cat([my_features, new_features.abs()])
            for step in range(self.path_length - 1):    # path length - 1 because the first coordinate counts too
                with torch.no_grad():
                    action = (
                        self.policy_net(new_features).max(1).indices.cpu().numpy()
                    )  # determine the action according to the policy network

                coord = self.neighbors.iloc[coord, action[0] + 1]
                new_features = feature_function[[coord]].to(self.device)

                my_features = torch.cat([my_features, new_features.abs()])
                if coord == 624:
                    # find how many moves we've made
                    zeros = torch.zeros(1, self.n_observations)
                    points_remaining = self.path_length - 2 - step
                    to_append = zeros.repeat(points_remaining, 1).to(self.device)
                    my_features = torch.cat([my_features, to_append])
                    finishes += 1
                    break

                elif coord == 625:
                    maxis = my_features[-1]
                    points_remaining = self.path_length - 2 - step
                    to_append = maxis.repeat(points_remaining, 1)
                    # print(my_features.shape)
                    # print(to_append.shape)
                    my_features = torch.cat([my_features, to_append.abs()])
                    failures += 1
                    break

                else:
                    continue

        # wandb.log({'q_learning_finishes': finishes})
        # wandb.log({'q_learning_failures': failures})

        return my_features.view(-1, len(my_features), self.n_observations)
