import copy

import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import namedtuple, deque
import random
from torch import nn
import torch.nn.functional as func
from torch import optim
import math
from itertools import count

from IRL_architecture import feature_avg, CustomPolicyDataset
from IRL_utilities import new_position, MyLogger

log = MyLogger(logging=False, debug_msgs=True)
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
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

        # for a single threat field
        self.starting_coords = [341, 126, 26, 620, 299, 208, 148, 150, 27, 302, 134, 460, 513, 200, 1, 598, 69, 309,
                                111, 504, 393, 588, 83, 27, 250]

    def select_action(self, loc, features):
        state = features[0, loc]
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(
            -1 * self.steps_done / self.EPS_DECAY
        )
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
                )
                return action
        else:
            # act = np.random.randint(4)
            # return np.random.randint(4)
            # return torch.tensor([[act]], device=self.device, dtype=torch.long)

            # find the neighboring cell with the smallest threat

            threat_old = -np.inf
            action = np.inf
            neighbors = self.neighbors.loc[loc][1:5]
            for i, neighbor in enumerate(neighbors.to_numpy(dtype=np.uint32)):
                threat = features[0, neighbor, 0]
                if threat > threat_old:
                    action = i
                    threat_old = threat
            return action

    def find_next_state(self, loc, action, features):
        next_loc = self.neighbors.iloc[loc, action + 1]
        state = features[loc]
        if next_loc == 624:
            terminated = False
            finished = True
            next_state = features[next_loc].to(self.device)
            reward = self.reward(next_state[:4]).unsqueeze(0)
            next_state = next_state.unsqueeze(0)

        elif next_loc == 625:
            terminated = True
            finished = False
            next_state = features[next_loc].to(self.device)
            reward = self.reward(next_state[:4]).unsqueeze(0)
            next_state = None

        else:
            terminated = False
            finished = False
            next_state = features[next_loc].to(self.device)
            reward = self.reward(next_state[:4]).unsqueeze(0)
            next_state = next_state.unsqueeze(0)

        # formatting
        state = state.to(self.device).unsqueeze(0)
        action = torch.from_numpy(np.array([action])).to(self.device).unsqueeze(0)
        return terminated, finished, next_state, reward, state, action, next_loc

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
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

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )

        expected_state_action_values = (next_state_values.unsqueeze(1) * self.gamma) + reward_batch.unsqueeze(1)
        loss = self.criterion(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return loss

    def run_q_learning(self, features):
        if self.loss > 0.05:     # todo: try 0.05 instead of 0.5...might lead to better convergence
            self.policy_net = DQN(
                n_observations=self.n_observations, n_actions=self.n_actions
            ).to(self.device)
            self.target_net = DQN(
                n_observations=self.n_observations, n_actions=self.n_actions
            ).to(self.device)
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.LR, amsgrad=True)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        # reinitialize relevant parameters: need to reset the learning networks, memory, optimizer, and
        # epsilon parameters
        self.memory = ReplayMemory(capacity=self.memory_length)

        self.steps_done = 0

        for episode in range(self.num_epochs):
            # pick a random place to start
            loc = np.random.randint(624)
            feature = features[episode % len(features)]
            action = self.select_action(loc, features=features)
            terminated, finished, next_state, reward, state, action, loc = (
                self.find_next_state(loc=loc, action=action, features=feature)
            )

            # if not terminated:
            self.memory.push(state, action, next_state, reward)

            loss = self.optimize_model()

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
            # if terminated or finished or (t > 25):
            #     if loss:
            # if episode % 20 == 0:
            #     log.debug(
            #         "Epoch: \t"
            #         + str(episode)
            #         + " \t Final Loss Calculated: \t"
            #         + str(np.round(loss, 6))
            #     )
            #         loss = loss.item()
            #     else:
            #         loss = 10
                # if finished:
                #     log.debug(color='red', message='Successfully finished \t Path Length: \t' + str(t))
                # break
            if loss < self.min_accuracy:
                break
        log.debug(color='red', message='Final loss: \t' + str(np.round(loss, 4)))
        self.loss = loss
        sums = self.find_feature_expectation(feature_function=features)
        return sums

    def find_feature_expectation(self, feature_function):
        # want 2 steps: 3 total points per path
        # tile the starting coordinates
        n_threats = len(feature_function)
        feature_function = feature_function.view(-1, self.n_observations)

        coords = np.tile(self.starting_coords, n_threats)
        coords_conv = np.repeat(626 * np.arange(0, n_threats, 1), len(self.starting_coords))
        # 626 because we've added a 626th row to the feature function for outside the boundary

        my_features = (
            feature_function[coords + coords_conv]
        )
        new_features = copy.deepcopy(my_features).to(self.device)
        my_features = my_features[:, :4].view(-1, 4).to(self.device)
        mask = np.ones(coords.shape, dtype=bool)
        finished_mask = np.ones(coords.shape, dtype=bool)

        for step in range(self.path_length - 1):

            with torch.no_grad():
                # log.debug(coords[[1, 3, 5, 9]])
                action = (
                    self.target_net(new_features).max(1).indices.cpu().numpy()
                )  # this should be max(1) for multi-threat

                coords[mask] = list(
                    map(
                        lambda index: self.neighbors.iloc[
                            index[1], action[index[0]] + 1
                        ],
                        enumerate(coords[mask]),
                    )
                )

                ind = np.where(coords == self.target_loc)
                if ind:
                    # finished = np.append(finished, ind)
                    # finished = np.unique(finished)
                    mask[ind] = False
                    finished_mask[ind] = False
                failures = np.where(coords == self.target_loc + 1)
                if failures:
                    # fail_ind = np.append(fail_ind, failures)
                    # fail_ind = np.unique(fail_ind)
                    mask[failures] = False

                new_features = (
                    feature_function[coords[finished_mask] + coords_conv[finished_mask]].view(-1, self.n_observations).to(self.device)
                )
                # todo: note, changed this to step + 1...gamma is raised to the 0, 1, 2... and we start on the 2nd val
                my_features[finished_mask] += self.gamma ** (step + 1) * new_features[:, :4]
        log.debug(color='red', message='Number of failures \t' + str(len(coords) - sum(mask)))
        log.debug(color='red', message='Number of successes \t' + str(len(coords) - sum(finished_mask)))
        n_returns = len(self.starting_coords)
        reshaped_features = my_features.view(-1, n_returns, my_features.size(1))
        feature_sums = reshaped_features.sum(dim=1) / len(self.starting_coords)
        return feature_sums

    # class DeepQ:
    #     def __init__(
    #         self,
    #         target_loc,
    #         policy_net,
    #         target_net,
    #         gamma,
    #         criterion,
    #         tau,
    #         batch_size,
    #         epochs,
    #         device,
    #         num_features,
    #         LR,
    #         epsilon,
    #         neighbor_info,
    #         starting,
    #     ):
    #         self.target_loc = np.array(target_loc)
    #         self.policy_net = policy_net
    #         self.target_net = target_net
    #         self.gamma = gamma
    #         self.criterion = criterion
    #         self.optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LR)
    #         self.tau = tau
    #         self.batch_size = batch_size
    #         self.epochs = epochs
    #         self.device = device
    #         self.num_features = num_features
    #         self.epsilon = epsilon
    #         self.dataloader = None
    #         self.sample_points = np.array(starting)
    #         self.n_paths = len(starting)
    #         self.sample_features = None
    #         self.neighbors = neighbor_info
    #
    #         # set up parameters for Q-learning
    #
    #     def find_next_state(self, state, features):
    #
    #         return 0
    #
    #     def find_features_threats(self, state, threat_fields):
    #         # state is a vector that repeats itself, so the first n entries correspond to the first threat field
    #         # then the second n entries correspond to the second threat field
    #         # todo: NEED TO FIX THIS FUNCTION, currently assumes that all states are repeated across all threat fields
    #         # my_points = self.neighbors.loc[state]
    #         # print(my_points)
    #
    #         # left neighbors: threat values
    #         _left = list(map(lambda state_vec: self.neighbors.loc[state_vec].left, state))
    #         left_vals = tuple(
    #             map(
    #                 lambda index: threat_fields[index[0]][index[1]].unsqueeze(0),
    #                 enumerate(_left),
    #             )
    #         )
    #         left_vals = torch.unsqueeze(torch.cat(left_vals), 1)
    #
    #         # right neighbors: threat values
    #         _right = list(map(lambda state_vec: self.neighbors.loc[state_vec].right, state))
    #         right_vals = tuple(
    #             map(
    #                 lambda index: threat_fields[index[0]][index[1]].unsqueeze(0),
    #                 enumerate(_right),
    #             )
    #         )
    #         right_vals = torch.unsqueeze(torch.cat(right_vals), 1)
    #
    #         # up neighbors
    #         _up = list(map(lambda state_vec: self.neighbors.loc[state_vec].up, state))
    #         up_vals = tuple(
    #             map(
    #                 lambda index: threat_fields[index[0]][index[1]].unsqueeze(0),
    #                 enumerate(_up),
    #             )
    #         )
    #         up_vals = torch.unsqueeze(torch.cat(up_vals), 1)
    #
    #         # down neighbors
    #         _down = list(map(lambda state_vec: self.neighbors.loc[state_vec].down, state))
    #         down_vals = tuple(
    #             map(
    #                 lambda index: threat_fields[index[0]][index[1]].unsqueeze(0),
    #                 enumerate(_down),
    #             )
    #         )
    #         down_vals = torch.unsqueeze(torch.cat(down_vals), 1)
    #
    #         my_vals = torch.cat((left_vals, right_vals, up_vals, down_vals), 1)
    #
    #         my_threat = tuple(
    #             map(
    #                 lambda index: threat_fields[index[0]][index[1]].unsqueeze(0),
    #                 enumerate(state),
    #             )
    #         )
    #         my_threat = torch.unsqueeze(torch.cat(my_threat), 1)
    #
    #         x_distance = list(
    #             map(lambda state_vec: self.neighbors.loc[state_vec].x_dist, state)
    #         )
    #
    #         x_distance = -torch.unsqueeze(torch.tensor(x_distance), 1).to(self.device)
    #
    #         y_distance = list(
    #             map(lambda state_vec: self.neighbors.loc[state_vec].y_dist, state)
    #         )
    #         y_distance = -torch.unsqueeze(torch.tensor(y_distance), 1).to(self.device)
    #
    #         features = torch.cat((my_threat, my_vals, x_distance, y_distance), 1)
    #
    #         return features
    #
    #     def new_state(self, state, action):
    #         new = list(
    #             map(
    #                 lambda ind: self.neighbors.iloc[ind[1], action[ind[0]] + 1],
    #                 enumerate(state),
    #             )
    #         )
    #         return np.array(new)
    #
    #     def my_policy(self, state, threat_field):
    #         # df first col = array[df first col vals]
    #         # return the following: features at my state, the appropriate next state, and the features at that state.
    #         # state is a numpy array (1D indices) and threat_field is a torch tensor
    #         # noting here for clarity: df order is left, right, up, down; corresponding indices are 0, 1, 2, 3
    #
    #         my_points = copy.deepcopy(self.neighbors.loc[state]).reset_index(drop=True)
    #
    #         # left neighbors: threat values
    #         _left = my_points.left.to_numpy()
    #         left_vals = torch.unsqueeze(threat_field[_left], 1)
    #
    #         # right neighbors: threat values
    #         _right = my_points.right.to_numpy()
    #         right_vals = torch.unsqueeze(threat_field[_right], 1)
    #
    #         # up neighbors
    #         _up = my_points.up.to_numpy()
    #         up_vals = torch.unsqueeze(threat_field[_up], 1)
    #
    #         # down neighbors
    #         _down = my_points.down.to_numpy()
    #         down_vals = torch.unsqueeze(threat_field[_down], 1)
    #
    #         my_vals = torch.cat((left_vals, right_vals, up_vals, down_vals), 1)
    #         action = my_vals.min(1).indices.cpu().numpy()
    #         # note: adding one to i_col[1] to account for the fact that the first col is the actual state not "left"
    #         new_location = list(
    #             map(
    #                 lambda i_col: my_points.iloc[i_col[0], i_col[1] + 1],
    #                 enumerate(action),
    #             )
    #         )
    #
    #         # new_location = my_points.right.to_numpy()  # todo: changed this
    #         # new_location[new_location > 624] = my_points.left.to_numpy()[new_location > 624]
    #
    #         # find the original features
    #         my_threat = torch.unsqueeze(threat_field[state], 1)
    #         x_distance = -torch.unsqueeze(torch.tensor(my_points.x_dist.values), 1).to(
    #             self.device
    #         )
    #         y_distance = -torch.unsqueeze(torch.tensor(my_points.y_dist.values), 1).to(
    #             self.device
    #         )
    #         my_features = torch.cat((my_threat, my_vals, x_distance, y_distance), 1)
    #
    #         # find the next features
    #         next_features = self.find_features_states(new_location, threat_field)
    #
    #         return my_features, next_features, torch.from_numpy(action).to(self.device)
    #
    #     def optimize_model(self, states, actions, rewards, next_states):
    #         actions = actions.type(torch.int64)
    #         q_pi = self.policy_net(states).gather(1, actions)
    #         with torch.no_grad():
    #             # next_state_values = self.target_net(next_states).max(1).values
    #             next_state_values = self.target_net(next_states).max(1).values
    #         q_pi_prime = (next_state_values * self.gamma) + rewards
    #         q_pi_prime = q_pi_prime.view(-1, 1)
    #         loss = self.criterion(q_pi, q_pi_prime)
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
    #         self.optimizer.step()
    #         return loss
    #
    #     def model(self, threat_fields, reward):
    #         # threat fields is a list of tensors...I think
    #         # generate 100 random coordinates in the threat field
    #         starting_coords = np.random.randint(0, 624, size=15)
    #         curr_features = torch.tensor([]).to(self.device)
    #         next_features = torch.tensor([]).to(self.device)
    #         curr_rewards = torch.tensor([]).to(self.device)
    #         actions = torch.tensor([]).to(self.device)
    #         self.policy_net = DQN(n_observations=7, n_actions=4).to(self.device)
    #         self.target_net = DQN(n_observations=7, n_actions=4).to(self.device)
    #         # todo: this is slowing the code down
    #         for threat in threat_fields:
    #             my_feat, next_feat, decisions = self.my_policy(starting_coords, threat)
    #             curr_features = torch.cat((curr_features, my_feat), 0)
    #             next_features = torch.cat((next_features, next_feat), 0)
    #             actions = torch.cat((actions, decisions), 0)
    #             my_reward = reward(my_feat).detach()
    #             curr_rewards = torch.cat((curr_rewards, my_reward), 0)
    #
    #         # create the dataloader
    #         dataset = CustomPolicyDataset(
    #             my_features=curr_features,
    #             actions=actions,
    #             rewards=curr_rewards,
    #             next_features=next_features,
    #         )
    #         self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    #         log.info("Created Q-learning dataloader")
    #
    #         for epoch in range(self.epochs):
    #             losses = []
    #             for batch_num, input_data in enumerate(self.dataloader):
    #                 my_features, my_actions, my_rewards, next_features = input_data
    #                 # my_features = my_features.to(self.device).float()
    #                 # my_actions = my_actions.to(self.device).float()
    #                 # my_rewards = my_rewards.to(self.device).float()
    #                 # next_features = next_features.to(self.device).float()
    #
    #                 loss = self.optimize_model(
    #                     states=my_features,
    #                     actions=my_actions,
    #                     rewards=my_rewards,
    #                     next_states=next_features,
    #                 )
    #                 losses.append(loss)
    #
    #                 target_net_state_dict = self.target_net.state_dict()
    #                 policy_net_state_dict = self.policy_net.state_dict()
    #                 for key in policy_net_state_dict:
    #                     target_net_state_dict[key] = policy_net_state_dict[
    #                         key
    #                     ] * self.tau + target_net_state_dict[key] * (1 - self.tau)
    #                 self.target_net.load_state_dict(target_net_state_dict)
    #
    #             if (epoch + 1) % 20 == 0:
    #                 log.debug("\t \tEpoch %d | Loss %4.5f" % (epoch + 1, losses[-1]))
    #             if losses[-1] < self.epsilon:
    #                 break
    #
    #         # log.info("Epoch %d | Loss %4.5f" % (epoch, losses[-1]))
    #         return self.expected_feature_function(threat_fields=threat_fields)

    # def expected_feature_function(self, threat_fields):
    #     # NOTE: threat fields has to include multiple threat fields
    #     # current position is a stacked array: each row corresponds to a threat field
    #     # todo: stack the threats and the actions??
    #     # NOTE: I need to return a feature vector for each threat field
    #     # don't pick either the "fake" point or the target loc
    #
    #     threats = torch.repeat_interleave(threat_fields, self.n_paths, dim=0)
    #     points = np.tile(self.sample_points, len(threat_fields))
    #     features = self.find_features_threats(points, threats)
    #     next_position = points
    #     mask = np.ones(next_position.shape, dtype=bool)
    #     feature_sum = features
    #
    #     finished = np.array([], dtype=int)
    #     fail_ind = np.array([], dtype=int)
    #
    #     for i in range(4):
    #         actions = self.target_net(features).max(1).indices.cpu().numpy()
    #         next_position[mask] = self.new_state(points[mask], actions[mask])
    #
    #         ind = np.where(next_position == self.target_loc)[0]
    #         if ind.size:
    #             finished = np.append(finished, ind)
    #             finished = np.unique(finished)
    #             mask[finished] = False
    #         failures = np.where(next_position == self.target_loc + 1)[0]
    #         if failures.size:
    #             fail_ind = np.append(fail_ind, failures)
    #             fail_ind = np.unique(fail_ind)
    #             mask[fail_ind] = False
    #
    #         next_position[np.logical_not(mask)] = self.target_loc
    #
    #         features = self.find_features_threats(next_position, threats)
    #         feature_sum[mask] += self.gamma**i * features[mask]
    #
    #     n_returns = len(self.sample_points)
    #     reshaped_features = feature_sum.view(-1, n_returns, feature_sum.size(1))
    #     feature_sums = reshaped_features.sum(dim=1) / len(self.sample_points)
    #
    #     return feature_sums

    # raise Warning("Have not edited past this point")
    #
    # feature_return = torch.tensor([]).to(self.device)
    # for threat in threat_fields:
    #     feature_threat = torch.zeros((1, self.num_features)).to(self.device)
    #     for start in starts:
    #         position = start
    #         feature_path = torch.zeros(self.num_features).to(self.device)
    #         for i in range(25):
    #             feature = self.find_features_states([position], threat)
    #             action = self.target_net(feature).max(1).indices.cpu().numpy()
    #             position = self.neighbors.iloc[position, action + 1].values[0]
    #             feature_path += self.gamma**i * feature[0]
    #             if position == 625:
    #                 break
    #             if position == self.target_loc:
    #                 break
    #         feature_threat += feature_path / len(feature_path)
    #     feature_return = torch.cat((feature_return, feature_threat), 0)
    # feature_vals = features.view(
    #     -1, self.n_paths, features.size(1)
    # )  # to do: verify this
    # feature_return = feature_vals.sum(dim=1)
    # return feature_return

    # print(threat_fields)
    # current_position = torch.randint(0, 624, (50,)).repeat(len(threat_fields), 1)
    # features = []
    #     for _ in range(25):
    #         current_features = self.find_features_threats(
    #             current_position, threat_fields
    #         ).to(self.device)
    #         print(current_features)
    #         # todo: can check this, but this should be a 1D vector right now
    #         action = self.target_net(current_features).max(1).indices
    #         print(action)
    #         next_state = self.new_state(current_position, action.cpu())
    #
    # # feature_vector = np.zeros((len(threat_fields), self.num_features))
    #
    # # for each point, we went to generate a path 25 points long
    # # if we reach the destination, stop that path
    # for k, threat in enumerate(threat_fields):
    #     current_position = starting_coords
    #     my_features = self.find_features_states(current_position, threat)
    #     next_position = self.target_net(my_features)
    #
    #     current_features = np.zeros((50, self.num_features))
    #     for i, point in enumerate(self.sample_points):
    #         current_features[i] = feature_avg(point, threat, self.target_loc)
    #         feature_vector[k] += current_features[i]
    #     for i, point in enumerate(self.sample_points):
    #         for _ in range(25):
    #             if (np.array(current_position[i]) == self.target_loc).all():
    #                 break
    #             my_feature = (
    #                 torch.from_numpy(current_features[i]).float().to(self.device)
    #             )
    #             action = self.target_net(my_feature).max(0).indices
    #             current_position[i], success = new_position(
    #                 point, action.item(), log
    #             )
    #             current_features[i] = feature_avg(
    #                 current_position[i], threat, self.target_loc
    #             )
    #             if success is False:
    #                 break
    #             feature_vector[k] += current_features[i]
    # return feature_vector / len(self.sample_points)
