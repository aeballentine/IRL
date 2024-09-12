import copy

import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import namedtuple, deque
from IRL_utilities import neighbors_of_four
import random
from torch import nn
import torch.nn.functional as func
from torch import optim
import pandas as pd
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
        self.layer1 = nn.Linear(n_observations, 4)
        # self.layer2 = nn.Linear(128, 128)
        # self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        # x = (self.layer1(x))
        # x = (self.layer2(x))
        # x = func.relu(x)
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
        self.EPS_DECAY = 1000  # this was originally 1000

        # for movement tracking
        self.neighbors = neighbors  # dataframe of neighbors

        # for reward calculations
        self.gamma = gamma  # discount factor

        # for a single threat field
        self.starting_coords = np.arange(0, 624, 1)

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
            return np.random.randint(4)
            # return torch.tensor([[act]], device=self.device, dtype=torch.long)

            # find the neighboring cell with the smallest threat

            # threat_old = np.inf
            # action = np.inf
            # neighbors = self.neighbors.loc[loc][1:5]
            # for i, neighbor in enumerate(neighbors.to_numpy(dtype=np.uint32)):
            #     threat = features[0, neighbor, 0]
            #     if threat < threat_old:
            #         action = i
            #         threat_old = threat
            # return action

    def find_next_state(self, loc, action, features):
        next_loc = self.neighbors.iloc[loc, action + 1]
        state = features[loc]
        if next_loc == 624:
            terminated = False
            finished = True
            next_state = features[next_loc].to(self.device)
            reward = 100 - state[0].unsqueeze(0).to(self.device)
            next_state = next_state.unsqueeze(0)

        elif next_loc == 625:
            terminated = True
            finished = False
            next_state = features[next_loc].to(self.device)
            reward = 100 - state[0].unsqueeze(0).to(self.device)
            next_state = None

        else:
            terminated = False
            finished = False
            next_state = features[next_loc].to(self.device)
            reward = 100 - state[0].unsqueeze(0).to(self.device)
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
                self.target_net(non_final_next_states).max(1).values    # todo: changed this to min
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

        loss_memory = []

        for episode in range(self.num_epochs):
            # pick a random place to start
            loc = np.random.randint(624)
            if 500 < episode < 550:
                loc = episode % 25
            if 1000 < episode < 1050:
                loc = 623 - episode % 25
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
            log.debug(
                "Epoch: \t"
                + str(episode)
                + " \t Final Loss Calculated: \t"
                + str(np.round(loss, 6))
            )
            #         loss = loss.item()
            #     else:
            #         loss = 10
                # if finished:
                #     log.debug(color='red', message='Successfully finished \t Path Length: \t' + str(t))
                # break
            if loss < self.min_accuracy:
                break
            loss_memory.append(loss)
        # print(loss_memory)
        log.debug(color='red', message='Final loss: \t' + str(np.round(loss, 4)))
        self.loss = loss
        sums = self.find_feature_expectation(feature_function=features)
        return sums

    def find_feature_expectation(self, feature_function):
        # want 2 steps: 3 total points per path
        # tile the starting coordinates
        n_threats = len(feature_function)
        feature_function = feature_function.view(-1, self.n_observations)    # todo: this may need to change

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

        errors = 0

        for step in range(self.path_length - 1):

            with torch.no_grad():
                # log.debug(coords[[1, 3, 5, 9]])
                # print(self.policy_net(new_features))
                action = (
                    self.policy_net(new_features).max(1).indices.cpu().numpy()
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

                print(new_features)
                print(action)

                for i, act in enumerate(action):
                    act_feat = new_features[i]
                    print('......')
                    print(act_feat)
                    desired_action = act_feat[1:].min(0).indices
                    print(desired_action)
                    print(act)
                    print('......')
                    if desired_action != act:
                        errors += 1

                new_features = (
                    feature_function[coords[finished_mask] + coords_conv[finished_mask]].view(-1, self.n_observations).to(self.device)
                )
                # print(new_features)
                print(coords[finished_mask] + coords_conv[finished_mask])
                print(new_features[:, [0]])
                print('~~~~~~~~~~~~~~')
                # todo: note, changed this to step + 1...gamma is raised to the 0, 1, 2... and we start on the 2nd val
                my_features[finished_mask] += self.gamma ** (step + 1) * new_features[:, :4]
        log.debug(color='red', message='Number of failures \t' + str(len(coords) - sum(mask[finished_mask])))
        log.debug(color='red', message='Number of successes \t' + str(len(coords) - sum(finished_mask)))
        print(errors)
        n_returns = len(self.starting_coords)
        reshaped_features = my_features.view(-1, n_returns, my_features.size(1))
        feature_sums = reshaped_features.sum(dim=1) / len(self.starting_coords)
        return feature_sums


if __name__ == "__main__":
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # PARAMETERS

    # DATA PARAMETERS
    # threat field
    target_loc = 624  # final location in the threat field
    gamma = 0.99  # discount factor
    path_length = 2  # maximum number of points to keep along expert generated paths
    dims = (25, 25)

    # feature dimensions
    feature_dims = (
        4  # number of features to take into account (for the reward function)
    )

    # MACHINE LEARNING PARAMETERS
    # reward function
    batch_size = 400  # number of samples to take per batch
    learning_rate = 0.5  # learning rate
    epochs = 1000  # number of epochs for the main training loop

    # value function
    tau = (
        0.0001  # rate at which to update the target_net variable inside the Q-learning module
    )
    LR = 1  # learning rate for Q-learning
    q_criterion = (
        nn.MSELoss()
    )  # criterion to determine the loss during training (otherwise try hinge embedding)
    q_batch_size = 400  # batch size
    num_features = 5  # number of features to take into consideration
    q_epochs = 2000  # number of epochs to iterate through for Q-learning
    min_accuracy = 1.5e-2  # value to terminate Q-learning (if value is better than this)
    memory_length = 1000

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # NEIGHBORS OF FOUR
    neighbors = neighbors_of_four(dims=dims, target=target_loc)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # LOAD THE DATA
    data = pd.read_pickle('expert_demonstrations/multi_threat.pkl')

    feature_averages = data.expert_feat
    feature_function = data.feature_map
    threat_fields = data.threat_field

    log.info("Expert feature average calculated")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    q_learning = DeepQ(
        n_observations=num_features,
        n_actions=4,
        device=device,
        LR=LR,
        neighbors=neighbors,
        gamma=gamma,
        target_loc=target_loc,
        min_accuracy=min_accuracy,
        memory_length=memory_length,
        tau=tau,
        num_epochs=q_epochs,
        batch_size=q_batch_size,
        criterion=q_criterion,
        path_length=path_length
    )

    torch.set_printoptions(linewidth=200)

    feature_function = np.reshape(feature_function[0][:, [0, 2, 4, 6, 8]], (1, 626, 5))
    print(feature_function)
    q_learning.run_q_learning(features=torch.from_numpy(feature_function).float())
