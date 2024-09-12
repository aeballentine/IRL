"""
This is the module used to _test_ the q-learning network.
Goal: always choose the smallest threat
Possible movements: stay at the same location or move left, right, up, or down
"""
import copy
import numpy as np
import pandas as pd
import math
import random
from collections import namedtuple, deque
import torch
from torch import nn
from torch import optim

from IRL_utilities import MyLogger
from IRL_utilities import neighbors_of_four


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

    def sample(self, batch):
        return random.sample(self.memory, batch)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    """
    Deep Q-Learning network
    For this application, using a single linear layer
    """
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, n_actions)

    def forward(self, x):
        return self.layer1(x)


class DeepQ:
    """
    This is the class with the training and assessment functionality
    """
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
        self.memory = ReplayMemory(capacity=memory_length)  # memory class

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
        state = features[0, loc]    # features is a (1, 626, 5) vector, so choose the row corresponding to the loc
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
                )   # choose an action according to the policy network
                return int(action)
        else:
            return np.random.randint(5)   # return a random action otherwise

    def find_next_state(self, loc, action, features):
        next_loc = self.neighbors.iloc[loc, action] # given a known action, find the corresponding location
        next_state = features[next_loc].to(self.device)
        reward = 10 - next_state[0].unsqueeze(0).to(self.device)     # reward associated with the next state

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

        # formatting
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
        self.steps_done = 0
        loss_memory = []

        for episode in range(self.num_epochs):
            # pick a random place to start
            loc = np.random.randint(624)

            feature = features[0]

            # choose an action based on the starting location
            action = self.select_action(loc, features=features)
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

            log.info(
                "Epoch: \t"
                + str(episode)
                + " \t Final Loss Calculated: \t"
                + str(np.round(loss, 6))
            )

            loss_memory.append(loss)

            if loss < self.min_accuracy:
                break

        log.debug(color='red', message='Final loss: \t' + str(np.round(loss, 4)))
        self.loss = loss
        self.find_feature_expectation(feature_function=features)

    def find_feature_expectation(self, feature_function):
        n_threats = 1
        feature_function = feature_function.view(-1, self.n_observations)

        coords = np.tile(self.starting_coords, n_threats)
        coords_conv = np.repeat(626 * np.arange(0, n_threats, 1), len(self.starting_coords))
        # 626 because we've added a 626th row to the feature function for outside the boundary

        my_features = (
            feature_function[coords + coords_conv]
        )   # features at all the starting coordinates
        new_features = copy.deepcopy(my_features).to(self.device)   # feature values to use to decide each action
        my_features = my_features[:, :4].view(-1, 4).to(self.device)

        errors = 0

        for step in range(self.path_length - 1):
            with torch.no_grad():
                action = (
                    self.policy_net(new_features).max(1).indices.cpu().numpy()
                )  # this should be max(1) for multi-threat

                coords = list(
                    map(
                        lambda index: self.neighbors.iloc[
                            index[1], action[index[0]] + 1
                        ],
                        enumerate(coords),
                    )
                )

                for i, act in enumerate(action):
                    act_feat = new_features[i]
                    print('......')
                    print(act_feat)
                    desired_action = act_feat.min(0).indices
                    print(desired_action)
                    print(act)
                    print('......')
                    if desired_action != act:
                        errors += 1

                new_features = (
                    feature_function[coords + coords_conv].view(-1, self.n_observations).to(self.device)
                )
                print('~~~~~~~~~~~~~~')
                my_features += self.gamma ** (step + 1) * new_features[:, :4]
        print(errors)


if __name__ == "__main__":
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # PARAMETERS

    # DATA PARAMETERS
    # threat field
    target_loc_ = 624  # final location in the threat field
    gamma_ = 0.99  # discount factor
    path_length_ = 2  # maximum number of points to keep along expert generated paths
    dims = (25, 25)

    # MACHINE LEARNING PARAMETERS
    q_tau = (
        0.0005  # rate at which to update the target_net variable inside the Q-learning module
    )
    q_lr = 0.1  # learning rate
    q_criterion = (
        nn.MSELoss()
    )  # criterion to determine the loss during training (otherwise try hinge embedding)
    q_batch_size = 400  # batch size
    q_features = 5  # number of features to take into consideration
    q_epochs = 2000  # number of epochs to iterate through for Q-learning
    q_accuracy = 1.5e-2  # value to terminate Q-learning (if value is better than this)
    q_memory = 1000

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # NEIGHBORS OF FOUR
    neighbors_ = neighbors_of_four(dims=dims, target=target_loc_)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # LOAD THE DATA
    data = pd.read_pickle('expert_demonstrations/multi_threat.pkl')
    feature_function_ = data.feature_map

    log.info("Expert feature average calculated")

    device_ = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    q_learning = DeepQ(
        n_observations=q_features,
        n_actions=5,
        device=device_,
        LR=q_lr,
        neighbors=neighbors_,
        gamma=gamma_,
        target_loc=target_loc_,
        min_accuracy=q_accuracy,
        memory_length=q_memory,
        tau=q_tau,
        num_epochs=q_epochs,
        batch_size=q_batch_size,
        criterion=q_criterion,
        path_length=path_length_
    )

    torch.set_printoptions(linewidth=200)

    feature_function_ = np.reshape(feature_function_[0][:, [0, 4, 8, 12, 16]], (1, 626, 5))
    # print(feature_function)
    q_learning.run_q_learning(features=torch.from_numpy(feature_function_).float())
