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
from itertools import count
from collections import namedtuple, deque
import torch
from torch import nn
from torch import optim
import torch.nn.functional as func
import matplotlib.pyplot as plt

from IRL_utilities import MyLogger
from IRL_utilities import neighbors_of_four

log = MyLogger(logging=True, debug_msgs=True)
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
    """
    This is the class with the training and assessment functionality
    """

    def __init__(
            self, n_observations, n_actions, device, LR, neighbors, gamma, target_loc, min_accuracy, memory_length, tau,
            num_epochs, batch_size, criterion, path_length, expert_paths
    ):
        # basic parameters
        self.n_observations = n_observations  # number of characteristics of the state
        self.n_actions = n_actions  # number of possible actions
        self.LR = LR  # learning rate
        self.min_accuracy = min_accuracy  # value to terminate Q-learning
        self.batch_size = batch_size  # number of datapoints per epoch
        self.num_epochs = num_epochs  # number of epochs to run
        self.tau = tau  # parameter to update the target network
        self.target_loc = target_loc  # target location
        self.path_length = path_length
        self.expert_paths = expert_paths[0]

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
        self.device = device  # should always be mps
        self.reward = None  # reward neural network (updated from main code)

        # epsilon parameters
        self.steps_done = 0  # to track for decay
        self.EPS_START = 0.91  # starting value
        self.EPS_END = 0.051  # lowest possible value
        self.EPS_DECAY = 1000  # this was originally 1000

        # for movement tracking
        self.neighbors = neighbors  # dataframe of neighbors

        # for reward calculations
        self.gamma = gamma  # discount factor

        # for a single threat field
        self.starting_coords = np.arange(0, 624, 1)
        self.nn_calc = None

    def select_action(self, loc, features):
        state = features[loc]  # features is a (1, 626, 5) vector, so choose the row corresponding to the loc
        sample = random.random()  # random number generator
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(
            -1 * self.steps_done / self.EPS_DECAY
        )  # threshold, based on iterations of the network, so this decreases as the network learns
        self.steps_done += 1
        if sample > eps_threshold:
            self.nn_calc += 1
            with torch.no_grad():
                action = (
                    self.policy_net(state.to(self.device))
                    .max(0)
                    .indices.clone()
                    .detach()
                    .cpu()
                    .numpy()
                )  # choose an action according to the policy network todo: here changed this to min
                return int(action)
        else:
            # return np.random.randint(4)  # return a random action otherwise
            pick_action = np.random.randint(4)
            opposite_actions = np.array([1, 0, 3, 2])
            if self.neighbors.iloc[loc, pick_action + 1] != 625:
                return pick_action
            else:
                return int(opposite_actions[pick_action])

    def find_next_state(self, loc, action, features):
        next_loc = self.neighbors.iloc[loc, action + 1]  # given a known action, find the corresponding location
        next_state = features[next_loc].to(self.device)
        # R(s, a, s') = 4 * (10 - c(s') - b)
        # where b is 1 if not at the goal and 0 at the goal state
        # reward = 5 * (10 - next_state[0].unsqueeze(0).to(self.device))
        # reward = -(next_state[0].unsqueeze(0) + 2 * next_state[1].unsqueeze(0))   # next_state[0].unsqueeze(0).to(self.device) # + next_state[1].to(self.device)    # todo: made this positive
        # - 2 * next_state[1])  # reward associated with the next state
        reward = (
                1.5671 * next_state[0] - 1.1445 * next_state[1] -0.5255 * next_state[2] + 2.9470 * next_state[3]
        ).unsqueeze(0)

        # formatting
        state = features[loc].to(self.device).unsqueeze(0)
        action = torch.from_numpy(np.array([action])).to(self.device).unsqueeze(0)

        if next_loc == 624:
            terminated = False
            finished = True
            next_state = next_state.unsqueeze(0)
            # reward += 100   # cost is zero at the destination

        elif next_loc == 625:
            terminated = True
            finished = False
            # next_state = None
            next_state = next_state.unsqueeze(0)
            # reward += 50
            # reward -= 5

        else:
            terminated = False
            finished = False
            next_state = next_state.unsqueeze(0)
            # reward -= 5

        # formatting
        return terminated, finished, next_state, reward, state, action, next_loc

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return 100

        transitions = self.memory.sample(self.batch_size)  # generate a random sample for training
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
            )  # value at the next state   todo: changed this to a negative

        # want loss between q_{my state} and R + gamma * q_{next state}
        expected_state_action_values = (next_state_values.unsqueeze(1) * self.gamma) + reward_batch.unsqueeze(1)

        loss = self.criterion(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()

        # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        return loss.item()

    def run_q_learning(self, features):
        self.steps_done = 0
        # loss_memory = []
        num_features = len(features)
        possible_actions = np.array([-1, 1, 25, -25])

        cumulative_loss = []

        for episode in range(self.num_epochs):
            # hard update
            if episode % 50 == 0:
                policy_net_state_dict = self.policy_net.state_dict()
                self.target_net.load_state_dict(policy_net_state_dict)

            loc = np.random.randint(624)
            self.nn_calc = 0

            feature_ind = np.random.randint(num_features)
            feature = features[feature_ind]

            path_indexer = 0
            path_num = np.random.randint(len(self.expert_paths))

            loss_memory = []
            for t in count():
                # pick a random place to start
                # feature = features[0]

                if (episode % 15 == 0) and episode < 100:
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

                # if loc in path_history:
                #     next_state = None

                # add the action to memory
                self.memory.push(state, action, next_state, reward)

                # run the optimizer
                loss = self.optimize_model()

                # update the target network with a soft update: θ′ ← τ θ + (1 - τ )θ′
                # target_net_state_dict = self.target_net.state_dict()
                # policy_net_state_dict = self.policy_net.state_dict()
                # for key in policy_net_state_dict:
                #     target_net_state_dict[key] = (policy_net_state_dict[key] * self.tau +
                #                                   target_net_state_dict[key] * (1 - self.tau))
                # self.target_net.load_state_dict(target_net_state_dict)

                # if not loss:
                #     loss = 10
                # else:
                #     pass

                loss_memory.append(loss)
                cumulative_loss.append(loss)

                if loc == 624:
                    log.debug(color='blue', message='Found finish, total iterations: \t' + str(t))
                    break
                elif loc == 625:
                    log.debug(color='red', message='Exited the graph, total iterations: \t' + str(t))
                    break
                elif t > 75:
                    break
                # elif loc in path_history:
                #     break

            log.info(
                "Epoch: \t"
                + str(episode)
                + " \t Final Loss Calculated: \t"
                + str(np.average(np.array(loss_memory)))
            )
            if np.average(np.array(loss_memory)) < self.min_accuracy:
                break
            # log.debug(color='red', message=str(self.nn_calc))
            log.debug(message='Final location \t' + str(loc) + '\t Final Threat \t' + str(feature[loc][0]))
            print(' ')

            # if np.mean(loss_memory[-10:]) < self.min_accuracy:
            #     break

        log.debug(color='red', message='Final loss: \t' + str(np.round(loss, 4)))
        self.loss = loss

        # plot the loss over time
        plt.plot(np.arange(0, len(cumulative_loss), 1), cumulative_loss)
        # self.find_feature_expectation(feature_function=features)
        torch.save(self.policy_net, 'q_learning/policy_net_more_samples.pth')
        self.check_convergence(feature_function=features)

    def check_convergence(self, feature_function):
        coords = np.random.randint(0, 624, size=(40,))
        feature_function = feature_function.view(-1, self.n_observations)

        for coord in coords:
            new_coord = coord
            new_feat = feature_function[new_coord].to(self.device)
            for step in range(100):
                with torch.no_grad():
                    action = (
                        self.policy_net(new_feat).max(0).indices.clone().detach().cpu().numpy()     # todo: this is now min
                    )  # this should be max(1) for multi-threat
                # print("Neural network-chosen action: \t", action)
                # neural network rewards
                new_coord = self.neighbors.iloc[new_coord, int(action) + 1]

                if new_coord == 625:
                    print('terminated outside the graph')
                    break

                elif new_coord == 624:
                    print("Success!")
                    break

                # print("Neural network next coordinate: \t", new_coord)
                new_feat = feature_function[new_coord].to(self.device)

                if step == 99:
                    print('failed')

    def find_feature_expectation(self, feature_function):
        n_threats = 1
        feature_function = feature_function.view(-1, self.n_observations)

        coords = np.tile(self.starting_coords, n_threats)
        coords_conv = np.repeat(626 * np.arange(0, n_threats, 1), len(self.starting_coords))
        # 626 because we've added a 626th row to the feature function for outside the boundary

        my_features = (
            feature_function[coords + coords_conv]
        )  # features at all the starting coordinates
        new_features = copy.deepcopy(my_features).to(self.device)  # feature values to use to decide each action
        # my_features = my_features[:, :4].view(-1, 4).to(self.device)

        errors = 0
        rewards_vec = []
        min_threat_rewards = []

        for coord in coords:
        # for coord in [0, 1]:
            features = feature_function[coord].to(self.device)
            nn_reward = 0
            my_reward = 0

            new_coord = copy.deepcopy(coord)
            new_feat = feature_function[coord].to(self.device)
            # log.debug("Starting coordinate:  \t" + str(coord))
            # print("Starting features: \t", new_feat)

            i = 0

            for step in range(self.path_length - 1):

                i += 1

                with torch.no_grad():
                    action = (
                        self.policy_net(new_feat).max(0).indices.clone().detach().cpu().numpy()
                    )  # this should be max(1) for multi-threat

                # print("Neural network-chosen action: \t", action)
                # neural network rewards
                new_coord = self.neighbors.iloc[new_coord, int(action) + 1]

                if new_coord == 625:
                    break
                # print("Neural network next coordinate: \t", new_coord)
                new_feat = feature_function[new_coord].to(self.device)

                # print("Neural network next feature: \t", new_feat)
                reward = 10 - new_feat[0]
                nn_reward += reward.cpu().numpy()

                # my rewards: moving only toward the minimum threat
                my_action = feature_function[coord].min(0).indices
                # print("Action according to the minimum threat: \t", action)
                coord = self.neighbors.iloc[coord, int(my_action) + 1]
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
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # PARAMETERS

    # DATA PARAMETERS
    # threat field
    target_loc_ = 624  # final location in the threat field
    gamma_ = 1  # discount factor
    path_length_ = 50  # maximum number of points to keep along expert generated paths
    dims = (25, 25)

    # MACHINE LEARNING PARAMETERS
    q_tau = (
        0.8  # rate at which to update the target_net variable inside the Q-learning module
    )
    q_lr = 0.0001  # learning rate
    q_criterion = (
        nn.HuberLoss()
    )  # criterion to determine the loss during training (otherwise try hinge embedding)
    q_batch_size = 500  # batch size
    q_features = 20  # number of features to take into consideration
    q_epochs = 550  # number of epochs to iterate through for Q-learning
    q_accuracy = 0.001  # value to terminate Q-learning (if value is better than this)
    q_memory = 500

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # NEIGHBORS OF FOUR
    neighbors_ = neighbors_of_four(dims=dims, target=target_loc_)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # LOAD THE DATA
    data = pd.read_pickle('expert_demonstrations/single_threat_sample_paths.pkl')
    feature_function_ = data.feature_map.to_numpy()
    feature_function_ = np.concatenate(feature_function_)
    expert_paths_ = data.sample_paths

    log.info("Expert feature average calculated")

    device_ = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    q_learning = DeepQ(
        n_observations=q_features,
        n_actions=4,
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
        path_length=path_length_,
        expert_paths=expert_paths_,
    )

    torch.set_printoptions(linewidth=200)
    feature_function_ = torch.from_numpy(feature_function_).view(1, 626, q_features).float()
    # feature_function_ = np.reshape(feature_function_, (-1, 626, q_features))
    # feature_function = np.abs(feature_function_)
    # print(feature_function)
    q_learning.run_q_learning(features=feature_function_)
