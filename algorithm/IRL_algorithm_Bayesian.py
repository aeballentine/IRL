"""
UPDATED
Inverse reinforcement learning: learn the reward function from expert demonstrations
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from skopt import gp_minimize

from IRL_architecture import RewardFunction, CustomRewardDataset
from IRL_utilities import neighbors_of_four
from Qlearning_algorithm import DeepQ, log
import matplotlib.pyplot as plt

log.info("Initializing code")
torch.set_printoptions(linewidth=800)

# QUESTION: why does the loss decrease when there are more paths that terminate - let this run more and then look into
# it again once we have a larger sample set
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PARAMETERS

# DATA PARAMETERS
# threat field
target_loc = 624  # final location in the threat field
gamma = 0.9  # discount factor
path_length = 50  # maximum number of points to keep along expert generated paths
dims = (25, 25)

# feature dimensions
feature_dims = (
    4  # number of features to take into account (for the reward function)
)

# MACHINE LEARNING PARAMETERS
# reward function
batch_size = 1  # number of samples to take per batch
learning_rate = 0.01   # learning rate
epochs = 1000  # number of epochs for the main training loop

# value function
q_tau = (
    0.9  # rate at which to update the target_net variable inside the Q-learning module
)
q_lr = 0.001  # learning rate for Q-learning
q_criterion = (
    nn.HuberLoss()
)  # criterion to determine the loss during training (otherwise try hinge embedding)
q_batch_size = 500  # batch size
q_features = 20  # number of features to take into consideration
q_epochs = 550  # number of epochs to iterate through for Q-learning
q_accuracy = 0.1  # value to terminate Q-learning (if value is better than this)
q_memory = 500     # memory length for Q-learning

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NEIGHBORS OF FOUR
neighbors = neighbors_of_four(dims=dims, target=target_loc)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# LOAD THE DATA
data = pd.read_pickle('expert_demonstrations/single_threat_long_path.pkl')

feature_averages = data.expert_feat
feature_function = data.feature_map
threat_fields = data.threat_field
expert_paths = data.expert_paths

log.info("Expert feature average calculated")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# select the device to use: cpu or mps (mps is faster)
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
log.info("The device is: " + str(device))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# constants for the network & initialize the reward model
rewards = RewardFunction(feature_dim=feature_dims, device=device)
# criterion = nn.CrossEntropyLoss().to(device)   # weight=torch.tensor([2, 1.6, 0.2, 0.2,
                                               #         0.5, 0.4, 0.05, 0.05,
                                               #         0.5, 0.4, 0.05, 0.05,
                                               #         0.5, 0.4, 0.05, 0.05,
                                               #         0.5, 0.4, 0.05, 0.05,]).to(device)
                                # )  # criterion to determine the loss
criterion = nn.HuberLoss()
log.info(rewards)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# create the dataloader and the testloader
dataset = CustomRewardDataset(feature_map=feature_function, expert_expectation=feature_averages)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)  # todo: changed this to false

log.info("The dataloaders are created")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# set up the deep Q network
q_learning = DeepQ(
    n_observations=q_features,
    n_actions=4,
    device=device,
    LR=q_lr,
    neighbors=neighbors,
    gamma=gamma,
    target_loc=target_loc,
    min_accuracy=q_accuracy,
    memory_length=q_memory,
    tau=q_tau,
    num_epochs=q_epochs,
    batch_size=q_batch_size,
    criterion=q_criterion,
    path_length=path_length,
    expert_paths=expert_paths
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# train the model
# log.info("Beginning training")
#
#
# feature_function = torch.from_numpy(feature_function[0]).float().view(1, 626, q_features)
# feature_averages = torch.from_numpy(feature_averages[0]).float().to(device).view(1, 1250, q_features)
#
#
# def obj(x):
#     rewards.weight_update(x)
#     q_learning.reward = rewards
#     output = q_learning.run_q_learning(features=feature_function)
#     loss = criterion(output, feature_averages)
#     log.debug(
#         color="blue",
#         message="Current Performance: %6.4f" % (loss.item()),
#     )
#     return loss.item()
#
#
# res = gp_minimize(obj,
#                   [(-10, 10), (-10, 10), (-10, 10), (-10, 10)],
#                   n_calls=100, n_random_starts=20)
# print(res.x)
# print(res.fun)

losses_total = [np.inf]
for epoch in range(epochs):
    losses = []
    for batch_num, input_data in enumerate(dataloader):
        x, y = (
            input_data  # x is the threat field and y is the expert average feature expectation
        )

        y = y.to(device).float()

        log.info("Beginning Q-learning module")
        q_learning.reward = rewards
        output = q_learning.run_q_learning(features=x)
        log.info("Q-learning completed")

        loss = criterion(output, y)
        log.info(message=output[:5])
        log.info(message=y[:5])
        log.debug(message=rewards.state_dict())

        loss.requires_grad = True
        loss.backward()
        losses.append(loss.item())
        losses_total.append(loss.item())
#
#         # can try this, but parameters look to be independent now
#         # torch.nn.utils.clip_grad_value_(rewards.parameters(), 100)
#
        optimizer.step()
        log.debug(message=rewards.state_dict())
#
#         # # variable learning rate
#         # if loss.item() < 50:
#         #     new_rate = learning_rate / 100
#         # elif loss.item() < 10:
#         #     new_rate = learning_rate / 1000
#         # else:
#         #     new_rate = learning_rate
#         #
#         # # update the learning rate accordingly
#         # for g in optimizer.param_groups:
#         #     g['lr'] = new_rate
#
#         log.debug(
#             color="blue",
#             message="Epoch %d | Loss %6.4f" % (epoch, sum(losses) / len(losses)),
#         )
#
# log.info(message=rewards.state_dict())
# losses = np.array(losses_total)
# plt.plot(losses)
# plt.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# save the model parameters
# torch.save(rewards, "results/reward_model_updated_two.pth")
# torch.save(q_learning.policy_net, "results/policy_model_updated_two.pth")