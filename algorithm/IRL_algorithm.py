"""
UPDATED
Inverse reinforcement learning: learn the reward function from expert demonstrations
"""

import pandas as pd
import numpy as np
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from IRL_architecture import feature_avg, RewardFunction, CustomRewardDataset, DQN
from IRL_utilities import neighbors_of_four
from Qlearning_algorithm import DeepQ, log
import matplotlib.pyplot as plt

# todo: variable learning rate
# NOTE: CHANGED THE INDICES TO NOW BE THE RIGHT NEIGHBORS OF FOUR: THREAT FIELD FILLS LEFT TO RIGHT AND UP
# TODO: GAMMA GAMMA GAMMA for the discount feature expectation

log.info("Initializing code")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PARAMETERS

# DATA PARAMETERS
# threat field
target_loc = 624  # final location in the threat field
gamma = 0.95  # discount factor
path_length = 25  # maximum number of points to keep along expert generated paths
dims = (25, 25)

# feature dimensions
feature_dims = (
    7  # number of features to take into account (for the reward function and policy)
)

# MACHINE LEARNING PARAMETERS
# reward function
batch_size = 1  # number of samples to take per batch
learning_rate = 1e-4  # learning rate
epochs = 400  # number of epochs for the main training loop
criterion = nn.MSELoss()  # criterion to determine the loss during training

# value function
tau = (
    0.8  # rate at which to update the target_net variable inside the Q-learning module
)
LR = 1e-3  # learning rate for Q-learning
q_criterion = (
    nn.L1Loss()
)  # criterion to determine the loss during training (otherwise hinge embedding)
q_batch_size = 64  # batch size
num_features = 7  # number of features to take into consideration
q_epochs = 100  # number of epochs to iterate through for Q-learning
min_accuracy = 1e-1  # value to terminate Q-learning (if value is better than this)
memory_length = 500

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
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# select the device to use: cpu or mps (mps is faster)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
log.info("The device is: " + str(device))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# constants for the network & initialize the reward model
# my_features = torch.zeros(feature_dims)
rewards = RewardFunction(feature_dim=feature_dims).to(device)
log.info(rewards)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# create the dataloader and the testloader
dataset = CustomRewardDataset(feature_map=feature_function, expert_expectation=feature_averages)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

log.info("The dataloaders are created")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# select the optimizer
optimizer = torch.optim.Adam(rewards.parameters(), lr=learning_rate, amsgrad=True)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# set up the deep Q network
# policy_net = DQN(n_observations=num_features, n_actions=4).to(device)
# target_net = DQN(n_observations=num_features, n_actions=4).to(device)
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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# train the model
rewards.train()
log.info("Beginning training")
losses_total = [np.inf]
for epoch in range(epochs):
    losses = []
    for batch_num, input_data in enumerate(dataloader):
        optimizer.zero_grad()
        x, y = (
            input_data  # x is the threat field and y is the expert average feature expectation
        )
        # x = x.to(device).float()
        # print(x)
        y = y.to(device).float()

        log.info("Beginning Q-learning module")

        # to numpy array: x.clone().detach().cpu().numpy()
        q_learning.reward = rewards

        output = q_learning.run_q_learning(features=x)

        # output = torch.from_numpy(output).float().to(device)
        log.info("Q-learning completed")

        loss = criterion(output, y)
        loss.requires_grad = True
        loss.backward()
        losses.append(loss.item())
        losses_total.append(loss.item())

        optimizer.step()

        # log.debug(
        #     color="red",
        #     message="\tEpoch %d | Batch %d | Loss %6.2f"
        #     % (epoch, batch_num, loss.item()),
        # )
    log.debug(
        color="blue",
        message="Epoch %d | Loss %6.4f" % (epoch, sum(losses) / len(losses)),
    )

losses = np.array(losses_total)
plt.plot(losses)
plt.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# save the model parameters
torch.save(rewards, "../results/reward_model_SINGLE.pth")
torch.save(q_learning.policy_net, "../results/policy_model_SINGLE.pth")
