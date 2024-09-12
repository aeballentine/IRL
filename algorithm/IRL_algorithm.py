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
from IRL_architecture import feature_avg, RewardFunction, CustomRewardDataset, WeightClipper
from IRL_utilities import neighbors_of_four
from Qlearning_algorithm import DeepQ, log
import matplotlib.pyplot as plt
import copy

# todo: variable learning rate
# NOTE: CHANGED THE INDICES TO NOW BE THE RIGHT NEIGHBORS OF FOUR: THREAT FIELD FILLS LEFT TO RIGHT AND UP
# TODO: GAMMA GAMMA GAMMA for the discount feature expectation

log.info("Initializing code")
torch.set_printoptions(linewidth=400)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PARAMETERS

# DATA PARAMETERS
# threat field
target_loc = 624  # final location in the threat field
gamma = 0.99  # discount factor
path_length = 10  # maximum number of points to keep along expert generated paths
dims = (25, 25)

# feature dimensions
feature_dims = (
    20  # number of features to take into account (for the reward function)
)

# MACHINE LEARNING PARAMETERS
# reward function
batch_size = 400  # number of samples to take per batch
learning_rate = 0.0001  # learning rate
epochs = 1000  # number of epochs for the main training loop

# value function
tau = (
    0.0001  # rate at which to update the target_net variable inside the Q-learning module
)
LR = 0.25  # learning rate for Q-learning
q_criterion = (
    nn.HuberLoss()
)  # criterion to determine the loss during training (otherwise try hinge embedding)
q_batch_size = 400  # batch size
num_features = 20  # number of features to take into consideration
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
# my_features = torch.zeros(feature_dims)
rewards = RewardFunction(feature_dim=feature_dims).to(device)
criterion = nn.CrossEntropyLoss(weight=torch.tensor([2, 1.6, 0.2, 0.2,
                                                     0.5, 0.4, 0.05, 0.05,
                                                     0.5, 0.4, 0.05, 0.05,
                                                     0.5, 0.4, 0.05, 0.05,
                                                     0.5, 0.4, 0.05, 0.05,]).to(device)
                                )  # criterion to determine the loss
clipper = WeightClipper()
log.info(rewards)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# create the dataloader and the testloader
dataset = CustomRewardDataset(feature_map=feature_function, expert_expectation=feature_averages)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)  # todo: changed this to false

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
        # print(output)
        # print(y)

        loss = criterion(output, y)
        log.debug(message=output[:5])
        log.debug(message=y[:5])
        print(rewards.state_dict())

        loss.requires_grad = True
        loss.backward()
        losses.append(loss.item())
        losses_total.append(loss.item())

        # for param in rewards.parameters():
        #     print(param.requires_grad)
        #
        # for param in rewards.parameters():
        #     if param.grad is None:
        #         print("Grad is None")
        #     else:
        #         print(param.grad)

        # torch.nn.utils.clip_grad_norm_(rewards.parameters(), max_norm=1.0)
        optimizer.step()
        print(rewards.state_dict())
        # rewards.apply(clipper)
        # with torch.no_grad():
        #     for param in rewards.parameters():
        #         param.copy_(param.abs())  # Take absolute value of the weights
        # print(rewards.state_dict())

        if loss.item() < 50:
            new_rate = learning_rate / 1000
        elif loss.item() < 10:
            new_rate = learning_rate / 100000
        else:
            new_rate = learning_rate

        for g in optimizer.param_groups:
            g['lr'] = new_rate

        # log.debug(
        #     color="red",
        #     message="\tEpoch %d | Batch %d | Loss %6.2f"
        #     % (epoch, batch_num, loss.item()),
        # )
    log.debug(
        color="blue",
        message="Epoch %d | Loss %6.4f" % (epoch, sum(losses) / len(losses)),
    )

print(rewards.state_dict())
losses = np.array(losses_total)
plt.plot(losses)
plt.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# save the model parameters
torch.save(rewards, "results/reward_model_updated_two.pth")
torch.save(q_learning.policy_net, "results/policy_model_updated_two.pth")
