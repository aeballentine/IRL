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

from IRL_architecture import CustomRewardDataset
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
q_accuracy = 4  # value to terminate Q-learning (if value is better than this)
q_memory = 500     # memory length for Q-learning

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NEIGHBORS OF FOUR
neighbors = neighbors_of_four(dims=dims, target=target_loc)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# LOAD THE DATA
data = pd.read_pickle('expert_demonstrations/single_threat_sample_paths.pkl')

feature_averages = data.expert_feat
feature_function = data.feature_map
threat_fields = data.threat_field
expert_paths = data.sample_paths
test_points = data.test_points[0]

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


class RewardFunction(nn.Module):
    """
    Assuming the reward function is a linear combination of the features
    R(s, a, s') = w^T * phi(s)
    """

    def __init__(self, feature_dim):
        super(RewardFunction, self).__init__()
        # initialize the weights as ones
        self.weights = nn.Parameter(torch.tensor([-4, -5, -7, 2]).float())

    def forward(self, features):
        # return the anticipated reward function
        f1 = torch.matmul(features, self.weights)   # using matmul to allow for 2d inputs
        return f1


rewards = RewardFunction(feature_dim=feature_dims).to(device)
criterion = nn.HuberLoss()
optimizer = torch.optim.Adam(rewards.parameters(), lr=learning_rate, amsgrad=True)
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
    expert_paths=expert_paths,
    starting_coords=test_points,
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# train the model
log.info("Beginning training")

losses_total = [np.inf]
best_loss = np.inf
best_reward = None
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

        output = rewards(output[0, :, :4])
        y = rewards(y[0, :, :4])

        loss = criterion(output, y)
        log.debug(message='Current loss: \t' + str(loss))
        log.debug(message=output[:5])
        log.debug(message=y[:5])
        log.debug(message=rewards.state_dict())

        loss.backward()
        losses.append(loss.item())
        losses_total.append(loss.item())
#
#         # can try this, but parameters look to be independent now
#         # torch.nn.utils.clip_grad_value_(rewards.parameters(), 100)
#
        optimizer.step()
        log.debug(message=rewards.state_dict())

        if loss.item() < best_loss:
            torch.save(rewards, "results/reward_best_model_more_samples.pth")
            torch.save(q_learning.policy_net, "results/policy_model_best_more_samples.pth")
            best_loss = loss.item()
            best_reward = rewards.state_dict()

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
losses = np.array(losses_total)
plt.plot(losses)
plt.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# save the model parameters
torch.save(rewards, "results/reward_model_final_more_samples.pth")
torch.save(q_learning.policy_net, "results/policy_model_final_more_samples.pth")
log.debug("Best loss: \t" + str(best_loss))
log.debug(rewards.state_dict())
