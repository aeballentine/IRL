"""
UPDATED
Inverse reinforcement learning: learn the reward function from expert demonstrations
"""

import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import datetime

from IRL_architecture import CustomRewardDataset
from create_expert_demonstrations import get_expert_demos
from IRL_utilities import neighbors_of_four
from Qlearning_algorithm import DeepQ, log
from evaluation_dijkstra import dijkstra_evaluation

log.info("Initializing code")
torch.set_printoptions(linewidth=800)
print(datetime.datetime.now())
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
learning_rate = 0.01  # learning rate
epochs = 400  # number of epochs for the main training loop
criterion = nn.HuberLoss()

# value function
q_tau = (
    0.8  # rate at which to update the target_net variable inside the Q-learning module
)
q_lr = 0.0001  # learning rate for Q-learning
q_criterion = (
    nn.HuberLoss()
)  # criterion to determine the loss during training (otherwise try hinge embedding)
q_batch_size = 400  # batch size
q_features = 20  # number of features to take into consideration
q_epochs = 400  # number of epochs to iterate through for Q-learning
q_accuracy = 2  # value to terminate Q-learning (if value is better than this)
q_memory = 750  # memory length for Q-learning

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NEIGHBORS OF FOUR
neighbors = neighbors_of_four(dims=dims, target=target_loc)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# REWARD FUNCTION CLASS
class RewardFunction(nn.Module):
    """
    Assuming the reward function is a linear combination of the features
    R(s, a, s') = w^T * phi(s)
    """

    def __init__(self, feature_dim):
        super(RewardFunction, self).__init__()
        # initialize the weights as ones
        self.weights = nn.Parameter(torch.tensor([-2, -2, -3, 1]).float())

    def forward(self, features):
        # return the anticipated reward function
        f1 = torch.matmul(features, self.weights ** 2)  # using matmul to allow for 2d inputs
        return -f1


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# select the device to use: cpu or mps (mps is faster)
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
log.info("The device is: " + str(device))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# WEIGHTS AND BIASES
num_sample_points = [10, 50, 100, 200, 300, 400, 500, 600]
wandb.login(key='77fd51534f63a49b4afb5879ce07f92f39d9e590')
# wandb.login()

for num in num_sample_points:
    run = wandb.init(project='inverse-reinforcement-learning',
                     name=str(num) + '-sample-points',
                     config={
                         'gamma': 1,
                         'reward_learning_rate': learning_rate,
                         'reward_epochs': epochs,
                         'reward_features': feature_dims,
                         'q_learning_rate': q_lr,
                         'q_batch_size': q_batch_size,
                         'q_memory': q_memory,
                         'q_epochs': q_epochs,
                         'q_tau': q_tau,
                         'q_features': q_features
                     })

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # LOAD THE DATA
    data = get_expert_demos(num_paths=num, training_percent=0.1)

    feature_averages = data.expert_feat
    feature_function = data.feature_map
    threat_fields = data.threat_field
    expert_paths = data.sample_paths
    test_points = data.test_points[0]

    log.info("Expert feature average calculated")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # constants for the network & initialize the reward model
    rewards = RewardFunction(feature_dim=feature_dims).to(device)

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
    for epoch in range(epochs):
        for batch_num, input_data in enumerate(dataloader):
            x, y = (
                input_data  # x is the threat field and y is the expert average feature expectation
            )

            y = y.to(device).float()
            q_learning.reward = rewards
            output, q_learning_loss, q_learning_failures, q_learning_finishes = q_learning.run_q_learning(features=x)
            # if loss > 100:
            #     continue

            total_cost_NN = torch.sum(output[0, :, 0]).unsqueeze(0)
            total_cost_ideal = torch.sum(y[0, :, 0]).unsqueeze(0)

            output = rewards(output[0, :, :4])
            output = torch.concat([output, total_cost_NN])
            y = rewards(y[0, :, :4])
            y = torch.concat([y, total_cost_ideal])

            loss = criterion(output, y)

            loss.backward()
            wandb.log({'reward_loss': loss, 'final_q_learning_loss': q_learning_loss,
                       'q_learning_failures': q_learning_failures, 'q_learning_finishes': q_learning_finishes,
                       'rewards_values_threat': rewards.weights.cpu().detach().numpy()[0],
                       'rewards_values_distance': rewards.weights.cpu().detach().numpy()[1],
                       'rewards_values_grad1': rewards.weights.cpu().detach().numpy()[2],
                       'rewards_values_grad2': rewards.weights.cpu().detach().numpy()[3]})

            optimizer.step()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # evaluate against Dijkstra's
    dijkstra_evaluation(policy_net=q_learning.policy_net, device=device,
                        feature_function_=feature_function[0], neighbors=neighbors)
    run.finish()
