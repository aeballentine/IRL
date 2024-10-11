import wandb
from IRL_algorithm_gradient import find_reward_function


if __name__=="__main__":
    # we want to run examples with n=10, 50, 100, 200, 300, 400, 500, 600 points


    # hyperparameters
    # threat field
    target_loc = 624  # final location in the threat field
    gamma = 1  # discount factor
    path_length = 50  # maximum number of points to keep along expert generated paths
    dims = (25, 25)

    # feature dimensions
    feature_dims = (
        4  # number of features to take into account (for the reward function)
    )

    # reward function
    batch_size = 1  # number of samples to take per batch
    learning_rate = 0.01  # learning rate
    epochs = 600  # number of epochs for the main training loop

    # value function
    q_tau = (
        0.9  # rate at which to update the target_net variable inside the Q-learning module
    )
    q_lr = 0.0001  # learning rate for Q-learning
    q_batch_size = 500  # batch size
    q_features = 20  # number of features to take into consideration
    q_epochs = 600  # number of epochs to iterate through for Q-learning
    q_accuracy = 2  # value to terminate Q-learning (if value is better than this)
    q_memory = 500  # memory length for Q-learning
