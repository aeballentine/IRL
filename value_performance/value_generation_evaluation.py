import seaborn as sns
import math
import torch
from data_formatting import *


def to_2d(location, dims):
    # loc is a one-dimensional index
    # NOTE: this indexes from the lower left corner
    row, col = location // dims[1], location % dims[1]
    return [int(row), int(col)]


dataloader = torch.load("pytorch_models/dataloader_baseline.pth")
testloader = torch.load("pytorch_models/testloader_baseline.pth")
model = torch.load("pytorch_models/model_baseline.pth")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("The device is: ", device)

# # want to pick a random selection of points (say 4)
# num = np.random.randint(0, 625, 50)
# dim = (25, 25)
#
#
# def map_val_to_index(val, dims):
#     col = math.floor(val / dims[1])
#     row = val % dims[0]
#     if row == 0:
#         row = row
#     else:
#         row -= 1
#     return [row, col]

points_1d = np.arange(0, 624, 1)
points = []
for i, point in enumerate(points_1d):
    points.append(to_2d(point, (25, 25)))

points = [
    [19, 18],
    [18, 8],
    [8, 21],
    [14, 20],
    [10, 16],
    [13, 7],
    [2, 7],
    [16, 24],
    [2, 12],
    [7, 10],
    [5, 7],
    [22, 8],
    [23, 21],
    [21, 20],
    [13, 14],
    [20, 16],
    [16, 20],
    [0, 11],
    [15, 19],
    [6, 9],
    [20, 3],
    [2, 17],
    [3, 7],
    [12, 13],
    [5, 9],
    [23, 13],
    [22, 9],
    [5, 0],
    [19, 15],
    [22, 4],
    [12, 20],
    [21, 13],
    [19, 21],
    [22, 5],
    [2, 6],
    [12, 6],
    [10, 12],
    [23, 5],
    [18, 20],
    [7, 4],
    [10, 9],
    [17, 21],
    [16, 11],
    [18, 20],
    [9, 6],
    [6, 15],
    [5, 9],
    [17, 19],
    [4, 18],
    [8, 6],
]

# difference in the initial value functions
error_value_function = []

# error between the value function and the actual cost of the path
error_value_to_cost = []

# error between the analytical value function and actual path:
error_path_actual = []
value_func = []

percent_error = []

# number of times that the path finding algorithms fail
by4_fail = 0

total_paths = 0

for sample, ground_truth in testloader:
    sample = sample
    ground_truth = ground_truth
    break

for i, my_threat in enumerate(sample):
    my_ground_truth = ground_truth[i]
    for loc in points:
        total_paths += 1
        # print(loc)

        # create a prediction for the value function based on the threat field
        threat = my_threat.to(device)
        preds = model(threat)

        # detach the prediction and threat field
        prediction = preds.detach().cpu().numpy()
        threat = threat.detach().cpu().numpy()

        # format the threat field and value functions, note that the *_map is a DataFrame and the other is a
        # 2D numpy array
        dim = (25, 25)
        pred_map, pred = create_dataframe(prediction, dim)
        value_map, value = create_dataframe(my_ground_truth.numpy(), dim)
        threat_map, threat = create_dataframe(threat, dim)

        # find the error between the generated and analytical value functions
        value_ana = value[loc[0], loc[1]]
        value_gen = pred[loc[0], loc[1]]
        error_value_function.append((value_gen - value_ana))

        # set the starting location and the final location
        # this should be the row and column of the value function, etc
        start_index = [loc[0], loc[1]]
        end_index = [0, 24]

        # find the optimal path according to the predicted threat field:
        optimal_generated, costgen_4by, success_4by = optimal_path(
            value_function=pred,
            threat=threat,
            start_index=start_index,
            end_index=end_index,
            eightby=False,
            dim=dim,
            offset=0,
            disp=False,
            max_points=75,
        )

        # find the optimal path according to the actual threat field
        optimal, cost_4by, __ = optimal_path(
            value_function=value,
            threat=threat,
            start_index=start_index,
            end_index=end_index,
            eightby=False,
            dim=dim,
            offset=0,
            disp=False,
            max_points=75,
        )

        if success_4by:
            # find the error between the value function and the path
            error_value_to_cost.append((costgen_4by - value_gen))
            error_path_actual.append(costgen_4by - cost_4by)
            value_func.append(costgen_4by)
            percent_error.append(100 * (costgen_4by - cost_4by) / cost_4by)
            if costgen_4by - cost_4by < 0:
                print("Cost gen smaller: ", costgen_4by - cost_4by)
        else:
            # note that the path failed
            by4_fail += 1

print(total_paths)
# find the mean and standard deviation:
# full threat field:
error_val = np.array(error_value_function, dtype=np.float64)
mean_val = np.round(np.mean(error_val), 3)
std_val = np.round(np.std(error_val), 3)
var_val = np.round(np.var(error_val), 3)

error_cost = np.array(error_value_to_cost, dtype=np.float64)
mean_cost = np.round(np.mean(error_cost), 3)
std_cost = np.round(np.std(error_cost), 3)
var_cost = np.round(np.var(error_cost), 3)

error_path = np.array(error_path_actual, dtype=np.float64)
mean_path = np.round(np.mean(error_path), 3)
std_path = np.round(np.std(error_path), 3)
var_path = np.round(np.var(error_path), 3)

print("Total Paths Calculated:", total_paths)
print("~~~~~~~~~~~~~~~~~~~~~~~")
print("Full Threat Field Input:")
print("Error: Generated Value - Analytical Value:")
print("Mean:", mean_val)
print("Standard Deviation:", std_val)
print("Variance:", var_val)
print("~~~~~~~~~~~~~~~~~~~~~~~")

print("4by: Actual Cost - Generated Value:")
print("Mean:", mean_cost)
print("Standard Deviation:", std_cost)
print("Variance:", var_cost)
print("~~~~~~~~~~~~~~~~~~~~~~~")

print("4by: Actual Cost - Analytical Value:")
print("Mean:", mean_path)
print("Standard Deviation:", std_path)
print("Variance:", var_path)
print("~~~~~~~~~~~~~~~~~~~~~~~")

print("Number of Failures (Neighbors of 4):", by4_fail)

print("~~~~~~~~~~~~~~~~~~~~~~~")
print("~~~~~~~~~~~~~~~~~~~~~~~")

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams['font.size'] = 18

# plot the difference between the neural network and Dijkstra's algorithm (positive means Dijkstra's won)
plt.scatter(value_func, percent_error, c='tab:blue', s=7.5)
plt.xlabel('Minimum Cost', fontdict={'size': 20})
plt.ylabel(r'Percent Error - $J_{NN}$ and $J^*$', fontdict={'size': 20})
plt.ylim([-0.1, 40])
plt.xlim([0, 150])

plt.annotate(r'$\bar{x}$: ' + str(np.round(np.mean(np.array(percent_error)), 3)), xy=(5, 38), horizontalalignment='left', verticalalignment='top', fontsize=15)
plt.annotate(r'$\sigma$: ' + str(np.round(np.std(np.array(percent_error)), 3)), xy=(5, 36), horizontalalignment='left', verticalalignment='top', fontsize=15)
plt.annotate(r'Non-Convergent Paths: ' + str(by4_fail), xy=(5, 34), horizontalalignment='left', verticalalignment='top', fontsize=15)
plt.show()
