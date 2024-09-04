import glob
import numpy as np
import pandas as pd
from IRL_utilities import neighbors_of_four, to_2d
import pickle


def find_optimal_path(
    value_func,
    threat_map,
    start_index,
    destination,
    neighbor_coords,
    dim=(25, 25),
    disp=True,
    max_points=500,
):
    # value function should be a numpy array with dimensions as specified
    # start and end index should be an array. The index references the value function (1D array)
    my_position = start_index

    # does the path converge?
    success = True

    # value function of the starting point (use this to check the total cost of the path)
    value = value_func[start_index]

    # this will be updated throughout the function. There is no incurred cost from the starting location
    total_cost = 0

    # array to hold the path
    path = [start_index]

    while my_position != destination:
        adjacent = neighbor_coords.iloc[my_position].to_numpy()[1:]
        mini = float("Inf")
        cost = float("Inf")
        for coords in adjacent:
            if value_func[coords] < mini:
                mini = value_func[coords]
                cost = threat_map[coords]
                my_position = coords
            elif value_func[coords] == mini:
                if threat_map[coords] < cost:
                    cost = threat_map[coords]
                    my_position = coords

        path.append(my_position)
        total_cost += threat_map[my_position]
        if len(path) > max_points:
            success = False
            print("Error - path failed")
    if disp:
        print("The final value function: ", np.round(value_func[my_position], 3))
        print("The final coordinate:", to_2d(my_position, dims))
        print("The value function from the initial location:", np.round(value, 3))
        print(
            "The value function adding the cost along the way:", np.round(total_cost, 3)
        )

    if np.abs(total_cost - value) > 1e-3:
        success = False
        # print("Discrepancy: ", np.abs(total_cost - value))
    return path, success


def create_feature_map(my_field, my_neighbors):
    # current threat
    my_threat = np.reshape(my_field[:-1], (625, 1))

    # four neighbors
    left_vals = np.reshape(my_field[my_neighbors.left.to_numpy()], (625, 1))
    right_vals = np.reshape(my_field[my_neighbors.right.to_numpy()], (625, 1))
    up_vals = np.reshape(my_field[my_neighbors.up.to_numpy()], (625, 1))
    down_vals = np.reshape(my_field[my_neighbors.down.to_numpy()], (625, 1))

    # euclidean distance
    x_vals = np.reshape(my_neighbors.x_dist.to_numpy(), (625, 1)) / 12
    y_vals = np.reshape(my_neighbors.y_dist.to_numpy(), (625, 1)) / 12
    distance = (x_vals ** 2 + y_vals ** 2) ** 0.5

    return np.concatenate(
        (my_threat, left_vals, right_vals, up_vals, down_vals, distance), axis=1
    )


def find_feature_expectation(coords, feature_function, discount):
    relevant_features = feature_function[coords]
    relevant_features = relevant_features[:, [0, -1]]
    discount_factor = np.reshape(
        np.array(list(map(lambda x: pow(discount, x), range(len(coords))))),
        (len(coords), 1),
    )

    discount_expectation = discount_factor * relevant_features

    return np.sum(discount_expectation, axis=0)


# parameters of the threat field
dims = (25, 25)  # dimension of the threat field
starting_coords = np.random.randint(0, 623, size=25)  # random points for path planning
end_index = 624  # index of the final location
path_length = 10
gamma = 0.95

# neighbors dataframe: this records all the neighbors of four
neighbors = neighbors_of_four(dims, end_index)

# parameters to load the relevant files
data_path = "cost_function/"
file_list = glob.glob(data_path + "*")

# counters: for failures
failures = 0
counter = 0

# these are the variables we'll save to a dataframe
features = []  # average feature expectation
feature_map = []  # feature map for each point in the threat field
threat_map = []  # need this to map paths for visualization

for file in [file_list[0]]:
    threat = pd.read_csv(file)  # all the data for the threat field

    # save the threat field
    threat_map.append(threat["Threat Intensity"].to_numpy())
    max_threat = 10 * max(threat["Threat Intensity"].to_numpy())

    # formatting for our pathfinder
    threat_field = np.append(threat["Threat Intensity"].to_numpy(), max_threat)
    value_function = np.append(threat["Value"].to_numpy(), np.inf)

    # create the feature map for this threat field
    my_feature_map = create_feature_map(my_field=threat_field, my_neighbors=neighbors)
    feature_map.append(my_feature_map)
    my_features = np.zeros(2)

    for loc in starting_coords:
        path, status = find_optimal_path(
            value_func=value_function,
            threat_map=threat_field,
            start_index=loc,
            destination=end_index,
            disp=False,
            neighbor_coords=neighbors,
        )  # find the optimal path using the threat field and value function
        path = path[:path_length]  # arbitrarily shorten the path

        # find the feature expectation of the path
        my_features += find_feature_expectation(
            coords=path, feature_function=my_feature_map, discount=gamma
        )

        if not status:
            # print("FAILED")
            failures += 1

    my_features /= len(starting_coords)
    features.append(my_features)
print(failures)

# save to a pkl file

expert_information = {
    "expert_feat": features,
    "feature_map": feature_map,
    "threat_field": threat_map,
}
expert_information = pd.DataFrame(expert_information)
expert_information.to_pickle("expert_demonstrations/single_threat.pkl")

print(starting_coords)
# note: most recent call -> starting coords: [43, 150, 232, 509, 474, 483, 347, 358, 112, 147, 338, 452,  92, 204, 391,
# 341, 308, 437, 557, 619, 235, 174, 584, 596, 485]

# single threat field example -> [154, 557,  34, 588, 188, 372, 616, 268, 31, 452, 338, 418, 13, 58, 266, 44, 20, 193,
#  304, 513, 323, 198, 291, 200, 109]
