import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale
from IRL_utilities import neighbors_of_four, to_2d


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
        adjacent = neighbor_coords.iloc[my_position].to_numpy(dtype=np.uint32)[1:5]

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

        path.append(int(my_position))
        total_cost += threat_map[my_position]
        if len(path) > max_points:
            success = False
            print("Error - path failed")
    if disp:
        print("The final value function: ", np.round(value_func[my_position], 3))
        print("The final coordinate:", to_2d(np.array([my_position]), dim))
        print("The value function from the initial location:", np.round(value, 3))
        print(
            "The value function adding the cost along the way:", np.round(total_cost, 3)
        )

    if np.abs(total_cost - value) > 1e-3:
        success = False
        # print("Discrepancy: ", np.abs(total_cost - value))
    return path, success


def create_feature_map(my_field, my_neighbors, grad_x, grad_y):
    # my_field = max(my_field) - my_field
    # normalization on [0, 1]
    my_field = minmax_scale(my_field, feature_range=(0, 1))
    my_field[-1] = 10 * max(my_field)   # last value represents the values outside the threat field, increasing this

    # neighbors
    left_ind = my_neighbors.left.to_numpy()
    right_ind = my_neighbors.right.to_numpy()
    up_ind = my_neighbors.up.to_numpy()
    down_ind = my_neighbors.down.to_numpy()

    # current threat
    my_threat = np.reshape(my_field[:-1], (625, 1))
    left_vals = np.reshape(my_field[left_ind], (625, 1))
    right_vals = np.reshape(my_field[right_ind], (625, 1))
    up_vals = np.reshape(my_field[up_ind], (625, 1))
    down_vals = np.reshape(my_field[down_ind], (625, 1))

    # euclidean distance
    max_distance = max(my_neighbors.dist.to_numpy())
    distance = np.append(my_neighbors.dist.to_numpy(), 2 * max_distance)
    distance = minmax_scale(distance, feature_range=(0, 5))
    left_dist = np.reshape(distance[left_ind], (625, 1))
    right_dist = np.reshape(distance[right_ind], (625, 1))
    up_dist = np.reshape(distance[up_ind], (625, 1))
    down_dist = np.reshape(distance[down_ind], (625, 1))
    distance = np.reshape(distance[:-1], (625, 1))

    # x gradient
    my_gradx = np.reshape(grad_x[:-1], (625, 1))
    left_gradx = np.reshape(grad_x[left_ind], (625, 1))
    right_gradx = np.reshape(grad_x[right_ind], (625, 1))
    up_gradx = np.reshape(grad_x[up_ind], (625, 1))
    down_gradx = np.reshape(grad_x[down_ind], (625, 1))

    # y gradient
    my_grady = np.reshape(grad_y[:-1], (625, 1))
    left_grady = np.reshape(grad_y[left_ind], (625, 1))
    right_grady = np.reshape(grad_y[right_ind], (625, 1))
    up_grady = np.reshape(grad_y[up_ind], (625, 1))
    down_grady = np.reshape(grad_y[down_ind], (625, 1))

    high_threat = max(my_field)  # we already appended this value in the main part of this file
    max_distance = max(distance)[0]  # because the distance decreases toward the final destination
    high_gradx = max(grad_x)
    high_grady = max(grad_y)
    outside_cell = np.array([[high_threat, max_distance,  # high_gradx, high_grady,
                              high_threat, max_distance,  # high_gradx, high_grady,
                              high_threat, max_distance,  # high_gradx, high_grady,
                              high_threat, max_distance,  # high_gradx, high_grady,
                              high_threat, max_distance, ]])  # high_gradx, high_grady,]])

    # want to group by cell, not by value type to make calling the reward function easier
    feature_func = np.concatenate(
        (my_threat, distance,  # my_gradx, my_grady,
         left_vals, left_dist,  # left_gradx, left_grady,
         right_vals, right_dist,  # right_gradx, right_grady,
         up_vals, up_dist,  # up_gradx, up_grady,
         down_vals, down_dist,), axis=1  # down_gradx, down_grady), axis=1
    )
    return np.concatenate((feature_func, outside_cell), axis=0)


def find_feature_expectation(coords, feature_function, discount):
    relevant_features = feature_function[coords]
    relevant_features = relevant_features
    discount_factor = np.reshape(
        np.array(list(map(lambda x: pow(discount, x), range(len(coords))))),
        (len(coords), 1),
    )

    discount_expectation = discount_factor * np.abs(relevant_features)

    if len(discount_expectation) < path_length:
        zeros = np.zeros((1, len(feature_function[0])))
        points_missing = path_length - len(discount_expectation)
        zeros = np.repeat(zeros, points_missing, axis=0)
        discount_expectation = np.concatenate((discount_expectation, zeros))

    # return np.sum(discount_expectation, axis=0)
    return discount_expectation


if __name__ == "__main__":
    # parameters of the threat field
    dims = (25, 25)  # dimension of the threat field
    # starting_coords = np.random.randint(0, 623, size=25)  # random points for path planning
    starting_coords = [341, 126, 26, 620, 299, 208, 148, 150, 27, 302, 134, 460, 513, 200, 1, 598, 69, 309,
                       111, 504, 393, 588, 83, 27, 250]
    end_index = 624  # index of the final location

    path_length = 10  # maximum number of points to keep
    gamma = 1  # discount factor

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

    for file in file_list[:50]:
        threat = pd.read_csv(file)  # all the data for the threat field

        # save the threat field
        threat_map.append(threat["Threat Intensity"].to_numpy())
        max_threat = max(threat["Threat Intensity"].to_numpy())

        # formatting for our pathfinder
        threat_field = np.append(threat["Threat Intensity"].to_numpy(), max_threat)
        value_function = np.append(threat["Value"].to_numpy(), np.inf)
        grad_x1 = np.append(threat['Threat Gradient x_1'], 0)
        grad_x2 = np.append(threat['Threat Gradient x_2'], 0)

        # create the feature map for this threat field
        my_feature_map = create_feature_map(my_field=threat_field, my_neighbors=neighbors, grad_x=grad_x1,
                                            grad_y=grad_x2)
        feature_map.append(my_feature_map)
        my_features = []

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
            my_features.append(find_feature_expectation(
                coords=path, feature_function=my_feature_map, discount=gamma
            ))

            if not status:
                # print("FAILED")
                failures += 1

        # my_features /= len(starting_coords)
        my_features = np.concatenate(my_features, axis=0)
        features.append(my_features)
    print(failures)  # NOTE: not sure why these are failures: there are very small discrepancies

    # save to a pkl file
    expert_information = {
        "expert_feat": features,
        "feature_map": feature_map,
        "threat_field": threat_map,
    }
    expert_information = pd.DataFrame(expert_information)
    expert_information.to_pickle("expert_demonstrations/multi_threat.pkl")

    # print(', '.join(map(lambda x: str(x), starting_coords)))
