import matplotlib.pyplot as plt
import seaborn as sns
from optimal_path import *


def create_dataframe(values, dims):
    # values should be a one-dimensional vector
    # dims is a tuple: number of x points by number of y points

    # reshape the input vector. The input is listed from the lower left corner, left to right, and up
    # reshape the input to the specified dimensions, and then flip the vector
    values = np.flip(np.reshape(values, dims), axis=0)

    # create the names of the columns and the rows. The columns will become the x-axis coordinates and the rows will
    # become the y-axis
    columns = np.round(np.linspace(-1, 1, dims[0]), 2)
    indices = np.round(np.linspace(1, -1, dims[1]), 2)
    ind = []
    for x in indices:
        if np.where(indices == x)[0][0] % 2 == 0:
            ind.append(str(x))
        else:
            ind.append(" ")

    cols = []
    for x in columns:
        if np.where(columns == x)[0][0] % 2 == 0:
            cols.append(str(x))
        else:
            cols.append(" ")
    # indices = [str(x) for x in indices]

    # create a data frame with the x coordinates as the columns, the y coordinates as the indices, and the values as
    # the input vector
    value_map = pd.DataFrame(values, columns=cols, index=ind)

    return value_map, values


# TODO: def sparse_dataframe():


# data = pd.read_csv("cost_function/200.csv")
# threat = data["Threat Intensity"].to_numpy()
# threat = np.flip(np.reshape(threat, (25, 25)), axis=0)
# col_names = np.round(np.linspace(-1, 1, 25), 2)
# col_names = [str(x) for x in col_names]
#
# index_names = np.round(np.linspace(1, -1, 25), 2)
# index_names = [str(x) for x in index_names]
# threat_map = pd.DataFrame(threat, columns=col_names, index=index_names)
#
# value_function = data["Value"].to_numpy()
# value = np.flip(np.reshape(value_function, (25, 25)), axis=0)
#
# val_map = pd.DataFrame(value, columns=col_names, index=index_names)
# val_map.to_csv("value_200")
#
# print(value[24, 0])
#
# start_index = [24, 0]
# end_index = [0, 24]
#
# optimal, cost_4by = optimal_path_fourby(value, threat, start_index, end_index)
#
# # Todo: changed the order of x and y (in 4by)
#
# fig, (ax1) = plt.subplots(1, 1)
# # sns.heatmap(threat_map, annot=False, ax=ax1)
# sns.heatmap(val_map, annot=False, ax=ax1)
# ax1.plot(optimal.x_coord + 0.5, optimal.y_coord + 0.5, lw=3)
# ax1.scatter([24.5], [24.5])
# # ax1.plot([24, 14], [0, 1], lw=3)
# # plt.show()
