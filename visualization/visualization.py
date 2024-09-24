import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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


data = pd.read_pickle('multi_threat.pkl')
threat_field = data.threat_field.to_numpy()[0]
# threat_field = np.arange(0, 625, 1)
# print(threat_field)
threat_field, _ = create_dataframe(threat_field, (25, 25))

sns.heatmap(threat_field, cbar=True, annot=False, fmt='g')
plt.show()
# print(threat_field)
