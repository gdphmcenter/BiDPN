import numpy as np

def oneD_Normalize(data):

    data = data.T

    mins = data.min(0)
    maxs = data.max(0)

    ranges = maxs - mins

    normData = np.zeros(data.shape)

    row = data.shape[0]

    normData = data - np.tile(mins, (row, 1))
    normData = normData / np.tile(ranges, (row, 1))

    return normData.T